"""ADR-0026 paper-only collar recommendations and pre-wall research input validation."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from types import MappingProxyType
from typing import Any

from .lifecycle import STUDY_WALL_DATE
from .tca_queue_parity import (
    DRIFT_WARN,
    FORBIDDEN_LIVE_LANES,
    PAPER_ACCOUNT_LANES,
    PARITY_GREEN,
    TCA_QUEUE_PARITY_SCHEMA,
    THROTTLE_RED,
    _canonical_sha256,
    _require_count,
    _require_finite_real,
    _require_label,
    _require_prewall_date,
    _round_derived,
)

COLLAR_RECALIBRATION_SCHEMA = "sota.collar-recalibration.v1"
POST_WALL_PROVENANCE = "POST_WALL_LIVE_PAPER_ADVISORY"
ALLOWED_PAPER_LANES = frozenset(PAPER_ACCOUNT_LANES)
_SESSION_CEILINGS = MappingProxyType({"RTH": 25.0, "OVERNIGHT_BOATS": 35.0, "OVERNIGHT_23X5_SIP": 35.0})
_WALL = datetime(2026, 10, 16, 4, tzinfo=UTC)
_UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_WALL_NS = ((_WALL - _UNIX_EPOCH) // timedelta(microseconds=1)) * 1000
_ISO_INSTANT = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(Z|[+-]\d{2}:\d{2})")
_PREWALL_SOURCES = frozenset({"PRE_WALL_HISTORICAL", "PRE_WALL_SIMULATION"})


def _instant_ns(value: datetime | str | int) -> int:
    if type(value) is int:
        if 0 < value < 10**17:
            raise ValueError("ambiguous integer timestamp: supply Unix nanoseconds or an explicit-offset timestamp")
        return value
    fraction_ns = 0
    if isinstance(value, str):
        match = _ISO_INSTANT.fullmatch(value)
        if match is None:
            raise ValueError("expected an ISO timestamp with an explicit UTC offset")
        clock, fraction, offset = match.groups()
        if offset != "Z" and (int(offset[1:3]) > 23 or int(offset[4:]) > 59):
            raise ValueError("invalid UTC offset")
        value = datetime.fromisoformat(clock + offset)
        fraction_ns = int((fraction or "").ljust(9, "0"))
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("timezone-aware timestamp or integer Unix nanoseconds required")
    delta = value - _UNIX_EPOCH
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000 + fraction_ns


def _reject_advisory(value: object) -> None:
    if isinstance(value, CollarDecision):
        raise ValueError("post-wall advisory is not a discovery input")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in ("provenance_tag", "source_provenance") and (
                not isinstance(child, str) or child not in _PREWALL_SOURCES
            ):
                raise ValueError("unverified discovery source provenance")
            if key in ("schema_version", "provenance_tag", "source_provenance", "paper_policy") and child in (
                COLLAR_RECALIBRATION_SCHEMA,
                POST_WALL_PROVENANCE,
                "NEVER_GATED_ADVISORY",
            ):
                raise ValueError("post-wall provenance in discovery record")
            _reject_advisory(child)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_advisory(item)


def validate_prewall_temporal_boundary(
    event_ts_utc: datetime | str | int,
    label_maturity_ts_utc: datetime | str | int,
    source_provenance: object,
    *,
    source_record: object = None,
    receipt_ts_utc: datetime | str | int | None = None,
    decision_ts_utc: datetime | str | int | None = None,
    trade_date: str | None = None,
) -> None:
    """Raise before building discovery features or labels from a causally ineligible record."""
    if not isinstance(source_provenance, str) or source_provenance not in _PREWALL_SOURCES:
        raise ValueError("explicit pre-wall historical or simulation provenance required")
    if source_record is not None:
        if not isinstance(source_record, Mapping):
            raise ValueError("a discovery source record must be a decoded historical mapping")
        _reject_advisory(source_record)
    event, maturity = _instant_ns(event_ts_utc), _instant_ns(label_maturity_ts_utc)
    if not 0 <= event <= maturity < _WALL_NS:
        raise ValueError("event and label must mature inside the pre-wall interval")
    if receipt_ts_utc is not None or decision_ts_utc is not None:
        if receipt_ts_utc is None or decision_ts_utc is None:
            raise ValueError("receipt and decision timestamps must be provided together")
        if not event <= _instant_ns(receipt_ts_utc) <= _instant_ns(decision_ts_utc) <= maturity:
            raise ValueError("knowledge time or decision outside the causal interval")
    if trade_date is not None:
        _require_prewall_date(trade_date, "trade_date")


@dataclass(frozen=True, slots=True)
class CollarStratum:
    strategy_id: str
    account_lane: str
    execution_session: str
    side: str
    liquidity_regime: str
    collar_version: str

    def __post_init__(self) -> None:
        for name in ("strategy_id", "liquidity_regime", "collar_version"):
            _require_label(getattr(self, name), name)
        if self.account_lane in FORBIDDEN_LIVE_LANES or self.account_lane not in ALLOWED_PAPER_LANES:
            raise ValueError("only a known paper lane is eligible")
        if self.execution_session not in _SESSION_CEILINGS or self.side not in ("BUY", "SELL"):
            raise ValueError("unsupported execution session or side")

    @property
    def stream(self) -> str:
        # The spending clock and cohort watermark span all policy versions.
        return _json([self.strategy_id, self.account_lane, self.execution_session, self.side, self.liquidity_regime])


@dataclass(frozen=True, slots=True)
class CollarBaseline:
    collar_bps: float
    max_event_date: str
    sim_engine: str
    snapshot_sha256: str
    max_collar_bps: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "collar_bps", _require_finite_real(self.collar_bps, "collar_bps"))
        object.__setattr__(self, "max_event_date", _require_prewall_date(self.max_event_date, "max_event_date"))
        _require_label(self.sim_engine, "sim_engine")
        if not isinstance(self.snapshot_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.snapshot_sha256):
            raise ValueError("expected the SHA-256 of the frozen baseline snapshot")
        if self.max_collar_bps is not None:
            cap = _require_finite_real(self.max_collar_bps, "max_collar_bps")
            if cap <= 0:
                raise ValueError("max_collar_bps must be positive")
            object.__setattr__(self, "max_collar_bps", cap)


@dataclass(frozen=True, slots=True)
class SessionBlock:
    """One independent completed session with unique fill weights and mean matched slippage drift."""

    session_date: str
    fill_weights: tuple[float, ...]
    slippage_drift_bps: float


@dataclass(frozen=True, slots=True)
class CollarEpoch:
    stratum: CollarStratum
    epoch_receipt_id: str
    epoch_index: int
    observed_collar_version: str
    observed_collar_bps: float
    blocks: tuple[SessionBlock, ...]
    stale_inputs: bool
    denominators_complete: bool
    direction_supported: bool
    consecutive_windows: int
    sampling_interval_s: float
    max_markout_lag_60s_s: float
    parity_receipt_sha256: str
    baseline_snapshot_sha256: str
    rho_u: float = 1.0


def effective_sample_size(block_weights: Sequence[Sequence[float]], rho_u: float = 1.0) -> Fraction:
    """Exact weight/design-effect ratio, invariant to permutations, without float square overflow."""
    rho = Fraction(_require_finite_real(rho_u, "rho_u"))
    if not 0 <= rho <= 1:
        raise ValueError("rho_u must be in [0, 1]")
    total = within = between = Fraction(0)
    for block in block_weights:
        weights = [Fraction(_require_finite_real(w, "fill weight")) for w in block]
        if not weights or min(weights) <= 0:
            raise ValueError("each independent session block needs positive finite weights")
        subtotal = sum(weights, Fraction(0))
        total += subtotal
        within += sum((w * w for w in weights), Fraction(0))
        between += subtotal * subtotal
    return total * total / ((1 - rho) * within + rho * between) if total else Fraction(0)


def session_sign_test(adverse_blocks: int, blocks: int, epoch_index: int) -> tuple[Fraction, Fraction]:
    """Exact one-sided binomial p and lifetime alpha allocation, even below float resolution."""
    x, g, k = (
        _require_count(v, name)
        for v, name in ((adverse_blocks, "adverse_blocks"), (blocks, "blocks"), (epoch_index, "epoch_index"))
    )
    if not 0 <= x <= g or k == 0:
        raise ValueError("require 0 <= X <= G and k >= 1")
    denominator = 1 << g
    # Symmetry permits summing the shorter lower tail without a loss of precision.
    short_tail = min(x, g - x + 1)
    mass, term = 0, 1
    for j in range(short_tail):
        mass += term
        term = term * (g - j) // (j + 1)
    numerator = denominator - mass if x <= g - x + 1 else mass
    return Fraction(numerator, denominator), Fraction(1, 20 * k * (k + 1))


@dataclass(frozen=True, slots=True)
class CollarProgress:
    collar_bps: float
    adverse_streak: int = 0
    green_streak: int = 0
    alarm_active: bool = False


def _advance(prev: CollarProgress, anchor: float, signal: str) -> CollarProgress:
    if signal == THROTTLE_RED:
        if prev.adverse_streak == 1:
            return CollarProgress(max(5.0, anchor - 4.0, prev.collar_bps - 2.0), 0, 0, True)
        return CollarProgress(prev.collar_bps, 1, 0, prev.alarm_active)
    green = min(2, prev.green_streak + 1) if signal == PARITY_GREEN else 0
    return CollarProgress(prev.collar_bps, 0, green, prev.alarm_active and green < 2)


@dataclass(frozen=True, slots=True)
class CollarDecision:
    stratum: CollarStratum
    epoch_receipt_id: str
    epoch_index: int
    c_0_bps: float
    c_prev_bps: float
    c_out_bps: float
    ceiling_bps: float
    action: str
    reason: str
    adverse_streak: int
    green_streak: int
    alarm_active: bool
    last_session: str
    n_eff: float
    p_value: float | None
    alpha_k: float
    baseline_sha256: str
    parity_receipt_sha256: str | None
    receipt_sha256: str = ""
    schema_version: str = COLLAR_RECALIBRATION_SCHEMA
    provenance_tag: str = POST_WALL_PROVENANCE
    paper_policy: str = "NEVER_GATED_ADVISORY"
    halts_paper_farm: bool = False
    stop_distance_mutated: bool = False


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _input_value(value: Any, *, preserve_zero_sign: bool = False) -> Any:
    """Encode even nonfinite observations for identity without allowing nonfinite receipt JSON."""
    if isinstance(value, float):
        return {"float": (value if preserve_zero_sign else value or 0.0).hex()}
    if isinstance(value, SessionBlock):
        weights = _input_value(value.fill_weights)
        if isinstance(weights, list):
            weights.sort(key=_json)
        return {"session_block": [_input_value(value.session_date), weights, _input_value(value.slippage_drift_bps)]}
    if isinstance(value, Mapping):
        return {
            "mapping": {key: _input_value(child, preserve_zero_sign=preserve_zero_sign) for key, child in value.items()}
        }
    if isinstance(value, (list, tuple)):
        return [_input_value(child, preserve_zero_sign=preserve_zero_sign) for child in value]
    return value


def _input_sha256(epoch: CollarEpoch, fleet: Mapping[str, Any]) -> str:
    payload = {f.name: _input_value(getattr(epoch, f.name)) for f in fields(epoch) if f.name != "stratum"}
    payload["stratum"] = asdict(epoch.stratum)
    if isinstance(payload["blocks"], list):
        payload["blocks"].sort(key=_json)
    # Preserve the numeric representation covered by the producer's checksums.
    fleet_payload = _input_value(fleet, preserve_zero_sign=True)
    if isinstance(fleet, Mapping) and isinstance(rows := fleet_payload["mapping"].get("rows"), list):
        rows.sort(key=_json)
    return _canonical_sha256({"epoch": payload, "fleet": fleet_payload})


def _read_decision(body: str) -> CollarDecision:
    fields = json.loads(body)
    digest = fields.pop("receipt_sha256")
    if digest != _canonical_sha256(fields):
        raise ValueError("stored collar decision checksum mismatch")
    fields["stratum"] = CollarStratum(**fields["stratum"])
    return CollarDecision(**fields, receipt_sha256=digest)


@dataclass(slots=True)
class _Evidence:
    reason: str = "INSUFFICIENT_EVIDENCE"
    signal: str = "HOLD"
    n_eff: float = 0.0
    p: float | None = None
    last_session: str = ""
    row_hash: str | None = None


def _evaluate(
    epoch: CollarEpoch,
    fleet: Mapping[str, Any],
    prev: CollarProgress,
    baseline: CollarBaseline,
    ceiling: float,
    last_session: str,
) -> _Evidence:
    result = _Evidence(last_session=last_session)
    if not 5 <= baseline.collar_bps <= ceiling:
        result.reason = "OBSERVATIONAL_OUT_OF_DOMAIN"
        return result
    scope = epoch.stratum
    if epoch.observed_collar_version != scope.collar_version or (
        _require_finite_real(epoch.observed_collar_bps, "observed collar") != prev.collar_bps
    ):
        result.reason = "UNACKNOWLEDGED_COLLAR"
        return result
    if (
        epoch.stale_inputs is not False
        or epoch.denominators_complete is not True
        or type(epoch.direction_supported) is not bool
        or _require_count(epoch.consecutive_windows, "windows") < 3
    ):
        return result
    if not (
        0 < _require_finite_real(epoch.sampling_interval_s, "sampling interval") <= 60
        and 0 <= _require_finite_real(epoch.max_markout_lag_60s_s, "60s markout lag") <= 6
    ):
        result.reason = "MARKOUT_TIMING"
        return result
    if (
        fleet["schema_version"] != TCA_QUEUE_PARITY_SCHEMA
        or fleet["record_type"] != "fleet_summary"
        or fleet["paper_policy"] != "NEVER_GATED_ADVISORY"
        or fleet["halts_paper_farm"] is not False
    ):
        raise ValueError("incompatible fleet summary")
    rows = [
        r for r in fleet["rows"] if r["strategy_id"] == scope.strategy_id and r["account_lane"] == scope.account_lane
    ]
    if len(rows) != 1:
        result.reason = "NO_UNIQUE_MATCHED_LANE"
        return result
    row = dict(rows[0])
    digest = row.pop("receipt_sha256")
    if (
        digest != epoch.parity_receipt_sha256
        or epoch.baseline_snapshot_sha256 != baseline.snapshot_sha256
        or digest != _canonical_sha256(row)
        or row["record_type"] != "lane_reconciliation"
        or row["schema_version"] != TCA_QUEUE_PARITY_SCHEMA
        or row["halts_paper_farm"] is not False
        or row["paper_policy"] != "NEVER_GATED_ADVISORY"
        or row["prewall_cutoff_date"] != baseline.max_event_date
    ):
        raise ValueError("unverified parity row")
    result.row_hash = digest
    dates = [_session_date(b.session_date) for b in epoch.blocks]
    if len(set(dates)) != len(dates) or (dates and min(dates) <= last_session):
        result.reason = "OVERLAPPING_OR_OLD_COHORT"
        return result
    weights = [b.fill_weights for b in epoch.blocks]
    n = _require_count(row["live_sample_n"], "live_sample_n")
    if n != sum(map(len, weights)):
        result.reason = "FILL_COUNT_MISMATCH"
        return result
    ess = effective_sample_size(weights, epoch.rho_u)
    result.n_eff = float(ess)
    signs = [
        _round_derived(_require_finite_real(b.slippage_drift_bps, "block drift"), "block drift") > 10.0
        for b in epoch.blocks
    ]
    p, alpha = session_sign_test(sum(signs), len(signs), epoch.epoch_index)
    result.p = float(p)
    if dates:
        result.last_session = max(dates)
    if n < 30 or len(dates) < 3 or ess < 20:
        result.reason = "INSUFFICIENT_SAMPLE"
        return result
    status = row["parity_status"]
    drift = _require_finite_real(row["slippage_drift_bps"], "drift")
    gap = _require_finite_real(row["fill_rate_gap_ratio"], "fill-rate gap")
    markout = _require_finite_real(row["live_markout_60s_bps"], "markout")
    edge = _require_finite_real(row["eb_shrunk_net_edge_bps"], "net edge")
    if abs(gap) > 0.15 or markout < -3 or edge <= 0:
        result.reason = "DIRECTION_UNRESOLVED"
    elif status == PARITY_GREEN and drift <= 4:
        result.signal, result.reason = PARITY_GREEN, "ELIGIBLE_GREEN"
    elif status == THROTTLE_RED and drift > 10 and epoch.direction_supported and p <= alpha:
        result.signal, result.reason = THROTTLE_RED, "QUALIFIED_ADVERSE"
    elif status == DRIFT_WARN:
        result.reason = "DRIFT_WARN"
    else:
        result.reason = "ADVERSE_UNCONFIRMED"
    return result


def _session_date(value: str) -> str:
    parsed = date.fromisoformat(value)
    if parsed <= STUDY_WALL_DATE or parsed.isoformat() != value:
        raise ValueError("expected a canonical post-wall session date")
    return value


class CollarLedger:
    """Transactional paper journal on a dedicated caller-owned SQLite connection.

    Use separate connections to the same private database for concurrent workers.
    Construction never opens a path, creates broker clients, or reads a research lake.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._db = connection
        with self._transaction():
            connection.execute("""CREATE TABLE IF NOT EXISTS collar_policies (
                stream TEXT NOT NULL, version TEXT NOT NULL, baseline TEXT NOT NULL,
                UNIQUE(stream, version))""")
            connection.execute("""CREATE TABLE IF NOT EXISTS collar_decisions (
                stream TEXT NOT NULL, k INTEGER NOT NULL, receipt_id TEXT NOT NULL, decision TEXT NOT NULL,
                input_sha256 TEXT, PRIMARY KEY(stream, k), UNIQUE(stream, receipt_id))""")
            if "input_sha256" not in {row[1] for row in connection.execute("PRAGMA table_info(collar_decisions)")}:
                connection.execute("ALTER TABLE collar_decisions ADD COLUMN input_sha256 TEXT")
            connection.execute("""CREATE TABLE IF NOT EXISTS collar_lane_steps (
                lane TEXT NOT NULL, session_date TEXT NOT NULL, PRIMARY KEY(lane, session_date))""")

    @contextmanager
    def _transaction(self) -> Generator[None, None, None]:
        if self._db.in_transaction:
            raise ValueError("a dedicated idle SQLite connection is required")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise

    def _active(self, stream: str) -> tuple[str, str] | None:
        return self._db.execute(
            "SELECT version, baseline FROM collar_policies WHERE stream=? ORDER BY rowid DESC LIMIT 1",
            (stream,),
        ).fetchone()

    def register(
        self, stratum: CollarStratum, baseline: CollarBaseline, *, previous_version: str | None = None
    ) -> None:
        """Commission a frozen private anchor. Version changes require an explicit predecessor."""
        if not isinstance(stratum, CollarStratum) or not isinstance(baseline, CollarBaseline):
            raise ValueError("expected a collar stratum and a pre-wall baseline")
        body = _json(asdict(baseline))
        with self._transaction():
            current = self._active(stratum.stream)
            if current == (stratum.collar_version, body):
                return
            if (current[0] if current else None) != previous_version:
                raise ValueError("baseline version compare-and-swap conflict")
            if self._db.execute(
                "SELECT 1 FROM collar_policies WHERE stream=? AND version=?", (stratum.stream, stratum.collar_version)
            ).fetchone():
                raise ValueError("a policy version cannot be reused or re-anchored")
            self._db.execute(
                "INSERT INTO collar_policies VALUES (?, ?, ?)", (stratum.stream, stratum.collar_version, body)
            )

    def consume(self, epoch: CollarEpoch, fleet: Mapping[str, Any]) -> CollarDecision:
        """Consume one prospective epoch, or return a prior decision for an identical request."""
        if not isinstance(epoch, CollarEpoch) or not isinstance(epoch.stratum, CollarStratum):
            raise ValueError("expected an epoch bound to a validated paper stratum")
        scope = epoch.stratum
        _require_label(epoch.epoch_receipt_id, "epoch_receipt_id")
        with self._transaction():
            prior = self._db.execute(
                "SELECT decision, input_sha256 FROM collar_decisions WHERE stream=? AND receipt_id=?",
                (scope.stream, epoch.epoch_receipt_id),
            ).fetchone()
            if prior:
                saved = _read_decision(prior[0])
                if saved.stratum != scope:
                    raise ValueError("receipt belongs to another policy version")
                if prior[1] is None or prior[1] != _input_sha256(epoch, fleet):
                    raise ValueError("receipt payload differs or its original input checksum is unavailable")
                return saved
            active = self._active(scope.stream)
            if not active or active[0] != scope.collar_version:
                raise ValueError("policy version is not active")
            latest = self._db.execute(
                "SELECT decision FROM collar_decisions WHERE stream=? ORDER BY k DESC LIMIT 1", (scope.stream,)
            ).fetchone()
            last = _read_decision(latest[0]) if latest else None
            k = _require_count(epoch.epoch_index, "epoch index")
            if k != (last.epoch_index if last else 0) + 1:
                raise ValueError("epoch index compare-and-swap conflict")
            baseline = CollarBaseline(**json.loads(active[1]))
            prev = CollarProgress(baseline.collar_bps)
            if last is not None and last.stratum == scope:
                prev = CollarProgress(last.c_out_bps, last.adverse_streak, last.green_streak, last.alarm_active)
            ceiling = _SESSION_CEILINGS[scope.execution_session]
            if baseline.max_collar_bps is not None:
                ceiling = min(ceiling, baseline.max_collar_bps)
            watermark = last.last_session if last else ""
            try:
                evidence = _evaluate(epoch, fleet, prev, baseline, ceiling, watermark)
            except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
                evidence = _Evidence(reason="INVALID_OBSERVATION", last_session=watermark)
            after = _advance(prev, baseline.collar_bps, evidence.signal)
            if after.collar_bps < prev.collar_bps:
                reserved = self._db.execute(
                    "INSERT OR IGNORE INTO collar_lane_steps VALUES (?, ?)", (scope.account_lane, evidence.last_session)
                )
                if not reserved.rowcount:
                    evidence.reason = "LANE_ALREADY_ADJUSTED_THIS_SESSION"
                    after = _advance(prev, baseline.collar_bps, "HOLD")
            decision = CollarDecision(
                stratum=scope,
                epoch_receipt_id=epoch.epoch_receipt_id,
                epoch_index=k,
                c_0_bps=baseline.collar_bps,
                c_prev_bps=prev.collar_bps,
                c_out_bps=after.collar_bps,
                ceiling_bps=ceiling,
                action="TIGHTEN" if after.collar_bps < prev.collar_bps else "HOLD",
                reason=evidence.reason,
                adverse_streak=after.adverse_streak,
                green_streak=after.green_streak,
                alarm_active=after.alarm_active,
                last_session=evidence.last_session,
                n_eff=evidence.n_eff,
                p_value=evidence.p,
                alpha_k=1 / (20 * k * (k + 1)),
                baseline_sha256=baseline.snapshot_sha256,
                parity_receipt_sha256=evidence.row_hash,
            )
            payload = asdict(decision)
            del payload["receipt_sha256"]
            decision = replace(decision, receipt_sha256=_canonical_sha256(payload))
            self._db.execute(
                "INSERT INTO collar_decisions (stream, k, receipt_id, decision, input_sha256) VALUES (?, ?, ?, ?, ?)",
                (scope.stream, k, epoch.epoch_receipt_id, _json(asdict(decision)), _input_sha256(epoch, fleet)),
            )
            return decision
