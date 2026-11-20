"""ADR-0026 uses synthetic paper cohorts and the real ADR-0025 producer."""

import hashlib
import json
import math
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from fractions import Fraction
from itertools import product
from threading import Barrier

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sota_compounding import (
    CollarBaseline,
    CollarEpoch,
    CollarLedger,
    CollarStratum,
    LivePaperTcaObservation,
    QueueParityBaseline,
    SessionBlock,
    compute_hlz_t_hurdle,
    effective_sample_size,
    reconcile_fleet_parity,
    validate_prewall_temporal_boundary,
)
from sota_compounding.collar_recalibration import (
    POST_WALL_PROVENANCE,
    CollarProgress,
    _advance,
    session_sign_test,
)

SCOPE = CollarStratum("fixture", ":8787", "RTH", "BUY", "liquid", "v1")
BASE = CollarBaseline(15.0, "2026-10-15", "queue-fixture", "ab" * 32)
FIRST_SESSION = date(2026, 10, 19)


def fleet(scope=SCOPE, *, drift=11.0, gap=0.0, markout=0.0, n=30, gross=100.0):
    baseline = QueueParityBaseline(scope.strategy_id, "2026-10-15", 0.5, 1.0, 100, 100, 103)
    obs = LivePaperTcaObservation(
        scope.strategy_id, scope.account_lane, n, 0.5 + gap, 1 + drift, 0, 0, markout, 0, gross
    )
    return reconcile_fleet_parity([baseline], [obs])


def epoch(k, *, scope=SCOPE, observed=15.0, drift=11.0, summary=None):
    # Ten completed non-overlapping synthetic session blocks of three independent fills each.
    start = FIRST_SESSION + timedelta(days=21 * (k - 1))
    sessions = [start + timedelta(days=(i // 5) * 7 + i % 5) for i in range(10)]
    blocks = tuple(SessionBlock(d.isoformat(), (1.0, 1.0, 1.0), drift) for d in sessions)
    matched = fleet(scope, drift=drift) if summary is None else summary
    return CollarEpoch(
        scope,
        f"epoch-{k}",
        k,
        scope.collar_version,
        observed,
        blocks,
        stale_inputs=False,
        denominators_complete=True,
        direction_supported=True,
        consecutive_windows=3,
        sampling_interval_s=1.0,
        max_markout_lag_60s_s=1.0,
        parity_receipt_sha256=matched["rows"][0]["receipt_sha256"],
        baseline_snapshot_sha256=BASE.snapshot_sha256,
        rho_u=0.0,
    )


def ledger(c0=15.0, *, scope=SCOPE, cap=None):
    db = CollarLedger(sqlite3.connect(":memory:"))
    db.register(scope, replace(BASE, collar_bps=c0, max_collar_bps=cap))
    return db


def test_two_reds_two_greens_and_receipt_replay():
    db = ledger()
    red, green = fleet(), fleet(drift=4)
    first = db.consume(epoch(1), red)
    second = db.consume(epoch(2), red)
    third = db.consume(epoch(3, observed=13, drift=4), green)
    fourth = db.consume(epoch(4, observed=13, drift=4), green)
    assert [r.c_out_bps for r in (first, second, third, fourth)] == [15, 13, 13, 13]
    assert second.alarm_active and third.alarm_active and not fourth.alarm_active
    assert second.action == "TIGHTEN"
    assert db.consume(epoch(2), red) == second
    data = asdict(second)
    digest = data.pop("receipt_sha256")
    assert (
        digest
        == hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    )
    assert second.paper_policy == "NEVER_GATED_ADVISORY"
    assert not second.halts_paper_farm and not second.stop_distance_mutated
    assert second.provenance_tag == POST_WALL_PROVENANCE


@pytest.mark.parametrize(("c0", "cap"), [(2, None), (3, 10), (11, 10), (26, None), (5, 4), (0, 10)])
def test_out_of_domain_is_observational(c0, cap):
    db = ledger(c0, cap=cap)
    result = db.consume(epoch(1, observed=c0), fleet())
    assert result.reason == "OBSERVATIONAL_OUT_OF_DOMAIN"
    assert (result.action, result.c_prev_bps, result.c_out_bps, result.c_0_bps) == ("HOLD", c0, c0, c0)


@pytest.mark.parametrize(
    ("regime", "c0", "ceiling"), [("RTH", 25, 25), ("OVERNIGHT_BOATS", 35, 35), ("OVERNIGHT_23X5_SIP", 35, 35)]
)
def test_ceilings_and_persistent_red_budget(regime, c0, ceiling):
    scope = replace(SCOPE, execution_session=regime)
    db = ledger(c0, scope=scope)
    c = c0
    for k in range(1, 7):
        result = db.consume(epoch(k, scope=scope, observed=c), fleet(scope))
        c = result.c_out_bps
        assert c == max(c0 - 4, c0 - 2 * (k // 2))
        assert result.ceiling_bps == ceiling


@pytest.mark.parametrize("lane", [":8798", ":4001", ":7496", ":9999", "paper", ""])
def test_unknown_or_live_lane_refused(lane):
    with pytest.raises(ValueError, match="paper lane"):
        replace(SCOPE, account_lane=lane)


@pytest.mark.parametrize("lane", [":8787", ":8799", ":8797", ":4002"])
def test_paper_lanes_and_no_objective_or_trial_mutation(lane):
    scope = replace(SCOPE, account_lane=lane)
    fl = fleet(scope)
    original = json.dumps(fl, sort_keys=True)
    db = ledger(scope=scope)
    decision = db.consume(epoch(1, scope=scope), fl)
    assert decision.stratum == scope
    assert fl["rows"][0]["hlz_trials_n"] == 103 and compute_hlz_t_hurdle(103) == 4.2063
    assert json.dumps(fl, sort_keys=True) == original
    assert not ({"hlz_trials_n", "optimum", "target_return", "new_stop_distance"} & set(asdict(decision)))
    with pytest.raises(ValueError):
        reconcile_fleet_parity([decision], [])


@pytest.mark.parametrize(("rho", "expected"), [(0, 20), (0.1, 12.738853503184714), (1, 2.985074626865672)])
def test_effective_n_776(rho, expected):
    blocks = [[1] * n for n in (7, 7, 6)]
    assert float(effective_sample_size(blocks, rho)) == pytest.approx(expected)
    assert effective_sample_size(blocks) == effective_sample_size(blocks, 1)


def test_exact_statistics_include_ties_boundaries_and_subnormal_scales():
    assert effective_sample_size([[1e308] * 10, [1e308] * 10], 0) == 20
    assert effective_sample_size([[5e-324] * 10, [5e-324] * 10], 0) == 20
    assert session_sign_test(3, 3, 1) == (Fraction(1, 8), Fraction(1, 40))
    assert session_sign_test(10, 10, 4) == (Fraction(1, 1024), Fraction(1, 400))
    for g in range(18):
        for x in range(g + 1):
            assert session_sign_test(x, g, 1)[0] == Fraction(sum(math.comb(g, j) for j in range(x, g + 1)), 2**g)
    with pytest.raises(ValueError):
        session_sign_test(True, 3, 1)
    with pytest.raises(ValueError):
        session_sign_test(4, 3, 0)
    # Positive mass below binary64 resolution still gets an exact decision.
    assert session_sign_test(1100, 1100, 1)[0] == Fraction(1, 2**1100)


@pytest.mark.parametrize(
    "change",
    [
        {"stale_inputs": True},
        {"stale_inputs": 0},
        {"denominators_complete": False},
        {"denominators_complete": 1},
        {"direction_supported": False},
        {"direction_supported": 1},
        {"consecutive_windows": 2},
        {"consecutive_windows": True},
        {"sampling_interval_s": 61},
        {"max_markout_lag_60s_s": 6.000001},
        {"max_markout_lag_60s_s": -1},
        {"observed_collar_bps": 13},
        {"observed_collar_version": "old"},
        {"rho_u": 1},
        {"rho_u": 1.1},
        {"rho_u": False},
        {"baseline_snapshot_sha256": "cc" * 32},
        {"parity_receipt_sha256": "cd" * 32},
    ],
)
def test_missed_gates_hold_and_break_confirmation(change):
    db = ledger()
    db.consume(epoch(1), fleet())
    hold = db.consume(replace(epoch(2), **change), fleet())
    assert hold.action == "HOLD" and hold.c_out_bps == 15 and hold.adverse_streak == 0
    assert db.consume(epoch(3), fleet()).action == "HOLD"


@pytest.mark.parametrize(("n", "block_sizes"), [(0, []), (29, [3] * 9 + [2]), (30, [15, 15]), (30, [9] * 3)])
def test_actual_raw_fills_unique_sessions_and_denominators(n, block_sizes):
    db = ledger()
    fl = fleet(n=n)
    ep = epoch(1, summary=fl)
    blocks = tuple(replace(b, fill_weights=(1,) * count) for b, count in zip(ep.blocks, block_sizes, strict=False))
    result = db.consume(replace(ep, blocks=blocks), fl)
    assert (
        result.c_out_bps == 15
        and result.adverse_streak == 0
        and result.reason in ("INSUFFICIENT_SAMPLE", "FILL_COUNT_MISMATCH")
    )


@pytest.mark.parametrize(
    ("drift", "gap", "mark", "gross"),
    [
        (10, 0, 0, 100),
        (10.000001, 0.150001, 0, 100),
        (11, -0.150001, 0, 100),
        (11, 0, -3.000001, 100),
        (11, 0, 0, -200),
        (1, -0.31, 0, 100),
        (1, 0, -9, 100),
    ],
)
def test_red_or_warn_without_specific_adverse_evidence_holds(drift, gap, mark, gross):
    db = ledger()
    db.consume(epoch(1), fleet())
    fl = fleet(drift=drift, gap=gap, markout=mark, gross=gross)
    assert db.consume(epoch(2, drift=drift, summary=fl), fl).action == "HOLD"


def test_sign_test_requires_strict_adversity_and_three_blocks_cannot_qualify():
    db = ledger()
    ep = epoch(1)
    ties = tuple(replace(b, slippage_drift_bps=10) for b in ep.blocks)
    assert db.consume(replace(ep, blocks=ties), fleet()).p_value == 1
    for k in (2, 3):
        ep = epoch(k)
        blocks = tuple(replace(b, fill_weights=(1,) * 10) for b in ep.blocks[:3])
        r = db.consume(replace(ep, blocks=blocks), fleet())
        assert (r.n_eff, r.p_value) == (30, 0.125)
        assert (r.action, r.reason, r.c_out_bps, r.adverse_streak) == ("HOLD", "ADVERSE_UNCONFIRMED", 15.0, 0)


def test_ten_adverse_blocks_cannot_pass_epoch_41_spending_gate():
    db = ledger()
    warn = fleet(drift=6)
    for k in range(1, 41):
        assert db.consume(epoch(k, drift=6, summary=warn), warn).reason == "DRIFT_WARN"
    r = db.consume(epoch(41), fleet())
    assert (r.epoch_index, r.n_eff, r.p_value, r.alpha_k) == (41, 30, 0.0009765625, 1 / (20 * 41 * 42))
    assert (r.action, r.reason, r.c_out_bps, r.adverse_streak) == ("HOLD", "ADVERSE_UNCONFIRMED", 15.0, 0)


@pytest.mark.parametrize(("block_drift", "p"), [(10.00000049, 1), (10.00000051, 2**-10)])
def test_block_sign_uses_the_same_precision_as_adr0025(block_drift, p):
    red = fleet()
    ep = epoch(1, drift=block_drift, summary=red)
    assert fleet(drift=block_drift)["rows"][0]["slippage_drift_bps"] == round(block_drift, 6)
    result = ledger().consume(ep, red)
    assert result.p_value == p
    assert result.reason == ("ADVERSE_UNCONFIRMED" if p == 1 else "QUALIFIED_ADVERSE")


def test_full_six_dimensional_stratum_and_lane_session_budget():
    db = ledger()
    first = SCOPE
    others = [
        replace(SCOPE, **kw)
        for kw in (
            {"strategy_id": "other"},
            {"account_lane": ":8799"},
            {"execution_session": "OVERNIGHT_BOATS"},
            {"side": "SELL"},
            {"liquidity_regime": "thin"},
        )
    ]
    for scope in [first, *others]:
        db.register(scope, BASE)
        assert db.consume(epoch(1, scope=scope), fleet(scope)).adverse_streak == 1
        r = db.consume(epoch(2, scope=scope), fleet(scope))
        assert r.action == ("TIGHTEN" if scope == first or scope.account_lane != first.account_lane else "HOLD")
    # New, coincident strategy cannot consume an unmatching row even on the same lane.
    wrong = replace(SCOPE, strategy_id="unmatched")
    db.register(wrong, BASE)
    assert db.consume(epoch(1, scope=wrong), fleet()).reason == "NO_UNIQUE_MATCHED_LANE"


def test_cohorts_must_be_disjoint_unique_and_match_baseline():
    db = ledger()
    ep = epoch(1)
    db.consume(ep, fleet())
    repeated = replace(ep, epoch_receipt_id="another-id", epoch_index=2)
    assert db.consume(repeated, fleet()).reason == "OVERLAPPING_OR_OLD_COHORT"
    ep3 = epoch(3)
    twice = (ep3.blocks[0],) + ep3.blocks[:-1]
    assert db.consume(replace(ep3, blocks=twice), fleet()).reason == "OVERLAPPING_OR_OLD_COHORT"
    broken = fleet()
    broken["rows"][0]["live_markout_60s_bps"] = 999
    assert db.consume(epoch(4), broken).reason == "INVALID_OBSERVATION"
    assert db.consume(epoch(5), reconcile_fleet_parity([], [])).reason == "NO_UNIQUE_MATCHED_LANE"
    with pytest.raises(ValueError):
        replace(BASE, max_event_date="2026-10-16")


def test_spending_survives_restart_rebaseline_and_compare_and_swap(tmp_path):
    path = tmp_path / "private-journal.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        db = CollarLedger(conn)
        db.register(SCOPE, BASE)
        first = db.consume(epoch(1), fleet())
    with closing(sqlite3.connect(path)) as conn:
        db = CollarLedger(conn)
        db.register(SCOPE, BASE)
        assert db.consume(epoch(1), fleet()) == first
        with pytest.raises(ValueError, match="compare-and-swap"):
            db.consume(replace(epoch(1), epoch_receipt_id="late"), fleet())
        new = replace(SCOPE, collar_version="v2")
        db.register(new, replace(BASE, collar_bps=10), previous_version="v1")
        with pytest.raises(ValueError, match="reused"):
            db.register(SCOPE, BASE, previous_version="v2")
        assert db.consume(epoch(1), fleet()) == first
        with pytest.raises(ValueError, match="not active"):
            db.consume(epoch(2), fleet())
        ep = epoch(2, scope=new, observed=10)
        r = db.consume(ep, fleet())
        assert (r.epoch_index, r.alpha_k, r.c_out_bps, r.adverse_streak) == (2, 1 / 120, 10, 1)


@pytest.mark.parametrize("duplicate", [True, False])
def test_concurrent_receipts_have_one_atomic_winner(tmp_path, duplicate):
    path = tmp_path / "cas.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        db = CollarLedger(conn)
        db.register(SCOPE, BASE)
        db.consume(epoch(1), fleet())
    barrier = Barrier(2)

    def submit(i):
        with closing(sqlite3.connect(path)) as conn:
            db = CollarLedger(conn)
            barrier.wait(timeout=5)
            ep = replace(epoch(2), epoch_receipt_id="same" if duplicate else f"race-{i}")
            try:
                return db.consume(ep, fleet())
            except ValueError as error:
                assert "compare-and-swap" in str(error)
                return None

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(submit, range(2)))
    winners = [r for r in results if r is not None]
    assert len(winners) == (2 if duplicate else 1)
    assert all(r == winners[0] and r.c_out_bps == 13 for r in winners)
    with closing(sqlite3.connect(path)) as conn:
        assert conn.execute("SELECT count(*) FROM collar_decisions").fetchone() == (2,)


def test_canonical_receipt_ignores_row_block_weight_order_and_signed_zero():
    a, b = ledger(), ledger()
    fl = fleet()
    other = fleet(replace(SCOPE, strategy_id="other"))
    fl["rows"].extend(other["rows"])
    ep = epoch(1)
    blocks = tuple(replace(block, fill_weights=(1.5, 1.0, 0.5)) for block in ep.blocks)
    r1 = a.consume(replace(ep, blocks=blocks, max_markout_lag_60s_s=-0.0), fl)
    fl["rows"].reverse()
    permuted = tuple(replace(block, fill_weights=block.fill_weights[::-1]) for block in reversed(blocks))
    r2 = b.consume(replace(ep, blocks=permuted, max_markout_lag_60s_s=0.0), fl)
    assert r1 == r2 and len(r1.receipt_sha256) == 64
    assert a.consume(replace(ep, blocks=permuted, max_markout_lag_60s_s=0.0), fl) == r1


def test_replay_compares_epoch_and_matched_fleet_evidence_after_restart(tmp_path):
    path = tmp_path / "journal.sqlite"
    ep, fl = epoch(1), fleet()
    with closing(sqlite3.connect(path)) as conn:
        db = CollarLedger(conn)
        db.register(SCOPE, BASE)
        saved = db.consume(ep, fl)
    with closing(sqlite3.connect(path)) as conn:
        db = CollarLedger(conn)
        changed_row = fleet()
        changed_row["rows"][0]["live_sample_n"] += 1  # keep its claimed row digest
        changed_zero = fleet()
        changed_zero["rows"][0]["fill_rate_gap_ratio"] = -0.0  # ADR-0025 checks the actual row bytes
        for candidate, summary in (
            (replace(ep, epoch_index=2), fl),
            (replace(ep, stale_inputs=True), fl),
            (replace(ep, blocks=(replace(ep.blocks[0], fill_weights=(1, 1, 2)), *ep.blocks[1:])), fl),
            (replace(ep, parity_receipt_sha256="12" * 32), fl),
            (replace(ep, baseline_snapshot_sha256="34" * 32), fl),
            (ep, changed_row),
            (ep, changed_zero),
            (ep, fl | {"receipt_sha256": "56" * 32}),
            (ep, fl | {"fleet_status": "PARITY_GREEN"}),
            (ep, fl | {"halts_paper_farm": True}),
        ):
            with pytest.raises(ValueError, match="receipt payload"):
                db.consume(candidate, summary)
        assert db.consume(ep, fl) == saved
        assert db.consume(epoch(2), fl).c_out_bps == 13


def test_legacy_receipt_rejects_unverifiable_retry_and_preserves_spending():
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("""CREATE TABLE collar_decisions (
            stream TEXT NOT NULL, k INTEGER NOT NULL, receipt_id TEXT NOT NULL, decision TEXT NOT NULL,
            PRIMARY KEY(stream, k), UNIQUE(stream, receipt_id))""")
        original = ledger().consume(epoch(1), fleet())
        conn.execute(
            "INSERT INTO collar_decisions VALUES (?, ?, ?, ?)",
            (SCOPE.stream, 1, original.epoch_receipt_id, json.dumps(asdict(original))),
        )
        conn.commit()
        db = CollarLedger(conn)
        db.register(SCOPE, BASE)
        with pytest.raises(ValueError, match="receipt payload"):
            db.consume(epoch(1), fleet())
        assert db.consume(epoch(2), fleet()).c_out_bps == 13


T = "2026-10-16T04:00:00Z"
BEFORE = "2026-10-16T03:59:59.999999999Z"
AFTER = "2026-10-16T04:00:00.000000001Z"
HIST = "PRE_WALL_HISTORICAL"


def test_integer_firewall_rejects_unit_confusion_in_every_clock_and_preserves_epoch_zero():
    validate_prewall_temporal_boundary(0, 0, HIST, receipt_ts_utc=0, decision_ts_utc=0)
    validate_prewall_temporal_boundary(10**17, 10**17, HIST)
    validate_prewall_temporal_boundary("1970-01-01T00:00:00.000000001Z", BEFORE, HIST)
    with pytest.raises(ValueError, match="pre-wall interval"):
        validate_prewall_temporal_boundary(-1, 0, HIST)
    later = int(datetime(2026, 11, 1, tzinfo=UTC).timestamp())
    end = int(datetime(2026, 10, 16, 4, tzinfo=UTC).timestamp()) * 10**9 - 1
    for ambiguous in (1, later, later * 1000, later * 10**6, 10**17 - 1):
        for clocks in (
            (ambiguous, end, end, end),
            (0, ambiguous, end, end),
            (0, 0, ambiguous, end),
            (0, 0, 0, ambiguous),
        ):
            ev, received, decision, maturity = clocks
            with pytest.raises(ValueError, match="nanoseconds"):
                validate_prewall_temporal_boundary(
                    ev, maturity, HIST, receipt_ts_utc=received, decision_ts_utc=decision
                )


def test_temporal_firewall_nanoseconds_offsets_and_tagged_decisions():
    validate_prewall_temporal_boundary(BEFORE, BEFORE, HIST, trade_date="2026-10-15")
    validate_prewall_temporal_boundary("2026-10-15T23:59:59.999999999-04:00", BEFORE, HIST)
    for ev, mat in [(T, T), (AFTER, AFTER), (BEFORE, T), (BEFORE, AFTER), (datetime(2026, 10, 15), BEFORE)]:
        with pytest.raises(ValueError):
            validate_prewall_temporal_boundary(ev, mat, HIST)
    ns = int(datetime(2026, 10, 16, 4, tzinfo=UTC).timestamp()) * 10**9
    validate_prewall_temporal_boundary(ns - 2, ns - 1, HIST, receipt_ts_utc=ns - 2, decision_ts_utc=ns - 1)
    for know, decision in [(ns, ns - 1), (ns - 1, ns), (None, ns - 1)]:
        with pytest.raises(ValueError):
            validate_prewall_temporal_boundary(ns - 2, ns - 1, HIST, receipt_ts_utc=know, decision_ts_utc=decision)
    output = ledger().consume(epoch(1), fleet())
    for record in [output, asdict(output), {"upstream": [asdict(output)]}, json.dumps(asdict(output))]:
        with pytest.raises(ValueError):
            validate_prewall_temporal_boundary(BEFORE, BEFORE, HIST, source_record=record)
    for tag in [POST_WALL_PROVENANCE, output, asdict(output), "live_paper", None]:
        with pytest.raises(ValueError):
            validate_prewall_temporal_boundary(BEFORE, BEFORE, tag)


def test_e4_exhaustive_six_epoch_trajectories():
    paths = transitions = 0
    max_delta = max_total = 0.0
    for upper in (25, 35):
        for c0 in (5, 5.5, 8, 15, upper):
            for signals in product(("THROTTLE_RED", "PARITY_GREEN", "DRIFT_WARN", "HOLD"), repeat=6):
                state = CollarProgress(c0)
                total = 0
                for signal in signals:
                    after = _advance(state, c0, signal)
                    delta = state.collar_bps - after.collar_bps
                    total += delta
                    assert 0 <= delta <= 2 and 5 <= after.collar_bps <= c0 <= upper
                    assert total == c0 - after.collar_bps <= 4
                    max_delta, max_total = max(max_delta, delta), max(max_total, total)
                    state = after
                    transitions += 1
                paths += 1
    assert (paths, transitions, max_delta, max_total) == (40960, 245760, 2, 4)


@given(
    st.floats(min_value=5, max_value=10),
    st.lists(st.sampled_from(["THROTTLE_RED", "PARITY_GREEN", "DRIFT_WARN", "HOLD"]), min_size=1, max_size=100),
)
def test_directional_monotonicity_optional_cap_and_budget(c0, signals):
    state = CollarProgress(c0)
    for signal in signals:
        red = _advance(state, c0, "THROTTLE_RED")
        warn = _advance(state, c0, "DRIFT_WARN")
        green = _advance(state, c0, "PARITY_GREEN")
        assert 5 <= red.collar_bps <= warn.collar_bps <= green.collar_bps <= c0 <= 10
        after = _advance(state, c0, signal)
        assert max(5, c0 - 4, state.collar_bps - 2) <= after.collar_bps <= state.collar_bps
        state = after


@settings(max_examples=50, deadline=None)
@given(st.one_of(st.floats(allow_nan=True, allow_infinity=True), st.booleans()))
def test_observations_are_finite_and_bools_are_not_numbers(value):
    db = ledger()
    ep = epoch(1)
    ep = replace(ep, max_markout_lag_60s_s=value)
    result = db.consume(ep, fleet())
    assert db.consume(ep, fleet()) == result
    if type(value) is bool or not math.isfinite(value) or not 0 <= value <= 6:
        assert result.adverse_streak == 0 and result.action == "HOLD"
    json.dumps(asdict(result), allow_nan=False)


def test_journal_rolls_back_lane_reservation_and_spending_on_write_failure():
    with closing(sqlite3.connect(":memory:")) as conn:
        db = CollarLedger(conn)
        db.register(SCOPE, BASE)
        db.consume(epoch(1), fleet())
        conn.execute("""CREATE TRIGGER fixture_interrupt BEFORE INSERT ON collar_decisions
                     BEGIN SELECT RAISE(ABORT, 'fixture interruption'); END""")
        with pytest.raises(sqlite3.IntegrityError, match="fixture interruption"):
            db.consume(epoch(2), fleet())
        assert conn.execute("SELECT count(*) FROM collar_lane_steps").fetchone() == (0,)
        conn.execute("DROP TRIGGER fixture_interrupt")
        assert db.consume(epoch(2), fleet()).c_out_bps == 13


def test_one_hundred_noisy_epochs_do_not_ratchet_or_loosen():
    db = ledger(10, cap=10)
    for k in range(1, 101):
        drift = (11, 8, 3, 11, 1)[(k - 1) % 5]
        fl = fleet(drift=drift)
        ep = epoch(k, observed=10, drift=drift, summary=fl)
        result = db.consume(ep, fl)
        assert result.action == "HOLD" and result.c_0_bps == result.c_out_bps == result.ceiling_bps == 10
        assert result.epoch_index == k and result.alpha_k == 1 / (20 * k * (k + 1))


def test_alarm_recovery_requires_consecutive_eligible_green():
    db = ledger()
    db.consume(epoch(1), fleet())
    db.consume(epoch(2), fleet())
    green = fleet(drift=3)
    for k in range(3, 7):
        ep = epoch(k, observed=13, drift=3, summary=green)
        r = db.consume(replace(ep, stale_inputs=(k == 4)), green)
        assert r.alarm_active == (k < 6)
        assert r.c_out_bps == 13


@pytest.mark.parametrize("number", [True, False, float("nan"), float("inf"), float("-inf"), 10**400])
def test_invalid_anchor_and_weights_rejected(number):
    with pytest.raises(ValueError):
        replace(BASE, collar_bps=number)
    with pytest.raises(ValueError):
        effective_sample_size([[number]], 0)
    with pytest.raises(ValueError):
        effective_sample_size([[1]], number)


def test_date_provenance_receipt_and_calendar_cannot_be_laundered():
    for source in [{"enrichment": {"source_provenance": "live_paper"}}, {"source_provenance": None}]:
        with pytest.raises(ValueError):
            validate_prewall_temporal_boundary(BEFORE, BEFORE, HIST, source_record=source)
    with pytest.raises(ValueError):
        validate_prewall_temporal_boundary(BEFORE, BEFORE, HIST, trade_date="2026-10-16")
    with pytest.raises(ValueError):
        validate_prewall_temporal_boundary(True, BEFORE, HIST)


def test_unrelated_summary_keys_cannot_optimize_collar_or_submit_an_order():
    first, second = ledger(), ledger()
    ep, fl = epoch(1), fleet()
    changed = fl | {"target_pnl": 1e100, "desired_collar_bps": 4, "submit": True, "fleet_status": "THROTTLE_RED"}
    assert first.consume(ep, fl) == second.consume(ep, changed)


def test_corrupt_checkpoint_cannot_supply_a_new_anchor_or_duplicate_receipt():
    with closing(sqlite3.connect(":memory:")) as conn:
        db = CollarLedger(conn)
        db.register(SCOPE, BASE)
        result = db.consume(epoch(1), fleet())
        bad = asdict(result) | {"c_out_bps": 500}
        conn.execute("UPDATE collar_decisions SET decision=?", (json.dumps(bad),))
        conn.commit()
        for k in (1, 2):
            with pytest.raises(ValueError, match="checksum mismatch"):
                db.consume(epoch(k), fleet())
        assert conn.execute("SELECT count(*) FROM collar_decisions").fetchone() == (1,)
