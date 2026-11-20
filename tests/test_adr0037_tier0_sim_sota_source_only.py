"""Tests for ADR-0037 Sim Tier-0 SOTA-Source-Only Enforcement."""

from __future__ import annotations

from sota_compounding.upstream_lib_strategy_e2e import (
    TIER0_PURGED_SELF_INVENTED_SIM_STRATEGIES,
    enforce_tier0_sota_source_only_sim_gate,
)


def test_sim_tier0_rejects_stdlib_self_built_and_all_purged_anti_patterns() -> None:
    for anti in TIER0_PURGED_SELF_INVENTED_SIM_STRATEGIES:
        res = enforce_tier0_sota_source_only_sim_gate(
            strategy_id=anti,
            upstream_repo="nautechsystems/nautilus_trader",
            upstream_pin_sha="da6b57aefdd62b44f675e434ffcfefca7f4025dd",
            primary_literature="Zarattini et al. (2024, SSRN 4729284): 5m ORB",
        )
        assert res["allowed"] is False
        assert any("TIER0_PURGED_SELF_INVENTED" in r for r in res["rejection_reasons"])

    res_std = enforce_tier0_sota_source_only_sim_gate(
        strategy_id="ZARATTINI_2024_5M_ORB",
        upstream_repo="cpython-stdlib",
        upstream_pin_sha="3.14.10",
        primary_literature="Zarattini et al. (2024, SSRN 4729284): 5m ORB",
    )
    assert res_std["allowed"] is False
    assert any("TIER0_FORBIDDEN_NON_SOTA_UPSTREAM" in r for r in res_std["rejection_reasons"])


def test_sim_tier0_accepts_pinned_sota_upstream_with_primary_citation() -> None:
    res = enforce_tier0_sota_source_only_sim_gate(
        strategy_id="ZARATTINI_2024_5M_ORB",
        upstream_repo="nautechsystems/nautilus_trader",
        upstream_pin_sha="da6b57aefdd62b44f675e434ffcfefca7f4025dd",
        primary_literature="Zarattini, Aziz & Barbon (2024, SSRN 4729284): 5m ORB Stocks-in-Play",
    )
    assert res["allowed"] is True
    assert res["adr"] == "ADR-0037"
    assert res["rejection_reasons"] == []
