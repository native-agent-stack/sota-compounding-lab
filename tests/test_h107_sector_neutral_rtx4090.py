"""Unit and property tests for Trial #103 (H107) SIC Sector-Neutral Liquid Multi-Feed CPCV."""

from __future__ import annotations

import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_h107_sector_neutral_liquid_multifeed_rtx4090 as h107  # noqa: E402


def test_h107_preregistration_constants_and_universe_guards() -> None:
    assert h107.H107_ID == "H107_SECTOR_NEUTRAL_LIQUID_MULTIFEED_RESIDUAL_RTX4090_CPCV"
    assert len(h107.H107_SPEC_SHA256) == 64
    assert h107.H107_TRIAL_NUMBER == 103
    assert h107.MIN_LIQUID_DVOL_USD == 25_000_000.0
    assert h107.MIN_SECTOR_GROUP_SIZE >= 6
    assert h107.SHORT_BORROW_1X_BPS == 5.0


def test_within_sector_neutral_ranks_are_bounded_and_isolated() -> None:
    np = pytest.importorskip("numpy")
    raw = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [100.0, -5.0],
            [200.0, -4.0],
            [300.0, -3.0],
            [400.0, -2.0],
            [500.0, -1.0],
            [600.0, 0.0],
        ],
        dtype=np.float32,
    )
    date_codes = np.zeros(12, dtype=np.int64)
    sic2_codes = np.array([35] * 6 + [73] * 6, dtype=np.int64)
    r1 = h107.build_within_sector_neutral_ranks(raw, date_codes, sic2_codes, demean_within_sector=True)
    assert np.all(np.isfinite(r1))
    assert np.all((r1 >= -1.0001) & (r1 <= 1.0001))
    assert np.allclose(r1[:6, 0], np.linspace(-1.0, 1.0, 6, dtype=np.float32), atol=1e-5)

    # Mutate SIC 73 wildly -> SIC 35 must remain bit-identical
    raw2 = raw.copy()
    raw2[6:, 0] *= -50.0
    r2 = h107.build_within_sector_neutral_ranks(raw2, date_codes, sic2_codes, demean_within_sector=True)
    assert np.array_equal(r1[:6], r2[:6])


def test_daily_sector_neutral_returns_dollar_neutral_and_non_diluting_longs() -> None:
    np = pytest.importorskip("numpy")
    n = 30
    date_codes = np.zeros(n, dtype=np.int64)
    sic2_codes = np.array([28] * 15 + [36] * 15, dtype=np.int64)
    inv_vol = np.linspace(1.0, 2.0, n, dtype=np.float64)
    has_90d_dil = np.zeros(n, dtype=np.float32)
    # Highest-scored name in SIC 28 is actively diluting -> MUST be excluded from long leg
    has_90d_dil[14] = 1.0

    groups = h107.precompute_date_sector_groups(date_codes, sic2_codes, inv_vol, has_90d_dil)
    assert 0 in groups and len(groups[0]) == 2

    scores = np.arange(n, dtype=np.float64)
    long_net_1x = np.full(n, 10.0, dtype=np.float64)
    long_net_2x = np.full(n, 8.0, dtype=np.float64)
    long_net_1x[14] = 999.0  # diluting row must NOT enter long basket
    short_net_1x = np.full(n, 4.0, dtype=np.float64)
    short_net_2x = np.full(n, 2.0, dtype=np.float64)

    idx_slice = np.arange(n, dtype=np.int64)
    dcs, ls1, ls2, l2, s2 = h107.compute_daily_sector_neutral_returns(
        scores,
        idx_slice,
        date_codes=date_codes,
        date_sector_groups=groups,
        long_net_1x=long_net_1x,
        long_net_2x=long_net_2x,
        short_net_1x=short_net_1x,
        short_net_2x=short_net_2x,
    )
    assert dcs == [0]
    assert np.isclose(ls1[0], 0.5 * 10.0 + 0.5 * 4.0)
    assert np.isclose(ls2[0], 0.5 * 8.0 + 0.5 * 2.0)
    assert np.isclose(l2[0], 8.0)
    assert np.isclose(s2[0], 2.0)
