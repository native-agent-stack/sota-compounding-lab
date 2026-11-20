# SOTA Upstream-Library Strategy E2E Sim & Regime-Conditioned Per-Symbol Manifest (`ADR-0035`)

- **Schema:** `sota.upstream-lib-strategy-e2e-regime-persymbol.v1` · **Version:** `1.0.0` · **Study Wall:** `<= 2026-10-15`
- **SHA-256 Digest:** `37c4e0c13f50b85424b6f090c61889a6929d81676f97b0be7a82cdfca7bef682`
- **Governance:** `PROTECTIVE_FILTER_OR_GOVERNANCE` · `standalone_alpha_claim=False` · `hlz_trial_consumed=False` · Next HLZ `t*(110) = 4.2189`
- **Safety Invariants:** `ADR-0022 <= 2.00%` · `take_profit_cap_enabled=False` · `paper_only_live_refused=True` (live ports `[4001, 7496, 8796, 8798]` refused)

## 1. Live Multi-Horizon Market Situation

| Session | Macro Regime | Tactical VIX Regime | Composite Situation | VIX 9d/3m | Soft Log-Odds | Cederburg Brake | Short Overlay |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| `POST_CLOSE` | `CONTRACTION` | `CRISIS_SPIKE` | `CONTRACTION_RISK_OFF` | `1.4034` | `-6.0000` | `0.4554x` | `False` |

## 2. Executed Pinned SOTA Upstream Libraries

| Upstream Library | Executed | Version / Engine | Key Pre-Wall Verification Result |
| :--- | :--- | :--- | :--- |
| `bashtage_arch` | `True` | `9.0.0` | Hansen SPA p=0.0520; MCS M*=['VBT_20D_BREAKOUT', 'VBT_IBS_LT_15', 'VBT_IBS_LT_20', 'VBT_IBS_LT_25'] |
| `hftbacktest_queue_parity_ref` | `True` | `STOP_CHECK_BEFORE_MFE_UPDATE + Almgren-Chriss 1x/2x` | 30 pre-wall symbol x strategy queue-parity sims (0 intrabar lookahead) |
| `ml4t_diagnostic` | `True` | `0.2.0` | DSR=0.0065, WRC p=0.0775, MinTRL finite=True (honest fail-closed) |
| `nautilus_trader` | `True` | `2.0.0rc9` | BacktestEngine Rust-core XNAS NETTING/MARGIN verified (run_ok=True, iterations=0) |
| `polars_duckdb_pandera_xnys` | `True` | `polars 2.1.0 / duckdb 2.0.0 / pandera 0.35.0 / xcals 4.13.2` | 700 official XNYS pre-wall sessions (2024-01-02..2026-10-15), 7,836,606 1m bars, 0 post-wall bytes, Pandera Float64 PASS |
| `skfolio` | `True` | `1.12.7` | CombinatorialPurgedCV C(6,2)=15 purged+embargoed splits; median OOS SR=+1.198 |
| `statsmodels` | `True` | `0.15.0` | OLS Newey-West HAC max_t=+2.0612 across 4 strategies |
| `vectorbt` | `True` | `1.2.0` | 4 plateau sweeps over 15 symbols x 700 pre-wall sessions |

## 3. Real Pre-Wall (`<= 2026-10-15`) Queue-Parity E2E Simulation Summary

| Symbol | Strategy | Dir | Cadence | Events | 1x Cost bps | Mean 1x bps | Mean 2x bps | WR 2x % | PF 2x | Max DD 2x % |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `AAPL` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 74 | 8.00 | +36.71 | +28.71 | 48.6% | 1.34 | 16.21% |
| `AAPL` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 33 | 8.00 | -29.99 | -37.99 | 42.4% | 0.61 | 14.58% |
| `AMD` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 117 | 10.02 | +77.49 | +67.46 | 35.0% | 1.57 | 23.88% |
| `AMD` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 35 | 10.02 | +72.64 | +62.62 | 42.9% | 1.56 | 12.88% |
| `AVGO` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 100 | 10.02 | +17.49 | +7.47 | 33.0% | 1.06 | 23.94% |
| `AVGO` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 36 | 10.02 | +108.15 | +98.13 | 41.7% | 1.95 | 7.66% |
| `IWM` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 79 | 8.00 | -33.49 | -41.49 | 36.7% | 0.56 | 33.89% |
| `IWM` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 23 | 8.00 | -26.53 | -34.53 | 43.5% | 0.64 | 9.50% |
| `META` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 103 | 10.02 | +34.21 | +24.19 | 36.9% | 1.23 | 21.15% |
| `META` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 31 | 10.02 | +13.03 | +3.01 | 48.4% | 1.03 | 11.88% |
| `MSFT` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 85 | 8.00 | -8.00 | -16.00 | 41.2% | 0.82 | 19.64% |
| `MSFT` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 27 | 8.00 | +16.03 | +8.03 | 48.1% | 1.10 | 5.67% |
| `MU` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 114 | 10.02 | +67.45 | +57.42 | 32.5% | 1.50 | 22.47% |
| `MU` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 54 | 10.02 | +28.69 | +18.66 | 40.7% | 1.17 | 13.09% |
| `NVDA` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 105 | 8.00 | +60.22 | +52.22 | 42.9% | 1.52 | 10.37% |
| `NVDA` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 37 | 8.00 | +93.67 | +85.67 | 56.8% | 2.04 | 5.32% |
| `PLTR` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 98 | 10.02 | +84.87 | +74.84 | 39.8% | 1.67 | 21.43% |
| `PLTR` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 48 | 10.02 | +48.38 | +38.36 | 43.8% | 1.36 | 21.39% |
| `QQQ` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 76 | 8.00 | +6.68 | -1.32 | 43.4% | 0.99 | 16.24% |
| `QQQ` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 30 | 8.00 | +15.94 | +7.94 | 46.7% | 1.08 | 11.00% |
| `SMH` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 103 | 10.02 | +41.67 | +31.65 | 41.8% | 1.30 | 11.15% |
| `SMH` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 45 | 10.02 | +21.15 | +11.12 | 46.7% | 1.12 | 12.87% |
| `SPY` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 66 | 8.00 | +12.62 | +4.62 | 48.5% | 1.07 | 9.00% |
| `SPY` | `S04_MINERVINI_VCP_QULLAMAGGIE` | `LONG` | `DAILY_SWING` | 33 | 8.00 | -22.28 | -30.28 | 42.4% | 0.68 | 17.80% |
| `SVIX` | `S09_CONNORS_IBS_CONFORMAL_MR` | `LONG` | `DAILY_SWING` | 87 | 10.02 | +86.50 | +76.47 | 43.7% | 1.79 | 17.03% |

## 4. Live Regime-Conditioned Per-Symbol Factor & Strategy Routing

- **Evaluated:** `54` · **Eligible:** `5` (Long `5`, Short `0`) · **Blocked:** `49` · **Pre-Wall Calibrated:** `18`

| Symbol | Cluster | Eligible | Dir | Strategy | Score | EB Post bps | Stop % | Trail bps | Size Scale | Paper Lane | Reasons |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `MU` | `HIGH_BETA_SEMI` | `True` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 75.4 | +41.91 | 0.82% | 13.5 | 0.46x | `:8787` | `OK` |
| `SQQQ` | `VOL_HEDGE_ETF` | `True` | `LONG` | `VIX_SPIKE_AND_CRUSH_REGIME_SURFER` | 70.9 | +23.79 | 0.60% | 10.0 | 0.46x | `:8787` | `OK` |
| `SOXL` | `VOL_HEDGE_ETF` | `True` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 63.4 | +2.00 | 1.94% | 14.0 | 0.41x | `:8787` | `OK` |
| `TQQQ` | `VOL_HEDGE_ETF` | `True` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 60.5 | +2.00 | 0.58% | 9.7 | 0.41x | `:8787` | `OK` |
| `NVDA` | `HIGH_BETA_SEMI` | `True` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 57.7 | +45.74 | 0.41% | 6.8 | 0.46x | `:8787` | `OK` |
| `AVGO` | `HIGH_BETA_SEMI` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 73.4 | +49.87 | 0.50% | 8.5 | 0.46x | `:8787` | `SEC_90D_DILUTION_LONG_BLOCK, SPREAD_EXCEEDS_COLLAR:27.7>6.8` |
| `SSMR` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 63.2 | +1.00 | 1.53% | 12.0 | 0.40x | `:8787` | `SPREAD_EXCEEDS_COLLAR:2661.5>23.8` |
| `UVXY` | `VOL_HEDGE_ETF` | `False` | `LONG` | `VIX_SPIKE_AND_CRUSH_REGIME_SURFER` | 61.3 | +95.48 | 0.80% | 10.5 | 0.46x | `:8787` | `SPREAD_EXCEEDS_COLLAR:89.5>10.2` |
| `NAVI` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 59.1 | -0.75 | 0.97% | 14.0 | 0.38x | `:8787` | `SPREAD_EXCEEDS_COLLAR:597.0>12.75` |
| `PXLW` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 58.9 | +1.00 | 1.52% | 12.0 | 0.40x | `:8787` | `SPREAD_EXCEEDS_COLLAR:461.7>23.8` |
| `GPN` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 58.1 | -2.82 | 1.56% | 12.0 | 0.36x | `:8787` | `SPREAD_EXCEEDS_COLLAR:420.2>23.8` |
| `DAMD` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 56.9 | +16.87 | 1.55% | 12.0 | 0.46x | `:8787` | `SPREAD_EXCEEDS_COLLAR:25.8>23.8` |
| `CVS` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 56.7 | +0.58 | 1.56% | 12.0 | 0.39x | `:8787` | `SPREAD_EXCEEDS_COLLAR:141.2>23.8` |
| `BIPC` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 56.2 | -2.06 | 1.57% | 12.0 | 0.37x | `:8787` | `SPREAD_EXCEEDS_COLLAR:144.7>23.8` |
| `SEI` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 55.6 | +0.50 | 1.55% | 12.0 | 0.39x | `:8787` | `SPREAD_EXCEEDS_COLLAR:675.1>23.8` |
| `TSM` | `HIGH_BETA_SEMI` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 55.6 | +11.75 | 0.68% | 10.7 | 0.46x | `:8787` | `SPREAD_EXCEEDS_COLLAR:12.4>8.5` |
| `AKAN` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 53.3 | -7.82 | 1.37% | 12.0 | 0.32x | `:8787` | `SPREAD_EXCEEDS_COLLAR:45.6>23.8` |
| `BILL` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 51.7 | +0.50 | 1.01% | 14.0 | 0.39x | `:8787` | `SPREAD_EXCEEDS_COLLAR:32.5>12.75` |
| `CMCT` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 50.6 | +4.58 | 1.52% | 12.0 | 0.43x | `:8787` | `INSUFFICIENT_LONG_OFI_ALIGNMENT, SPREAD_EXCEEDS_COLLAR:2677.2>23.8` |
| `AMD` | `HIGH_BETA_SEMI` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 50.0 | +46.74 | 0.73% | 11.5 | 0.46x | `:8787` | `SEC_90D_DILUTION_LONG_BLOCK, INSUFFICIENT_LONG_OFI_ALIGNMENT, SPREAD_EXCEEDS_COLLAR:16.1>8.5` |
| `EVH` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 49.9 | -1.44 | 1.54% | 12.0 | 0.37x | `:8787` | `SPREAD_EXCEEDS_COLLAR:204.1>23.8` |
| `YSWY` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 49.2 | -4.53 | 1.54% | 12.0 | 0.35x | `:8787` | `SPREAD_EXCEEDS_COLLAR:478.7>23.8` |
| `VIXY` | `VOL_HEDGE_ETF` | `False` | `LONG` | `VIX_SPIKE_AND_CRUSH_REGIME_SURFER` | 49.1 | +21.36 | 0.82% | 13.8 | 0.46x | `:8787` | `SPREAD_EXCEEDS_COLLAR:97.5>12.75` |
| `GAP` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 47.3 | -1.18 | 0.90% | 14.0 | 0.38x | `:8787` | `SPREAD_EXCEEDS_COLLAR:123.6>12.75` |
| `SOXS` | `VOL_HEDGE_ETF` | `False` | `LONG` | `VIX_SPIKE_AND_CRUSH_REGIME_SURFER` | 46.4 | +34.52 | 1.21% | 14.0 | 0.46x | `:8787` | `SPREAD_EXCEEDS_COLLAR:20.0>12.75` |
| `BNAI` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 45.8 | -4.99 | 1.54% | 12.0 | 0.34x | `:8787` | `SPREAD_EXCEEDS_COLLAR:152.7>23.8` |
| `CARL` | `CATALYST_MIDCAP` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 45.3 | +1.00 | 1.55% | 12.0 | 0.40x | `:8787` | `SPREAD_EXCEEDS_COLLAR:1472.0>23.8` |
| `SNDK` | `HIGH_BETA_SEMI` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 44.4 | +1.50 | 1.60% | 14.0 | 0.40x | `:8787` | `INSUFFICIENT_LONG_OFI_ALIGNMENT` |
| `GLOB` | `BROAD_LIQUID` | `False` | `LONG` | `H08_FIRST_HAND_CATALYST_IGNITION` | 44.2 | -6.83 | 1.56% | 12.0 | 0.32x | `:8787` | `SPREAD_EXCEEDS_COLLAR:549.0>23.8` |
| `VXX` | `VOL_HEDGE_ETF` | `False` | `LONG` | `VIX_SPIKE_AND_CRUSH_REGIME_SURFER` | 43.9 | +23.70 | 0.82% | 13.3 | 0.46x | `:8787` | `INSUFFICIENT_LONG_OFI_ALIGNMENT, SPREAD_EXCEEDS_COLLAR:71.0>12.75` |
