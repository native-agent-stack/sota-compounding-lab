# SOTA Unified Sim (`S1/S2`) -> Paper (`P1–P4`) -> Live (`:8798`/`:8796`) Real Performance & Merit Resolution Manifest (`ADR-0034`)

- **Schema / Canonical SHA-256:** `sota.sim-to-paper-to-live-merit-manifest.v1` · `19d7bf6cf34183e1ce3ec45b1afc15867fa5f02eb5057d97a1f02d0a8036c9e7`
- **Generated (UTC):** `2026-11-20T16:27:06.769313Z` | **Pre-Wall Study Wall:** `<= 2026-10-15` (`0` post-wall bytes)
- **HLZ Multiple-Testing Ledger:** `N = 107` completed trials (`t*(107) = 4.2127`; next `Trial #108` `t* = 4.2148`)
- **Governance Track:** `PROTECTIVE_FILTER_OR_GOVERNANCE` (`standalone_alpha_claim = False` · `ADR-0022 <= 2.00%` · `TP OFF` · strictly dimensionless)

## 1. Executive Merit Resolution Summary

- **Total Real-Data Strategies Evaluated (`700` Pre-Wall + `W5` Paper):** `19`
- **Autonomous Live `:8798` Eligible (`Sim HLZ PASS` + `Paper W5 PASS`):** **`0`** (`[]`)
- **Positive Net `1x` Candidates (Multi-Day / Selective Swing & Shields):** `3` — `S04_MINERVINI_VCP_QULLAMAGGIE, S09_CONNORS_IBS_CONFORMAL_MR, S12_PEAD_DRIFT_COIL_BREAKOUT`
- **Cost-Killed Intraday Strategies (`Net 1x <= 0`):** `15`
- **Live `W5` Paper Concentrated Cohort (`P2 :8787`):** `34` closed RTs (`W5 >= 30 Ready: True`), WR `52.9%`, PF `0.59`, Mean `-11.9 bps`, Payoff `0.46x`, Fill Latency `3981.9 ms`, Dynamic Trail Shield `-0.76%` (`0` tail breaches)
- **Live `:8798` + `:8796` Read-Only State:** `mode = HALTED_DAILY_DRAWDOWN_BREAKER`, `daily_pnl_pct = -3.577%` vs `-1.5%` breaker (`halted = True`), `auto_adopt_owner_entries = True`, `:8796` `open_positions = 6`, `untracked = 0`, `exits_fired = 13`

## 2. Real Pre-Wall Sim (`<= 2026-10-15`) + Live Paper (`W5`) Strategy Merit Matrix

| Strategy ID | Cadence | Sim Ev / Days | Gross bps/d | Net 1x bps/d | Net 2x bps/d | t-stat (2x) | DSR (2x) | Paper RTs (WR / ExpR) | Hedge Wt | Merit Tier | Live Auto |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| `S04_MINERVINI_VCP_QULLAMAGGIE` | `DAILY_SWING` | 147 / 111 | +28.76 | +16.24 | +3.72 | +0.16 | n/a | 0 | 0.0663 | `STAGE3_PAPER_CRUCIBLE_HIGH_PRIORITY_SWING_OR_SHIELD` | `False` |
| `S09_CONNORS_IBS_CONFORMAL_MR` | `DAILY_SWING` | 323 / 172 | +24.98 | +13.26 | +1.54 | +0.13 | n/a | 0 | 0.0645 | `STAGE3_PAPER_CRUCIBLE_HIGH_PRIORITY_SWING_OR_SHIELD` | `False` |
| `S12_PEAD_DRIFT_COIL_BREAKOUT` | `DAILY_SWING` | 10 / 10 | +17.22 | +4.39 | -8.43 | -0.08 | n/a | 0 | 0.0592 | `STAGE3_PAPER_CRUCIBLE_HIGH_PRIORITY_SWING_OR_SHIELD` | `False` |
| `S03_STOCKBEE_4PCT_BURST_EP` | `DAILY_SWING` | 144 / 121 | +12.24 | -1.22 | -14.68 | -0.33 | n/a | 0 | 0.0561 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `H110_TRIAL107_DIAG_T1_TO_T0_RESIDUAL_LEADLAG` | `INTRADAY_1M` | 10480 / 524 | +3.73 | -6.06 | -15.86 | -10.00 | 0.000 | 0 | 0.0536 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `H110_TRIAL107_PRIMARY_DIRECTED_LEADLAG` | `INTRADAY_1M` | 6740 / 337 | +3.47 | -6.19 | -15.84 | -8.63 | 0.000 | 0 | 0.0535 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `H112_TRIAL106_424B5_DILUTION_SHORT` | `INTRADAY_1M` | 2078 / 596 | +2.88 | -7.44 | -17.75 | -4.72 | 0.000 | 0 | 0.0529 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `H110_TRIAL107_DIAG_EOD_LETF_CLOSEFLOW_8BPS` | `INTRADAY_1M` | 1171 / 402 | -1.09 | -9.88 | -18.67 | -9.97 | 0.000 | 0 | 0.0517 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S01_ZARATTINI_2024_ORB` | `INTRADAY_5M` | 10381 / 699 | +1.39 | -11.08 | -23.55 | -21.74 | n/a | 3 (100.0% / +0.12R) | 0.0511 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S06_VIX_TERM_STRUCTURE_SURFER` | `INTRADAY_5M` | 899 / 385 | +0.30 | -11.14 | -22.58 | -6.13 | n/a | 1 (0.0% / -0.52R) | 0.0511 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `H109_TRIAL105_BOATS_VWAP_FADE_SHORT` | `INTRADAY_1M` | 3182 / 439 | -1.51 | -11.96 | -22.41 | -5.02 | 0.000 | 0 | 0.0507 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S02_ZARATTINI_2025_NOISE_VWAP` | `INTRADAY_5M` | 1333 / 585 | -2.14 | -12.55 | -22.97 | -13.07 | n/a | 0 | 0.0504 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S08_BOATS_VWAP_HOLD_LONG` | `INTRADAY_5M` | 1547 / 499 | -0.53 | -13.89 | -27.25 | -9.05 | n/a | 0 | 0.0498 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S07_LETF_CLOSEFLOW_1530` | `INTRADAY_5M` | 2979 / 613 | -1.79 | -14.09 | -26.39 | -18.78 | n/a | 0 | 0.0497 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S13_VIX_DUAL_LONGSHORT_UNCAPPED` | `INTRADAY_5M` | 9132 / 447 | -3.45 | -15.79 | -28.14 | -22.93 | n/a | 0 | 0.0489 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S14_SOTA_EARLY_MOVER_BRACKET` | `INTRADAY_5M` | 3967 / 669 | -4.38 | -16.54 | -28.69 | -20.14 | n/a | 0 | 0.0485 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S05_BJZZ_INST_DARK_OFI_PASSIVE` | `INTRADAY_5M` | 5452 / 696 | -4.83 | -17.31 | -29.78 | -20.49 | n/a | 20 (65.0% / -0.06R) | 0.0482 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S10_TTM_SQUEEZE_KLINGER_KVO` | `INTRADAY_5M` | 1182 / 518 | -4.85 | -17.51 | -30.16 | -13.80 | n/a | 0 | 0.0481 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |
| `S11_CROSSASSET_LEADLAG_RS` | `INTRADAY_5M` | 879 / 300 | -10.94 | -22.91 | -34.87 | -12.39 | n/a | 0 | 0.0457 | `STAGE3_PAPER_P3_SHADOW_OR_SHIELD_ONLY` | `False` |

## 3. Governance Separation & Remediation Law

- **Owner-Order Protection:** ACTIVE_ALWAYS_ON: Every manual or TradingView order placed by the owner on Live (:8798/:8796) is automatically adopted with ADR-0022 <= 2.00% stop, >=10% exit reserve, and uncapped Chandelier.
- **Autonomous Live Gate:** FAIL_CLOSED_MERIT_GATED: Strictly requires BOTH Pre-Wall Sim HLZ/DSR/CPCV/PBO/SPA PASS AND Real Paper W5 (>=30 RTs, >=10 sessions, PF>=1.25, Payoff>=1.10, EB>0). Zero n=0/1 seed canaries allowed.
- **Audit Remediation:** Retire live :8787 merit_farm_status n=0/n=1 STAGE5_LOSSLESS_FOUNDATION_CANARY_10PCT exemptions (SOTA_LOSSLESS_FOUNDATION_RUNNER n=0, SOTA_ZERO_SELL_VWAP_BRACKET n=1, VIX_SPIKE_AND_CRUSH n=1) so autonomous live routing obeys strict W5 + HLZ provenance.
