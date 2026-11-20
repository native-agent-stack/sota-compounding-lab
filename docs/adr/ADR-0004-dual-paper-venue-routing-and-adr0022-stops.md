# ADR-0004: Dual Paper Venue Routing (`200/min` Alpaca + `3,000/min` IBKR `:4002`) & `ADR-0022` Stops

- **Status:** Accepted
- **Date:** 2026-11-17
- **Organization:** `https://github.com/native-agent-stack`
- **Enforced By:** `quantize_nms_stop`, `export_broker_strategy_manifest`, `resolve_cutover_regime`

## Context

Autonomous paper execution spans regular, extended (`04:00–20:00 ET`), and overnight (`20:00–04:00 ET` `BOATS` / `Dec 6, 2026` `23/5` SIP) sessions while strictly prohibiting live real-money execution.

## Decision

1. Enforce `alpaca_order_rpm_cap <= 200` (`15/min` protective-exit reserve, isolated from market-data RPM) on `paper-api.alpaca.markets` (`PA...`) and `ibkr_port == 4002` (`<= 3,000 msg/min`, `DU...`); hard-refuse `api.alpaca.markets` and live ports `4001` / `7496`.
2. Quantize every protective stop via `quantize_nms_stop` to US Reg NMS Rule 612 ticks (`$0.01` at `>= $1.00`, `$0.0001` below `$1.00`) conservatively inward so realized stop distance never exceeds `ADR-0022 <= 2.00%`.
3. Tag forward paper attribution by regime (`pre_2026_12_06_24x5_boats` vs. `post_2026_12_06_23x5_sip`) gated by `enable_dec6_23x5_sip`.

## Verification & Evidence
- Backfilled under `sota-upstream-shelf-lock ADR-0023` (2026-11-20): anchored to repository `UPSTREAM.md`, `upstream-map.toml`, and git history of `adr0023-19adr-evidence` (`shelf_lock.hard_upstream_rule` verified).
