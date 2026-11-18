"""SOTA 5-Stage Quantitative Strategy Compounding & Paper-Broker Manifest Engine."""

from sota_compounding.lifecycle import (
    ADR0022_MAX_STOP_PCT,
    ALPACA_PAPER_MAX_RPM,
    IBKR_PAPER_MAX_MSG_PER_MIN,
    STUDY_WALL_DATE,
    BrokerStrategyManifest,
    EmpiricalBayesTuningResult,
    LifecycleEvaluation,
    PaperExecutionMetrics,
    PrewallBacktestMetrics,
    SimStageMetrics,
    evaluate_full_lifecycle,
    export_broker_strategy_manifest,
    hlz_dynamic_t_hurdle,
    quantize_nms_stop,
    tune_empirical_bayes_ucb,
)
from sota_compounding.strategy_template import (
    ARCHETYPES,
    scaffold_converged_strategy,
    validate_converged_strategy_template,
)

__all__ = [
    "ADR0022_MAX_STOP_PCT",
    "ALPACA_PAPER_MAX_RPM",
    "ARCHETYPES",
    "IBKR_PAPER_MAX_MSG_PER_MIN",
    "STUDY_WALL_DATE",
    "BrokerStrategyManifest",
    "EmpiricalBayesTuningResult",
    "LifecycleEvaluation",
    "PaperExecutionMetrics",
    "PrewallBacktestMetrics",
    "SimStageMetrics",
    "evaluate_full_lifecycle",
    "export_broker_strategy_manifest",
    "hlz_dynamic_t_hurdle",
    "quantize_nms_stop",
    "scaffold_converged_strategy",
    "tune_empirical_bayes_ucb",
    "validate_converged_strategy_template",
]
