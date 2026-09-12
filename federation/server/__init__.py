"""
Flower server coordinator, custom aggregation strategies, and CLI execution for FedMed.
"""

from federation.server.fl_server import (
    create_fedavg_strategy,
    get_evaluate_config_fn,
    get_fit_config_fn,
    get_initial_parameters,
    start_flower_server,
)
from federation.server.model_manager import GlobalModelManager
from federation.server.run_server import (
    main as run_server_main,
    parse_args as parse_server_args,
)
from federation.server.strategy import (
    FedMedStrategy,
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
    aggregate_weighted_parameters,
    create_fedmed_strategy,
    get_server_eval_fn,
)
from federation.server.sync_manager import (
    PendingClientUpdate,
    RoundSyncManager,
    StalePolicy,
)

__all__ = [
    "create_fedavg_strategy",
    "get_initial_parameters",
    "get_fit_config_fn",
    "get_evaluate_config_fn",
    "start_flower_server",
    "run_server_main",
    "parse_server_args",
    "FedMedStrategy",
    "create_fedmed_strategy",
    "aggregate_weighted_parameters",
    "aggregate_fit_metrics",
    "aggregate_evaluate_metrics",
    "get_server_eval_fn",
    "GlobalModelManager",
    "RoundSyncManager",
    "PendingClientUpdate",
    "StalePolicy",
]


