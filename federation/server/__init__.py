"""
Flower server coordinator, aggregation strategies, and CLI execution for FedMed.
"""

from federation.server.fl_server import (
    create_fedavg_strategy,
    get_evaluate_config_fn,
    get_fit_config_fn,
    get_initial_parameters,
    start_flower_server,
)
from federation.server.run_server import (
    main as run_server_main,
    parse_args as parse_server_args,
)

__all__ = [
    "create_fedavg_strategy",
    "get_initial_parameters",
    "get_fit_config_fn",
    "get_evaluate_config_fn",
    "start_flower_server",
    "run_server_main",
    "parse_server_args",
]
