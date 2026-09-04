import flwr as fl


class FedMedStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy used by the FedMed federated server."""

    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
        )