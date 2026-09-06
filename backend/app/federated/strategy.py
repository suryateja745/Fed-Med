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

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client updates and report federated training metrics."""

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated[0] is not None:
            print(
                f"[Round {server_round}] "
                f"Federated aggregation completed "
                f"for {len(results)} hospitals"
            )

        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation metrics from all hospitals."""

        aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if aggregated[0] is not None:
            print(
                f"[Round {server_round}] "
                f"Global evaluation loss: {aggregated[0]:.4f}"
            )

        return aggregated