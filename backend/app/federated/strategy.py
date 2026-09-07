import flwr as fl

from app.federated.metrics import (
    record_round,
    set_hospital_status,
    set_training_status,
)


class FedMedStrategy(fl.server.strategy.FedAvg):
    """FedAvg strategy used by the FedMed federated learning server."""

    def __init__(self):
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
        )

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client updates and record training status."""

        set_training_status("training")

        for client, _ in results:
            hospital_id = client.cid

            set_hospital_status(
                hospital_id,
                "trained",
            )

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
        """Aggregate evaluation metrics."""

        aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        loss = None

        if aggregated[0] is not None:
            loss = float(aggregated[0])

            print(
                f"[Round {server_round}] "
                f"Global evaluation loss: {loss:.4f}"
            )

        record_round(
            round_number=server_round,
            loss=loss,
        )

        if server_round >= 3:
            set_training_status("completed")

        return aggregated