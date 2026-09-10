from __future__ import annotations

from typing import Any

import flwr as fl

from app.federated.metrics import (
    record_round,
    set_hospital_status,
    set_training_status,
)


KNOWN_HOSPITALS = {
    "hospital-1",
    "hospital-2",
    "hospital-3",
}


def fit_config(server_round: int) -> dict[str, int]:
    """Send the current federated round to each client."""
    return {
        "server_round": server_round,
    }


def evaluate_config(server_round: int) -> dict[str, int]:
    """Send the current federated round to each client."""
    return {
        "server_round": server_round,
    }


class FedMedStrategy(fl.server.strategy.FedAvg):
    """FedMed FedAvg strategy with node failure handling."""

    def __init__(self) -> None:
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,

            # Three hospitals exist, but two successful clients
            # are enough to complete a round.
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=3,

            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate successful updates and record failed nodes."""

        print(
            f"[Round {server_round}] "
            f"Received {len(results)} successful fit result(s)"
        )

        print(
            f"[Round {server_round}] "
            f"Received {len(failures)} failure(s)"
        )

        set_training_status(
            "training",
            server_round,
        )

        # ---------------------------------------------------------
        # Successful hospitals
        # ---------------------------------------------------------
        successful_hospitals: set[str] = set()

        for _, fit_res in results:
            hospital_id = fit_res.metrics.get(
                "hospital_id"
            )

            if hospital_id:
                hospital_id = str(hospital_id)

                successful_hospitals.add(
                    hospital_id
                )

                set_hospital_status(
                    hospital_id,
                    "trained",
                    round_number=server_round,
                )

                print(
                    f"[Round {server_round}] "
                    f"{hospital_id} → trained"
                )

        # ---------------------------------------------------------
        # Failed hospitals
        #
        # Flower can return failures in different forms:
        #
        #   (ClientProxy, failure)
        #
        # OR a direct exception such as GrpcBridgeClosed.
        # ---------------------------------------------------------
        failed_hospitals: set[str] = set()

        for failure in failures:

            if (
                isinstance(failure, tuple)
                and len(failure) == 2
            ):
                failed_client, failure_detail = (
                    failure
                )

                hospital_id = (
                    self._extract_hospital_id(
                        failed_client
                    )
                )

                if hospital_id:
                    failed_hospitals.add(
                        hospital_id
                    )

                print(
                    f"[Round {server_round}] "
                    f"Client failure: "
                    f"{failure_detail}"
                )

            else:
                print(
                    f"[Round {server_round}] "
                    f"Client failure: {failure}"
                )

        # ---------------------------------------------------------
        # Infer the missing hospital when all 3 clients were
        # selected but only 2 returned results.
        # ---------------------------------------------------------
        if (
            len(results) + len(failures)
            >= len(KNOWN_HOSPITALS)
        ):
            missing_hospitals = (
                KNOWN_HOSPITALS
                - successful_hospitals
            )

            failed_hospitals.update(
                missing_hospitals
            )

        # ---------------------------------------------------------
        # Persist failure/timeout state
        # ---------------------------------------------------------
        for hospital_id in sorted(
            failed_hospitals
        ):
            set_hospital_status(
                hospital_id,
                "timeout",
                round_number=server_round,
            )

            print(
                f"[Round {server_round}] "
                f"{hospital_id} → timeout/failure"
            )

        # ---------------------------------------------------------
        # Normal FedAvg aggregation
        # ---------------------------------------------------------
        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated is None:
            print(
                f"[Round {server_round}] "
                f"FedAvg aggregation returned no result"
            )

            return None

        parameters, metrics = aggregated

        print(
            f"[Round {server_round}] "
            f"FedAvg aggregation completed with "
            f"{len(results)} successful client(s)"
        )

        return parameters, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        """Aggregate evaluations and save round metrics."""

        print(
            f"[Round {server_round}] "
            f"Received {len(results)} evaluation result(s)"
        )

        print(
            f"[Round {server_round}] "
            f"Received {len(failures)} evaluation failure(s)"
        )

        aggregated = super().aggregate_evaluate(
            server_round,
            results,
            failures,
        )

        if aggregated is None:
            print(
                f"[Round {server_round}] "
                f"No aggregated evaluation result"
            )

            return None

        loss, metrics = aggregated

        round_status = (
            "completed_with_failure"
            if failures
            else "completed"
        )

        record_round(
            round_number=server_round,
            loss=float(loss),
            status=round_status,
        )

        if server_round >= 3:
            set_training_status(
                "completed",
                server_round,
            )

        print(
            f"[Round {server_round}] "
            f"Evaluation loss: {float(loss):.4f}"
        )

        return aggregated

    @staticmethod
    def _extract_hospital_id(
        client_proxy: Any,
    ) -> str | None:
        """Try to obtain hospital ID from a Flower client proxy."""

        for attribute in (
            "cid",
            "node_id",
            "client_id",
        ):
            value = getattr(
                client_proxy,
                attribute,
                None,
            )

            if value:
                value = str(value)

                if value in KNOWN_HOSPITALS:
                    return value

        return None