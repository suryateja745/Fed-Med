from __future__ import annotations

from typing import Any

import flwr as fl
import numpy as np

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from app.federated.metrics import (
    record_round,
    record_security_updates,
    set_hospital_status,
    set_training_status,
)

from app.federated.secure_aggregation import (
    aggregate_encrypted_updates,
)


KNOWN_HOSPITALS = {
    "hospital-1",
    "hospital-2",
    "hospital-3",
}


def fit_config(
    server_round: int,
) -> dict[str, int]:
    return {
        "server_round": server_round,
    }


def evaluate_config(
    server_round: int,
) -> dict[str, int]:
    return {
        "server_round": server_round,
    }


class FedMedStrategy(
    fl.server.strategy.FedAvg
):
    """FedMed federated strategy."""

    def __init__(self) -> None:
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
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
        """Aggregate TenSEAL-encrypted client updates."""

        print(
            f"[Round {server_round}] "
            f"Received {len(results)} "
            f"successful fit result(s)"
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
                hospital_id = str(
                    hospital_id
                )

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
                    f"{hospital_id} -> trained"
                )

        # ---------------------------------------------------------
        # Failed hospitals
        # ---------------------------------------------------------
        failed_hospitals: set[str] = set()

        for failure in failures:
            if (
                isinstance(failure, tuple)
                and len(failure) == 2
            ):
                failed_client, detail = (
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
                    f"Client failure: {detail}"
                )
            else:
                print(
                    f"[Round {server_round}] "
                    f"Client failure: {failure}"
                )

        # Infer missing hospital if necessary.
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

        # Persist failures.
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
                f"{hospital_id} -> timeout/failure"
            )

        # ---------------------------------------------------------
        # Secure aggregation
        # ---------------------------------------------------------
        if not results:
            print(
                f"[Round {server_round}] "
                f"No successful encrypted updates"
            )
            return None

        encrypted_updates: list[bytes] = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(
                fit_res.parameters
            )

            for array in arrays:
                encrypted_updates.append(
                    np.asarray(
                        array,
                        dtype=np.uint8,
                    ).tobytes()
                )

        print(
            f"[Round {server_round}] "
            f"Received {len(encrypted_updates)} "
            f"encrypted update(s)"
        )

        print(
            f"[Round {server_round}] "
            f"Performing TenSEAL CKKS secure aggregation"
        )

        try:
            encrypted_sum = (
                aggregate_encrypted_updates(
                    encrypted_updates
                )
            )
        except Exception as exc:
            print(
                f"[Round {server_round}] "
                f"Encrypted aggregation failed: {exc}"
            )
            set_training_status(
                "error",
                server_round,
            )
            return None

        # Convert encrypted aggregate sum to FedAvg mean.
        aggregate_sum = np.asarray(
            encrypted_sum,
            dtype=np.float32,
        )

        aggregate_mean = (
            aggregate_sum / len(results)
        )

        aggregated_parameters = (
            ndarrays_to_parameters(
                [aggregate_mean]
            )
        )

        record_security_updates(
            len(encrypted_updates)
        )

        aggregation_metrics = {
            "encrypted_updates": len(
                encrypted_updates
            ),
            "secure_aggregation": True,
            "encryption": "TenSEAL-CKKS",
            "plaintext_updates_exposed": False,
        }

        print(
            f"[Round {server_round}] "
            f"TenSEAL secure aggregation completed"
        )

        print(
            f"[Round {server_round}] "
            f"FedAvg mean computed from "
            f"{len(results)} client(s)"
        )

        return (
            aggregated_parameters,
            aggregation_metrics,
        )

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ):
        print(
            f"[Round {server_round}] "
            f"Received {len(results)} "
            f"evaluation result(s)"
        )

        print(
            f"[Round {server_round}] "
            f"Received {len(failures)} "
            f"evaluation failure(s)"
        )

        aggregated = (
            super().aggregate_evaluate(
                server_round,
                results,
                failures,
            )
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
            f"Evaluation loss: "
            f"{float(loss):.4f}"
        )

        return aggregated

    @staticmethod
    def _extract_hospital_id(
        client_proxy: Any,
    ) -> str | None:
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