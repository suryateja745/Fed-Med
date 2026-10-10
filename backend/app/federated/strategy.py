from __future__ import annotations

from typing import Any

import flwr as fl
import numpy as np

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from app.federated.metrics import (
    record_hospital_metrics,
    record_round,
    record_security_updates,
    set_hospital_status,
    set_training_status,
)

from app.federated.secure_aggregation import (
    aggregate_encrypted_chunks,
)


KNOWN_HOSPITALS = {
    "hospital-1",
    "hospital-2",
    "hospital-3",
}


def fit_config(
    server_round: int,
) -> dict[str, Any]:
    """Configuration sent to hospitals for local training."""
    return {
        "server_round": server_round,
        "epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "dp_max_norm": 1.0,
        "dp_noise_multiplier": 0.1,
    }


def evaluate_config(
    server_round: int,
) -> dict[str, Any]:
    """Configuration sent to hospitals for evaluation."""
    return {
        "server_round": server_round,
        "batch_size": 1,
    }


class FedMedStrategy(fl.server.strategy.FedAvg):
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
        """Record hospital metrics and aggregate encrypted updates."""

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

        successful_hospitals: set[str] = set()

        # ---------------------------------------------------------
        # Record the actual metrics returned by each successful
        # hospital. Do not manufacture missing metric values.
        # ---------------------------------------------------------
        for _, fit_res in results:
            fit_metrics = dict(fit_res.metrics or {})

            hospital_id = fit_metrics.get(
                "hospital_id"
            )

            if hospital_id:
                hospital_id = str(hospital_id)

                record_hospital_metrics(
                    hospital_id=hospital_id,
                    round_number=server_round,
                    metrics=fit_metrics,
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
                    f"{hospital_id} -> metrics recorded"
                )

        # ---------------------------------------------------------
        # Identify clients that failed during the round.
        # ---------------------------------------------------------
        failed_hospitals: set[str] = set()

        for failure in failures:
            if (
                isinstance(failure, tuple)
                and len(failure) == 2
            ):
                failed_client, detail = failure

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

        # Identify expected hospitals that did not report a
        # successful fit result when the full client set responded.
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

        # There is nothing to aggregate when no client succeeded.
        if not results:
            print(
                f"[Round {server_round}] "
                "No successful encrypted updates"
            )

            return None

        # ---------------------------------------------------------
        # Convert the encrypted payloads back to byte chunks.
        # Chunk order must remain identical across hospitals.
        # ---------------------------------------------------------
        client_encrypted_updates: list[
            list[bytes]
        ] = []

        for _, fit_res in results:
            arrays = parameters_to_ndarrays(
                fit_res.parameters
            )

            client_chunks: list[bytes] = []

            for array in arrays:
                client_chunks.append(
                    np.asarray(
                        array,
                        dtype=np.uint8,
                    ).tobytes()
                )

            client_encrypted_updates.append(
                client_chunks
            )

        chunk_count = len(
            client_encrypted_updates[0]
        )

        print(
            f"[Round {server_round}] "
            f"Received {len(results)} encrypted client update(s)"
        )

        print(
            f"[Round {server_round}] "
            f"Encrypted chunks per client: {chunk_count}"
        )

        print(
            f"[Round {server_round}] "
            "Performing TenSEAL CKKS aggregation"
        )

        try:
            encrypted_sum = (
                aggregate_encrypted_chunks(
                    client_encrypted_updates
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

        # Preserve existing dashboard update-count semantics.
        record_security_updates(
            len(results)
        )

        aggregation_metrics = {
            "encrypted_updates": len(results),
            "encrypted_chunks": chunk_count,
            "secure_aggregation": True,
            "encryption": "TenSEAL-CKKS",
            "plaintext_updates_exposed": False,
            "dp_enabled": True,
        }

        print(
            f"[Round {server_round}] "
            "TenSEAL aggregation completed"
        )

        print(
            f"[Round {server_round}] "
            f"FedAvg mean computed from {len(results)} client(s)"
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
        """Aggregate evaluation and save the global round result."""

        print(
            f"[Round {server_round}] "
            f"Received {len(results)} evaluation result(s)"
        )

        print(
            f"[Round {server_round}] "
            f"Received {len(failures)} evaluation failure(s)"
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
                "No aggregated evaluation result"
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
        """Extract a known hospital ID from the client proxy."""

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