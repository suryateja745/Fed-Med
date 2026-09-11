from __future__ import annotations

import time
from typing import Any

import flwr as fl
import numpy as np

from app.federated.secure_aggregation import encrypt_update


class HospitalClient(fl.client.NumPyClient):
    """Mock Flower client representing one hospital."""

    def __init__(
        self,
        hospital_id: str,
        fail: bool = False,
        retry_count: int = 0,
    ) -> None:
        self.hospital_id = hospital_id

        # Local mock model state.
        self.parameters = [
            np.array([0.0], dtype=np.float32)
        ]

        self.fail = bool(fail)
        self.retry_limit = max(0, int(retry_count))

        self.retry_count = 0
        self.failed_rounds: list[int] = []

    def get_properties(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "hospital_id": self.hospital_id,
            "node_type": "hospital",
            "status": "connected",
            "retry_count": self.retry_count,
        }

    def get_parameters(
        self,
        config: dict[str, Any],
    ):
        print(f"[{self.hospital_id}] get_parameters")
        return self.parameters

    def fit(
        self,
        parameters,
        config: dict[str, Any],
    ):
        """Perform local training, then encrypt the update."""

        server_round = int(
            config.get("server_round", 0)
        )

        print(
            f"[{self.hospital_id}] "
            f"training round {server_round}"
        )

        # Permanent failure simulation.
        if self.fail:
            print(
                f"[{self.hospital_id}] "
                f"simulated timeout/failure "
                f"in round {server_round}"
            )

            self.failed_rounds.append(
                server_round
            )

            time.sleep(2)

            raise RuntimeError(
                f"{self.hospital_id} failed "
                f"during round {server_round}"
            )

        # Temporary retry simulation.
        if (
            self.retry_limit > 0
            and self.retry_count < self.retry_limit
        ):
            self.retry_count += 1

            print(
                f"[{self.hospital_id}] "
                f"temporary failure in round "
                f"{server_round}; "
                f"retry {self.retry_count}/"
                f"{self.retry_limit}"
            )

            time.sleep(1)

            print(
                f"[{self.hospital_id}] "
                f"retry succeeded for round "
                f"{server_round}"
            )

        # ---------------------------------------------------------
        # Mock local training
        # ---------------------------------------------------------
        updated_parameters = [
            np.asarray(
                parameters[0],
                dtype=np.float32,
            ) + 0.1
        ]

        # Plaintext remains local to the hospital.
        self.parameters = updated_parameters

        # ---------------------------------------------------------
        # Encrypt before sending to Flower
        # ---------------------------------------------------------
        encrypted_parameters = []

        for parameter in updated_parameters:
            encrypted_bytes = encrypt_update(
                parameter
            )

            encrypted_parameters.append(
                np.frombuffer(
                    encrypted_bytes,
                    dtype=np.uint8,
                )
            )

        print(
            f"[{self.hospital_id}] "
            f"round {server_round} training completed"
        )

        print(
            f"[{self.hospital_id}] "
            f"update encrypted with TenSEAL CKKS"
        )

        return (
            encrypted_parameters,
            1,
            {
                "hospital_id": self.hospital_id,
                "status": "encrypted",
                "round": server_round,
                "retry_count": self.retry_count,
                "encryption": "TenSEAL-CKKS",
            },
        )

    def evaluate(
        self,
        parameters,
        config: dict[str, Any],
    ):
        """Evaluate the current global model."""

        server_round = int(
            config.get("server_round", 0)
        )

        print(
            f"[{self.hospital_id}] "
            f"evaluate round {server_round}"
        )

        loss = 0.5

        return (
            loss,
            1,
            {
                "hospital_id": self.hospital_id,
                "status": "evaluated",
                "round": server_round,
            },
        )