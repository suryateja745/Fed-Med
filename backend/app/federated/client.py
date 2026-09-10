from __future__ import annotations

import time
from typing import Any

import flwr as fl
import numpy as np


class HospitalClient(fl.client.NumPyClient):
    """Mock Flower client representing one hospital."""

    def __init__(
        self,
        hospital_id: str,
        fail: bool = False,
        retry_count: int = 0,
    ) -> None:
        self.hospital_id = hospital_id

        # Mock model parameters.
        self.parameters = [
            np.array([0.0], dtype=np.float32)
        ]

        # Failure/retry configuration.
        self.fail = fail
        self.retry_limit = max(0, int(retry_count))

        # Runtime counters.
        self.retry_count = 0
        self.failed_rounds: list[int] = []

    def get_properties(
        self,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Return hospital metadata to the Flower server."""

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
        """Return the current global/model parameters."""

        print(f"[{self.hospital_id}] get_parameters")

        return self.parameters

    def fit(
        self,
        parameters,
        config: dict[str, Any],
    ):
        """Perform mock local training for one federated round."""

        server_round = int(config.get("server_round", 0))

        print(
            f"[{self.hospital_id}] "
            f"training round {server_round}"
        )

        # Permanent failure mode.
        #
        # This is useful for demonstrating that the Flower server
        # can continue with the remaining hospitals.
        if self.fail:
            print(
                f"[{self.hospital_id}] "
                f"simulated timeout/failure "
                f"in round {server_round}"
            )

            self.failed_rounds.append(server_round)

            # A small delay allows the server round timeout to
            # demonstrate real failure handling without waiting
            # indefinitely.
            time.sleep(2)

            raise RuntimeError(
                f"{self.hospital_id} failed "
                f"during round {server_round}"
            )

        # Optional client-side retry simulation.
        #
        # This models a temporary local failure followed by a
        # successful retry.
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

            # Simulate a short recovery interval.
            time.sleep(1)

            print(
                f"[{self.hospital_id}] "
                f"retry succeeded for round "
                f"{server_round}"
            )

        # Mock local model update.
        updated_parameters = [
            np.asarray(parameters[0]) + 0.1
        ]

        self.parameters = updated_parameters

        print(
            f"[{self.hospital_id}] "
            f"round {server_round} training completed"
        )

        return (
            updated_parameters,
            1,
            {
                "hospital_id": self.hospital_id,
                "status": "trained",
                "round": server_round,
                "retry_count": self.retry_count,
            },
        )

    def evaluate(
        self,
        parameters,
        config: dict[str, Any],
    ):
        """Evaluate the current global model."""

        server_round = int(config.get("server_round", 0))

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