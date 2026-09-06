from __future__ import annotations

import numpy as np
import flwr as fl


class HospitalClient(fl.client.NumPyClient):
    """Mock Flower client representing one hospital."""

    def __init__(self, hospital_id: str) -> None:
        self.hospital_id = hospital_id
        self.parameters = [
            np.array([0.0], dtype=np.float32)
        ]

    def get_properties(self, config):
        return {
            "hospital_id": self.hospital_id,
            "node_type": "hospital",
            "status": "connected",
        }

    def get_parameters(self, config):
        print(f"[{self.hospital_id}] get_parameters")
        return self.parameters

    def fit(self, parameters, config):
        server_round = config.get("server_round", 0)

        print(
            f"[{self.hospital_id}] "
            f"training round {server_round}"
        )

        updated_parameters = [
            np.asarray(parameters[0]) + 0.1
        ]

        self.parameters = updated_parameters

        return (
            updated_parameters,
            1,
            {
                "hospital_id": self.hospital_id,
                "status": "trained",
                "round": server_round,
            },
        )

    def evaluate(self, parameters, config):
        server_round = config.get("server_round", 0)

        print(
            f"[{self.hospital_id}] "
            f"evaluating round {server_round}"
        )

        return (
            0.5,
            1,
            {
                "hospital_id": self.hospital_id,
                "status": "evaluated",
                "round": server_round,
            },
        )