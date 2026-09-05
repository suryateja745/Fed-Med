from __future__ import annotations

import numpy as np
import flwr as fl


class HospitalClient(fl.client.NumPyClient):
    """Mock Flower client representing one hospital."""

    def __init__(self, hospital_id: str) -> None:
        self.hospital_id = hospital_id

        # Mock model parameters for today's connectivity task.
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
        print(f"[{self.hospital_id}] fit")

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
            },
        )

    def evaluate(self, parameters, config):
        print(f"[{self.hospital_id}] evaluate")

        return (
            0.5,
            1,
            {
                "hospital_id": self.hospital_id,
                "status": "evaluated",
            },
        )