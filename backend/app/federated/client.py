from __future__ import annotations

import time
from typing import Any

import flwr as fl
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector

from app.federated.privacy import (
    privatize_update,
)
from app.federated.secure_aggregation import (
    encrypt_update_chunks,
)
from app.ml.federated_training import (
    evaluate_from_global_vector,
    initial_parameter_vector,
    train_from_global_vector,
)


class HospitalClient(fl.client.NumPyClient):
    """Flower client representing one hospital."""

    def __init__(
        self,
        hospital_id: str,
        fail: bool = False,
        retry_count: int = 0,
    ) -> None:
        self.hospital_id = hospital_id

        torch.manual_seed(42)

        # Initial model parameters.
        self.parameters = [
            initial_parameter_vector()
        ]

        self.fail = bool(fail)

        self.retry_limit = max(
            0,
            int(retry_count),
        )

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
        print(
            f"[{self.hospital_id}] "
            "get_parameters"
        )

        return self.parameters

    def fit(
        self,
        parameters,
        config: dict[str, Any],
    ):
        server_round = int(
            config.get(
                "server_round",
                0,
            )
        )

        print(
            f"[{self.hospital_id}] "
            f"training round {server_round}"
        )

        if self.fail:
            print(
                f"[{self.hospital_id}] "
                "simulated timeout/failure "
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

        if (
            self.retry_limit > 0
            and self.retry_count
            < self.retry_limit
        ):
            self.retry_count += 1

            print(
                f"[{self.hospital_id}] "
                "temporary failure in round "
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

        global_vector = np.asarray(
            parameters[0],
            dtype=np.float32,
        ).reshape(-1)

        epochs = int(
            config.get(
                "epochs",
                1,
            )
        )

        batch_size = int(
            config.get(
                "batch_size",
                1,
            )
        )

        learning_rate = float(
            config.get(
                "learning_rate",
                1e-3,
            )
        )

        # ---------------------------------------------------------
        # REAL LOCAL 3D U-NET TRAINING
        # ---------------------------------------------------------
        training_result = (
            train_from_global_vector(
                global_vector=global_vector,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
        )

        local_vector = np.asarray(
            training_result[
                "updated_vector"
            ],
            dtype=np.float32,
        )

        # ---------------------------------------------------------
        # DIFFERENTIAL PRIVACY ON LOCAL UPDATE
        # ---------------------------------------------------------
        local_delta = (
            local_vector - global_vector
        )

        hospital_seed = (
            server_round * 100
            + self._hospital_number()
        )

        private_delta = privatize_update(
            local_delta,
            max_norm=float(
                config.get(
                    "dp_max_norm",
                    1.0,
                )
            ),
            noise_multiplier=float(
                config.get(
                    "dp_noise_multiplier",
                    0.1,
                )
            ),
            seed=hospital_seed,
        )

        private_vector = (
            global_vector + private_delta
        ).astype(np.float32)

        self.parameters = [
            private_vector
        ]

        # ---------------------------------------------------------
        # ENCRYPT AFTER DP
        # ---------------------------------------------------------
        encrypted_payloads = (
            encrypt_update_chunks(
                private_vector
            )
        )

        encrypted_parameters = [
            np.frombuffer(
                payload,
                dtype=np.uint8,
            )
            for payload in encrypted_payloads
        ]

        print(
            f"[{self.hospital_id}] "
            f"train loss="
            f"{training_result['train_loss']:.4f}, "
            f"val loss="
            f"{training_result['val_loss']:.4f}, "
            f"Dice="
            f"{training_result['dice']:.4f}"
        )

        print(
            f"[{self.hospital_id}] "
            f"round {server_round} "
            "training completed"
        )

        print(
            f"[{self.hospital_id}] "
            "DP applied before encryption"
        )

        print(
            f"[{self.hospital_id}] "
            "update encrypted with TenSEAL CKKS"
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
                "dp_enabled": True,
                "dp_max_norm": float(
                    config.get(
                        "dp_max_norm",
                        1.0,
                    )
                ),
                "dp_noise_multiplier": float(
                    config.get(
                        "dp_noise_multiplier",
                        0.1,
                    )
                ),
                "train_loss": float(
                    training_result["train_loss"]
                ),
                "val_loss": float(
                    training_result["val_loss"]
                ),
                "dice": float(
                    training_result["dice"]
                ),
            },
        )

    def evaluate(
        self,
        parameters,
        config: dict[str, Any],
    ):
        server_round = int(
            config.get(
                "server_round",
                0,
            )
        )

        global_vector = np.asarray(
            parameters[0],
            dtype=np.float32,
        ).reshape(-1)

        loss, dice = (
            evaluate_from_global_vector(
                global_vector,
                batch_size=int(
                    config.get(
                        "batch_size",
                        1,
                    )
                ),
            )
        )

        print(
            f"[{self.hospital_id}] "
            f"evaluate round {server_round}: "
            f"loss={loss:.4f}, "
            f"Dice={dice:.4f}"
        )

        return (
            float(loss),
            1,
            {
                "hospital_id": self.hospital_id,
                "status": "evaluated",
                "round": server_round,
                "dice": float(dice),
            },
        )

    def _hospital_number(self) -> int:
        try:
            return int(
                self.hospital_id.rsplit(
                    "-",
                    1,
                )[1]
            )
        except (
            ValueError,
            IndexError,
        ):
            return 1