from __future__ import annotations

import flwr as fl

from app.federated.strategy import FedMedStrategy


SERVER_ADDRESS = "0.0.0.0:8080"
NUM_ROUNDS = 3
ROUND_TIMEOUT = 30


def start_server() -> None:
    """Start the FedMed Flower federated learning server."""

    strategy = FedMedStrategy()

    print("FedMed Flower Server starting...")
    print("FedAvg strategy initialized")
    print(f"Federated rounds: {NUM_ROUNDS}")
    print(f"Round timeout: {ROUND_TIMEOUT} seconds")
    print("Waiting for federated clients...")

    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(
            num_rounds=NUM_ROUNDS,
            round_timeout=ROUND_TIMEOUT,
        ),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()