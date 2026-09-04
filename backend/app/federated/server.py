import flwr as fl

from app.federated.strategy import FedMedStrategy


def start_server() -> None:
    """Start the FedMed Flower federated learning server."""

    strategy = FedMedStrategy()

    print("FedMed Flower Server starting...")
    print("FedAvg strategy initialized")
    print("Waiting for federated clients...")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(
            num_rounds=1,
        ),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server() 