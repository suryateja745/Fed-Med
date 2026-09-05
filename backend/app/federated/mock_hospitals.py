from __future__ import annotations

import argparse

import flwr as fl

from app.federated.client import HospitalClient


SERVER_ADDRESS = "127.0.0.1:8080"

HOSPITALS = {
    "hospital-1": {
        "name": "Hospital 1",
        "dataset": "mock-dataset-1",
    },
    "hospital-2": {
        "name": "Hospital 2",
        "dataset": "mock-dataset-2",
    },
    "hospital-3": {
        "name": "Hospital 3",
        "dataset": "mock-dataset-3",
    },
}


def start_hospital(hospital_id: str) -> None:
    info = HOSPITALS[hospital_id]

    print("=" * 50)
    print(f"Starting {info['name']}")
    print(f"Hospital ID: {hospital_id}")
    print(f"Dataset: {info['dataset']}")
    print(f"Flower server: {SERVER_ADDRESS}")
    print("=" * 50)

    client = HospitalClient(hospital_id)

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
        insecure=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedMed mock hospital client"
    )

    parser.add_argument(
        "hospital_id",
        choices=list(HOSPITALS.keys()),
    )

    args = parser.parse_args()

    start_hospital(args.hospital_id)


if __name__ == "__main__":
    main()