from __future__ import annotations

import argparse

import flwr as fl

from app.federated.client import HospitalClient


SERVER_ADDRESS = "127.0.0.1:8080"

HOSPITALS = {
    "hospital-1": {
        "name": "Hospital 1",
        "location": "Hyderabad",
    },
    "hospital-2": {
        "name": "Hospital 2",
        "location": "Warangal",
    },
    "hospital-3": {
        "name": "Hospital 3",
        "location": "Nizamabad",
    },
}


def start_hospital(
    hospital_id: str,
    fail: bool = False,
    retry_count: int = 0,
) -> None:
    """Start one mock hospital Flower client."""

    if hospital_id not in HOSPITALS:
        raise ValueError(
            f"Unknown hospital: {hospital_id}"
        )

    hospital = HOSPITALS[hospital_id]

    print("=" * 60)
    print(f"Starting {hospital_id}")
    print(f"Name: {hospital['name']}")
    print(f"Location: {hospital['location']}")
    print(f"Failure simulation: {fail}")
    print(f"Retry count: {retry_count}")
    print(f"Flower server: {SERVER_ADDRESS}")
    print("=" * 60)

    client = HospitalClient(
        hospital_id=hospital_id,
        fail=fail,
        retry_count=retry_count,
    )

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client(),
        insecure=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FedMed mock hospital Flower client"
    )

    parser.add_argument(
        "hospital_id",
        choices=list(HOSPITALS.keys()),
        help="Hospital node to start",
    )

    parser.add_argument(
        "--fail",
        action="store_true",
        help="Continuously simulate hospital failure",
    )

    parser.add_argument(
        "--retry",
        type=int,
        default=0,
        help=(
            "Number of temporary client-side retries "
            "to demonstrate before continuing"
        ),
    )

    args = parser.parse_args()

    start_hospital(
        hospital_id=args.hospital_id,
        fail=args.fail,
        retry_count=args.retry,
    )


if __name__ == "__main__":
    main()