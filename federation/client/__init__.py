"""
Flower client worker implementations for local hospital nodes in FedMed.
"""

from federation.client.fl_client import (
    FedMedClient,
    create_client,
    start_fedmed_client,
)

__all__ = [
    "FedMedClient",
    "create_client",
    "start_fedmed_client",
]
