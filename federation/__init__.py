"""
FedMed Federated Learning Module.
Flower-based federated training framework for privacy-preserving medical imaging.
"""

from federation.api_bridge import FedMedAPIBridge
from federation.simulate import generate_simulation_plots, run_simulation

__version__ = "0.1.0"
__all__ = ["run_simulation", "generate_simulation_plots", "FedMedAPIBridge"]


