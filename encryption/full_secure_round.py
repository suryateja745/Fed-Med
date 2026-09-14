import torch
import sys, os

sys.path.append(os.path.dirname(__file__))
from pytorch_encryption import create_context, decrypt_tensor
from secure_pipeline import secure_transmit
from federated_aggregation import aggregate_encrypted_weights

def run_secure_federated_round(hospital_weights_list, epsilon=5.0):
    """
    Simulates one complete federated learning round with full privacy protection:
    1. Each hospital adds DP noise + encrypts locally
    2. Server aggregates encrypted updates (never sees raw data)
    3. Final global model is decrypted for use
    """
    context = create_context()
    encrypted_updates = []
    shape = None

    print("--- Local hospital processing (DP + Encryption) ---")
    for i, weights in enumerate(hospital_weights_list):
        encrypted, shape = secure_transmit(context, weights, epsilon=epsilon)
        encrypted_updates.append(encrypted)
        print(f"Hospital {i+1}: local weights protected and sent to server.")

    print("\n--- Server-side secure aggregation ---")
    aggregated = aggregate_encrypted_weights(encrypted_updates)
    print("Server averaged all hospital updates without decrypting any of them.")

    global_model = decrypt_tensor(aggregated, shape)
    print(f"\n--- New global model (Round complete) ---\n{global_model}")
    return global_model

if __name__ == "__main__":
    hospital_data = [
        torch.tensor([[0.10, 0.20], [0.30, 0.40]]),
        torch.tensor([[0.12, 0.18], [0.29, 0.42]]),
        torch.tensor([[0.11, 0.22], [0.31, 0.38]]),
    ]

    final_model = run_secure_federated_round(hospital_data, epsilon=5.0)
    print("\nFederated round completed with full privacy protection (DP + Homomorphic Encryption).")