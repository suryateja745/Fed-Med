import torch
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "security"))
from pytorch_encryption import create_context, encrypt_tensor, decrypt_tensor
from secure_pipeline import secure_transmit

def aggregate_encrypted_weights(encrypted_list):
    """
    Server-side aggregation: sums encrypted weights from all hospitals
    and averages them — without ever decrypting individual contributions.
    """
    total = encrypted_list[0]
    for enc in encrypted_list[1:]:
        total = total + enc
    average = total * (1.0 / len(encrypted_list))
    return average

if __name__ == "__main__":
    context = create_context()

    # Simulate 3 hospitals with slightly different local weights
    hospital_weights = [
        torch.tensor([[0.10, 0.20], [0.30, 0.40]]),
        torch.tensor([[0.12, 0.18], [0.29, 0.42]]),
        torch.tensor([[0.11, 0.22], [0.31, 0.38]]),
    ]

    encrypted_updates = []
    shape = None
    for i, weights in enumerate(hospital_weights):
        encrypted, shape = secure_transmit(context, weights, epsilon=5.0)
        encrypted_updates.append(encrypted)
        print(f"Hospital {i+1}: weights secured (DP + encryption) and sent to server.")

    print("\nServer aggregating encrypted updates from all 3 hospitals...")
    aggregated = aggregate_encrypted_weights(encrypted_updates)

    final_global_model = decrypt_tensor(aggregated, shape)
    print(f"\nNew global model weights (after secure aggregation):\n{final_global_model}")
