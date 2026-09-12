import torch
import sys, os

sys.path.append(os.path.dirname(__file__))
from pytorch_encryption import create_context, encrypt_tensor, decrypt_tensor
from secure_pipeline import add_noise

def test_encryption_hides_data():
    """Confirms encrypted data cannot be read as plain numbers."""
    context = create_context()
    weights = torch.tensor([0.5, 0.6, 0.7])
    encrypted, _ = encrypt_tensor(context, weights)

    # An "attacker" without the right context tries to peek at raw internal data
    raw_representation = str(encrypted)
    is_hidden = "0.5" not in raw_representation and "0.6" not in raw_representation

    print(f"Test: Encrypted data hides original values -> {'PASS' if is_hidden else 'FAIL'}")

def test_differential_privacy_changes_values():
    """Confirms DP noise actually changes individual values (no exact leakage)."""
    original = torch.tensor([0.5, 0.6, 0.7])
    noisy = add_noise(original, epsilon=1.0)

    is_different = not torch.allclose(original, noisy)
    print(f"Test: Differential privacy alters original values -> {'PASS' if is_different else 'FAIL'}")

def test_decryption_recovers_correct_value():
    """Confirms legitimate decryption (with correct context) still works correctly."""
    context = create_context()
    weights = torch.tensor([[1.0, 2.0]])
    encrypted, shape = encrypt_tensor(context, weights)
    decrypted = decrypt_tensor(encrypted, shape)

    is_close = torch.allclose(weights, decrypted, atol=1e-2)
    print(f"Test: Legitimate decryption recovers correct values -> {'PASS' if is_close else 'FAIL'}")

if __name__ == "__main__":
    test_encryption_hides_data()
    test_differential_privacy_changes_values()
    test_decryption_recovers_correct_value()