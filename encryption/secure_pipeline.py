import torch
import numpy as np
import sys, os

sys.path.append(os.path.dirname(__file__))
from pytorch_encryption import create_context, encrypt_tensor, decrypt_tensor

def add_noise(tensor, epsilon=1.0, sensitivity=1.0):
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, size=tensor.numel())
    noisy_flat = tensor.flatten() + torch.tensor(noise, dtype=tensor.dtype)
    return noisy_flat.reshape(tensor.shape)

def secure_transmit(context, tensor, epsilon=1.0):
    """
    Full secure pipeline: add differential privacy noise, then encrypt.
    This is what a hospital node would run before sending weights to the server.
    """
    noisy_tensor = add_noise(tensor, epsilon=epsilon)
    encrypted, shape = encrypt_tensor(context, noisy_tensor)
    return encrypted, shape

if __name__ == "__main__":
    context = create_context()

    model_weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    print(f"Original model weights:\n{model_weights}")

    encrypted, shape = secure_transmit(context, model_weights, epsilon=5.0)
    print("\nWeights protected with differential privacy AND encrypted.")

    received = decrypt_tensor(encrypted, shape)
    print(f"\nServer receives (after decryption, still has DP noise):\n{received}")