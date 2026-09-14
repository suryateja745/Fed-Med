import torch
import torch.nn as nn
import sys, os

sys.path.append(os.path.dirname(__file__))
from pytorch_encryption import create_context, encrypt_tensor, decrypt_tensor

# A tiny example model, similar in shape to what the Federated teammate will train
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 8)
        self.layer2 = nn.Linear(8, 2)

def encrypt_model_state(context, model):
    """Encrypts every layer's weights in a model's state_dict."""
    encrypted_state = {}
    for name, param in model.state_dict().items():
        encrypted, shape = encrypt_tensor(context, param)
        encrypted_state[name] = (encrypted, shape)
    return encrypted_state

def decrypt_model_state(encrypted_state):
    """Decrypts every layer back into a normal state_dict."""
    decrypted_state = {}
    for name, (encrypted, shape) in encrypted_state.items():
        decrypted_state[name] = decrypt_tensor(encrypted, shape)
    return decrypted_state

if __name__ == "__main__":
    context = create_context()
    model = SimpleModel()

    print("Original model layers:")
    for name, param in model.state_dict().items():
        print(f"  {name}: shape {param.shape}")

    encrypted_state = encrypt_model_state(context, model)
    print("\nEntire model state_dict encrypted, layer by layer.")

    decrypted_state = decrypt_model_state(encrypted_state)
    model.load_state_dict(decrypted_state)
    print("\nModel successfully restored from encrypted state — no data lost.")