import torch
import sys, os

sys.path.append(os.path.dirname(__file__))
from pytorch_encryption import create_context
from model_encryption import encrypt_model_state

def prepare_weights_for_transmission(model, context=None, epsilon=5.0):
    """
    This is the function the Federated teammate's hospital-node code will call
    after local training finishes, before sending weights to the central server.

    Returns: (encrypted_state_dict, context, shape_info)
    """
    if context is None:
        context = create_context()

    # Add differential privacy noise directly to the state_dict before encrypting
    from secure_pipeline import add_noise
    noisy_state = {}
    for name, param in model.state_dict().items():
        noisy_state[name] = add_noise(param, epsilon=epsilon)

    # Now encrypt the noisy weights
    encrypted_state = {}
    for name, tensor in noisy_state.items():
        from pytorch_encryption import encrypt_tensor
        encrypted, shape = encrypt_tensor(context, tensor)
        encrypted_state[name] = (encrypted, shape)

    return encrypted_state, context

if __name__ == "__main__":
    import torch.nn as nn

    class HospitalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(4, 8)

    model = HospitalModel()
    print("Hospital Node: local model trained. Preparing secure transmission...")

    encrypted_weights, context = prepare_weights_for_transmission(model, epsilon=5.0)

    print(f"Success: {len(encrypted_weights)} layer(s) encrypted and privacy-protected.")
    print("Ready to send to central server (context/keys stay only with authorized parties).")