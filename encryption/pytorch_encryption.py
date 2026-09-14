import torch
import tenseal as ts

def create_context():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def encrypt_tensor(context, tensor):
    """Encrypts a PyTorch tensor by flattening it first."""
    flat = tensor.flatten().tolist()
    return ts.ckks_vector(context, flat), tensor.shape

def decrypt_tensor(encrypted_vector, original_shape):
    """Decrypts and reshapes back to the original tensor shape."""
    flat = encrypted_vector.decrypt()
    return torch.tensor(flat).reshape(original_shape)

if __name__ == "__main__":
    context = create_context()

    fake_model_weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    print(f"Original tensor:\n{fake_model_weights}")

    encrypted, shape = encrypt_tensor(context, fake_model_weights)
    print("\nTensor encrypted successfully.")

    decrypted = decrypt_tensor(encrypted, shape)
    print(f"\nDecrypted tensor:\n{decrypted}")