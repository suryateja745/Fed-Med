import tenseal as ts

def create_context():
    """Creates and returns a TenSEAL encryption context (the 'keys' for encrypt/decrypt)."""
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.generate_galois_keys()
    context.global_scale = 2**40
    return context

def encrypt_weights(context, weights):
    """Encrypts a list of model weights."""
    return ts.ckks_vector(context, weights)

def decrypt_weights(encrypted_weights):
    """Decrypts back to plain numbers."""
    return encrypted_weights.decrypt()

if __name__ == "__main__":
    context = create_context()

    # Simulate 2 hospitals' model weight updates (small example)
    hospital_1_weights = [0.10, 0.20, 0.30]
    hospital_2_weights = [0.15, 0.25, 0.35]

    # Each hospital encrypts their own weights before sending
    encrypted_1 = encrypt_weights(context, hospital_1_weights)
    encrypted_2 = encrypt_weights(context, hospital_2_weights)

    print("Both hospitals' weights encrypted.")

    # The server averages them WITHOUT ever decrypting (this is the key trick)
    encrypted_sum = encrypted_1 + encrypted_2
    encrypted_average = encrypted_sum * 0.5

    # Only now do we decrypt, to see the final combined result
    final_result = decrypt_weights(encrypted_average)
    print(f"Federated average (computed on encrypted data): {final_result}")

    # Sanity check: does it match a normal, unencrypted average?
    expected = [(a + b) / 2 for a, b in zip(hospital_1_weights, hospital_2_weights)]
    print(f"Expected average (plain math, for comparison): {expected}")
    