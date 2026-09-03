import tenseal as ts

context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

weights = [0.25, 0.13, 0.89, 0.42]

encrypted_weights = ts.ckks_vector(context, weights)
print("Encrypted successfully. Original values are now hidden.")

decrypted_weights = encrypted_weights.decrypt()
print(f"Decrypted values: {decrypted_weights}")
