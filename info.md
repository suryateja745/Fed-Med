# Module Information

Developer Name: Meghana Rose
Branch Name: Security
Module Name: Privacy & Encryption

## Install Command
pip install -r security/requirements.txt

## Run Command
python security/model_encryption.py

## Port Number
Not Applicable

## Environment Variables
None

## Additional Notes
- Uses TenSEAL for homomorphic encryption (CKKS scheme) — allows the server to aggregate encrypted model weights without ever decrypting them
- Uses Laplace-noise differential privacy to protect against model inversion attacks
- Files:
  - test_setup.py — environment verification
  - pytorch_encryption.py — encrypt/decrypt individual tensors
  - secure_pipeline.py — combines differential privacy + encryption
  - model_encryption.py — encrypts/decrypts a full model state_dict
  - security_tests.py — QA tests confirming protections work correctly
  - encryption/homomorphic_encryption.py — basic encrypted averaging demo
  - privacy/differential_privacy.py — noise-based privacy demo
- No API key required, runs entirely locally