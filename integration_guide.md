# Security Module — Integration Guide (for Federated Learning teammate)

## How to use this module

​```python
from security.hospital_client import prepare_weights_for_transmission

# After your local training loop finishes:
encrypted_weights, context = prepare_weights_for_transmission(your_trained_model, epsilon=5.0)

# Send `encrypted_weights` to the central server.
# Keep `context` safe — it's needed to decrypt the final aggregated result.
​```

## Function signature

`prepare_weights_for_transmission(model, context=None, epsilon=5.0)`

**Input:**
- `model` — your trained PyTorch model (any architecture)
- `epsilon` — privacy strength (lower = more private, more noise; default 5.0 is a reasonable balance)

**Output:**
- `encrypted_weights` — dictionary of encrypted layer weights, safe to transmit
- `context` — encryption context/keys, needed later for decryption (keep secure, don't share publicly)

## Server-side aggregation

Once the server collects `encrypted_weights` from all hospital nodes, use:
​```python
from security.federated_aggregation import aggregate_encrypted_weights
aggregated = aggregate_encrypted_weights([hospital1_weights, hospital2_weights, hospital3_weights])
​```

## Notes
- No API key or environment variables required — runs entirely locally
- Differential privacy (epsilon) and homomorphic encryption are applied automatically inside this one function
- Tested and validated against a simulated model inversion attack (see `attack_simulation.py`)