# FedMed - Federated Learning Framework Documentation

FedMed is a privacy-preserving Federated Learning system designed for 3D Brain Tumor MRI Segmentation (MONAI 3D U-Net, PyTorch, Flower, and TenSEAL). It enables multiple healthcare institutions (e.g. Hospital A, Hospital B, Hospital C) to collaboratively train medical AI models without sharing raw volumetric patient data.

---

## 🏗️ Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │     FedMed Central Coordinator      │
                          │   (fl_server.py / start_server.py)  │
                          ├─────────────────────────────────────┤
                          │ • FedMedStrategy (Weighted Dice)    │
                          │ • RoundSyncManager (Locks & Sync)   │
                          │ • AutoDispatchTrigger               │
                          │ • AutoAggregateTrigger              │
                          │ • GlobalModelManager (Checkpoints)  │
                          │ • FedMedAPIBridge (JSON Telemetry)  │
                          └──────────────────┬──────────────────┘
                                             │
                      gRPC Parameter Exchange│ (Encrypted/Plain)
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         │                                   │                                   │
         ▼                                   ▼                                   ▼
┌─────────────────────────┐         ┌─────────────────────────┐         ┌─────────────────────────┐
│       Hospital A        │         │       Hospital B        │         │       Hospital C        │
│    (start_client.py)    │         │    (start_client.py)    │         │    (start_client.py)    │
├─────────────────────────┤         ├─────────────────────────┤         ├─────────────────────────┤
│ • Private MRI Data      │         │ • Private MRI Data      │         │ • Private MRI Data      │
│ • Local MONAI 3D U-Net  │         │ • Local MONAI 3D U-Net  │         │ • Local MONAI 3D U-Net  │
│ • FedMedClient (Flower) │         │ • FedMedClient (Flower) │         │ • FedMedClient (Flower) │
│ • Checkpoint Fallback   │         │ • Checkpoint Fallback   │         │ • Checkpoint Fallback   │
│ • SecureClientHook (DP) │         │ • SecureClientHook (DP) │         │ • SecureClientHook (DP) │
└─────────────────────────┘         └─────────────────────────┘         └─────────────────────────┘
```

---

## 🚀 Quick Start & CLI Reference

### 1. Central Server Coordinator
Start the Flower coordination server listening on port `8080` with automated dispatch and aggregation triggers:

```bash
# Launch server with FedMedStrategy, 10 rounds, min 2 clients
python start_server.py --host 127.0.0.1 --port 8080 --rounds 10 --min-clients 2

# Fast offline dry-run self-test
python start_server.py --dry-run
```

**Key Arguments for `start_server.py`:**
- `--host` (default: `127.0.0.1`): Server binding IP.
- `--port` (default: `8080`): Listening port for Flower gRPC connections.
- `--rounds` (default: `10`): Number of federated training rounds.
- `--min-clients` (default: `2`): Minimum hospital clients required before aggregation triggers.
- `--strategy` (default: `FedMedStrategy`): `FedMedStrategy` (quality-weighted Dice averaging) or `FedAvg`.
- `--dry-run`: Validates coordinator initialization and API exports without blocking.

---

### 2. Local Hospital Client Worker
Launch a hospital worker node pointing to its local dataset folder:

```bash
# Hospital A Node
python start_client.py --hospital-id hospital_a --data-dir ./data/hospital_a --server 127.0.0.1:8080 --epochs 1

# Hospital B Node
python start_client.py --hospital-id hospital_b --data-dir ./data/hospital_b --server 127.0.0.1:8080 --epochs 1

# Fast offline dry-run self-test
python start_client.py --hospital-id hospital_a --dry-run
```

**Key Arguments for `start_client.py`:**
- `--hospital-id`: Unique identifier (e.g. `hospital_a`, `hospital_b`, `hospital_c`).
- `--data-dir`: Local folder containing volumetric MRI scans (`.npy`, `.npz`, `.nii.gz`).
- `--server`: Coordinator address (`<host>:<port>`).
- `--epochs`: Local training epochs per round.
- `--create-synthetic`: Automatically generates synthetic 3D MRI scans if local folder is empty.

---

### 3. Multi-Hospital Simulation Testbed
Simulate the entire federated learning lifecycle across 3+ hospitals with non-IID partitioned datasets in a single process:

```bash
python federation/simulate.py --num-clients 3 --num-rounds 5 --partition-type quantity_skew
```
This automatically generates:
- `reports/round_vs_dice.png` (Convergence of Multi-Region Dice: Mean, TC, WT, ET).
- `reports/round_vs_loss.png` (Training Loss vs Validation Loss).
- `reports/simulation_summary.json` (Consolidated experiment metadata).

---

## 📡 API Bridge & Live Dashboard Telemetry

The `FedMedAPIBridge` (`federation/api_bridge.py`) exposes programmatic status endpoints and periodically exports `logs/fedmed_live_dashboard.json` for live UI polling.

### Python API Usage
```python
from federation.api_bridge import FedMedAPIBridge

bridge = FedMedAPIBridge(checkpoint_dir="./checkpoints", logs_dir="./logs")

# Query State
current_round = bridge.get_current_round()
checkpoint_info = bridge.get_latest_checkpoint_info()
history = bridge.get_metrics_history()
active_hospitals = bridge.get_active_hospitals()
system_health = bridge.get_system_health()

# Export Full Dashboard JSON
dashboard_state = bridge.get_full_dashboard_state()
bridge.export_dashboard_json("./logs/fedmed_live_dashboard.json")
```

### Dashboard JSON Specification (`logs/fedmed_live_dashboard.json`)
```json
{
  "project_name": "FedMed - Federated 3D Brain Tumor MRI Segmentation",
  "timestamp": "2026-09-21T14:00:00Z",
  "federation_summary": {
    "current_round": 3,
    "total_rounds_target": 10,
    "best_dice_score": 0.885,
    "best_round": 2,
    "active_hospitals_count": 3,
    "min_clients_required": 2
  },
  "latest_round_metrics": {
    "train_loss": 0.38,
    "val_loss": 0.42,
    "val_dice_mean": 0.885,
    "val_dice_tc": 0.86,
    "val_dice_wt": 0.91,
    "val_dice_et": 0.84
  },
  "checkpoint_info": {
    "latest_round": 3,
    "best_round": 2,
    "best_dice_score": 0.885,
    "has_best_checkpoint": true,
    "best_checkpoint_path": "./checkpoints/best_global_model.pth"
  },
  "participating_hospitals": [
    {"hospital_id": "hospital_a", "status": "CONNECTED", "latest_round": 3, "best_local_dice": 0.87},
    {"hospital_id": "hospital_b", "status": "CONNECTED", "latest_round": 3, "best_local_dice": 0.89}
  ],
  "system_health": {
    "status": "HEALTHY",
    "cuda_available": true,
    "gpu_name": "NVIDIA GeForce RTX 2050",
    "gpu_vram_gb": 4.0,
    "cpu_cores": 12
  }
}
```

---

## 🔒 Security & Privacy

1. **Differential Privacy (`federation/security/encryption.py`)**:
   - $L_2$ norm gradient and parameter clipping.
   - Calibrated Gaussian noise addition: $\sigma = \frac{C \sqrt{2 \ln(1.25/\delta)}}{\epsilon}$.
2. **Homomorphic Encryption (TenSEAL CKKS)**:
   - Supports parameter ciphertext vector encryption.
   - Server-side `secure_aggregate_encrypted()` computes weighted averages directly in the ciphertext domain without decrypting client weights.

---

## ⚙️ Concurrency & Triggers

- **`RoundSyncManager` (`federation/server/sync_manager.py`)**: Re-entrant mutex (`threading.RLock`) protects parameter ingestion from simultaneous client uploads and rejects stale parameters from completed rounds.
- **`AutoDispatchTrigger` (`federation/server/triggers.py`)**: Streams `best_global_model.pth` or preset baseline weights to connecting clients when the admin is offline. Supports authenticated AES-256-GCM encrypted payload streaming.
- **`AutoAggregateTrigger` (`federation/server/triggers.py`)**: Automatically triggers `FedMedStrategy.aggregate_fit()`, saves encrypted checkpoints (`global_model_round_X.pth.enc`), and publishes round events once client upload quota is reached.

---

## 🔒 Global Model Weight Encryption & Key Management

FedMed implements zero-trust authenticated encryption at rest and in transit:
1. **AES-256-GCM Authenticated Encryption**: 256-bit symmetric cipher in Galois/Counter Mode with 12-byte random nonces and 16-byte authentication tags.
2. **HMAC-SHA256 Model Signatures**: Digital signatures computed over ciphertext and metadata headers; tampering with any single bit raises `CryptographicIntegrityError`.
3. **Key Management (`GlobalKeyManager`)**:
   - Master key stored securely at `checkpoints/fedmed_global_model.key`.
   - PBKDF2-HMAC-SHA256 derivation with 100,000 rounds.
   - Key rotation via `/api/security/keys/rotate`.
   - 8-character hex key fingerprint for auditing.
4. **Encrypted Model Serialization**:
   - `GlobalModelManager` automatically saves `.pth.enc` bundles for every round and latest/best checkpoints.
   - Transparent in-memory decryption directly into PyTorch models without writing plaintext weights to insecure storage.

---

## 🌐 Production FastAPI Backend & Real-Time Telemetry

The FedMed platform includes a complete, production-grade FastAPI REST and WebSocket backend (`backend/`):

### Starting the Backend
```bash
# Launch FastAPI backend with Uvicorn (HTTP: 8000, WebSocket: /ws/telemetry)
python start_backend.py --host 0.0.0.0 --port 8000 --reload

# Offline route validation self-test
python start_backend.py --dry-run
```

### Interactive Documentation & Schema
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

### REST API Endpoints Overview
| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/federation/dashboard` | `GET` | Full telemetry dashboard state (rounds, Dice scores, nodes, health) |
| `/api/federation/status` | `GET` | Real-time federation state (IDLE / TRAINING, current round) |
| `/api/federation/metrics` | `GET` | Round-by-round convergence history (train/val losses, multi-region Dice) |
| `/api/hospitals` | `GET` | List all connected and registered hospital client nodes |
| `/api/hospitals/register` | `POST` | Register hospital node with hardware specs and public key |
| `/api/hospitals/{id}/heartbeat` | `POST` | Hospital node keep-alive heartbeat ping |
| `/api/models/global` | `GET` | Global 3D U-Net architecture specs, parameter counts, and version registry |
| `/api/models/download/latest` | `GET` | Stream latest encrypted global model (`.pth.enc`) with `X-Model-HMAC` headers |
| `/api/models/download/best` | `GET` | Stream historical best encrypted global model |
| `/api/models/verify` | `POST` | Cryptographically verify HMAC-SHA256 signature and integrity of model file |
| `/api/models/dispatch/{id}` | `POST` | Trigger automated encrypted model dispatch to hospital node |
| `/api/security/status` | `GET` | Global encryption status, cipher mode, key fingerprint, and DP settings |
| `/api/security/keys/rotate` | `POST` | Rotate master encryption key and re-encrypt latest/best checkpoints |
| `/api/security/keys/export-hospital-key` | `POST` | Provision authorized hospital node with decryption credential |
| `/api/control/train/start` | `POST` | Initiate federated training orchestration round |
| `/api/control/train/stop` | `POST` | Halt active federated training background jobs |
| `/api/control/simulate` | `POST` | Trigger multi-hospital simulation testbed in background thread |
| `/api/control/system/health` | `GET` | Inspect server hardware, PyTorch CUDA GPU VRAM, and CPU cores |
| `/ws/telemetry` | `WebSocket` | Real-time bi-directional telemetry broadcast to Frontend UI |

### Frontend Real-Time WebSocket Integration (React / TypeScript Example)
```typescript
import React, { useEffect, useState } from 'react';

export const FedMedLiveDashboard = () => {
  const [telemetry, setTelemetry] = useState<any>(null);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/telemetry');

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'INITIAL_STATE' || msg.type === 'TELEMETRY_UPDATE') {
        setTelemetry(msg.data);
      }
    };

    return () => ws.close();
  }, []);

  if (!telemetry) return <div>Connecting to FedMed live telemetry...</div>;

  return (
    <div>
      <h1>FedMed Federated AI Coordinator</h1>
      <p>Round: {telemetry.federation_summary.current_round} / {telemetry.federation_summary.total_rounds_target}</p>
      <p>Best Global Dice: {telemetry.federation_summary.best_dice_score}</p>
      <p>Encryption: {telemetry.security.global_weight_encryption.cipher} (Active)</p>
    </div>
  );
};
```

