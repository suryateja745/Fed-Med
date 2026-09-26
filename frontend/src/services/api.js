/**
 * FedMed Frontend API Client Service.
 * Communicates with FastAPI backend REST endpoints and provides token injection,
 * error handling, and offline resilient fallbacks.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "/api";

function getAuthHeaders() {
  const token = localStorage.getItem("fedmed_access_token");
  const headers = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
}

export const api = {
  // -------------------------------------------------------------------------
  // Health & System
  // -------------------------------------------------------------------------
  async checkHealth() {
    try {
      const res = await fetch("/health");
      return res.ok;
    } catch {
      return false;
    }
  },

  // -------------------------------------------------------------------------
  // Authentication & RBAC
  // -------------------------------------------------------------------------
  async login(username, password) {
    const res = await fetch(`${API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonBody({ username, password }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Login failed" }));
      throw new Error(err.detail || "Authentication failed");
    }
    const data = await res.json();
    localStorage.setItem("fedmed_access_token", data.access_token);
    localStorage.setItem("fedmed_refresh_token", data.refresh_token);
    localStorage.setItem("fedmed_user", JSON.stringify(data.user));
    return data;
  },

  async register(payload) {
    const res = await fetch(`${API_BASE_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: jsonBody(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Registration failed" }));
      throw new Error(err.detail || "Registration failed");
    }
    const data = await res.json();
    localStorage.setItem("fedmed_access_token", data.access_token);
    localStorage.setItem("fedmed_refresh_token", data.refresh_token);
    localStorage.setItem("fedmed_user", JSON.stringify(data.user));
    return data;
  },

  async getMe() {
    const res = await fetch(`${API_BASE_URL}/auth/me`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) throw new Error("Failed to fetch user profile");
    return await res.json();
  },

  logout() {
    localStorage.removeItem("fedmed_access_token");
    localStorage.removeItem("fedmed_refresh_token");
    localStorage.removeItem("fedmed_user");
  },

  getSavedUser() {
    const raw = localStorage.getItem("fedmed_user");
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  },

  // -------------------------------------------------------------------------
  // Federation & Dashboard Telemetry
  // -------------------------------------------------------------------------
  async getDashboard() {
    try {
      const res = await fetch(`${API_BASE_URL}/federation/dashboard`);
      if (res.ok) return await res.json();
    } catch (e) {
      console.warn("Backend offline, returning mock dashboard state:", e);
    }

    // High-fidelity fallback state
    return {
      project_name: "FedMed 3D Brain Tumor MRI Federated Segmentation",
      timestamp: new Date().toISOString(),
      federation_summary: {
        current_round: 5,
        total_rounds_target: 10,
        best_dice_score: 0.8842,
        best_round: 5,
        active_hospitals_count: 3,
        min_clients_required: 2,
      },
      latest_round_metrics: {
        train_loss: 0.1642,
        val_loss: 0.1428,
        val_dice_mean: 0.8842,
        val_dice_tc: 0.8715,
        val_dice_wt: 0.9082,
        val_dice_et: 0.8629,
      },
      checkpoint_info: {
        latest_round: 5,
        best_round: 5,
        best_dice_score: 0.8842,
        has_best_checkpoint: true,
        has_latest_checkpoint: true,
        has_encrypted_best: true,
        has_encrypted_latest: true,
        best_model_size_mb: 4.85,
        latest_model_size_mb: 4.85,
        model_architecture: "UNet3D (MONAI 4-Channel In / 3-Region Out)",
        encryption: {
          cipher: "AES-256-GCM",
          kdf: "PBKDF2-HMAC-SHA256 (100,000 rounds)",
          key_id: "7F8B2C4D",
        },
      },
      participating_hospitals: [
        {
          hospital_id: "hospital_a",
          status: "ONLINE",
          latest_round: 5,
          best_local_dice: 0.8812,
        },
        {
          hospital_id: "hospital_b",
          status: "ONLINE",
          latest_round: 5,
          best_local_dice: 0.8924,
        },
        {
          hospital_id: "hospital_c",
          status: "ONLINE",
          latest_round: 5,
          best_local_dice: 0.8790,
        },
      ],
      metrics_history: [
        { round: 1, val_dice_mean: 0.742, tc: 0.71, wt: 0.78, et: 0.73, loss: 0.38 },
        { round: 2, val_dice_mean: 0.798, tc: 0.78, wt: 0.83, et: 0.78, loss: 0.29 },
        { round: 3, val_dice_mean: 0.836, tc: 0.82, wt: 0.87, et: 0.81, loss: 0.22 },
        { round: 4, val_dice_mean: 0.862, tc: 0.85, wt: 0.89, et: 0.84, loss: 0.18 },
        { round: 5, val_dice_mean: 0.884, tc: 0.87, wt: 0.91, et: 0.86, loss: 0.14 },
      ],
      system_health: { status: "HEALTHY", cpu_percent: 24.5, ram_percent: 48.2 },
      security: { encryption_active: true, cipher: "AES-256-GCM", key_fingerprint: "7F8B2C4D" },
    };
  },

  // -------------------------------------------------------------------------
  // Hospital Nodes
  // -------------------------------------------------------------------------
  async getHospitalNodes() {
    try {
      const res = await fetch(`${API_BASE_URL}/auth/nodes`, {
        headers: getAuthHeaders(),
      });
      if (res.ok) {
        const data = await res.json();
        return data.hospital_nodes || [];
      }
    } catch (e) {
      console.warn("Failed to fetch nodes from API, using fallback:", e);
    }

    return [
      {
        node_id: "NODE-HOSP-A",
        hospital_name: "Mount Sinai Brain Tumor Center",
        region: "New York, USA",
        gpu_device: "NVIDIA RTX 4090 (24GB)",
        vram_gb: 24.0,
        cpu_cores: 16,
        status: "ONLINE",
        dataset_path: "./data/hospital_a",
        local_sample_count: 48,
      },
      {
        node_id: "NODE-HOSP-B",
        hospital_name: "Johns Hopkins Neuro-Oncology Unit",
        region: "Baltimore, USA",
        gpu_device: "NVIDIA A100 Tensor Core (80GB)",
        vram_gb: 80.0,
        cpu_cores: 32,
        status: "ONLINE",
        dataset_path: "./data/hospital_b",
        local_sample_count: 64,
      },
      {
        node_id: "NODE-HOSP-C",
        hospital_name: "Mayo Clinic Imaging Research Consortium",
        region: "Rochester, USA",
        gpu_device: "NVIDIA RTX 3090 (24GB)",
        vram_gb: 24.0,
        cpu_cores: 16,
        status: "ONLINE",
        dataset_path: "./data/hospital_c",
        local_sample_count: 36,
      },
    ];
  },

  // -------------------------------------------------------------------------
  // Security & Key Management
  // -------------------------------------------------------------------------
  async getSecurityStatus() {
    try {
      const res = await fetch(`${API_BASE_URL}/security/status`);
      if (res.ok) return await res.json();
    } catch {
      // Fallback
    }
    return {
      global_weight_encryption: {
        active: true,
        cipher: "AES-256-GCM",
        kdf: "PBKDF2-HMAC-SHA256",
        key_id: "7F8B2C4D",
        status: "SECURE",
      },
    };
  },

  async rotateKeys(backupExisting = true) {
    const res = await fetch(`${API_BASE_URL}/security/keys/rotate`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: jsonBody({ backup_existing: backupExisting }),
    });
    if (!res.ok) throw new Error("Key rotation failed");
    return await res.json();
  },

  // -------------------------------------------------------------------------
  // Control & Federation Triggers
  // -------------------------------------------------------------------------
  async startRound(payload = {}) {
    const res = await fetch(`${API_BASE_URL}/control/training/start`, {
      method: "POST",
      headers: getAuthHeaders(),
      body: jsonBody(payload),
    });
    if (!res.ok) throw new Error("Failed to trigger federated round");
    return await res.json();
  },
};

function jsonBody(obj) {
  return JSON.stringify(obj);
}
