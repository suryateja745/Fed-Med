import os
import sqlite3
from pathlib import Path

# Paths
db_path = Path("data/fedmed.db")

print(f"Connecting to database at {db_path}...")
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# 1. Truncate hospital_nodes
print("Truncating hospital_nodes...")
cur.execute("DELETE FROM hospital_nodes;")

# 2. Clear users except admin
print("Deleting non-admin users from users table...")
cur.execute("DELETE FROM users WHERE username != 'admin';")

# 3. Clear training_sessions & audit_logs
print("Truncating training_sessions & audit_logs...")
cur.execute("DELETE FROM training_sessions;")
cur.execute("DELETE FROM audit_logs;")

conn.commit()

# Print current counts
print("\n--- Current Database State ---")
for tbl in ["users", "hospital_nodes", "training_sessions", "audit_logs"]:
    cur.execute(f"SELECT COUNT(*) FROM {tbl};")
    cnt = cur.fetchone()[0]
    print(f"Table '{tbl}': {cnt} records")

cur.execute("SELECT user_id, username, role, institution_name FROM users;")
admin_row = cur.fetchall()
print(f"Active Users in DB: {admin_row}")

conn.close()
print("\nDatabase cleanup complete. Only admin credentials remain.")

# 4. Remove stale history and log files in logs/ and data/
print("\nPurging stale history logs...")
logs_dir = Path("logs")
data_dir = Path("data")

for d in [logs_dir, data_dir]:
    if d.exists():
        for f in d.glob("*_history.json"):
            try:
                f.unlink()
                print(f"  Removed: {f}")
            except Exception as e:
                print(f"  Could not remove {f}: {e}")

# Clear dispatch and aggregation events
for log_file in [logs_dir / "dispatch_audit.jsonl", logs_dir / "aggregation_events.jsonl"]:
    if log_file.exists():
        try:
            log_file.unlink()
            print(f"  Removed: {log_file}")
        except Exception as e:
            print(f"  Could not remove {log_file}: {e}")

# Overwrite fedmed_live_dashboard.json with zeroed state
dashboard_cache = logs_dir / "fedmed_live_dashboard.json"
clean_state = {
    "project_name": "FedMed - Federated 3D Brain Tumor MRI Segmentation",
    "timestamp": "2026-10-02T05:30:00.000000+00:00",
    "federation_summary": {
        "current_round": 0,
        "total_rounds_target": 10,
        "best_dice_score": 0.0,
        "best_round": 0,
        "active_hospitals_count": 0,
        "min_clients_required": 2
    },
    "latest_round_metrics": {
        "train_loss": 0.0,
        "val_loss": 0.0,
        "val_dice_mean": 0.0,
        "val_dice_tc": 0.0,
        "val_dice_wt": 0.0,
        "val_dice_et": 0.0
    },
    "checkpoint_info": {
        "latest_round": 0,
        "best_round": 0,
        "best_dice_score": 0.0,
        "has_best_checkpoint": False,
        "has_latest_checkpoint": False,
        "has_encrypted_best": False,
        "has_encrypted_latest": False,
        "best_model_size_mb": 0.0,
        "latest_model_size_mb": 0.0,
        "model_architecture": "UNet3D",
        "encryption": {"enabled": True, "cipher": "AES-256-GCM", "key_id": "INIT"}
    },
    "participating_hospitals": [],
    "metrics_history": []
}
import json
with open(dashboard_cache, "w", encoding="utf-8") as f:
    json.dump(clean_state, f, indent=2)
print("  Zeroed: logs/fedmed_live_dashboard.json")
print("\nAll stale hospital history files and telemetry caches purged!")

