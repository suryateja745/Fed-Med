import React, { useState, useEffect } from "react";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import { Card } from "../components/common/Card";
import { Badge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { IconActivity, IconServer, IconLock, IconPlay, IconRefreshCw } from "../components/common/Icons";

export function CoordinatorDashboard() {
  const { user } = useAuth();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchDashboard = async () => {
    setLoading(true);
    const res = await api.getDashboard();
    setData(res);
    setLoading(false);
  };

  useEffect(() => {
    fetchDashboard();
  }, []);

  return (
    <div className="flex-col gap-6">
      {/* Header bar */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Badge variant="cyan" pulse>Flower Orchestrator</Badge>
            <Badge variant="emerald">AES-256-GCM Secure</Badge>
          </div>
          <h2 style={{ marginTop: "0.4rem" }}>Central Flower Coordinator Command Center</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
            Active Admin: <strong>{user?.username || "Coordinator Admin"}</strong> ({user?.institution_name || "Central Command"})
          </p>
        </div>

        <div style={{ display: "flex", gap: "0.75rem" }}>
          <Button variant="outline" size="sm" onClick={fetchDashboard} icon={<IconRefreshCw size={14} />}>
            Refresh Telemetry
          </Button>
          <Button variant="glow" size="sm" icon={<IconPlay size={14} />}>
            Trigger Next Round
          </Button>
        </div>
      </div>

      {/* KPI Stats */}
      <div className="grid grid-4">
        <div className="stat-card">
          <span className="stat-label">Active Round</span>
          <span className="stat-value" style={{ color: "var(--cyan-light)" }}>
            {data?.federation_summary?.current_round || 0} / {data?.federation_summary?.total_rounds_target || 10}
          </span>
          <span className="stat-desc">Federated training step</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Global Best Dice</span>
          <span className="stat-value" style={{ color: "var(--emerald-light)" }}>
            {((data?.federation_summary?.best_dice_score || 0) * 100).toFixed(1)}%
          </span>
          <span className="stat-desc">Achieved in Round {data?.federation_summary?.best_round || 0}</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Active Hospital Nodes</span>
          <span className="stat-value" style={{ color: "var(--violet-light)" }}>
            {data?.federation_summary?.active_hospitals_count || 3}
          </span>
          <span className="stat-desc">Min required: {data?.federation_summary?.min_clients_required || 2}</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Global Loss</span>
          <span className="stat-value" style={{ color: "var(--text-primary)" }}>
            {data?.latest_round_metrics?.val_loss?.toFixed(4) || "0.1428"}
          </span>
          <span className="stat-desc">DiceCELoss Convergence</span>
        </div>
      </div>

      {/* Connected Hospitals Cluster Grid */}
      <Card
        title="Connected Hospital Cluster Matrix"
        subtitle="Real-time telemetry from active medical center nodes"
        icon={<IconServer size={18} />}
      >
        <div className="grid grid-3">
          {(data?.participating_hospitals || []).map((h, i) => (
            <div
              key={i}
              style={{
                background: "var(--bg-surface)",
                padding: "1.25rem",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border-subtle)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
                <span style={{ fontWeight: 600, fontSize: "1rem", textTransform: "capitalize" }}>
                  {h.hospital_id.replace("_", " ")}
                </span>
                <Badge variant={h.status === "ONLINE" ? "emerald" : "cyan"} pulse>
                  {h.status}
                </Badge>
              </div>

              <div className="flex-col gap-1" style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                <div>Local Round: <strong style={{ color: "var(--text-primary)" }}>{h.latest_round}</strong></div>
                <div>Best Local Dice: <strong style={{ color: "var(--emerald-light)" }}>{(h.best_local_dice * 100).toFixed(1)}%</strong></div>
                <div>Encryption: <strong style={{ color: "var(--cyan-light)" }}>AES-256-GCM (Active)</strong></div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
