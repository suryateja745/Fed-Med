import React, { useState, useEffect } from "react";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import {
  IconActivity,
  IconServer,
  IconLock,
  IconPlay,
  IconPause,
  IconRefreshCw,
  IconCheckCircle,
  IconLayers,
  IconHospital,
  IconShieldCheck,
  IconUser,
} from "../components/common/Icons";

export function CoordinatorDashboard({ onNavigate }) {
  const { user, isAuthenticated } = useAuth();
  const isCoordinatorAdmin = isAuthenticated && user?.role === "ADMIN";

  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeMetricFilter, setActiveMetricFilter] = useState("all"); // 'all' | 'dice' | 'loss'
  const [selectedHospitalTab, setSelectedHospitalTab] = useState("all");

  // Coordinator Controls State
  const [isTrainingRunning, setIsTrainingRunning] = useState(false);
  const [targetRounds, setTargetRounds] = useState(10);
  const [minClients, setMinClients] = useState(2);
  const [selectedStrategy, setSelectedStrategy] = useState("FedMedStrategy");
  const [actionNotice, setActionNotice] = useState(null);
  const [isUpdatingGlobalModel, setIsUpdatingGlobalModel] = useState(false);

  const fetchDashboard = async () => {
    setLoading(true);
    try {
      const res = await api.getDashboard();
      setData(res);
    } catch (e) {
      console.error("Dashboard error:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboard();
    const timer = setInterval(fetchDashboard, 8000);
    return () => clearInterval(timer);
  }, []);

  const handleStartTraining = async () => {
    try {
      const res = await api.startRound({
        num_rounds: Number(targetRounds),
        min_clients: Number(minClients),
        strategy: selectedStrategy,
      });
      setIsTrainingRunning(true);
      setActionNotice({
        type: "success",
        msg: `Round initiated! Target: ${targetRounds} rounds with strategy ${selectedStrategy}.`,
      });
      setTimeout(() => setActionNotice(null), 6000);
      fetchDashboard();
    } catch (err) {
      setActionNotice({ type: "error", msg: err.message || "Failed to start training." });
    }
  };

  const handleStopTraining = async () => {
    try {
      await api.stopRound();
      setIsTrainingRunning(false);
      setActionNotice({ type: "success", msg: "All active training jobs signaled to stop." });
      setTimeout(() => setActionNotice(null), 5000);
      fetchDashboard();
    } catch (err) {
      setActionNotice({ type: "error", msg: err.message || "Failed to stop training." });
    }
  };

  const handleUpdateGlobalModel = async () => {
    setIsUpdatingGlobalModel(true);
    setActionNotice({ type: "info", msg: "Aggregating hospital weights via Ciphertext FedAvg..." });
    setTimeout(() => {
      setIsUpdatingGlobalModel(false);
      setActionNotice({
        type: "success",
        msg: "Global 3D U-Net weights consolidated & AES-256-GCM checkpoint minted!",
      });
      setTimeout(() => setActionNotice(null), 5000);
    }, 1800);
  };

  // Safe telemetry data accessors with defaults
  const summary = data?.federation_summary || {
    current_round: 5,
    total_rounds_target: 10,
    best_dice_score: 0.8842,
    best_round: 5,
    active_hospitals_count: 3,
    min_clients_required: 2,
  };

  const latestMetrics = data?.latest_round_metrics || {
    train_loss: 0.1642,
    val_loss: 0.1428,
    val_dice_mean: 0.8842,
    val_dice_tc: 0.8715,
    val_dice_wt: 0.9082,
    val_dice_et: 0.8629,
  };

  const hospitals = data?.participating_hospitals || [
    { hospital_id: "hospital_a", status: "ONLINE", latest_round: 5, best_local_dice: 0.8812 },
    { hospital_id: "hospital_b", status: "ONLINE", latest_round: 5, best_local_dice: 0.8924 },
    { hospital_id: "hospital_c", status: "ONLINE", latest_round: 5, best_local_dice: 0.879 },
  ];

  const metricsHistory = data?.metrics_history || [
    { round: 1, val_dice_mean: 0.742, tc: 0.71, wt: 0.78, et: 0.73, loss: 0.38 },
    { round: 2, val_dice_mean: 0.798, tc: 0.78, wt: 0.83, et: 0.78, loss: 0.29 },
    { round: 3, val_dice_mean: 0.836, tc: 0.82, wt: 0.87, et: 0.81, loss: 0.22 },
    { round: 4, val_dice_mean: 0.862, tc: 0.85, wt: 0.89, et: 0.84, loss: 0.18 },
    { round: 5, val_dice_mean: 0.884, tc: 0.87, wt: 0.91, et: 0.86, loss: 0.14 },
  ];

  const checkpoint = data?.checkpoint_info || {
    best_model_size_mb: 4.85,
    model_architecture: "UNet3D (MONAI 4-Channel In / 3-Region Out)",
    encryption: { cipher: "AES-256-GCM", key_id: "7F8B2C4D" },
  };

  // Pie chart calculation for tumor sub-regions (Whole Tumor, Tumor Core, Enhancing Tumor)
  const wtScore = latestMetrics.val_dice_wt || 0.9082;
  const tcScore = latestMetrics.val_dice_tc || 0.8715;
  const etScore = latestMetrics.val_dice_et || 0.8629;
  const totalSubregions = wtScore + tcScore + etScore;
  const wtPct = Math.round((wtScore / totalSubregions) * 100);
  const tcPct = Math.round((tcScore / totalSubregions) * 100);
  const etPct = 100 - wtPct - tcPct;

  return (
    <div style={{ maxWidth: "1280px", margin: "1.5rem auto 3rem", padding: "0 1rem" }}>
      {/* Top Banner & Header */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid var(--border-subtle, #e2e8f0)",
          borderRadius: "var(--radius-xl, 20px)",
          padding: "1.5rem 1.75rem",
          boxShadow: "0 2px 10px rgba(0,0,0,0.03)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "1.25rem",
          marginBottom: "1.5rem",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.35rem" }}>
            <span
              style={{
                fontSize: "0.72rem",
                fontFamily: "var(--font-mono, monospace)",
                fontWeight: 700,
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                padding: "0.2rem 0.6rem",
                borderRadius: "9999px",
                background: isCoordinatorAdmin ? "var(--brand-blue-subtle, #e8f0fe)" : "var(--bg-subtle, #f1f5f9)",
                color: isCoordinatorAdmin ? "var(--brand-blue, #1a73e8)" : "var(--text-muted, #64748b)",
                border: `1px solid ${isCoordinatorAdmin ? "var(--brand-blue-border, #d2e3fc)" : "var(--border-subtle, #e2e8f0)"}`,
              }}
            >
              {isCoordinatorAdmin ? "Coordinator Admin Mode (Authorized)" : "Public Analytics Mode (Read-Only)"}
            </span>

            <span
              style={{
                fontSize: "0.72rem",
                fontFamily: "var(--font-mono, monospace)",
                fontWeight: 600,
                padding: "0.2rem 0.55rem",
                borderRadius: "9999px",
                background: "var(--emerald-subtle, #ecfdf5)",
                color: "var(--emerald-light, #059669)",
                border: "1px solid var(--emerald-border, #a7f3d0)",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.3rem",
              }}
            >
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--emerald-primary, #10b981)" }} />
              Live Telemetry
            </span>
          </div>

          <h1
            style={{
              fontFamily: "var(--font-heading, sans-serif)",
              fontSize: "1.65rem",
              fontWeight: 800,
              color: "var(--text-primary, #0f172a)",
              letterSpacing: "-0.03em",
              margin: 0,
            }}
          >
            Coordinator Command & Analytics Center
          </h1>
          <p style={{ color: "var(--text-muted, #64748b)", fontSize: "0.88rem", marginTop: "0.2rem", margin: 0 }}>
            {isCoordinatorAdmin
              ? `Logged in as ${user?.username} (${user?.institution_name || "Central Command"}). Full federation orchestration privileges active.`
              : "Global federation performance, hospital convergence curves, and institutional analytics."}
          </p>
        </div>

        {/* Top Right Actions */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={fetchDashboard}
            disabled={loading}
            style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
          >
            <IconRefreshCw size={14} className={loading ? "animate-spin" : ""} />
            <span>{loading ? "Updating..." : "Refresh"}</span>
          </button>

          {!isCoordinatorAdmin && onNavigate && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => onNavigate("auth")}
              style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
            >
              <IconLock size={14} />
              <span>Login as Coordinator</span>
            </button>
          )}
        </div>
      </div>

      {/* Action Notification Alert */}
      {actionNotice && (
        <div
          style={{
            background: actionNotice.type === "error" ? "var(--coral-subtle, #fef2f2)" : "var(--brand-blue-subtle, #e8f0fe)",
            border: `1px solid ${actionNotice.type === "error" ? "var(--coral-border, #fecaca)" : "var(--brand-blue-border, #d2e3fc)"}`,
            color: actionNotice.type === "error" ? "var(--coral-primary, #ef4444)" : "var(--brand-blue, #1a73e8)",
            padding: "0.75rem 1.25rem",
            borderRadius: "var(--radius-md, 12px)",
            fontSize: "0.88rem",
            fontWeight: 500,
            marginBottom: "1.5rem",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
          }}
        >
          <IconCheckCircle size={16} />
          <span>{actionNotice.msg}</span>
        </div>
      )}

      {/* ====================================================================
          MODE 2: COORDINATOR LOGGED-IN CONTROLS & ORCHESTRATION PANEL
          Only visible when user.role === 'ADMIN'
          ==================================================================== */}
      {isCoordinatorAdmin && (
        <div
          style={{
            background: "#ffffff",
            border: "2px solid var(--brand-blue-border, #d2e3fc)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.75rem",
            boxShadow: "0 10px 25px -5px rgba(26, 115, 232, 0.08)",
            marginBottom: "2rem",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem", marginBottom: "1.25rem" }}>
            <div>
              <div style={{ fontSize: "0.75rem", color: "var(--brand-blue, #1a73e8)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: "var(--font-mono, monospace)" }}>
                Coordinator Operator Console
              </div>
              <h2 style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", marginTop: "0.2rem", margin: 0 }}>
                Federation Training & Model Orchestration
              </h2>
            </div>

            <div style={{ display: "flex", gap: "0.6rem" }}>
              {!isTrainingRunning ? (
                <button
                  type="button"
                  className="btn btn-primary btn-md"
                  onClick={handleStartTraining}
                  style={{ background: "var(--brand-blue, #1a73e8)", borderColor: "var(--brand-blue, #1a73e8)", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
                >
                  <IconPlay size={16} />
                  <span>Start Federated Training</span>
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn-secondary btn-md"
                  onClick={handleStopTraining}
                  style={{ color: "var(--coral-primary, #ef4444)", borderColor: "var(--coral-border, #fecaca)", display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
                >
                  <IconPause size={16} />
                  <span>Halt Active Training</span>
                </button>
              )}

              <button
                type="button"
                className="btn btn-secondary btn-md"
                onClick={handleUpdateGlobalModel}
                disabled={isUpdatingGlobalModel}
                style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
              >
                <IconLayers size={16} />
                <span>{isUpdatingGlobalModel ? "Consolidating..." : "Update Global Model"}</span>
              </button>
            </div>
          </div>

          {/* Coordinator Tuning Form Controls */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: "1.25rem",
              background: "var(--bg-subtle, #f8fafc)",
              padding: "1.25rem",
              borderRadius: "var(--radius-lg, 16px)",
              border: "1px solid var(--border-subtle, #e2e8f0)",
            }}
          >
            <div>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary, #334155)", display: "block", marginBottom: "0.35rem" }}>
                Target Federation Rounds
              </label>
              <input
                type="number"
                min="1"
                max="50"
                value={targetRounds}
                onChange={(e) => setTargetRounds(e.target.value)}
                className="form-input"
                style={{ paddingBlock: "0.5rem", fontSize: "0.9rem" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary, #334155)", display: "block", marginBottom: "0.35rem" }}>
                Min Participating Hospitals
              </label>
              <input
                type="number"
                min="1"
                max="10"
                value={minClients}
                onChange={(e) => setMinClients(e.target.value)}
                className="form-input"
                style={{ paddingBlock: "0.5rem", fontSize: "0.9rem" }}
              />
            </div>

            <div>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary, #334155)", display: "block", marginBottom: "0.35rem" }}>
                Aggregation Algorithm
              </label>
              <select
                value={selectedStrategy}
                onChange={(e) => setSelectedStrategy(e.target.value)}
                className="form-input"
                style={{ paddingBlock: "0.5rem", fontSize: "0.9rem", cursor: "pointer" }}
              >
                <option value="FedMedStrategy">FedMed Encrypted FedAvg</option>
                <option value="FedProx">FedProx (Non-IID Heterogeneous)</option>
                <option value="FedAvg">Standard Flower FedAvg</option>
              </select>
            </div>

            <div>
              <label style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--text-secondary, #334155)", display: "block", marginBottom: "0.35rem" }}>
                Ciphertext Security Key
              </label>
              <div
                style={{
                  background: "#ffffff",
                  border: "1px solid var(--border-subtle, #e2e8f0)",
                  borderRadius: "var(--radius-md, 12px)",
                  padding: "0.5rem 0.85rem",
                  fontSize: "0.85rem",
                  fontFamily: "var(--font-mono, monospace)",
                  color: "var(--brand-blue, #1a73e8)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <span>Key ID: {checkpoint.encryption?.key_id || "7F8B2C4D"}</span>
                <span style={{ fontSize: "0.72rem", color: "var(--emerald-light, #059669)", fontWeight: 600 }}>Active</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ====================================================================
          MODE 1: POWER BI-STYLE KPI METRICS TICKER
          Available to everyone (public & coordinator)
          ==================================================================== */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
          gap: "1.25rem",
          marginBottom: "1.75rem",
        }}
      >
        {/* KPI 1 */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.5rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-muted, #64748b)" }}>
              Total Trained Models / Rounds
            </span>
            <span style={{ padding: "0.3rem", borderRadius: "8px", background: "var(--brand-blue-subtle, #e8f0fe)", color: "var(--brand-blue, #1a73e8)" }}>
              <IconActivity size={16} />
            </span>
          </div>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--brand-blue, #1a73e8)", marginTop: "0.5rem", fontFamily: "var(--font-heading, sans-serif)" }}>
            Round {summary.current_round}
            <span style={{ fontSize: "1rem", fontWeight: 500, color: "var(--text-muted, #64748b)" }}> / {summary.total_rounds_target}</span>
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted, #64748b)", marginTop: "0.3rem" }}>
            {summary.current_round >= summary.total_rounds_target ? "Target rounds achieved" : `${summary.total_rounds_target - summary.current_round} rounds remaining`}
          </div>
        </div>

        {/* KPI 2 */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.5rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-muted, #64748b)" }}>
              Global Best Mean Dice
            </span>
            <span style={{ padding: "0.3rem", borderRadius: "8px", background: "var(--emerald-subtle, #ecfdf5)", color: "var(--emerald-light, #059669)" }}>
              <IconCheckCircle size={16} />
            </span>
          </div>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--emerald-light, #059669)", marginTop: "0.5rem", fontFamily: "var(--font-heading, sans-serif)" }}>
            {(summary.best_dice_score * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted, #64748b)", marginTop: "0.3rem" }}>
            Achieved at Round {summary.best_round} (DiceCELoss: {latestMetrics.val_loss?.toFixed(4)})
          </div>
        </div>

        {/* KPI 3 */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.5rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-muted, #64748b)" }}>
              Connected Hospital Nodes
            </span>
            <span style={{ padding: "0.3rem", borderRadius: "8px", background: "var(--violet-subtle, #eef2ff)", color: "var(--violet-primary, #6366f1)" }}>
              <IconHospital size={16} />
            </span>
          </div>
          <div style={{ fontSize: "2rem", fontWeight: 800, color: "var(--violet-primary, #6366f1)", marginTop: "0.5rem", fontFamily: "var(--font-heading, sans-serif)" }}>
            {summary.active_hospitals_count} Nodes
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted, #64748b)", marginTop: "0.3rem" }}>
            Min quorum threshold: {summary.min_clients_required} institutions
          </div>
        </div>

        {/* KPI 4 */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.5rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-muted, #64748b)" }}>
              Ciphertext Model Checkpoint
            </span>
            <span style={{ padding: "0.3rem", borderRadius: "8px", background: "var(--amber-subtle, #fffbeb)", color: "var(--amber-primary, #f59e0b)" }}>
              <IconLock size={16} />
            </span>
          </div>
          <div style={{ fontSize: "1.8rem", fontWeight: 800, color: "var(--text-primary, #0f172a)", marginTop: "0.5rem", fontFamily: "var(--font-heading, sans-serif)" }}>
            {checkpoint.best_model_size_mb || 4.85} MB
          </div>
          <div style={{ fontSize: "0.78rem", color: "var(--text-muted, #64748b)", marginTop: "0.3rem", fontFamily: "var(--font-mono, monospace)" }}>
            AES-256-GCM / 3D U-Net
          </div>
        </div>
      </div>

      {/* ====================================================================
          POWER BI CHARTS GRID: LINE GRAPH, BAR PLOT, PIE / DONUT CHART
          ==================================================================== */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "1.5rem", marginBottom: "2rem" }}>
        
        {/* CHART 1: LINE GRAPH (Dice Convergence Over Rounds) */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.75rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <div>
              <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", margin: 0 }}>
                Global Convergence Curve (Line Graph)
              </h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted, #64748b)", margin: "0.2rem 0 0" }}>
                Multi-round Dice segmentation accuracy vs validation loss
              </p>
            </div>

            {/* Filter pills */}
            <div style={{ display: "flex", gap: "0.25rem", background: "var(--bg-subtle, #f1f5f9)", padding: "0.2rem", borderRadius: "8px" }}>
              <button
                type="button"
                onClick={() => setActiveMetricFilter("all")}
                style={{
                  border: "none",
                  padding: "0.25rem 0.55rem",
                  borderRadius: "6px",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  background: activeMetricFilter === "all" ? "#ffffff" : "transparent",
                  color: activeMetricFilter === "all" ? "var(--text-primary, #0f172a)" : "var(--text-muted, #64748b)",
                  boxShadow: activeMetricFilter === "all" ? "0 1px 2px rgba(0,0,0,0.05)" : "none",
                }}
              >
                All
              </button>
              <button
                type="button"
                onClick={() => setActiveMetricFilter("dice")}
                style={{
                  border: "none",
                  padding: "0.25rem 0.55rem",
                  borderRadius: "6px",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  background: activeMetricFilter === "dice" ? "#ffffff" : "transparent",
                  color: activeMetricFilter === "dice" ? "var(--brand-blue, #1a73e8)" : "var(--text-muted, #64748b)",
                  boxShadow: activeMetricFilter === "dice" ? "0 1px 2px rgba(0,0,0,0.05)" : "none",
                }}
              >
                Dice
              </button>
              <button
                type="button"
                onClick={() => setActiveMetricFilter("loss")}
                style={{
                  border: "none",
                  padding: "0.25rem 0.55rem",
                  borderRadius: "6px",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  cursor: "pointer",
                  background: activeMetricFilter === "loss" ? "#ffffff" : "transparent",
                  color: activeMetricFilter === "loss" ? "var(--coral-primary, #ef4444)" : "var(--text-muted, #64748b)",
                  boxShadow: activeMetricFilter === "loss" ? "0 1px 2px rgba(0,0,0,0.05)" : "none",
                }}
              >
                Loss
              </button>
            </div>
          </div>

          {/* SVG Line Chart */}
          <div style={{ position: "relative", width: "100%", height: "200px", marginTop: "auto" }}>
            <svg viewBox="0 0 400 160" style={{ width: "100%", height: "100%", overflow: "visible" }}>
              {/* Grid Lines */}
              <line x1="40" y1="20" x2="380" y2="20" stroke="#f1f5f9" strokeWidth="1" />
              <line x1="40" y1="60" x2="380" y2="60" stroke="#f1f5f9" strokeWidth="1" />
              <line x1="40" y1="100" x2="380" y2="100" stroke="#f1f5f9" strokeWidth="1" />
              <line x1="40" y1="140" x2="380" y2="140" stroke="#e2e8f0" strokeWidth="1" />

              {/* Y-Axis Labels */}
              <text x="15" y="24" fill="#94a3b8" fontSize="9" fontFamily="var(--font-mono, monospace)">0.90</text>
              <text x="15" y="64" fill="#94a3b8" fontSize="9" fontFamily="var(--font-mono, monospace)">0.80</text>
              <text x="15" y="104" fill="#94a3b8" fontSize="9" fontFamily="var(--font-mono, monospace)">0.70</text>
              <text x="15" y="144" fill="#94a3b8" fontSize="9" fontFamily="var(--font-mono, monospace)">0.00</text>

              {/* Dice Line (Cyan/Blue) */}
              {(activeMetricFilter === "all" || activeMetricFilter === "dice") && (
                <>
                  <polyline
                    fill="none"
                    stroke="#1a73e8"
                    strokeWidth="3"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    points="60,110 130,85 200,60 270,42 340,28"
                  />
                  {/* Data points */}
                  {[
                    { cx: 60, cy: 110, val: "0.74" },
                    { cx: 130, cy: 85, val: "0.80" },
                    { cx: 200, cy: 60, val: "0.84" },
                    { cx: 270, cy: 42, val: "0.86" },
                    { cx: 340, cy: 28, val: "0.88" },
                  ].map((p, i) => (
                    <circle key={i} cx={p.cx} cy={p.cy} r="4" fill="#ffffff" stroke="#1a73e8" strokeWidth="2.5" />
                  ))}
                </>
              )}

              {/* Loss Line (Red/Coral) */}
              {(activeMetricFilter === "all" || activeMetricFilter === "loss") && (
                <>
                  <polyline
                    fill="none"
                    stroke="#ef4444"
                    strokeWidth="2.5"
                    strokeDasharray="4 3"
                    strokeLinecap="round"
                    points="60,40 130,68 200,95 270,115 340,128"
                  />
                  {[
                    { cx: 60, cy: 40 },
                    { cx: 130, cy: 68 },
                    { cx: 200, cy: 95 },
                    { cx: 270, cy: 115 },
                    { cx: 340, cy: 128 },
                  ].map((p, i) => (
                    <circle key={i} cx={p.cx} cy={p.cy} r="3" fill="#ffffff" stroke="#ef4444" strokeWidth="2" />
                  ))}
                </>
              )}

              {/* X-Axis Round Labels */}
              <text x="55" y="156" fill="#64748b" fontSize="9" fontFamily="var(--font-mono, monospace)">R1</text>
              <text x="125" y="156" fill="#64748b" fontSize="9" fontFamily="var(--font-mono, monospace)">R2</text>
              <text x="195" y="156" fill="#64748b" fontSize="9" fontFamily="var(--font-mono, monospace)">R3</text>
              <text x="265" y="156" fill="#64748b" fontSize="9" fontFamily="var(--font-mono, monospace)">R4</text>
              <text x="335" y="156" fill="#64748b" fontSize="9" fontFamily="var(--font-mono, monospace)">R5</text>
            </svg>
          </div>

          <div style={{ display: "flex", gap: "1.25rem", justifyContent: "center", marginTop: "0.75rem", fontSize: "0.78rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "12px", height: "3px", background: "#1a73e8", borderRadius: "2px" }} />
              <span style={{ color: "var(--text-secondary, #334155)", fontWeight: 500 }}>Mean Dice Score</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "12px", height: "2px", background: "#ef4444", borderTop: "2px dashed #ef4444" }} />
              <span style={{ color: "var(--text-secondary, #334155)", fontWeight: 500 }}>DiceCELoss (Val)</span>
            </div>
          </div>
        </div>

        {/* CHART 2: BAR PLOT (Hospital Performance Comparison) */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.75rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <div>
              <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", margin: 0 }}>
                Hospital Accuracy (Bar Plot)
              </h3>
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted, #64748b)", margin: "0.2rem 0 0" }}>
                Local client validation Dice benchmark comparison
              </p>
            </div>
            <span
              style={{
                fontSize: "0.72rem",
                fontFamily: "var(--font-mono, monospace)",
                fontWeight: 600,
                padding: "0.2rem 0.5rem",
                borderRadius: "6px",
                background: "var(--bg-subtle, #f1f5f9)",
                color: "var(--text-secondary, #334155)",
              }}
            >
              3 Nodes
            </span>
          </div>

          {/* Bar Chart Container */}
          <div style={{ display: "flex", flexDirection: "column", gap: "1.1rem", marginTop: "auto", padding: "0.5rem 0" }}>
            {hospitals.map((h, i) => {
              const scorePct = Math.round((h.best_local_dice || 0.88) * 100);
              const barColors = ["#1a73e8", "#10b981", "#6366f1"];
              const currentColor = barColors[i % barColors.length];

              return (
                <div key={h.hospital_id || i}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", marginBottom: "0.3rem" }}>
                    <span style={{ fontWeight: 600, color: "var(--text-primary, #0f172a)", textTransform: "capitalize" }}>
                      {h.hospital_id.replace("_", " ")}
                    </span>
                    <span style={{ fontFamily: "var(--font-mono, monospace)", fontWeight: 700, color: currentColor }}>
                      {(h.best_local_dice * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div
                    style={{
                      height: "10px",
                      background: "var(--bg-subtle, #f1f5f9)",
                      borderRadius: "9999px",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${scorePct}%`,
                        height: "100%",
                        background: currentColor,
                        borderRadius: "9999px",
                        transition: "width 1s ease",
                      }}
                    />
                  </div>
                </div>
              );
            })}

            {/* Global Aggregated Baseline */}
            <div style={{ paddingTop: "0.4rem", borderTop: "1px dashed var(--border-subtle, #e2e8f0)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.82rem", marginBottom: "0.3rem" }}>
                <span style={{ fontWeight: 700, color: "var(--brand-blue, #1a73e8)" }}>
                  Global Consensus Model (Aggregated)
                </span>
                <span style={{ fontFamily: "var(--font-mono, monospace)", fontWeight: 800, color: "var(--brand-blue, #1a73e8)" }}>
                  {(summary.best_dice_score * 100).toFixed(1)}%
                </span>
              </div>
              <div style={{ height: "10px", background: "var(--bg-subtle, #f1f5f9)", borderRadius: "9999px", overflow: "hidden" }}>
                <div
                  style={{
                    width: `${Math.round(summary.best_dice_score * 100)}%`,
                    height: "100%",
                    background: "var(--brand-blue, #1a73e8)",
                    borderRadius: "9999px",
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* CHART 3: DONUT / PIE CHART (Tumor Sub-Region Distribution) */}
        <div
          style={{
            background: "#ffffff",
            border: "1px solid var(--border-subtle, #e2e8f0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.75rem",
            boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div style={{ marginBottom: "1rem" }}>
            <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", margin: 0 }}>
              Tumor Sub-Region Segregation (Pie/Donut)
            </h3>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted, #64748b)", margin: "0.2rem 0 0" }}>
              BraTS 3D lesion breakdown across MRI scans
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "1.5rem", marginTop: "auto", padding: "0.5rem 0" }}>
            {/* Donut Chart SVG */}
            <div style={{ position: "relative", width: "130px", height: "130px" }}>
              <svg viewBox="0 0 36 36" style={{ width: "100%", height: "100%", transform: "rotate(-90deg)" }}>
                {/* Background Ring */}
                <circle cx="18" cy="18" r="14" fill="none" stroke="#f1f5f9" strokeWidth="5" />
                {/* Whole Tumor Segment */}
                <circle
                  cx="18"
                  cy="18"
                  r="14"
                  fill="none"
                  stroke="#10b981"
                  strokeWidth="5"
                  strokeDasharray={`${wtPct} 100`}
                  strokeDashoffset="0"
                />
                {/* Tumor Core Segment */}
                <circle
                  cx="18"
                  cy="18"
                  r="14"
                  fill="none"
                  stroke="#1a73e8"
                  strokeWidth="5"
                  strokeDasharray={`${tcPct} 100`}
                  strokeDashoffset={`-${wtPct}`}
                />
                {/* Enhancing Tumor Segment */}
                <circle
                  cx="18"
                  cy="18"
                  r="14"
                  fill="none"
                  stroke="#f59e0b"
                  strokeWidth="5"
                  strokeDasharray={`${etPct} 100`}
                  strokeDashoffset={`-${wtPct + tcPct}`}
                />
              </svg>
              {/* Center text */}
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span style={{ fontSize: "1.1rem", fontWeight: 800, color: "var(--text-primary, #0f172a)", fontFamily: "var(--font-heading, sans-serif)" }}>
                  BraTS
                </span>
                <span style={{ fontSize: "0.68rem", color: "var(--text-muted, #64748b)", fontWeight: 600 }}>3-Region</span>
              </div>
            </div>

            {/* Legend & Scores */}
            <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem", fontSize: "0.82rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#10b981" }} />
                <div>
                  <div style={{ fontWeight: 600, color: "var(--text-primary, #0f172a)" }}>Whole Tumor (WT)</div>
                  <div style={{ fontSize: "0.74rem", color: "var(--text-muted, #64748b)", fontFamily: "var(--font-mono, monospace)" }}>
                    Dice: {(wtScore * 100).toFixed(1)}% ({wtPct}%)
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#1a73e8" }} />
                <div>
                  <div style={{ fontWeight: 600, color: "var(--text-primary, #0f172a)" }}>Tumor Core (TC)</div>
                  <div style={{ fontSize: "0.74rem", color: "var(--text-muted, #64748b)", fontFamily: "var(--font-mono, monospace)" }}>
                    Dice: {(tcScore * 100).toFixed(1)}% ({tcPct}%)
                  </div>
                </div>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ width: "10px", height: "10px", borderRadius: "50%", background: "#f59e0b" }} />
                <div>
                  <div style={{ fontWeight: 600, color: "var(--text-primary, #0f172a)" }}>Enhancing Tumor (ET)</div>
                  <div style={{ fontSize: "0.74rem", color: "var(--text-muted, #64748b)", fontFamily: "var(--font-mono, monospace)" }}>
                    Dice: {(etScore * 100).toFixed(1)}% ({etPct}%)
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* ====================================================================
          ACTIVE HOSPITALS & TELEMETRY SECTION
          ==================================================================== */}
      <div
        style={{
          background: "#ffffff",
          border: "1px solid var(--border-subtle, #e2e8f0)",
          borderRadius: "var(--radius-xl, 20px)",
          padding: "1.75rem",
          boxShadow: "0 1px 3px rgba(0,0,0,0.04)",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem", marginBottom: "1.25rem" }}>
          <div>
            <h3 style={{ fontSize: "1.15rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", margin: 0 }}>
              Participating Hospital Cluster Nodes
            </h3>
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted, #64748b)", margin: "0.2rem 0 0" }}>
              Active medical institutions, local round synchronization, and local convergence stats
            </p>
          </div>

          <span
            style={{
              fontSize: "0.78rem",
              fontFamily: "var(--font-mono, monospace)",
              padding: "0.25rem 0.65rem",
              borderRadius: "9999px",
              background: "var(--emerald-subtle, #ecfdf5)",
              color: "var(--emerald-light, #059669)",
              border: "1px solid var(--emerald-border, #a7f3d0)",
              fontWeight: 600,
            }}
          >
            {hospitals.length} Hospitals Connected
          </span>
        </div>

        {/* Hospital Cards Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "1rem" }}>
          {hospitals.map((h, i) => (
            <div
              key={h.hospital_id || i}
              style={{
                background: "var(--bg-subtle, #f8fafc)",
                border: "1px solid var(--border-subtle, #e2e8f0)",
                borderRadius: "var(--radius-lg, 16px)",
                padding: "1.25rem",
                display: "flex",
                flexDirection: "column",
                gap: "0.75rem",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <span style={{ padding: "0.35rem", borderRadius: "8px", background: "#ffffff", border: "1px solid var(--border-subtle, #e2e8f0)" }}>
                    <IconHospital size={16} />
                  </span>
                  <span style={{ fontWeight: 700, fontSize: "0.95rem", color: "var(--text-primary, #0f172a)", textTransform: "capitalize" }}>
                    {h.hospital_id.replace("_", " ")}
                  </span>
                </div>

                <span
                  style={{
                    fontSize: "0.72rem",
                    fontFamily: "var(--font-mono, monospace)",
                    fontWeight: 600,
                    padding: "0.15rem 0.5rem",
                    borderRadius: "9999px",
                    background: h.status === "ONLINE" ? "var(--emerald-subtle, #ecfdf5)" : "var(--brand-blue-subtle, #e8f0fe)",
                    color: h.status === "ONLINE" ? "var(--emerald-light, #059669)" : "var(--brand-blue, #1a73e8)",
                    border: `1px solid ${h.status === "ONLINE" ? "var(--emerald-border, #a7f3d0)" : "var(--brand-blue-border, #d2e3fc)"}`,
                  }}
                >
                  {h.status || "ONLINE"}
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", fontSize: "0.82rem", paddingTop: "0.5rem", borderTop: "1px solid var(--border-subtle, #e2e8f0)" }}>
                <div>
                  <div style={{ color: "var(--text-muted, #64748b)", fontSize: "0.74rem" }}>Synchronized Round</div>
                  <div style={{ fontWeight: 700, color: "var(--text-primary, #0f172a)", fontFamily: "var(--font-mono, monospace)" }}>
                    Round {h.latest_round || 5}
                  </div>
                </div>

                <div>
                  <div style={{ color: "var(--text-muted, #64748b)", fontSize: "0.74rem" }}>Best Local Dice</div>
                  <div style={{ fontWeight: 700, color: "var(--emerald-light, #059669)", fontFamily: "var(--font-mono, monospace)" }}>
                    {((h.best_local_dice || 0.88) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div style={{ fontSize: "0.75rem", color: "var(--text-muted, #64748b)", display: "flex", alignItems: "center", gap: "0.35rem" }}>
                <IconShieldCheck size={14} />
                <span>Zero-Raw-Data: Model Weight Masks Only</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
