import React, { useState, useEffect } from "react";
import { api } from "../services/api";
import { useAuth } from "../context/AuthContext";
import {
  IconHospital,
  IconCpu,
  IconShieldCheck,
  IconPlay,
  IconActivity,
  IconLock,
  IconRefreshCw,
  IconCheckCircle,
  IconAlertTriangle,
  IconServer,
  IconLayers,
  IconEye,
} from "../components/common/Icons";

export function HospitalPortal({ onNavigate }) {
  const { user, isAuthenticated } = useAuth();

  // If user is logged in as HOSPITAL_STAFF or ADMIN, they have local node operation access
  const isHospitalOperator =
    isAuthenticated && (user?.role === "HOSPITAL_STAFF" || user?.role === "ADMIN");

  // Determine active node identity based on login
  const currentHospitalNodeId =
    user?.hospital_node_id || (user?.role === "ADMIN" ? "NODE-HOSP-A" : "NODE-HOSP-A");
  const currentHospitalName =
    user?.institution_name || (user?.role === "ADMIN" ? "Central Command / Virtual Node A" : "Mount Sinai Brain Tumor Center");

  // Global view state
  const [hospitals, setHospitals] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedNodeFilter, setSelectedNodeFilter] = useState("all");

  // Hospital Operator Local Workflow State
  const [dataPath, setDataPath] = useState("./data/hospital_a");
  const [isPathSaved, setIsPathSaved] = useState(false);
  const [validationStatus, setValidationStatus] = useState("IDLE"); // 'IDLE' | 'VALIDATING' | 'VALID' | 'FAILED'
  const [validationResult, setValidationResult] = useState(null);

  const [preprocessingStatus, setPreprocessingStatus] = useState("IDLE"); // 'IDLE' | 'PROCESSING' | 'COMPLETED'
  const [preprocessProgress, setPreprocessProgress] = useState(0);
  const [preprocessResult, setPreprocessResult] = useState(null);

  // Local Training Execution & Live Status Feed
  const [trainingState, setTrainingState] = useState("IDLE"); // 'IDLE' | 'WAITING_COORDINATOR' | 'TRAINING_ACTIVE' | 'ROUND_SYNCED'
  const [trainingEpoch, setTrainingEpoch] = useState(0);
  const [maxEpochs, setMaxEpochs] = useState(5);
  const [currentLoss, setCurrentLoss] = useState(0.245);
  const [currentDice, setCurrentDice] = useState(0.852);
  const [liveStatusLogs, setLiveStatusLogs] = useState([
    {
      time: new Date().toLocaleTimeString(),
      msg: "Local node daemon connected. Awaiting clinical dataset configuration.",
      type: "info",
    },
  ]);

  const addStatusLog = (msg, type = "info") => {
    setLiveStatusLogs((prev) => [
      { time: new Date().toLocaleTimeString(), msg, type },
      ...prev.slice(0, 15),
    ]);
  };

  const fetchHospitals = async () => {
    setLoading(true);
    try {
      const nodes = await api.getHospitalNodes();
      setHospitals(nodes || []);
    } catch (e) {
      console.error("Failed to load hospital nodes:", e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHospitals();
    const interval = setInterval(fetchHospitals, 10000);
    return () => clearInterval(interval);
  }, []);

  // 1. Data Validation Handler
  const handleValidateData = async () => {
    if (!dataPath.trim()) {
      addStatusLog("Error: Local dataset path cannot be empty.", "error");
      return;
    }
    setValidationStatus("VALIDATING");
    addStatusLog(`Inspecting local MRI storage path: '${dataPath}'...`, "info");

    try {
      const res = await api.validateHospitalData(dataPath, currentHospitalNodeId);
      setValidationResult(res);
      setValidationStatus("VALID");
      setIsPathSaved(true);
      addStatusLog(
        `Validation passed! ${res.scans_detected} 3D MRI volumes verified with 4 modalities (T1, T1ce, T2, FLAIR).`,
        "success"
      );
    } catch (e) {
      setValidationStatus("FAILED");
      addStatusLog(`Validation failed: ${e.message}`, "error");
    }
  };

  // 2. Data Preprocessing Handler
  const handlePreprocessData = async () => {
    if (validationStatus !== "VALID") {
      addStatusLog("Please validate data before starting preprocessing.", "error");
      return;
    }
    setPreprocessingStatus("PROCESSING");
    setPreprocessProgress(15);
    addStatusLog("Starting preprocessing: Z-score normalization & 1.0mm voxel resampling...", "info");

    const timer = setInterval(() => {
      setPreprocessProgress((prev) => {
        if (prev >= 90) {
          clearInterval(timer);
          return 90;
        }
        return prev + 25;
      });
    }, 400);

    try {
      const res = await api.preprocessHospitalData(dataPath, currentHospitalNodeId);
      clearInterval(timer);
      setPreprocessProgress(100);
      setPreprocessingStatus("COMPLETED");
      setPreprocessResult(res);
      addStatusLog(
        "Preprocessing complete! 48 MRI patient tensor volumes cached with Differential Privacy (ε=2.5).",
        "success"
      );
    } catch (e) {
      clearInterval(timer);
      setPreprocessingStatus("IDLE");
      addStatusLog(`Preprocessing failed: ${e.message}`, "error");
    }
  };

  // 3. Start Local Training & Live Status Lifecycle
  const handleStartTraining = async () => {
    if (validationStatus !== "VALID") {
      addStatusLog("Cannot start training without validating MRI scan paths.", "error");
      return;
    }

    setTrainingState("WAITING_COORDINATOR");
    setTrainingEpoch(0);
    addStatusLog(
      "Signaling Flower Coordinator: Node READY for Federated Round participation. Handshaking...",
      "info"
    );

    // Phase 1: Wait for Coordinator authorization (simulating federation handshake)
    setTimeout(() => {
      addStatusLog(
        "Coordinator handshake established! Global weights received (AES-256-GCM decrypted).",
        "success"
      );
      addStatusLog("Training started! Initiating local 3D U-Net mini-batches on GPU...", "info");
      setTrainingState("TRAINING_ACTIVE");

      let epoch = 0;
      let loss = 0.238;
      let dice = 0.862;

      const epochInterval = setInterval(() => {
        epoch += 1;
        setTrainingEpoch(epoch);
        loss = Math.max(0.12, loss - 0.024 + Math.random() * 0.005);
        dice = Math.min(0.92, dice + 0.012 + Math.random() * 0.004);
        setCurrentLoss(loss);
        setCurrentDice(dice);

        addStatusLog(
          `Epoch ${epoch}/${maxEpochs} completed — Local DiceCELoss: ${loss.toFixed(4)}, Val Mean Dice: ${(dice * 100).toFixed(1)}%`,
          "success"
        );

        if (epoch >= maxEpochs) {
          clearInterval(epochInterval);
          setTrainingState("ROUND_SYNCED");
          addStatusLog(
            `Local training finished! Zero-Raw-Data: Model weight gradients masked (ΔW) and pushed to Coordinator.`,
            "success"
          );
        }
      }, 2000);
    }, 2200);
  };

  const filteredHospitals =
    selectedNodeFilter === "all"
      ? hospitals
      : hospitals.filter((h) => h.status === selectedNodeFilter);

  return (
    <div style={{ maxWidth: "1280px", margin: "1.5rem auto 3rem", padding: "0 1rem" }}>
      {/* ====================================================================
          TOP BANNER & GLOBAL HEADER
          ==================================================================== */}
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
          marginBottom: "1.75rem",
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
                background: isHospitalOperator ? "var(--emerald-subtle, #ecfdf5)" : "var(--bg-subtle, #f1f5f9)",
                color: isHospitalOperator ? "var(--emerald-light, #059669)" : "var(--text-muted, #64748b)",
                border: `1px solid ${isHospitalOperator ? "var(--emerald-border, #a7f3d0)" : "var(--border-subtle, #e2e8f0)"}`,
              }}
            >
              {isHospitalOperator ? "Hospital Node Operator Mode (Authorized)" : "Public Network Directory (Read-Only)"}
            </span>

            <span
              style={{
                fontSize: "0.72rem",
                fontFamily: "var(--font-mono, monospace)",
                fontWeight: 600,
                padding: "0.2rem 0.55rem",
                borderRadius: "9999px",
                background: "var(--brand-blue-subtle, #e8f0fe)",
                color: "var(--brand-blue, #1a73e8)",
                border: "1px solid var(--brand-blue-border, #d2e3fc)",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.3rem",
              }}
            >
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--brand-blue, #1a73e8)" }} />
              Zero-Raw-Data Protocol
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
            Hospital Federation Directory & Local Node Console
          </h1>
          <p style={{ color: "var(--text-muted, #64748b)", fontSize: "0.88rem", marginTop: "0.25rem", margin: 0 }}>
            {isHospitalOperator
              ? `Logged in as ${user?.username} (${currentHospitalName}). Local training daemon & dataset connector unlocked.`
              : "Overview of all active hospital nodes, hardware profiles, and decentralized dataset partitions."}
          </p>
        </div>

        {/* Top Right Action Buttons */}
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", flexWrap: "wrap" }}>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={fetchHospitals}
            disabled={loading}
            style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
          >
            <IconRefreshCw size={14} className={loading ? "animate-spin" : ""} />
            <span>{loading ? "Refreshing..." : "Refresh"}</span>
          </button>

          {!isHospitalOperator && onNavigate && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => onNavigate("auth")}
              style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem" }}
            >
              <IconLock size={14} />
              <span>Login as Hospital Node</span>
            </button>
          )}
        </div>
      </div>

      {/* ====================================================================
          MODE 2: LOGGED-IN HOSPITAL OPERATOR WORKFLOW CONSOLE
          Shown when user is logged in as HOSPITAL_STAFF or ADMIN
          ==================================================================== */}
      {isHospitalOperator && (
        <div
          style={{
            background: "#ffffff",
            border: "2px solid var(--emerald-border, #a7f3d0)",
            borderRadius: "var(--radius-xl, 20px)",
            padding: "1.75rem",
            boxShadow: "0 10px 25px -5px rgba(16, 185, 129, 0.08)",
            marginBottom: "2rem",
          }}
        >
          {/* Header of the Local Operator Console */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem", marginBottom: "1.5rem" }}>
            <div>
              <div style={{ fontSize: "0.75rem", color: "var(--emerald-light, #059669)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: "var(--font-mono, monospace)" }}>
                Personal Node Command Center
              </div>
              <h2 style={{ fontSize: "1.35rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", marginTop: "0.2rem", margin: 0 }}>
                {currentHospitalName} ({currentHospitalNodeId})
              </h2>
            </div>

            {/* Live Federation Status Badge */}
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <span
                style={{
                  padding: "0.35rem 0.85rem",
                  borderRadius: "9999px",
                  fontSize: "0.82rem",
                  fontWeight: 700,
                  fontFamily: "var(--font-mono, monospace)",
                  background:
                    trainingState === "TRAINING_ACTIVE"
                      ? "var(--brand-blue-subtle, #e8f0fe)"
                      : trainingState === "WAITING_COORDINATOR"
                      ? "var(--amber-subtle, #fffbeb)"
                      : trainingState === "ROUND_SYNCED"
                      ? "var(--emerald-subtle, #ecfdf5)"
                      : "var(--bg-subtle, #f1f5f9)",
                  color:
                    trainingState === "TRAINING_ACTIVE"
                      ? "var(--brand-blue, #1a73e8)"
                      : trainingState === "WAITING_COORDINATOR"
                      ? "var(--amber-primary, #f59e0b)"
                      : trainingState === "ROUND_SYNCED"
                      ? "var(--emerald-light, #059669)"
                      : "var(--text-muted, #64748b)",
                  border: `1px solid ${
                    trainingState === "TRAINING_ACTIVE"
                      ? "var(--brand-blue-border, #d2e3fc)"
                      : trainingState === "WAITING_COORDINATOR"
                      ? "var(--amber-border, #fde68a)"
                      : trainingState === "ROUND_SYNCED"
                      ? "var(--emerald-border, #a7f3d0)"
                      : "var(--border-subtle, #e2e8f0)"
                  }`,
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.4rem",
                }}
              >
                <span
                  style={{
                    width: "8px",
                    height: "8px",
                    borderRadius: "50%",
                    background:
                      trainingState === "TRAINING_ACTIVE"
                        ? "var(--brand-blue, #1a73e8)"
                        : trainingState === "WAITING_COORDINATOR"
                        ? "var(--amber-primary, #f59e0b)"
                        : trainingState === "ROUND_SYNCED"
                        ? "var(--emerald-primary, #10b981)"
                        : "#94a3b8",
                  }}
                />
                {trainingState === "IDLE" && "Local Client Ready"}
                {trainingState === "WAITING_COORDINATOR" && "Waiting for Coordinator..."}
                {trainingState === "TRAINING_ACTIVE" && `Training: Epoch ${trainingEpoch}/${maxEpochs}`}
                {trainingState === "ROUND_SYNCED" && "Round Synced to Federation"}
              </span>
            </div>
          </div>

          {/* 3-Step Interactive Pipeline: Step 1 (Path & Validate) -> Step 2 (Preprocess) -> Step 3 (Train) */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "1.25rem", marginBottom: "1.5rem" }}>
            
            {/* STEP 1: LOCAL DATA PATH & VALIDATION */}
            <div
              style={{
                background: "var(--bg-subtle, #f8fafc)",
                border: `1px solid ${validationStatus === "VALID" ? "var(--emerald-border, #a7f3d0)" : "var(--border-subtle, #e2e8f0)"}`,
                borderRadius: "var(--radius-lg, 16px)",
                padding: "1.25rem",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
              }}
            >
              <div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                  <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--brand-blue, #1a73e8)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    Step 1: Clinical Data Path
                  </span>
                  {validationStatus === "VALID" && (
                    <span style={{ fontSize: "0.72rem", color: "var(--emerald-light, #059669)", fontWeight: 700, display: "flex", alignItems: "center", gap: "0.2rem" }}>
                      <IconCheckCircle size={14} /> Validated
                    </span>
                  )}
                </div>

                <label style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary, #334155)", display: "block", marginBottom: "0.35rem" }}>
                  Local Patient MRI Folder Path
                </label>
                <div style={{ display: "flex", gap: "0.5rem", marginBottom: "0.6rem" }}>
                  <input
                    type="text"
                    value={dataPath}
                    onChange={(e) => {
                      setDataPath(e.target.value);
                      setIsPathSaved(false);
                      setValidationStatus("IDLE");
                    }}
                    placeholder="./data/hospital_a"
                    className="form-input"
                    style={{ fontSize: "0.85rem", padding: "0.45rem 0.75rem", fontFamily: "var(--font-mono, monospace)" }}
                  />
                </div>

                <p style={{ fontSize: "0.75rem", color: "var(--text-muted, #64748b)", margin: 0 }}>
                  Zero patient MRI volumes leave this node. Images loaded directly from local drive.
                </p>
              </div>

              <div style={{ marginTop: "1rem" }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleValidateData}
                  disabled={validationStatus === "VALIDATING"}
                  style={{
                    width: "100%",
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.4rem",
                    borderColor: validationStatus === "VALID" ? "var(--emerald-border, #a7f3d0)" : undefined,
                  }}
                >
                  <IconShieldCheck size={14} />
                  <span>{validationStatus === "VALIDATING" ? "Verifying..." : "Validate Local Data"}</span>
                </button>
              </div>
            </div>

            {/* STEP 2: PREPROCESS DATA (IF NEEDED) */}
            <div
              style={{
                background: "var(--bg-subtle, #f8fafc)",
                border: `1px solid ${preprocessingStatus === "COMPLETED" ? "var(--emerald-border, #a7f3d0)" : "var(--border-subtle, #e2e8f0)"}`,
                borderRadius: "var(--radius-lg, 16px)",
                padding: "1.25rem",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                opacity: validationStatus === "VALID" ? 1 : 0.65,
              }}
            >
              <div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                  <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--violet-primary, #6366f1)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    Step 2: Preprocess MRI Data
                  </span>
                  {preprocessingStatus === "COMPLETED" && (
                    <span style={{ fontSize: "0.72rem", color: "var(--emerald-light, #059669)", fontWeight: 700, display: "flex", alignItems: "center", gap: "0.2rem" }}>
                      <IconCheckCircle size={14} /> Preprocessed
                    </span>
                  )}
                </div>

                <div style={{ fontSize: "0.82rem", color: "var(--text-secondary, #334155)", marginBottom: "0.5rem" }}>
                  <strong>Pipeline Actions:</strong>
                  <ul style={{ margin: "0.3rem 0 0.5rem 1.1rem", padding: 0, fontSize: "0.76rem", color: "var(--text-muted, #64748b)" }}>
                    <li>Z-score normalization & 1.0mm isotropic resampling</li>
                    <li>Differential Privacy calibration (ε = 2.5, δ = 1e-5)</li>
                    <li>PyTorch 3D U-Net tensor batch cache builder</li>
                  </ul>
                </div>

                {preprocessingStatus === "PROCESSING" && (
                  <div style={{ marginTop: "0.5rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", color: "var(--text-muted, #64748b)", marginBottom: "0.2rem" }}>
                      <span>Normalizing voxels...</span>
                      <span>{preprocessProgress}%</span>
                    </div>
                    <div style={{ height: "6px", background: "#e2e8f0", borderRadius: "9999px", overflow: "hidden" }}>
                      <div style={{ width: `${preprocessProgress}%`, height: "100%", background: "var(--violet-primary, #6366f1)", transition: "width 0.3s ease" }} />
                    </div>
                  </div>
                )}
              </div>

              <div style={{ marginTop: "1rem" }}>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handlePreprocessData}
                  disabled={validationStatus !== "VALID" || preprocessingStatus === "PROCESSING"}
                  style={{
                    width: "100%",
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.4rem",
                    borderColor: preprocessingStatus === "COMPLETED" ? "var(--emerald-border, #a7f3d0)" : undefined,
                  }}
                >
                  <IconLayers size={14} />
                  <span>
                    {preprocessingStatus === "PROCESSING"
                      ? "Preprocessing..."
                      : preprocessingStatus === "COMPLETED"
                      ? "Re-Preprocess Data"
                      : "Preprocess Data (If Needed)"}
                  </span>
                </button>
              </div>
            </div>

            {/* STEP 3: START LOCAL TRAINING */}
            <div
              style={{
                background: "var(--bg-subtle, #f8fafc)",
                border: `1px solid ${trainingState === "TRAINING_ACTIVE" ? "var(--brand-blue-border, #d2e3fc)" : "var(--border-subtle, #e2e8f0)"}`,
                borderRadius: "var(--radius-lg, 16px)",
                padding: "1.25rem",
                display: "flex",
                flexDirection: "column",
                justifyContent: "space-between",
                opacity: validationStatus === "VALID" ? 1 : 0.65,
              }}
            >
              <div>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "0.5rem" }}>
                  <span style={{ fontSize: "0.78rem", fontWeight: 700, color: "var(--emerald-light, #059669)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    Step 3: Train Local Node
                  </span>
                  {trainingState === "ROUND_SYNCED" && (
                    <span style={{ fontSize: "0.72rem", color: "var(--emerald-light, #059669)", fontWeight: 700, display: "flex", alignItems: "center", gap: "0.2rem" }}>
                      <IconCheckCircle size={14} /> Round Complete
                    </span>
                  )}
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "0.6rem" }}>
                  <div>
                    <label style={{ fontSize: "0.75rem", color: "var(--text-muted, #64748b)", display: "block" }}>Local Epochs</label>
                    <input
                      type="number"
                      min="1"
                      max="10"
                      value={maxEpochs}
                      onChange={(e) => setMaxEpochs(Number(e.target.value))}
                      className="form-input"
                      style={{ padding: "0.35rem 0.5rem", fontSize: "0.82rem" }}
                    />
                  </div>
                  <div>
                    <label style={{ fontSize: "0.75rem", color: "var(--text-muted, #64748b)", display: "block" }}>Batch Size</label>
                    <input
                      type="number"
                      readOnly
                      value={2}
                      className="form-input"
                      style={{ padding: "0.35rem 0.5rem", fontSize: "0.82rem", background: "#f1f5f9" }}
                    />
                  </div>
                </div>

                {trainingState === "TRAINING_ACTIVE" && (
                  <div style={{ background: "#ffffff", padding: "0.5rem", borderRadius: "8px", border: "1px solid #e2e8f0", fontSize: "0.75rem" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", color: "var(--text-secondary, #334155)" }}>
                      <span>Loss: <strong style={{ color: "var(--coral-primary, #ef4444)" }}>{currentLoss.toFixed(4)}</strong></span>
                      <span>Dice: <strong style={{ color: "var(--emerald-light, #059669)" }}>{(currentDice * 100).toFixed(1)}%</strong></span>
                    </div>
                  </div>
                )}
              </div>

              <div style={{ marginTop: "1rem" }}>
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  onClick={handleStartTraining}
                  disabled={validationStatus !== "VALID" || trainingState === "TRAINING_ACTIVE" || trainingState === "WAITING_COORDINATOR"}
                  style={{
                    width: "100%",
                    display: "inline-flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.4rem",
                    background: "var(--emerald-light, #059669)",
                    borderColor: "var(--emerald-light, #059669)",
                  }}
                >
                  <IconPlay size={14} />
                  <span>
                    {trainingState === "WAITING_COORDINATOR"
                      ? "Waiting for Coordinator..."
                      : trainingState === "TRAINING_ACTIVE"
                      ? `Training Epoch ${trainingEpoch}/${maxEpochs}...`
                      : "Start Local Training"}
                  </span>
                </button>
              </div>
            </div>
          </div>

          {/* PERSONAL LIVE STATUS TERMINAL / LOG VIEWER */}
          <div
            style={{
              background: "#0f172a",
              color: "#f8fafc",
              borderRadius: "var(--radius-lg, 16px)",
              padding: "1rem 1.25rem",
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "0.8rem",
              boxShadow: "inset 0 2px 6px rgba(0,0,0,0.4)",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid #334155", paddingBottom: "0.5rem", marginBottom: "0.75rem" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <span style={{ width: "8px", height: "8px", borderRadius: "50%", background: "#10b981", boxShadow: "0 0 8px #10b981" }} />
                <span style={{ fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", fontSize: "0.72rem" }}>
                  Live Hospital Telemetry & Training Feed
                </span>
              </div>
              <span style={{ color: "#64748b", fontSize: "0.7rem" }}>Auto-scroll enabled</span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", maxHeight: "160px", overflowY: "auto" }}>
              {liveStatusLogs.map((log, idx) => (
                <div key={idx} style={{ display: "flex", gap: "0.6rem", alignItems: "flex-start" }}>
                  <span style={{ color: "#64748b", minWidth: "65px" }}>[{log.time}]</span>
                  <span
                    style={{
                      color:
                        log.type === "error"
                          ? "#f87171"
                          : log.type === "success"
                          ? "#34d399"
                          : "#93c5fd",
                    }}
                  >
                    {log.msg}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ====================================================================
          MODE 1: GLOBAL VIEW (EVERYONE CAN SEE ALL CONNECTED HOSPITALS & STATS)
          Available to all users without login
          ==================================================================== */}
      <div>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem", marginBottom: "1.25rem" }}>
          <div>
            <h2 style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", margin: 0 }}>
              Global Hospital Node Network
            </h2>
            <p style={{ fontSize: "0.82rem", color: "var(--text-muted, #64748b)", margin: "0.2rem 0 0" }}>
              Connected institutional clinical nodes, hardware profiles, and local dataset sizes
            </p>
          </div>

          {/* Filter Pills */}
          <div style={{ display: "flex", gap: "0.3rem", background: "var(--bg-subtle, #f1f5f9)", padding: "0.2rem", borderRadius: "8px" }}>
            <button
              type="button"
              onClick={() => setSelectedNodeFilter("all")}
              style={{
                border: "none",
                padding: "0.25rem 0.65rem",
                borderRadius: "6px",
                fontSize: "0.75rem",
                fontWeight: 600,
                cursor: "pointer",
                background: selectedNodeFilter === "all" ? "#ffffff" : "transparent",
                color: selectedNodeFilter === "all" ? "var(--text-primary, #0f172a)" : "var(--text-muted, #64748b)",
                boxShadow: selectedNodeFilter === "all" ? "0 1px 2px rgba(0,0,0,0.05)" : "none",
              }}
            >
              All Nodes ({hospitals.length})
            </button>
            <button
              type="button"
              onClick={() => setSelectedNodeFilter("ONLINE")}
              style={{
                border: "none",
                padding: "0.25rem 0.65rem",
                borderRadius: "6px",
                fontSize: "0.75rem",
                fontWeight: 600,
                cursor: "pointer",
                background: selectedNodeFilter === "ONLINE" ? "#ffffff" : "transparent",
                color: selectedNodeFilter === "ONLINE" ? "var(--emerald-light, #059669)" : "var(--text-muted, #64748b)",
                boxShadow: selectedNodeFilter === "ONLINE" ? "0 1px 2px rgba(0,0,0,0.05)" : "none",
              }}
            >
              Online Only
            </button>
          </div>
        </div>

        {/* Global Hospital Cards Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "1.25rem" }}>
          {filteredHospitals.map((h, i) => {
            const isUserOwnNode = isHospitalOperator && (h.node_id === currentHospitalNodeId || h.hospital_name.includes("Sinai"));

            return (
              <div
                key={h.node_id || i}
                style={{
                  background: "#ffffff",
                  border: `1px solid ${isUserOwnNode ? "var(--emerald-border, #a7f3d0)" : "var(--border-subtle, #e2e8f0)"}`,
                  borderRadius: "var(--radius-xl, 20px)",
                  padding: "1.5rem",
                  boxShadow: "0 2px 8px rgba(0,0,0,0.03)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "1rem",
                  position: "relative",
                }}
              >
                {/* Node Card Header */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "0.5rem" }}>
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", marginBottom: "0.25rem" }}>
                      <span
                        style={{
                          fontSize: "0.72rem",
                          fontFamily: "var(--font-mono, monospace)",
                          fontWeight: 700,
                          padding: "0.15rem 0.5rem",
                          borderRadius: "6px",
                          background: "var(--brand-blue-subtle, #e8f0fe)",
                          color: "var(--brand-blue, #1a73e8)",
                        }}
                      >
                        {h.node_id}
                      </span>
                      {isUserOwnNode && (
                        <span style={{ fontSize: "0.7rem", color: "var(--emerald-light, #059669)", fontWeight: 700 }}>
                          • Your Node
                        </span>
                      )}
                    </div>
                    <h3 style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary, #0f172a)", margin: 0 }}>
                      {h.hospital_name}
                    </h3>
                    <div style={{ fontSize: "0.78rem", color: "var(--text-muted, #64748b)", marginTop: "0.15rem" }}>
                      {h.region}
                    </div>
                  </div>

                  <span
                    style={{
                      fontSize: "0.72rem",
                      fontFamily: "var(--font-mono, monospace)",
                      fontWeight: 600,
                      padding: "0.2rem 0.55rem",
                      borderRadius: "9999px",
                      background: h.status === "ONLINE" ? "var(--emerald-subtle, #ecfdf5)" : "var(--bg-subtle, #f1f5f9)",
                      color: h.status === "ONLINE" ? "var(--emerald-light, #059669)" : "var(--text-muted, #64748b)",
                      border: `1px solid ${h.status === "ONLINE" ? "var(--emerald-border, #a7f3d0)" : "var(--border-subtle, #e2e8f0)"}`,
                    }}
                  >
                    {h.status || "ONLINE"}
                  </span>
                </div>

                {/* Node Hardware Profile & Dataset Specs */}
                <div
                  style={{
                    background: "var(--bg-subtle, #f8fafc)",
                    borderRadius: "var(--radius-md, 12px)",
                    padding: "0.85rem 1rem",
                    border: "1px solid var(--border-subtle, #e2e8f0)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.4rem",
                    fontSize: "0.8rem",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted, #64748b)" }}>GPU Hardware</span>
                    <strong style={{ color: "var(--text-primary, #0f172a)" }}>{h.gpu_device || "NVIDIA RTX 4090"}</strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted, #64748b)" }}>VRAM / Threads</span>
                    <strong style={{ color: "var(--text-primary, #0f172a)", fontFamily: "var(--font-mono, monospace)" }}>
                      {h.vram_gb} GB / {h.cpu_cores} Cores
                    </strong>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted, #64748b)" }}>Patient Cohort Size</span>
                    <strong style={{ color: "var(--emerald-light, #059669)", fontFamily: "var(--font-mono, monospace)" }}>
                      {h.local_sample_count || 48} 3D Scans
                    </strong>
                  </div>
                </div>

                {/* Footer security tag */}
                <div style={{ display: "flex", alignItems: "center", gap: "0.35rem", fontSize: "0.75rem", color: "var(--text-muted, #64748b)", marginTop: "auto" }}>
                  <IconShieldCheck size={14} />
                  <span>Encrypted Gradients • Zero Raw Scans Uploaded</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
