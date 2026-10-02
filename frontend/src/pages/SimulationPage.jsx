import React, { useState, useEffect } from "react";
import { Card } from "../components/common/Card";
import { Badge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { IconLayers, IconPlay, IconActivity, IconCheckCircle, IconServer } from "../components/common/Icons";
import { api } from "../services/api";

export function SimulationPage() {
  const [numClients, setNumClients] = useState(3);
  const [numRounds, setNumRounds] = useState(3);
  const [partition, setPartition] = useState("quantity_skew");
  const [simulating, setSimulating] = useState(false);
  const [simResult, setSimResult] = useState(null);
  const [simError, setSimError] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [loadingJobs, setLoadingJobs] = useState(true);

  const fetchJobs = async () => {
    try {
      const data = await api.getJobs();
      if (data && Array.isArray(data.jobs)) {
        // filter simulation jobs or sort newest first
        const simJobs = data.jobs.filter((j) => j.type === "SIMULATION").reverse();
        setJobs(simJobs);
      }
    } catch (err) {
      console.warn("Failed to fetch jobs:", err);
    } finally {
      setLoadingJobs(false);
    }
  };

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleLaunchSimulation = async () => {
    setSimulating(true);
    setSimResult(null);
    setSimError(null);
    try {
      const res = await api.triggerSimulation({
        num_clients: numClients,
        num_rounds: numRounds,
        partition_type: partition,
      });
      setSimResult(res);
      // Immediately refresh jobs
      await fetchJobs();
    } catch (err) {
      setSimError(err.message || "Failed to trigger simulation job.");
    } finally {
      setSimulating(false);
    }
  };

  const latestJob = jobs.length > 0 ? jobs[0] : null;

  return (
    <div className="flex-col gap-6">
      <div>
        <Badge variant="violet" pulse style={{ marginBottom: "0.5rem" }}>
          Flower Simulation Engine
        </Badge>
        <h2>Multi-Hospital Non-IID Simulation Testbed</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.95rem" }}>
          Simulate multi-client federated training locally with heterogeneous medical data distributions.
        </p>
      </div>

      {simResult && (
        <div
          style={{
            padding: "1rem 1.25rem",
            background: "rgba(16, 185, 129, 0.08)",
            border: "1px solid rgba(16, 185, 129, 0.25)",
            borderRadius: "var(--radius-md)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
            <IconCheckCircle size={20} style={{ color: "var(--emerald-primary)", flexShrink: 0 }} />
            <div>
              <div style={{ fontWeight: 600, color: "var(--emerald-primary)" }}>
                Simulation Job Dispatched Successfully
              </div>
              <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginTop: "2px" }}>
                Job ID: <code>{simResult.job_id}</code> &bull; Clients: {simResult.num_clients} &bull; Rounds: {simResult.num_rounds} &bull; Strategy: {simResult.partition_type}
              </div>
            </div>
          </div>
          <Badge variant="emerald">QUEUED</Badge>
        </div>
      )}

      {simError && (
        <div
          style={{
            padding: "1rem 1.25rem",
            background: "rgba(239, 68, 68, 0.08)",
            border: "1px solid rgba(239, 68, 68, 0.25)",
            borderRadius: "var(--radius-md)",
            display: "flex",
            alignItems: "center",
            gap: "0.75rem",
          }}
        >
          <div style={{ fontWeight: 600, color: "var(--rose-primary)" }}>
            Simulation Error: {simError}
          </div>
        </div>
      )}

      <div className="grid grid-sidebar">
        {/* Controls */}
        <Card title="Simulation Parameters" icon={<IconLayers size={18} />}>
          <div className="flex-col gap-3">
            <div className="form-group">
              <label className="form-label">Hospital Clients: {numClients}</label>
              <input
                type="range"
                min="2"
                max="8"
                value={numClients}
                onChange={(e) => setNumClients(Number(e.target.value))}
                style={{ width: "100%", accentColor: "var(--cyan-primary)" }}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Federated Rounds: {numRounds}</label>
              <input
                type="range"
                min="1"
                max="10"
                value={numRounds}
                onChange={(e) => setNumRounds(Number(e.target.value))}
                style={{ width: "100%", accentColor: "var(--cyan-primary)" }}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Data Partition Distribution</label>
              <select
                className="form-select"
                value={partition}
                onChange={(e) => setPartition(e.target.value)}
              >
                <option value="quantity_skew">Quantity Skew (Non-IID)</option>
                <option value="dirichlet">Dirichlet Distribution (α = 0.5)</option>
                <option value="iid">Uniform IID</option>
              </select>
            </div>

            <Button
              variant="glow"
              onClick={handleLaunchSimulation}
              loading={simulating}
              icon={<IconPlay size={16} />}
              style={{ marginTop: "0.5rem" }}
            >
              {simulating ? "Executing Simulation..." : "Launch Simulation Job"}
            </Button>
          </div>
        </Card>

        {/* Live Output & Visualizer preview */}
        <Card title="Simulation Job Monitor" icon={<IconActivity size={18} />}>
          {jobs.length === 0 ? (
            <div
              style={{
                background: "var(--bg-card)",
                borderRadius: "var(--radius-md)",
                padding: "2.5rem 1.5rem",
                textAlign: "center",
                border: "1px dashed var(--border-subtle)",
              }}
            >
              <IconServer size={40} style={{ color: "var(--text-muted)", margin: "0 auto 1rem" }} />
              <h4>No Active Simulation Jobs</h4>
              <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", maxWidth: "460px", margin: "0.5rem auto" }}>
                Select parameters on the left and launch a simulation to execute in-process multi-hospital federated rounds. Live convergence telemetry will render here.
              </p>
            </div>
          ) : (
            <div className="flex-col gap-4">
              {jobs.map((job) => {
                const isRunning = job.status === "RUNNING";
                const isCompleted = job.status === "COMPLETED";
                const isFailed = job.status === "FAILED";
                const badgeVariant = isCompleted ? "emerald" : isRunning ? "cyan" : isFailed ? "rose" : "amber";

                return (
                  <div
                    key={job.job_id}
                    style={{
                      background: "var(--bg-card)",
                      border: "1px solid var(--border-subtle)",
                      borderRadius: "var(--radius-md)",
                      padding: "1.25rem",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
                      <div>
                        <div style={{ fontWeight: 600, fontSize: "0.95rem" }}>
                          Job <code>{job.job_id}</code>
                        </div>
                        <div style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "2px" }}>
                          Clients: {job.num_clients} &bull; Target Rounds: {job.num_rounds} &bull; Partition: {job.partition_type}
                        </div>
                      </div>
                      <Badge variant={badgeVariant} pulse={isRunning}>
                        {job.status}
                      </Badge>
                    </div>

                    {isFailed && (
                      <div
                        style={{
                          fontSize: "0.82rem",
                          color: "var(--rose-primary)",
                          background: "rgba(239, 68, 68, 0.05)",
                          padding: "0.5rem 0.75rem",
                          borderRadius: "var(--radius-sm)",
                          marginTop: "0.5rem",
                        }}
                      >
                        {job.error || "Simulation encountered an execution error."}
                      </div>
                    )}

                    {isCompleted && job.summary && (
                      <div style={{ marginTop: "0.75rem", paddingTop: "0.75rem", borderTop: "1px solid var(--border-subtle)" }}>
                        <div style={{ fontSize: "0.82rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
                          Final Convergence Metrics
                        </div>
                        <div className="grid grid-3" style={{ gap: "0.75rem" }}>
                          <div style={{ background: "var(--bg-surface)", padding: "0.5rem 0.75rem", borderRadius: "var(--radius-sm)" }}>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Best Dice</div>
                            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--emerald-primary)" }}>
                              {job.summary.best_dice ? (job.summary.best_dice * 100).toFixed(2) + "%" : "N/A"}
                            </div>
                          </div>
                          <div style={{ background: "var(--bg-surface)", padding: "0.5rem 0.75rem", borderRadius: "var(--radius-sm)" }}>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Best Round</div>
                            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--cyan-primary)" }}>
                              {job.summary.best_round ?? "N/A"}
                            </div>
                          </div>
                          <div style={{ background: "var(--bg-surface)", padding: "0.5rem 0.75rem", borderRadius: "var(--radius-sm)" }}>
                            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Val Loss</div>
                            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--violet-primary)" }}>
                              {job.summary.final_metrics?.val_loss !== undefined
                                ? Number(job.summary.final_metrics.val_loss).toFixed(4)
                                : "N/A"}
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}
