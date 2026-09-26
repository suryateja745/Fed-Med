import React, { useState } from "react";
import { Card } from "../components/common/Card";
import { Badge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { IconLayers, IconPlay, IconActivity } from "../components/common/Icons";

export function SimulationPage() {
  const [numClients, setNumClients] = useState(3);
  const [numRounds, setNumRounds] = useState(3);
  const [partition, setPartition] = useState("quantity_skew");
  const [simulating, setSimulating] = useState(false);

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
                <option value="iid">Uniform Uniform IID</option>
              </select>
            </div>

            <Button
              variant="glow"
              onClick={() => {
                setSimulating(true);
                setTimeout(() => setSimulating(false), 2500);
              }}
              loading={simulating}
              icon={<IconPlay size={16} />}
              style={{ marginTop: "0.5rem" }}
            >
              {simulating ? "Executing Simulation..." : "Launch Simulation Job"}
            </Button>
          </div>
        </Card>

        {/* Live Output & Visualizer preview */}
        <Card title="Live Convergence Curves & Logs" icon={<IconActivity size={18} />}>
          <div
            style={{
              background: "rgba(0, 0, 0, 0.4)",
              borderRadius: "var(--radius-md)",
              padding: "2rem",
              textAlign: "center",
              border: "1px dashed var(--border-subtle)",
            }}
          >
            <IconActivity size={40} style={{ color: "var(--violet-primary)", margin: "0 auto 1rem" }} />
            <h4>Interactive Convergence Plots</h4>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", maxWidth: "500px", margin: "0.5rem auto" }}>
              Side-by-side Multi-Region Dice score curves and loss trajectories across simulated
              hospitals will stream live here (full interactive charting engine arriving in Commits 25 & 30).
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
}
