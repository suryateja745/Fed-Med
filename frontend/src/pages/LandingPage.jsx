import React from "react";
import { Badge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { Card } from "../components/common/Card";
import { IconBrain, IconShieldCheck, IconLock, IconActivity, IconHospital } from "../components/common/Icons";

export function LandingPage({ onNavigate }) {
  return (
    <div className="flex-col gap-8">
      {/* Hero Showcase Preview */}
      <div style={{ textAlign: "center", padding: "3rem 1rem 2rem", position: "relative" }}>
        <div style={{ display: "inline-flex", gap: "0.5rem", marginBottom: "1rem" }}>
          <Badge variant="cyan" pulse>
            Flower v1.8+ PPML Framework
          </Badge>
          <Badge variant="emerald" pulse>
            Zero-Leakage Architecture
          </Badge>
        </div>

        <h1 style={{ maxWidth: "900px", margin: "0 auto", fontSize: "3.2rem" }}>
          Decentralized 3D Brain Tumor MRI Federated AI
        </h1>

        <p style={{ maxWidth: "720px", margin: "1.25rem auto", fontSize: "1.15rem", color: "var(--text-secondary)" }}>
          Federated segmentation of multi-modal brain tumor MRI scans (FLAIR, T1, T1ce, T2) across
          hospitals with authenticated AES-256-GCM model weight encryption and verifiable Dice metrics.
        </p>

        <div style={{ display: "flex", gap: "1rem", justifyContent: "center", marginTop: "2rem" }}>
          <Button variant="glow" size="lg" onClick={() => onNavigate("coordinator")} icon={<IconActivity size={18} />}>
            Coordinator Command Center
          </Button>
          <Button variant="outline" size="lg" onClick={() => onNavigate("hospital")} icon={<IconHospital size={18} />}>
            Hospital Node Portal
          </Button>
        </div>
      </div>

      {/* KPI Ticker Bar */}
      <div className="grid grid-4">
        <div className="stat-card">
          <span className="stat-label">Federated MRI Scans</span>
          <span className="stat-value" style={{ color: "var(--cyan-light)" }}>1,480+</span>
          <span className="stat-desc">Across 3 hospital consortium nodes</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Global Mean Dice</span>
          <span className="stat-value" style={{ color: "var(--emerald-light)" }}>88.5%</span>
          <span className="stat-desc">TC: 87.1% • WT: 90.8% • ET: 86.3%</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Raw Data Leakage</span>
          <span className="stat-value" style={{ color: "var(--violet-light)" }}>0.00%</span>
          <span className="stat-desc">Strictly local parameter updates only</span>
        </div>

        <div className="stat-card">
          <span className="stat-label">Encryption Cipher</span>
          <span className="stat-value" style={{ fontSize: "1.6rem", color: "var(--text-primary)", paddingTop: "0.4rem" }}>AES-256-GCM</span>
          <span className="stat-desc">PBKDF2-HMAC-SHA256 authenticated</span>
        </div>
      </div>

      {/* Interactive 3D MRI Preview teaser */}
      <Card
        title="3D MRI Volumetric Multi-Modal Segmentation"
        subtitle="Coming in Commit 22: Interactive multi-sequence layer viewer (FLAIR, T1, T1ce, T2) and tumor masks (WT, TC, ET)"
        icon={<IconBrain size={20} />}
      >
        <div
          style={{
            background: "rgba(7, 11, 20, 0.6)",
            borderRadius: "var(--radius-lg)",
            border: "1px dashed var(--border-cyan)",
            padding: "3rem 2rem",
            textAlign: "center",
          }}
        >
          <IconBrain size={48} style={{ color: "var(--cyan-primary)", margin: "0 auto 1rem" }} />
          <h3 style={{ marginBottom: "0.5rem" }}>Publication-Ready 3D MRI Tumor Visualizer</h3>
          <p style={{ maxWidth: "600px", margin: "0 auto", fontSize: "0.95rem" }}>
            The full interactive WebGL brain tumor MRI slice viewer with live sequence toggling
            and tumor sub-region opacity sliders is scheduled for <strong>Commit 22</strong>.
          </p>
          <div style={{ marginTop: "1.5rem" }}>
            <Button variant="primary" size="sm" onClick={() => onNavigate("sandbox")}>
              Explore UI Primitives & Tokens Sandbox
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
