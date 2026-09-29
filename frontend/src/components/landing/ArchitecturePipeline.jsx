import React, { useState } from "react";
import { Badge } from "../common/Badge";
import {
  IconHospital,
  IconCpu,
  IconLock,
  IconActivity,
  IconShieldCheck,
  IconKey,
} from "../common/Icons";

export function ArchitecturePipeline() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      step: 1,
      title: "1. Private Local Hospital Data",
      subtitle: "Zero Raw Data Upload",
      icon: <IconHospital size={22} />,
      color: "var(--cyan-primary)",
      badge: "Local Storage",
      desc: "Patients' volumetric 3D MRI scans (FLAIR, T1, T1ce, T2) remain isolated within each hospital's private filesystem (e.g. ./data/hospital_a). No DICOM or NIfTI images are ever transmitted over the network.",
      metric: "0.00% Raw Data Leakage",
    },
    {
      step: 2,
      title: "2. Client-Side 3D U-Net Training",
      subtitle: "PyTorch & MONAI Execution",
      icon: <IconCpu size={22} />,
      color: "var(--violet-primary)",
      badge: "GPU Accelerated",
      desc: "Each hospital client trains a local MONAI 3D U-Net model on its private GPU hardware using AdamW optimizer and DiceCELoss, calculating gradient updates for BraTS tumor sub-regions.",
      metric: "16-32 Local Batches/Epoch",
    },
    {
      step: 3,
      title: "3. Parameter Masking & AES-256-GCM",
      subtitle: "Zero-Trust Encryption",
      icon: <IconLock size={22} />,
      color: "var(--emerald-primary)",
      badge: "Authenticated Cipher",
      desc: "Model weight deltas (ΔW = W_local - W_global) are compressed and encrypted at rest and in-transit using 256-bit AES-GCM and digitally signed with HMAC-SHA256 to reject any bit-flip tampering.",
      metric: "100,000 Rounds PBKDF2",
    },
    {
      step: 4,
      title: "4. Flower Server FedMedStrategy",
      subtitle: "Quality-Weighted Aggregation",
      icon: <IconActivity size={22} />,
      color: "var(--cyan-primary)",
      badge: "Ciphertext FedAvg",
      desc: "The central coordinator verifies hospital signatures and computes a weighted average of received parameter updates, prioritizing nodes with higher validation Dice scores without unencrypting individual gradients.",
      metric: "Weighted Multi-Region FedAvg",
    },
    {
      step: 5,
      title: "5. Verified Global Model Dispatch",
      subtitle: "Automated Publication",
      icon: <IconShieldCheck size={22} />,
      color: "var(--emerald-primary)",
      badge: "Verified Checkpoint",
      desc: "The updated global 3D U-Net model checkpoint (global_model_round_X.pth.enc) is published to the registry. The auto-dispatch trigger automatically streams the new weights to all connecting hospitals.",
      metric: "88.5% Global Mean Dice",
    },
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "2rem" }}>
      <div style={{ textAlign: "center" }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.4rem",
            padding: "0.25rem 0.8rem",
            borderRadius: "var(--radius-full)",
            background: "rgba(0, 240, 255, 0.12)",
            border: "1px solid rgba(0, 240, 255, 0.3)",
            color: "var(--cyan-light)",
            fontSize: "0.72rem",
            fontFamily: "var(--font-mono)",
            marginBottom: "0.6rem",
          }}
        >
          Zero-Trust Privacy Flow
        </div>
        <h2 style={{ fontSize: "2.2rem" }}>Decentralized Architecture Pipeline</h2>
        <p style={{ color: "var(--text-muted)", fontSize: "0.95rem", maxWidth: "680px", margin: "0.4rem auto" }}>
          How FedMed securely trains 3D brain tumor segmentation models across independent healthcare
          systems with zero raw patient data sharing.
        </p>
      </div>

      {/* Interactive Step Selector Tabs */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: "0.75rem",
          background: "rgba(10, 14, 23, 0.7)",
          backdropFilter: "blur(16px)",
          padding: "0.6rem",
          borderRadius: "var(--radius-lg)",
          border: "1px solid var(--border-subtle)",
        }}
      >
        {steps.map((s, idx) => (
          <button
            key={idx}
            onClick={() => setActiveStep(idx)}
            style={{
              background: activeStep === idx ? "rgba(0, 240, 255, 0.12)" : "transparent",
              border: activeStep === idx ? "1px solid var(--cyan-primary)" : "1px solid transparent",
              borderRadius: "var(--radius-md)",
              padding: "0.85rem 0.6rem",
              color: activeStep === idx ? "var(--cyan-light)" : "var(--text-muted)",
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: "0.4rem",
              transition: "all var(--transition-fast)",
            }}
          >
            <span style={{ color: s.color }}>{s.icon}</span>
            <span style={{ fontSize: "0.8rem", fontWeight: 600, textAlign: "center", lineHeight: 1.2 }}>
              Step {s.step}
            </span>
          </button>
        ))}
      </div>

      {/* Active Step Detailed Card */}
      <div
        style={{
          background: "rgba(15, 23, 42, 0.75)",
          backdropFilter: "blur(20px)",
          border: "1px solid var(--border-cyan)",
          borderRadius: "var(--radius-xl)",
          padding: "2.25rem",
          boxShadow: "0 12px 40px rgba(0, 0, 0, 0.5), 0 0 25px rgba(0, 240, 255, 0.15)",
          display: "grid",
          gridTemplateColumns: "72px 1fr 240px",
          gap: "2rem",
          alignItems: "center",
        }}
      >
        <div
          style={{
            width: "72px",
            height: "72px",
            borderRadius: "var(--radius-lg)",
            background: "rgba(10, 14, 23, 0.8)",
            border: "1px solid var(--border-cyan)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: steps[activeStep].color,
            boxShadow: "0 0 20px rgba(0, 240, 255, 0.2)",
          }}
        >
          {steps[activeStep].icon}
        </div>

        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "0.4rem" }}>
            <Badge variant="cyan">{steps[activeStep].badge}</Badge>
            <span style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
              {steps[activeStep].subtitle}
            </span>
          </div>
          <h3 style={{ fontSize: "1.45rem", marginBottom: "0.5rem" }}>
            {steps[activeStep].title}
          </h3>
          <p style={{ color: "var(--text-secondary)", fontSize: "0.95rem", lineHeight: 1.6 }}>
            {steps[activeStep].desc}
          </p>
        </div>

        <div
          style={{
            background: "rgba(0, 0, 0, 0.4)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-md)",
            padding: "1.25rem",
            textAlign: "center",
          }}
        >
          <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.06em", fontFamily: "var(--font-mono)" }}>
            Security &amp; Metric
          </div>
          <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--cyan-light)", marginTop: "0.4rem", fontFamily: "var(--font-mono)" }}>
            {steps[activeStep].metric}
          </div>
        </div>
      </div>
    </div>
  );
}
