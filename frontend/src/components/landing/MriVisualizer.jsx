import React, { useState, useEffect, useRef } from "react";
import { Button } from "../common/Button";
import { Badge } from "../common/Badge";
import {
  IconBrain,
  IconPlay,
  IconPause,
  IconLayers,
  IconEye,
  IconActivity,
  IconRefreshCw,
} from "../common/Icons";

export function MriVisualizer() {
  // MRI Modality sequence: 'FLAIR' | 'T1' | 'T1ce' | 'T2'
  const [modality, setModality] = useState("FLAIR");

  // Segmentation masks visibility & opacities
  const [showWT, setShowWT] = useState(true); // Whole Tumor (Edema + Core + Enhancing)
  const [showTC, setShowTC] = useState(true); // Tumor Core (Necrotic + Enhancing)
  const [showET, setShowET] = useState(true); // Enhancing Tumor (Active boundary)

  const [opacityWT, setOpacityWT] = useState(0.55);
  const [opacityTC, setOpacityTC] = useState(0.65);
  const [opacityET, setOpacityET] = useState(0.75);

  // Volumetric slice depth (1 to 155 slices)
  const [sliceDepth, setSliceDepth] = useState(78);
  const [isPlaying, setIsPlaying] = useState(false);

  // Orientation view
  const [plane, setPlane] = useState("AXIAL"); // 'AXIAL' | 'CORONAL' | 'SAGITTAL'

  // Cursor inspection
  const [mousePos, setMousePos] = useState({ x: 0, y: 0, active: false });
  const [inspectData, setInspectData] = useState({ x: 128, y: 128, val: 0, label: "Brain Parenchyma" });

  const canvasRef = useRef(null);

  // Cine playback loop
  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setSliceDepth((prev) => (prev >= 140 ? 40 : prev + 1));
      }, 70);
    }
    return () => clearInterval(timer);
  }, [isPlaying]);

  // Canvas drawing effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    const width = (canvas.width = 460);
    const height = (canvas.height = 460);
    const cx = width / 2;
    const cy = height / 2;

    // Clear background (medical dark viewport)
    ctx.fillStyle = "#050811";
    ctx.fillRect(0, 0, width, height);

    // Subtle grid calibration lines
    ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
    ctx.lineWidth = 1;
    for (let i = 40; i < width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(width, i);
      ctx.stroke();
    }

    // Anatomical size variations based on slice depth (Z-axis progression)
    // Slices 1-30: top of skull; 40-110: full cerebral hemispheres; 120-155: brainstem/base
    const zNorm = (sliceDepth - 1) / 154; // 0 to 1
    const brainScale = Math.sin(Math.min(Math.max(zNorm, 0.1), 0.95) * Math.PI);
    const rx = 150 * (0.65 + 0.35 * brainScale);
    const ry = 175 * (0.65 + 0.35 * brainScale);

    // 1. Draw Outer Cranium & Scalp Outline
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx + 14, ry + 14, 0, 0, Math.PI * 2);
    ctx.strokeStyle = modality === "T1" ? "rgba(240, 240, 245, 0.4)" : "rgba(180, 190, 210, 0.25)";
    ctx.lineWidth = 4;
    ctx.stroke();

    // 2. Draw Brain Parenchyma (Cerebral Cortex & White Matter)
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);

    let parenchymaColor;
    if (modality === "FLAIR") parenchymaColor = "#1a2233";
    else if (modality === "T1") parenchymaColor = "#2c384e";
    else if (modality === "T1ce") parenchymaColor = "#222d42";
    else parenchymaColor = "#182030"; // T2

    ctx.fillStyle = parenchymaColor;
    ctx.fill();

    // Sulcal patterns and gyri texture
    ctx.beginPath();
    ctx.ellipse(cx, cy, rx - 16, ry - 16, 0, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
    ctx.lineWidth = 8;
    ctx.stroke();

    // 3. Longitudinal Cerebral Fissure (Hemisphere separator)
    ctx.beginPath();
    ctx.moveTo(cx, cy - ry + 10);
    ctx.lineTo(cx, cy + ry - 10);
    ctx.strokeStyle = "rgba(5, 8, 17, 0.75)";
    ctx.lineWidth = 3;
    ctx.stroke();

    // 4. Lateral Ventricles (CSF Chambers)
    // Slices 50-100 have prominent ventricles
    const ventScale = Math.max(0, Math.sin((zNorm - 0.2) * 3.5));
    if (ventScale > 0.05) {
      const vWidth = 14 * ventScale;
      const vHeight = 44 * ventScale;
      const ventColor = modality === "T2" ? "#e0f2fe" : modality === "FLAIR" ? "#060913" : "#0d1320";

      // Left ventricle
      ctx.beginPath();
      ctx.ellipse(cx - 22, cy - 8, vWidth, vHeight, -0.15, 0, Math.PI * 2);
      ctx.fillStyle = ventColor;
      ctx.fill();

      // Right ventricle
      ctx.beginPath();
      ctx.ellipse(cx + 22, cy - 8, vWidth, vHeight, 0.15, 0, Math.PI * 2);
      ctx.fillStyle = ventColor;
      ctx.fill();
    }

    // 5. Glioblastoma Pathology / Tumor Sub-Regions
    // Tumor appears predominantly between slice depth 50 and 110
    const tumorFactor = Math.sin(Math.min(Math.max((sliceDepth - 45) / 65, 0), 1) * Math.PI);

    if (tumorFactor > 0.05) {
      // Right frontal-parietal lobe location
      const tx = cx + 55 * brainScale;
      const ty = cy - 35 * brainScale;

      const rWT = 52 * tumorFactor;
      const rTC = 32 * tumorFactor;
      const rET = 22 * tumorFactor;

      // Layer A: Whole Tumor (WT) - Edema Zone
      if (showWT) {
        ctx.beginPath();
        ctx.ellipse(tx, ty, rWT, rWT * 0.88, 0.25, 0, Math.PI * 2);
        if (modality === "FLAIR") {
          // FLAIR prominently illuminates edema
          ctx.fillStyle = `rgba(16, 185, 129, ${opacityWT})`; // Bio Emerald
        } else if (modality === "T2") {
          ctx.fillStyle = `rgba(16, 185, 129, ${opacityWT * 0.85})`;
        } else {
          ctx.fillStyle = `rgba(16, 185, 129, ${opacityWT * 0.5})`;
        }
        ctx.fill();
        ctx.strokeStyle = "rgba(16, 185, 129, 0.9)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Layer B: Tumor Core (TC) - Necrotic Core + Hypercellular mass
      if (showTC) {
        ctx.beginPath();
        ctx.ellipse(tx + 4, ty + 2, rTC, rTC * 0.9, -0.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(245, 158, 11, ${opacityTC})`; // Amber
        ctx.fill();
        ctx.strokeStyle = "rgba(245, 158, 11, 0.95)";
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      // Layer C: Enhancing Tumor (ET) - Contrast-Enhancing Vascularized Rim
      if (showET) {
        ctx.beginPath();
        ctx.ellipse(tx + 6, ty + 1, rET, rET * 0.85, 0.1, 0, Math.PI * 2);

        if (modality === "T1ce") {
          // T1ce shows extreme hyper-intensity in enhancing rim
          ctx.fillStyle = `rgba(239, 68, 68, ${opacityET})`; // Crimson
          ctx.shadowColor = "#ef4444";
          ctx.shadowBlur = 12;
        } else {
          ctx.fillStyle = `rgba(239, 68, 68, ${opacityET * 0.6})`;
          ctx.shadowBlur = 0;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.strokeStyle = "#f87171";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Inner necrotic cavity
        ctx.beginPath();
        ctx.arc(tx + 6, ty + 1, rET * 0.42, 0, Math.PI * 2);
        ctx.fillStyle = modality === "T1ce" ? "rgba(10, 15, 28, 0.85)" : "rgba(20, 28, 45, 0.7)";
        ctx.fill();
      }
    }

    // 6. Interactive Cursor Crosshair
    if (mousePos.active) {
      ctx.strokeStyle = "rgba(6, 182, 212, 0.6)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);

      // Vertical line
      ctx.beginPath();
      ctx.moveTo(mousePos.x, 0);
      ctx.lineTo(mousePos.x, height);
      ctx.stroke();

      // Horizontal line
      ctx.beginPath();
      ctx.moveTo(0, mousePos.y);
      ctx.lineTo(width, mousePos.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // Coordinate marker
      ctx.beginPath();
      ctx.arc(mousePos.x, mousePos.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "#22d3ee";
      ctx.fill();
    }

    // 7. Medical HUD Calibration Overlays
    ctx.font = "11px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(6, 182, 212, 0.85)";
    ctx.fillText(`SEQ: ${modality}`, 14, 24);
    ctx.fillText(`SLICE: ${sliceDepth}/155 [${plane}]`, 14, 40);
    ctx.fillText(`FOV: 240mm • THK: 1.0mm`, 14, 56);

    ctx.fillStyle = "rgba(255, 255, 255, 0.45)";
    ctx.fillText(`VOX: 256×256`, width - 100, 24);
    ctx.fillText(`W: 420 L: 110`, width - 100, 40);

    // Anatomical orientation labels
    ctx.font = "bold 13px 'Inter', sans-serif";
    ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
    ctx.fillText("A", cx - 4, 18);
    ctx.fillText("P", cx - 4, height - 10);
    ctx.fillText("R", 14, cy + 5);
    ctx.fillText("L", width - 22, cy + 5);
  }, [
    modality,
    showWT,
    showTC,
    showET,
    opacityWT,
    opacityTC,
    opacityET,
    sliceDepth,
    plane,
    mousePos,
  ]);

  // Handle canvas mousemove for voxel inspection
  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(e.clientX - rect.left);
    const y = Math.floor(e.clientY - rect.top);

    setMousePos({ x, y, active: true });

    // Determine voxel label based on distance from tumor center
    const cx = 230;
    const cy = 230;
    const tx = cx + 50;
    const ty = cy - 30;
    const distTumor = Math.sqrt((x - tx) ** 2 + (y - ty) ** 2);
    const distCenter = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);

    let label = "Background Air";
    let val = 12;

    if (distCenter < 165) {
      label = "Brain Parenchyma";
      val = 620;

      if (distTumor < 18) {
        label = "Enhancing Tumor (ET) [BraTS Class 3]";
        val = 1420;
      } else if (distTumor < 32) {
        label = "Tumor Core (TC) [BraTS Class 2]";
        val = 980;
      } else if (distTumor < 52) {
        label = "Peritumoral Vasogenic Edema (WT) [BraTS Class 1]";
        val = 840;
      }
    }

    setInspectData({ x, y, val, label });
  };

  const handleMouseLeave = () => {
    setMousePos((prev) => ({ ...prev, active: false }));
  };

  return (
    <div className="diagnostic-scanner-card" style={{ maxWidth: "600px", margin: "0 auto" }}>
      {/* Reticle Hairline HUD Corners */}
      <div className="reticle-corner reticle-tl" />
      <div className="reticle-corner reticle-tr" />
      <div className="reticle-corner reticle-bl" />
      <div className="reticle-corner reticle-br" />

      {/* 1. Scanner Top Bar & Dynamic Vector Coordinates */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          borderBottom: "1px solid var(--border-subtle)",
          paddingBottom: "0.6rem",
          marginBottom: "0.75rem",
          fontSize: "0.72rem",
          fontFamily: "var(--font-mono)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <span
            style={{
              width: "7px",
              height: "7px",
              borderRadius: "50%",
              backgroundColor: "var(--crimson-primary)",
              display: "inline-block",
              boxShadow: "0 0 8px var(--crimson-primary)",
              animation: "pingSlow 2s infinite",
            }}
          />
          <span style={{ color: "var(--cyan-light)", fontWeight: 700 }}>DICOM 3.0 STREAM</span>
          <span style={{ color: "var(--text-muted)" }}>
            :: [X: {mousePos.active ? mousePos.x : 128}, Y: {mousePos.active ? mousePos.y : 144}, Z: {sliceDepth}]
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "var(--text-faint)" }}>
          <span>FOV: 240mm</span>
          <span>•</span>
          <span style={{ color: "var(--cyan-light)" }}>1.0mm³ Isotropic</span>
        </div>
      </div>

      {/* 2. Modality Sequence Selector Buttons */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
        <div style={{ display: "flex", gap: "0.4rem" }}>
          {[
            { id: "FLAIR", label: "FLAIR (Edema)" },
            { id: "T1ce", label: "T1ce (+Gd)" },
            { id: "T2", label: "T2 (Core)" },
            { id: "T1", label: "T1 (Native)" },
          ].map((m) => (
            <button
              key={m.id}
              className={`modality-btn ${modality === m.id ? "modality-btn-active" : "modality-btn-inactive"}`}
              onClick={() => setModality(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>

        {/* Anatomical Planes */}
        <div style={{ display: "flex", gap: "0.25rem" }}>
          {["AXIAL", "SAGITTAL", "CORONAL"].map((p) => (
            <button
              key={p}
              onClick={() => setPlane(p)}
              style={{
                background: plane === p ? "rgba(0, 240, 255, 0.2)" : "rgba(255, 255, 255, 0.04)",
                border: plane === p ? "1px solid var(--cyan-primary)" : "1px solid var(--border-subtle)",
                color: plane === p ? "var(--cyan-light)" : "var(--text-muted)",
                borderRadius: "3px",
                padding: "0.2rem 0.45rem",
                fontSize: "0.65rem",
                fontFamily: "var(--font-mono)",
                cursor: "pointer",
              }}
            >
              {p.slice(0, 3)}
            </button>
          ))}
        </div>
      </div>

      {/* 3. Interactive MRI Scanner Canvas Stage with Reticles & Holographic Laser */}
      <div
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "1 / 1",
          maxHeight: "360px",
          borderRadius: "var(--radius-md)",
          overflow: "hidden",
          background: "#050811",
          border: "1px solid rgba(255, 255, 255, 0.08)",
          boxShadow: "inset 0 0 35px rgba(0, 0, 0, 0.9)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <canvas
          ref={canvasRef}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
          style={{ width: "100%", height: "100%", cursor: "crosshair", display: "block" }}
        />

        {/* Holographic Animated Scanning Laser Sweep */}
        <div
          className="animate-scan-sweep"
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            height: "2px",
            background: "linear-gradient(90deg, transparent 0%, rgba(0, 240, 255, 0.8) 50%, transparent 100%)",
            boxShadow: "0 0 15px rgba(0, 240, 255, 0.9)",
            pointerEvents: "none",
          }}
        />

        {/* Floating Reticle Crosshair HUD */}
        <div
          style={{
            position: "absolute",
            top: "10px",
            right: "10px",
            background: "rgba(9, 14, 26, 0.85)",
            backdropFilter: "blur(12px)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-sm)",
            padding: "0.35rem 0.6rem",
            fontSize: "0.7rem",
            fontFamily: "var(--font-mono)",
            color: "var(--cyan-light)",
            display: "flex",
            alignItems: "center",
            gap: "0.4rem",
            pointerEvents: "none",
          }}
        >
          <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--cyan-primary)" }} />
          <span>Z-DEPTH: {sliceDepth}/155</span>
        </div>

        {/* Live Segment Legend Floating HUD */}
        <div
          style={{
            position: "absolute",
            bottom: "10px",
            left: "10px",
            background: "rgba(9, 14, 26, 0.85)",
            backdropFilter: "blur(12px)",
            border: "1px solid var(--border-subtle)",
            borderRadius: "var(--radius-sm)",
            padding: "0.4rem 0.65rem",
            fontSize: "0.68rem",
            fontFamily: "var(--font-mono)",
            display: "flex",
            flexDirection: "column",
            gap: "0.25rem",
            pointerEvents: "none",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "2px", background: "#10b981", boxShadow: "0 0 6px #10b981" }} />
            <span style={{ color: "var(--text-secondary)" }}>WT (Edema): 48.2 cm³</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "2px", background: "#f59e0b", boxShadow: "0 0 6px #f59e0b" }} />
            <span style={{ color: "var(--text-secondary)" }}>TC (Core): 19.4 cm³</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
            <span style={{ width: "8px", height: "8px", borderRadius: "2px", background: "#f43f5e", boxShadow: "0 0 6px #f43f5e" }} />
            <span style={{ color: "var(--text-secondary)" }}>ET (Active): 8.7 cm³</span>
          </div>
        </div>
      </div>

      {/* 4. Bottom Control Bar: Depth Scrubber, Cine Playback & Metrics */}
      <div style={{ marginTop: "0.75rem", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "0.75rem", fontFamily: "var(--font-mono)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              style={{
                background: isPlaying ? "rgba(244, 63, 94, 0.2)" : "rgba(0, 240, 255, 0.15)",
                border: isPlaying ? "1px solid var(--crimson-primary)" : "1px solid var(--cyan-primary)",
                color: isPlaying ? "var(--crimson-primary)" : "var(--cyan-light)",
                borderRadius: "var(--radius-sm)",
                padding: "0.25rem 0.6rem",
                fontSize: "0.72rem",
                fontWeight: 600,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                gap: "0.3rem",
              }}
            >
              {isPlaying ? <IconPause size={12} /> : <IconPlay size={12} />}
              <span>{isPlaying ? "Pause Cine" : "Play Slices"}</span>
            </button>
            <span style={{ color: "var(--text-muted)" }}>Slice {sliceDepth} of 155 [Axial]</span>
          </div>

          <div style={{ color: "var(--emerald-light)", fontWeight: 600 }}>
            Dice: 88.42% • HD95: 4.2mm
          </div>
        </div>

        {/* Slice slider track */}
        <input
          type="range"
          min="1"
          max="155"
          value={sliceDepth}
          onChange={(e) => setSliceDepth(Number(e.target.value))}
          style={{ width: "100%", accentColor: "var(--cyan-primary)", cursor: "pointer", height: "4px" }}
        />

        {/* Hover telemetry crosshair status */}
        <div
          style={{
            background: "rgba(0, 0, 0, 0.3)",
            padding: "0.4rem 0.75rem",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border-subtle)",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "0.72rem",
            fontFamily: "var(--font-mono)",
            color: "var(--text-secondary)",
          }}
        >
          <span>Target: <strong style={{ color: "var(--cyan-light)" }}>{inspectData.label}</strong></span>
          <span style={{ color: "var(--emerald-light)" }}>Intensity: {inspectData.val} HU</span>
        </div>
      </div>

      {/* 5. Floating Overlapping KPI Pills */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem", marginTop: "0.75rem" }}>
        <div style={{ background: "rgba(15, 23, 42, 0.8)", border: "1px solid rgba(0, 240, 255, 0.2)", borderRadius: "var(--radius-sm)", padding: "0.5rem" }}>
          <div style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono)", color: "var(--cyan-light)", fontWeight: 600 }}>0.00% Egress</div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.1rem" }}>Zero-Knowledge Verified</div>
        </div>

        <div style={{ background: "rgba(15, 23, 42, 0.8)", border: "1px solid rgba(139, 92, 246, 0.2)", borderRadius: "var(--radius-sm)", padding: "0.5rem" }}>
          <div style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono)", color: "var(--violet-light)", fontWeight: 600 }}>AES-256-GCM + DP</div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.1rem" }}>Homomorphic Aggregation</div>
        </div>

        <div style={{ background: "rgba(15, 23, 42, 0.8)", border: "1px solid rgba(16, 185, 129, 0.2)", borderRadius: "var(--radius-sm)", padding: "0.5rem" }}>
          <div style={{ fontSize: "0.7rem", fontFamily: "var(--font-mono)", color: "var(--emerald-light)", fontWeight: 600 }}>3 Nodes Synced</div>
          <div style={{ fontSize: "0.65rem", color: "var(--text-muted)", marginTop: "0.1rem" }}>Mayo • Charité • JHU</div>
        </div>
      </div>
    </div>
  );
}
