import React, { useState } from "react";
import {
  IconBrain,
  IconActivity,
  IconShieldCheck,
  IconLock,
  IconKey,
  IconServer,
  IconHospital,
  IconCpu,
} from "../components/common/Icons";
import { Button } from "../components/common/Button";
import { Card } from "../components/common/Card";
import { Badge } from "../components/common/Badge";
import { Input } from "../components/common/Input";
import { Modal } from "../components/common/Modal";

export function DesignSandbox() {
  const [modalOpen, setModalOpen] = useState(false);
  const [inputText, setInputText] = useState("Mount Sinai Hospital Node");
  const [inputError, setInputError] = useState("");

  return (
    <div className="flex-col gap-8">
      {/* Title */}
      <div>
        <Badge variant="cyan" pulse style={{ marginBottom: "0.75rem" }}>
          FedMed UI Design System
        </Badge>
        <h1>Medical AI Design Tokens & Components</h1>
        <p style={{ marginTop: "0.5rem", maxWidth: "700px" }}>
          Obsidian dark theme, glassmorphic surfaces, curated medical HSL color tokens,
          and accessible interactive primitives.
        </p>
      </div>

      {/* Color Palette Tokens */}
      <Card title="Color Palette & Brand Accents" icon={<IconActivity size={18} />}>
        <div className="grid grid-4" style={{ gap: "1rem" }}>
          <div style={{ background: "var(--bg-card-solid)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ width: "100%", height: "40px", borderRadius: "6px", background: "var(--cyan-primary)", marginBottom: "0.5rem", boxShadow: "0 0 15px var(--cyan-glow)" }} />
            <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>Neon Cyan</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>#06B6D4 • Medical Tech</div>
          </div>

          <div style={{ background: "var(--bg-card-solid)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ width: "100%", height: "40px", borderRadius: "6px", background: "var(--emerald-primary)", marginBottom: "0.5rem", boxShadow: "0 0 15px var(--emerald-glow)" }} />
            <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>Bio Emerald</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>#10B981 • Validation & Dice</div>
          </div>

          <div style={{ background: "var(--bg-card-solid)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ width: "100%", height: "40px", borderRadius: "6px", background: "var(--violet-primary)", marginBottom: "0.5rem", boxShadow: "0 0 15px var(--violet-glow)" }} />
            <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>Glowing Violet</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>#8B5CF6 • Federated AI</div>
          </div>

          <div style={{ background: "var(--bg-card-solid)", padding: "1rem", borderRadius: "var(--radius-md)", border: "1px solid var(--border-subtle)" }}>
            <div style={{ width: "100%", height: "40px", borderRadius: "6px", background: "var(--crimson-primary)", marginBottom: "0.5rem", boxShadow: "0 0 15px var(--crimson-glow)" }} />
            <div style={{ fontWeight: 600, fontSize: "0.9rem" }}>Danger Crimson</div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>#EF4444 • Tamper Alert</div>
          </div>
        </div>
      </Card>

      {/* Button Variations */}
      <Card title="Interactive Button Variants" icon={<IconKey size={18} />}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "1rem", alignItems: "center" }}>
          <Button variant="primary" icon={<IconBrain size={16} />}>
            Primary Action
          </Button>

          <Button variant="glow" icon={<IconActivity size={16} />}>
            Glow Gradient
          </Button>

          <Button variant="secondary" icon={<IconServer size={16} />}>
            Secondary Neutral
          </Button>

          <Button variant="outline" icon={<IconHospital size={16} />}>
            Outline Cyan
          </Button>

          <Button variant="success" icon={<IconShieldCheck size={16} />}>
            Success Emerald
          </Button>

          <Button variant="danger" icon={<IconLock size={16} />}>
            Emergency Stop
          </Button>

          <Button variant="primary" size="sm">
            Small
          </Button>

          <Button variant="primary" size="lg">
            Large CTA
          </Button>

          <Button variant="primary" loading>
            Loading State
          </Button>
        </div>
      </Card>

      {/* Status Badges */}
      <Card title="Status & Telemetry Badges" icon={<IconShieldCheck size={18} />}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", alignItems: "center" }}>
          <Badge variant="cyan" pulse>
            Flower Orchestrator
          </Badge>

          <Badge variant="emerald" pulse>
            Connected (Online)
          </Badge>

          <Badge variant="violet" pulse>
            Round 5 Active
          </Badge>

          <Badge variant="amber">
            High VRAM Load
          </Badge>

          <Badge variant="crimson" pulse>
            Tamper Rejected
          </Badge>

          <Badge variant="cyan" icon={<IconLock size={12} />}>
            AES-256-GCM
          </Badge>

          <Badge variant="emerald" icon={<IconCpu size={12} />}>
            RTX 4090 • 24GB
          </Badge>
        </div>
      </Card>

      {/* Form Controls & Inputs */}
      <Card title="Form Inputs & Data Entry" icon={<IconServer size={18} />}>
        <div className="grid grid-2">
          <Input
            label="Hospital Institution Name"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            icon={<IconHospital size={18} />}
            helperText="Legal name of healthcare system joining the cluster"
          />

          <Input
            label="Local MRI Dataset Directory"
            value="./data/hospital_a"
            onChange={() => {}}
            icon={<IconBrain size={18} />}
            helperText="Path on local machine (zero network upload)"
          />

          <Input
            label="Cryptographic API Key"
            value="fedmed_live_key_982347293"
            type="password"
            onChange={() => {}}
            icon={<IconLock size={18} />}
          />

          <Input
            label="Simulated Error Validation"
            value="invalid_cluster_id"
            error="Cluster node ID format must match 'NODE-HOSP-X'"
            onChange={() => {}}
            icon={<IconKey size={18} />}
          />
        </div>
      </Card>

      {/* Modal Demonstration */}
      <Card
        title="Interactive Modals & Dialogs"
        headerAction={
          <Button variant="glow" size="sm" onClick={() => setModalOpen(true)}>
            Open Test Modal
          </Button>
        }
      >
        <p>
          FedMed uses backdrop blur modals with keyboard ESC dismissal for key rotation,
          scan inspection, and round confirmation.
        </p>

        <Modal
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          title="Cryptographic Key Rotation Notice"
          footer={
            <>
              <Button variant="secondary" onClick={() => setModalOpen(false)}>
                Cancel
              </Button>
              <Button
                variant="glow"
                onClick={() => {
                  alert("Key rotated live!");
                  setModalOpen(false);
                }}
              >
                Confirm Rotation
              </Button>
            </>
          }
        >
          <div className="flex-col gap-3">
            <p>
              Rotating the central 256-bit AES-GCM key will re-encrypt all global 3D U-Net checkpoints
              stored under <code style={{ color: "var(--cyan-light)" }}>./checkpoints</code> and publish
              a new key fingerprint to authenticated hospital workers.
            </p>
            <div style={{ background: "rgba(0,0,0,0.3)", padding: "1rem", borderRadius: "8px", border: "1px solid var(--border-subtle)" }}>
              <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Active Key Fingerprint</div>
              <div style={{ fontFamily: "var(--font-mono)", fontSize: "1.1rem", color: "var(--cyan-light)", marginTop: "0.25rem" }}>
                7F8B2C4D
              </div>
            </div>
          </div>
        </Modal>
      </Card>
    </div>
  );
}
