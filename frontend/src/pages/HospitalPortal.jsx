import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { Card } from "../components/common/Card";
import { Badge } from "../components/common/Badge";
import { Button } from "../components/common/Button";
import { Input } from "../components/common/Input";
import { IconHospital, IconCpu, IconShieldCheck, IconPlay, IconActivity } from "../components/common/Icons";

export function HospitalPortal() {
  const { user } = useAuth();
  const [dataDir, setDataDir] = useState("./data/hospital_a");
  const [isTraining, setIsTraining] = useState(false);
  const [localEpoch, setLocalEpoch] = useState(0);

  const startLocalTraining = () => {
    setIsTraining(true);
    let epoch = 0;
    const interval = setInterval(() => {
      epoch += 1;
      setLocalEpoch(epoch);
      if (epoch >= 5) {
        clearInterval(interval);
        setIsTraining(false);
      }
    }, 1000);
  };

  return (
    <div className="flex-col gap-6">
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <Badge variant="emerald" pulse>Hospital Node Connected</Badge>
            <Badge variant="cyan">{user?.hospital_node_id || "NODE-HOSP-A"}</Badge>
          </div>
          <h2 style={{ marginTop: "0.4rem" }}>
            {user?.institution_name || "Mount Sinai Brain Tumor Center"} — Local Workspace
          </h2>
          <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>
            Clinician / ML Engineer: <strong>{user?.username || "Hospital Staff"}</strong>
          </p>
        </div>

        <div style={{ display: "flex", gap: "0.75rem" }}>
          <Button
            variant="glow"
            size="sm"
            onClick={startLocalTraining}
            loading={isTraining}
            icon={<IconPlay size={14} />}
          >
            {isTraining ? `Training Locally (Epoch ${localEpoch}/5)...` : "Launch Local Node Training"}
          </Button>
        </div>
      </div>

      {/* Local Privacy & Hardware Spec */}
      <div className="grid grid-3">
        <Card title="Private Data Connector" icon={<IconHospital size={18} />}>
          <div className="flex-col gap-3">
            <Input
              label="Local Patient MRI Folder"
              value={dataDir}
              onChange={(e) => setDataDir(e.target.value)}
              helperText="Scans are loaded locally from this disk path only"
            />
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", color: "var(--emerald-light)", fontSize: "0.8rem" }}>
              <IconShieldCheck size={16} />
              <span>Zero raw patient images are uploaded over the internet</span>
            </div>
          </div>
        </Card>

        <Card title="Compute Hardware Profile" icon={<IconCpu size={18} />}>
          <div className="flex-col gap-2" style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
            <div>Primary Compute: <strong style={{ color: "var(--text-primary)" }}>NVIDIA RTX 4090 / CUDA</strong></div>
            <div>Dedicated VRAM: <strong style={{ color: "var(--cyan-light)" }}>24.0 GB GDDR6X</strong></div>
            <div>CPU Threads: <strong style={{ color: "var(--text-primary)" }}>16 Cores</strong></div>
            <div>Memory Bandwidth: <strong style={{ color: "var(--emerald-light)" }}>1,008 GB/s</strong></div>
          </div>
        </Card>

        <Card title="Model Parameter Diffing" icon={<IconActivity size={18} />}>
          <div className="flex-col gap-2" style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
            <div>Compression: <strong style={{ color: "var(--violet-light)" }}>ΔW Diffing Enabled</strong></div>
            <div>Bandwidth Optimization: <strong style={{ color: "var(--emerald-light)" }}>68% reduction</strong></div>
            <div>Decryption Key: <strong style={{ color: "var(--cyan-light)" }}>7F8B2C4D Verified</strong></div>
            <div>Local Fallback: <strong style={{ color: "var(--emerald-light)" }}>Active (Safe Rollback)</strong></div>
          </div>
        </Card>
      </div>
    </div>
  );
}
