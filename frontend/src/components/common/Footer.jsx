import React from "react";
import { IconShieldCheck, IconLock, IconActivity } from "./Icons";

export function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", marginBottom: "0.4rem" }}>
            <span style={{ fontWeight: 700, color: "var(--text-primary)", fontFamily: "var(--font-heading)" }}>
              FedMed AI Consortium
            </span>
            <span style={{ color: "var(--cyan-primary)" }}>•</span>
            <span>Decentralized 3D Brain Tumor MRI Federated Segmentation</span>
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-faint)", maxWidth: "600px" }}>
            Engineered with PyTorch, MONAI 3D U-Net, Flower Federated Learning, and zero-trust
            AES-256-GCM model weight encryption. Patient data never leaves local institutional storage.
          </p>
        </div>

        {/* Security / Compliance Badges */}
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.75rem", color: "var(--emerald-light)" }}>
            <IconShieldCheck size={16} />
            <span>Zero Raw Data Transfer</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.75rem", color: "var(--cyan-light)" }}>
            <IconLock size={16} />
            <span>AES-256-GCM + HMAC-SHA256</span>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", fontSize: "0.75rem", color: "var(--violet-light)" }}>
            <IconActivity size={16} />
            <span>Flower v1.8+ Engine</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
