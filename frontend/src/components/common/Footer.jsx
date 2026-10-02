import React from "react";
import { IconShieldCheck, IconLock, IconActivity } from "./Icons";

export function Footer({ onNavigate }) {
  return (
    <footer className="antigravity-footer">
      <div className="antigravity-footer-inner">
        {/* Top Row: Brand & Links */}
        <div className="footer-top-row">
          {/* Brand Info */}
          <div className="footer-brand-col">
            <div style={{ display: "flex", alignItems: "center", gap: "0.6rem" }}>
              <div
                style={{
                  width: "28px",
                  height: "28px",
                  borderRadius: "6px",
                  background: "var(--brand-blue)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                }}
              >
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="3" y="3" width="18" height="18" rx="4" />
                  <path d="M12 7v10" />
                  <path d="M7 12h10" />
                </svg>
              </div>
              <span
                style={{
                  fontFamily: "var(--font-heading)",
                  fontWeight: 800,
                  fontSize: "1.1rem",
                  color: "var(--text-primary)",
                }}
              >
                FedMed Consortium
              </span>
            </div>
            <p style={{ fontSize: "0.88rem", color: "var(--text-secondary)", lineHeight: 1.55 }}>
              Decentralized 3D Brain Tumor MRI Federated Segmentation Platform. Built with
              PyTorch, MONAI Core, and Flower FL to enable multi-institutional clinical research
              with zero patient data transfer.
            </p>
          </div>

          {/* Quick Nav Columns */}
          <div className="footer-nav-col">
            <div>
              <h4 className="footer-col-title">Platform</h4>
              <ul className="footer-links-list">
                <li>
                  <button
                    onClick={() => onNavigate && onNavigate("home")}
                    className="footer-link-item"
                    style={{ background: "none", border: "none", padding: 0 }}
                  >
                    Overview
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => onNavigate && onNavigate("coordinator")}
                    className="footer-link-item"
                    style={{ background: "none", border: "none", padding: 0 }}
                  >
                    Coordinator Hub
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => onNavigate && onNavigate("hospital")}
                    className="footer-link-item"
                    style={{ background: "none", border: "none", padding: 0 }}
                  >
                    Hospital Node
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => onNavigate && onNavigate("simulation")}
                    className="footer-link-item"
                    style={{ background: "none", border: "none", padding: 0 }}
                  >
                    Simulation Testbed
                  </button>
                </li>
                <li>
                  <button
                    onClick={() => onNavigate && onNavigate("auth")}
                    className="footer-link-item"
                    style={{ background: "none", border: "none", padding: 0 }}
                  >
                    Portal Sign In
                  </button>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="footer-col-title">Consortium</h4>
              <ul className="footer-links-list">
                <li>
                  <button
                    onClick={() => onNavigate && onNavigate("about")}
                    className="footer-link-item"
                    style={{ background: "none", border: "none", padding: 0 }}
                  >
                    About Us
                  </button>
                </li>
                <li>
                  <span className="footer-link-item">Wockhardt Hospital</span>
                </li>
                <li>
                  <span className="footer-link-item">Mayo Clinic</span>
                </li>
                <li>
                  <span className="footer-link-item">KIMS Kingsway</span>
                </li>
                <li>
                  <span className="footer-link-item">AIIMS (AIMS)</span>
                </li>
              </ul>
            </div>

            <div>
              <h4 className="footer-col-title">Compliance</h4>
              <ul className="footer-links-list">
                <li>
                  <span className="footer-link-item">HIPAA Verification</span>
                </li>
                <li>
                  <span className="footer-link-item">GDPR In-Situ Data</span>
                </li>
                <li>
                  <span className="footer-link-item">AES-256-GCM Proof</span>
                </li>
                <li>
                  <span className="footer-link-item">BraTS 2024 Tier-1</span>
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Bottom Row: Badges & Copyright */}
        <div className="footer-bottom-row">
          <div>
            &copy; {new Date().getFullYear()} FedMed AI Consortium. All clinical data rights reserved.
          </div>

          <div style={{ display: "flex", gap: "1.25rem", flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem", color: "var(--emerald-light)", fontWeight: 500 }}>
              <IconShieldCheck size={16} /> 0.00% Raw Data Leakage
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem", color: "var(--brand-blue)", fontWeight: 500 }}>
              <IconLock size={16} /> Authenticated AES-256-GCM
            </span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.35rem", color: "var(--violet-primary)", fontWeight: 500 }}>
              <IconActivity size={16} /> Flower FL 1.11 Engine
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}
