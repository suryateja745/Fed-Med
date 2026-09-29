import React from "react";
import {
  IconBrain,
  IconActivity,
  IconHospital,
  IconShieldCheck,
  IconServer,
  IconLock,
} from "./Icons";

export function Navbar({
  activeTab,
  onSelectTab,
  currentUser,
  onLogout,
  isBackendConnected = true,
}) {
  return (
    <header className="antigravity-navbar">
      <div className="antigravity-navbar-inner">
        {/* Very Left End: FedMed Brand Logo & Name */}
        <button
          className="navbar-brand-wrapper"
          onClick={() => onSelectTab("home")}
          aria-label="FedMed Home"
        >
          <div className="brand-icon-mark">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <rect x="3" y="3" width="18" height="18" rx="5" />
              <path d="M12 7v10" />
              <path d="M7 12h10" />
              <circle cx="12" cy="12" r="2.5" fill="white" />
            </svg>
          </div>
          <span className="brand-title-text">
            FedMed
            <span className="brand-badge-pill">AI</span>
          </span>
        </button>

        {/* Center: The Three Options (Coordinator, Hospitals, About Us) */}
        <nav className="navbar-nav-links">
          <button
            className={`nav-link-btn ${activeTab === "coordinator" ? "active" : ""}`}
            onClick={() => onSelectTab("coordinator")}
          >
            <IconServer size={16} />
            <span>Coordinator</span>
          </button>

          <button
            className={`nav-link-btn ${activeTab === "hospital" ? "active" : ""}`}
            onClick={() => onSelectTab("hospital")}
          >
            <IconHospital size={16} />
            <span>Hospitals</span>
          </button>

          <button
            className={`nav-link-btn ${activeTab === "about" ? "active" : ""}`}
            onClick={() => onSelectTab("about")}
          >
            <span>About Us</span>
          </button>
        </nav>

        {/* Very Right End: Live Status & Action Button */}
        <div className="navbar-right-actions">
          {/* Live indicator dot */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.45rem",
              fontSize: "0.8rem",
              fontFamily: "var(--font-mono)",
              color: isBackendConnected ? "var(--emerald-light)" : "var(--amber-primary)",
              padding: "0.3rem 0.75rem",
              borderRadius: "var(--radius-full)",
              background: isBackendConnected ? "var(--emerald-subtle)" : "var(--amber-subtle)",
              border: `1px solid ${isBackendConnected ? "var(--emerald-border)" : "var(--amber-border)"}`,
              fontWeight: 500,
            }}
          >
            <span
              style={{
                width: "7px",
                height: "7px",
                borderRadius: "50%",
                background: isBackendConnected ? "var(--emerald-primary)" : "var(--amber-primary)",
              }}
            />
            <span>{isBackendConnected ? "Consortium Live" : "Offline"}</span>
          </div>

          {/* CTA / Auth Button */}
          {currentUser ? (
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <span style={{ fontSize: "0.85rem", color: "var(--text-secondary)", fontWeight: 500 }}>
                {currentUser.username || currentUser.node_id || "Clinical Member"}
              </span>
              <button
                onClick={onLogout}
                className="nav-link-btn"
                style={{ padding: "0.4rem 0.8rem", fontSize: "0.8rem", border: "1px solid var(--border-subtle)" }}
              >
                Logout
              </button>
            </div>
          ) : (
            <button
              className="nav-cta-btn"
              onClick={() => onSelectTab("auth")}
            >
              <span>Start with FedMed</span>
              <span style={{ fontSize: "1.1rem", lineHeight: 1 }}>&rarr;</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
