import React from "react";
import { IconBrain, IconActivity, IconLock, IconUser, IconLogOut, IconLayers } from "./Icons";
import { Badge } from "./Badge";
import { Button } from "./Button";

export function Navbar({
  activeTab,
  onSelectTab,
  currentUser,
  onLogout,
  isBackendConnected = true,
}) {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        {/* Brand */}
        <div
          className="navbar-brand"
          style={{ cursor: "pointer" }}
          onClick={() => onSelectTab("home")}
        >
          <div
            style={{
              width: "36px",
              height: "36px",
              borderRadius: "10px",
              background: "linear-gradient(135deg, var(--cyan-primary), var(--violet-primary))",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 15px var(--cyan-glow)",
            }}
          >
            <IconBrain size={22} style={{ color: "#ffffff" }} />
          </div>
          <span className="navbar-brand-gradient">FedMed</span>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)", fontWeight: 400, marginTop: "2px" }}>
            v1.0 • 3D MRI AI
          </span>
        </div>

        {/* Navigation Tabs */}
        <nav className="navbar-nav">
          <button
            className={`nav-link ${activeTab === "home" ? "active" : ""}`}
            onClick={() => onSelectTab("home")}
          >
            <IconBrain size={16} />
            <span>Showcase</span>
          </button>

          <button
            className={`nav-link ${activeTab === "coordinator" ? "active" : ""}`}
            onClick={() => onSelectTab("coordinator")}
          >
            <IconActivity size={16} />
            <span>Coordinator</span>
          </button>

          <button
            className={`nav-link ${activeTab === "hospital" ? "active" : ""}`}
            onClick={() => onSelectTab("hospital")}
          >
            <IconLock size={16} />
            <span>Hospital Node</span>
          </button>

          <button
            className={`nav-link ${activeTab === "simulation" ? "active" : ""}`}
            onClick={() => onSelectTab("simulation")}
          >
            <IconLayers size={16} />
            <span>Simulation</span>
          </button>

          <button
            className={`nav-link ${activeTab === "sandbox" ? "active" : ""}`}
            onClick={() => onSelectTab("sandbox")}
          >
            <span>UI Sandbox</span>
          </button>
        </nav>

        {/* Right Actions: Telemetry status & User Session */}
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          {/* Connection status */}
          <Badge
            variant={isBackendConnected ? "emerald" : "crimson"}
            pulse={isBackendConnected}
          >
            {isBackendConnected ? "API LIVE" : "OFFLINE"}
          </Badge>

          {/* User profile / Login */}
          {currentUser ? (
            <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
              <div style={{ textAlign: "right", display: "flex", flexDirection: "column" }}>
                <span style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-primary)" }}>
                  {currentUser.username}
                </span>
                <span style={{ fontSize: "0.7rem", color: "var(--cyan-light)" }}>
                  {currentUser.role}
                </span>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={onLogout}
                icon={<IconLogOut size={14} />}
              >
                Logout
              </Button>
            </div>
          ) : (
            <Button
              variant="primary"
              size="sm"
              onClick={() => onSelectTab("auth")}
              icon={<IconUser size={14} />}
            >
              Portal Login
            </Button>
          )}
        </div>
      </div>
    </header>
  );
}
