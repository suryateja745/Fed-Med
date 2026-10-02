import React, { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { IconUser, IconLock } from "../components/common/Icons";

export function AuthPage({ onNavigate }) {
  const { login, register, isAuthenticated, user, logout } = useAuth();
  const [tab, setTab] = useState("login"); // 'login' | 'register'

  // Set page document title when inside Authentication Portal
  useEffect(() => {
    const prevTitle = document.title;
    document.title = "Authentication Portal";
    return () => {
      document.title = prevTitle;
    };
  }, []);

  // Login form state
  const [loginUsername, setLoginUsername] = useState("");
  const [loginPassword, setLoginPassword] = useState("");

  // Register form state: name | username | password | confirm password | role (hospital or coordinator)
  const [regName, setRegName] = useState("");
  const [regUsername, setRegUsername] = useState("");
  const [regPassword, setRegPassword] = useState("");
  const [regConfirmPassword, setRegConfirmPassword] = useState("");
  const [regRole, setRegRole] = useState("HOSPITAL_STAFF"); // 'HOSPITAL_STAFF' (hospital) | 'ADMIN' (coordinator)

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e) => {
    e?.preventDefault();
    setError("");

    if (!loginUsername.trim() || !loginPassword) {
      setError("Please enter your username and password.");
      return;
    }

    setLoading(true);
    try {
      const loggedUser = await login(loginUsername.trim(), loginPassword);
      // Query the database for existing credential and redirect to respective windows
      if (loggedUser.role === "ADMIN") {
        onNavigate("coordinator");
      } else {
        onNavigate("hospital");
      }
    } catch (err) {
      setError(err.message || "Invalid credentials. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e?.preventDefault();
    setError("");

    if (!regName.trim() || !regUsername.trim() || !regPassword) {
      setError("Please fill in all required fields.");
      return;
    }

    if (regPassword.length < 6) {
      setError("Password must be at least 6 characters long.");
      return;
    }

    if (regPassword !== regConfirmPassword) {
      setError("Passwords do not match. Please verify.");
      return;
    }

    setLoading(true);
    try {
      const cleanUsername = regUsername.trim().toLowerCase();
      const generatedEmail = `${cleanUsername}@fedmed.org`;

      const registered = await register({
        username: regUsername.trim(),
        email: generatedEmail,
        password: regPassword,
        role: regRole,
        institution_name: regName.trim(),
        hospital_node_id: regRole === "HOSPITAL_STAFF" ? `NODE-${cleanUsername.toUpperCase().slice(0, 8)}` : null,
      });

      // Redirect to respective windows after successful registration
      if (registered.role === "ADMIN") {
        onNavigate("coordinator");
      } else {
        onNavigate("hospital");
      }
    } catch (err) {
      setError(err.message || "Registration failed. Username may already exist.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        width: "100vw",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        background: "var(--bg-primary, #ffffff)",
        padding: "1rem",
        boxSizing: "border-box",
        position: "relative",
      }}
    >
      {/* Back to Home Link */}
      <button
        type="button"
        onClick={() => onNavigate("home")}
        style={{
          position: "absolute",
          top: "1.25rem",
          left: "1.5rem",
          background: "none",
          border: "none",
          color: "var(--text-muted, #64748b)",
          cursor: "pointer",
          fontSize: "0.88rem",
          fontWeight: 500,
          display: "flex",
          alignItems: "center",
          gap: "0.4rem",
          padding: "0.4rem 0.6rem",
          borderRadius: "var(--radius-sm, 6px)",
          transition: "all var(--transition-fast, 150ms)",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.color = "var(--text-primary, #0f172a)";
          e.currentTarget.style.background = "var(--bg-subtle, #f1f5f9)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.color = "var(--text-muted, #64748b)";
          e.currentTarget.style.background = "none";
        }}
      >
        &larr; Back to Home
      </button>

      {/* Main Container Card */}
      <div
        style={{
          width: "100%",
          maxWidth: "400px",
          background: "#ffffff",
          border: "1px solid var(--border-subtle, #e2e8f0)",
          borderRadius: "var(--radius-xl, 20px)",
          padding: "2rem",
          boxShadow: "0 10px 30px -10px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Title */}
        <div style={{ textAlign: "center", marginBottom: "1.25rem" }}>
          <h1
            style={{
              fontFamily: "var(--font-heading, sans-serif)",
              fontSize: "1.65rem",
              fontWeight: 800,
              color: "var(--text-primary, #0f172a)",
              letterSpacing: "-0.03em",
              margin: 0,
            }}
          >
            Authentication Portal
          </h1>
        </div>

        {/* If user is already authenticated */}
        {isAuthenticated && user ? (
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                padding: "0.85rem 1rem",
                background: "var(--bg-subtle, #f8fafc)",
                borderRadius: "var(--radius-md, 12px)",
                border: "1px solid var(--border-subtle, #e2e8f0)",
              }}
            >
              <div>
                <div style={{ fontSize: "0.72rem", color: "var(--text-muted, #64748b)", textTransform: "uppercase", letterSpacing: "0.05em", fontFamily: "var(--font-mono, monospace)" }}>
                  Signed in as
                </div>
                <div style={{ fontSize: "1.05rem", fontWeight: 700, color: "var(--text-primary, #0f172a)" }}>
                  {user.username}
                </div>
              </div>
              <span
                style={{
                  fontSize: "0.72rem",
                  fontFamily: "var(--font-mono, monospace)",
                  padding: "0.2rem 0.55rem",
                  borderRadius: "9999px",
                  background: user.role === "ADMIN" ? "var(--brand-blue-subtle, #e8f0fe)" : "var(--emerald-subtle, #ecfdf5)",
                  color: user.role === "ADMIN" ? "var(--brand-blue, #1a73e8)" : "var(--emerald-light, #059669)",
                  border: `1px solid ${user.role === "ADMIN" ? "var(--brand-blue-border, #d2e3fc)" : "var(--emerald-border, #a7f3d0)"}`,
                  fontWeight: 600,
                }}
              >
                {user.role === "ADMIN" ? "Coordinator" : "Hospital"}
              </span>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "0.6rem" }}>
              {user.role === "ADMIN" ? (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => onNavigate("coordinator")}
                  style={{ width: "100%", padding: "0.65rem", fontSize: "0.9rem" }}
                >
                  Go to Coordinator Window &rarr;
                </button>
              ) : (
                <button
                  type="button"
                  className="btn btn-primary"
                  onClick={() => onNavigate("hospital")}
                  style={{ width: "100%", padding: "0.65rem", fontSize: "0.9rem" }}
                >
                  Go to Hospital Window &rarr;
                </button>
              )}

              <button
                type="button"
                className="btn btn-secondary"
                onClick={logout}
                style={{ width: "100%", padding: "0.55rem", fontSize: "0.85rem" }}
              >
                Sign Out
              </button>
            </div>
          </div>
        ) : (
          <>
            {/* Two Options: Login and Register */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "0.25rem",
                padding: "0.25rem",
                background: "var(--bg-subtle, #f1f5f9)",
                borderRadius: "var(--radius-md, 10px)",
                marginBottom: "1.25rem",
              }}
            >
              <button
                type="button"
                onClick={() => {
                  setTab("login");
                  setError("");
                }}
                style={{
                  padding: "0.48rem 0.75rem",
                  borderRadius: "var(--radius-sm, 7px)",
                  border: "none",
                  cursor: "pointer",
                  fontSize: "0.88rem",
                  fontWeight: 600,
                  fontFamily: "var(--font-sans, sans-serif)",
                  background: tab === "login" ? "#ffffff" : "transparent",
                  color: tab === "login" ? "var(--text-primary, #0f172a)" : "var(--text-muted, #64748b)",
                  boxShadow: tab === "login" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
                  transition: "all var(--transition-fast, 150ms)",
                }}
              >
                Login
              </button>

              <button
                type="button"
                onClick={() => {
                  setTab("register");
                  setError("");
                }}
                style={{
                  padding: "0.48rem 0.75rem",
                  borderRadius: "var(--radius-sm, 7px)",
                  border: "none",
                  cursor: "pointer",
                  fontSize: "0.88rem",
                  fontWeight: 600,
                  fontFamily: "var(--font-sans, sans-serif)",
                  background: tab === "register" ? "#ffffff" : "transparent",
                  color: tab === "register" ? "var(--text-primary, #0f172a)" : "var(--text-muted, #64748b)",
                  boxShadow: tab === "register" ? "0 1px 3px rgba(0,0,0,0.06)" : "none",
                  transition: "all var(--transition-fast, 150ms)",
                }}
              >
                Register
              </button>
            </div>

            {/* Error message */}
            {error && (
              <div
                style={{
                  background: "var(--coral-subtle, #fef2f2)",
                  color: "var(--coral-primary, #ef4444)",
                  border: "1px solid var(--coral-border, #fecaca)",
                  padding: "0.55rem 0.75rem",
                  borderRadius: "var(--radius-sm, 8px)",
                  marginBottom: "0.9rem",
                  fontSize: "0.82rem",
                  lineHeight: 1.4,
                }}
              >
                {error}
              </div>
            )}

            {/* 1. LOGIN FORM */}
            {tab === "login" ? (
              <form onSubmit={handleLogin} style={{ display: "flex", flexDirection: "column", gap: "0.9rem" }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.82rem", marginBottom: "0.3rem" }}>Username</label>
                  <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                    <span
                      style={{
                        position: "absolute",
                        left: "0.85rem",
                        color: "var(--text-faint, #94a3b8)",
                        display: "flex",
                        alignItems: "center",
                        pointerEvents: "none",
                      }}
                    >
                      <IconUser size={16} />
                    </span>
                    <input
                      type="text"
                      className="form-input"
                      style={{ paddingLeft: "2.4rem", paddingBlock: "0.55rem", fontSize: "0.9rem" }}
                      placeholder="Enter username"
                      value={loginUsername}
                      onChange={(e) => setLoginUsername(e.target.value)}
                      autoComplete="username"
                      required
                    />
                  </div>
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.82rem", marginBottom: "0.3rem" }}>Password</label>
                  <div style={{ position: "relative", display: "flex", alignItems: "center" }}>
                    <span
                      style={{
                        position: "absolute",
                        left: "0.85rem",
                        color: "var(--text-faint, #94a3b8)",
                        display: "flex",
                        alignItems: "center",
                        pointerEvents: "none",
                      }}
                    >
                      <IconLock size={16} />
                    </span>
                    <input
                      type="password"
                      className="form-input"
                      style={{ paddingLeft: "2.4rem", paddingBlock: "0.55rem", fontSize: "0.9rem" }}
                      placeholder="Enter password"
                      value={loginPassword}
                      onChange={(e) => setLoginPassword(e.target.value)}
                      autoComplete="current-password"
                      required
                    />
                  </div>
                </div>

                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                  style={{
                    width: "100%",
                    padding: "0.68rem",
                    fontSize: "0.92rem",
                    marginTop: "0.35rem",
                  }}
                >
                  {loading ? "Authenticating..." : "Login"}
                </button>
              </form>
            ) : (
              /* 2. REGISTER FORM: Name | Username | Role | Password | Confirm Password */
              <form onSubmit={handleRegister} style={{ display: "flex", flexDirection: "column", gap: "0.68rem" }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.78rem", marginBottom: "0.2rem" }}>Name</label>
                  <input
                    type="text"
                    className="form-input"
                    style={{ paddingBlock: "0.45rem", fontSize: "0.86rem" }}
                    placeholder="Full name or institution"
                    value={regName}
                    onChange={(e) => setRegName(e.target.value)}
                    required
                  />
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.78rem", marginBottom: "0.2rem" }}>Username</label>
                  <input
                    type="text"
                    className="form-input"
                    style={{ paddingBlock: "0.45rem", fontSize: "0.86rem" }}
                    placeholder="Choose username"
                    value={regUsername}
                    onChange={(e) => setRegUsername(e.target.value)}
                    autoComplete="username"
                    required
                  />
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.78rem", marginBottom: "0.2rem" }}>Role</label>
                  <select
                    className="form-input"
                    value={regRole}
                    onChange={(e) => setRegRole(e.target.value)}
                    style={{ paddingBlock: "0.45rem", fontSize: "0.86rem", cursor: "pointer" }}
                  >
                    <option value="HOSPITAL_STAFF">Hospital</option>
                    <option value="ADMIN">Coordinator</option>
                  </select>
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.78rem", marginBottom: "0.2rem" }}>Password</label>
                  <input
                    type="password"
                    className="form-input"
                    style={{ paddingBlock: "0.45rem", fontSize: "0.86rem" }}
                    placeholder="Min 6 characters"
                    value={regPassword}
                    onChange={(e) => setRegPassword(e.target.value)}
                    autoComplete="new-password"
                    required
                  />
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label" style={{ fontSize: "0.78rem", marginBottom: "0.2rem" }}>Confirm Password</label>
                  <input
                    type="password"
                    className="form-input"
                    style={{ paddingBlock: "0.45rem", fontSize: "0.86rem" }}
                    placeholder="Confirm password"
                    value={regConfirmPassword}
                    onChange={(e) => setRegConfirmPassword(e.target.value)}
                    autoComplete="new-password"
                    required
                  />
                </div>

                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={loading}
                  style={{
                    width: "100%",
                    padding: "0.62rem",
                    fontSize: "0.9rem",
                    marginTop: "0.35rem",
                  }}
                >
                  {loading ? "Creating Account..." : "Register"}
                </button>
              </form>
            )}
          </>
        )}
      </div>
    </div>
  );
}
