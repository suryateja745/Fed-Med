import React, { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { Card } from "../components/common/Card";
import { Button } from "../components/common/Button";
import { Input } from "../components/common/Input";
import { Badge } from "../components/common/Badge";
import { IconUser, IconLock, IconShieldCheck, IconHospital } from "../components/common/Icons";

export function AuthPage({ onNavigate }) {
  const { login, register, isAuthenticated, user, logout } = useAuth();
  const [tab, setTab] = useState("login"); // 'login' | 'register'
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [institutionName, setInstitutionName] = useState("");
  const [role, setRole] = useState("HOSPITAL_STAFF");
  const [hospitalNodeId, setHospitalNodeId] = useState("NODE-HOSP-A");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = async (e) => {
    e?.preventDefault();
    setError("");
    setLoading(true);
    try {
      const loggedUser = await login(username, password);
      if (loggedUser.role === "ADMIN") {
        onNavigate("coordinator");
      } else {
        onNavigate("hospital");
      }
    } catch (err) {
      setError(err.message || "Login failed");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async (e) => {
    e?.preventDefault();
    setError("");
    setLoading(true);
    try {
      const registered = await register({
        username,
        email,
        password,
        role,
        institution_name: institutionName || "General Hospital",
        hospital_node_id: role === "HOSPITAL_STAFF" ? hospitalNodeId : null,
      });
      if (registered.role === "ADMIN") {
        onNavigate("coordinator");
      } else {
        onNavigate("hospital");
      }
    } catch (err) {
      setError(err.message || "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  const autofillCredentials = (u, p) => {
    setUsername(u);
    setPassword(p);
    setError("");
  };

  if (isAuthenticated && user) {
    return (
      <div className="content-narrow" style={{ paddingTop: "2rem" }}>
        <Card title="Active User Session" icon={<IconUser size={20} />}>
          <div className="flex-col gap-4">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <h3 style={{ fontSize: "1.3rem" }}>{user.username}</h3>
                <p style={{ color: "var(--text-muted)", fontSize: "0.9rem" }}>{user.email}</p>
                <div style={{ marginTop: "0.5rem", display: "flex", gap: "0.5rem" }}>
                  <Badge variant="cyan">{user.role}</Badge>
                  {user.hospital_node_id && <Badge variant="emerald">{user.hospital_node_id}</Badge>}
                </div>
              </div>
              <Button variant="danger" size="sm" onClick={logout}>
                Sign Out
              </Button>
            </div>

            <div style={{ borderTop: "1px solid var(--border-subtle)", paddingTop: "1rem" }}>
              <p style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "1rem" }}>
                Institution: <strong>{user.institution_name}</strong>
              </p>
              <div style={{ display: "flex", gap: "1rem" }}>
                {user.role === "ADMIN" ? (
                  <Button variant="glow" onClick={() => onNavigate("coordinator")}>
                    Open Coordinator Dashboard
                  </Button>
                ) : (
                  <Button variant="glow" onClick={() => onNavigate("hospital")}>
                    Open Hospital Node Portal
                  </Button>
                )}
              </div>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="content-narrow" style={{ paddingTop: "2rem" }}>
      <div style={{ textAlign: "center", marginBottom: "2rem" }}>
        <Badge variant="cyan" pulse style={{ marginBottom: "0.75rem" }}>
          FedMed Zero-Trust Access
        </Badge>
        <h2>Institutional Authentication Portal</h2>
        <p style={{ color: "var(--text-muted)", marginTop: "0.4rem" }}>
          Role-based segregation for Central Flower Coordinators, Hospital Clinicians, and Clinical Auditors.
        </p>
      </div>

      <Card glow="cyan">
        {/* Tab Switcher */}
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.5rem", borderBottom: "1px solid var(--border-subtle)", paddingBottom: "0.75rem" }}>
          <Button
            variant={tab === "login" ? "primary" : "outline"}
            size="sm"
            onClick={() => { setTab("login"); setError(""); }}
          >
            Institutional Sign In
          </Button>
          <Button
            variant={tab === "register" ? "primary" : "outline"}
            size="sm"
            onClick={() => { setTab("register"); setError(""); }}
          >
            Register Hospital Node
          </Button>
        </div>

        {error && (
          <div
            style={{
              background: "var(--crimson-subtle)",
              color: "var(--crimson-primary)",
              padding: "0.75rem 1rem",
              borderRadius: "var(--radius-md)",
              marginBottom: "1.25rem",
              fontSize: "0.85rem",
              border: "1px solid rgba(239, 68, 68, 0.3)",
            }}
          >
            {error}
          </div>
        )}

        {tab === "login" ? (
          <form onSubmit={handleLogin} className="flex-col gap-2">
            <Input
              label="Username or Email"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. admin or hosp_a_lead"
              required
              icon={<IconUser size={18} />}
            />

            <Input
              label="Password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              required
              icon={<IconLock size={18} />}
            />

            <Button
              type="submit"
              variant="glow"
              loading={loading}
              style={{ width: "100%", marginTop: "1rem" }}
            >
              Sign In to Medical Workspace
            </Button>
          </form>
        ) : (
          <form onSubmit={handleRegister} className="flex-col gap-2">
            <Input
              label="Choose Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. dr_meredith"
              required
              icon={<IconUser size={18} />}
            />

            <Input
              label="Institutional Email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="lead@hospital.org"
              required
              icon={<IconHospital size={18} />}
            />

            <Input
              label="Institution Legal Name"
              value={institutionName}
              onChange={(e) => setInstitutionName(e.target.value)}
              placeholder="Mount Sinai Health System"
              required
            />

            <div className="form-group">
              <label className="form-label">Role Assignment</label>
              <select
                className="form-select"
                value={role}
                onChange={(e) => setRole(e.target.value)}
              >
                <option value="HOSPITAL_STAFF">Hospital Node Clinician (Local Training)</option>
                <option value="ADMIN">Central Flower Coordinator Admin</option>
                <option value="AUDITOR">Clinical Quality & Ethics Auditor</option>
              </select>
            </div>

            <Input
              label="Password (min 6 characters)"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              required
              icon={<IconLock size={18} />}
            />

            <Button
              type="submit"
              variant="primary"
              loading={loading}
              style={{ width: "100%", marginTop: "1rem" }}
            >
              Provision Account & Generate Keys
            </Button>
          </form>
        )}

        {/* Quick Demo Credentials Autofill */}
        <div style={{ marginTop: "1.75rem", paddingTop: "1.25rem", borderTop: "1px solid var(--border-subtle)" }}>
          <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.6rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            1-Click Demo Accounts (Database Pre-Seeded):
          </p>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem" }}>
            <Button
              variant="outline"
              size="sm"
              onClick={() => autofillCredentials("admin", "Admin@FedMed2026!")}
            >
              Coordinator Admin
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => autofillCredentials("hosp_a_lead", "HospA@FedMed2026!")}
            >
              Hospital A (Mt. Sinai)
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => autofillCredentials("hosp_b_lead", "HospB@FedMed2026!")}
            >
              Hospital B (Hopkins)
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => autofillCredentials("auditor", "Audit@FedMed2026!")}
            >
              Auditor
            </Button>
          </div>
        </div>
      </Card>
    </div>
  );
}
