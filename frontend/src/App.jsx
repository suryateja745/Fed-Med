import React, { useState, useEffect } from "react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { Navbar } from "./components/common/Navbar";
import { Footer } from "./components/common/Footer";
import { LandingPage } from "./pages/LandingPage";
import { AuthPage } from "./pages/AuthPage";
import { CoordinatorDashboard } from "./pages/CoordinatorDashboard";
import { HospitalPortal } from "./pages/HospitalPortal";
import { SimulationPage } from "./pages/SimulationPage";
import { DesignSandbox } from "./pages/DesignSandbox";
import { api } from "./services/api";

function AppContent() {
  const { user, logout } = useAuth();
  const [activeTab, setActiveTab] = useState("home");
  const [isBackendConnected, setIsBackendConnected] = useState(false);

  // Poll backend health status
  useEffect(() => {
    let mounted = true;
    async function checkStatus() {
      const healthy = await api.checkHealth();
      if (mounted) setIsBackendConnected(healthy);
    }

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="app-container">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        onSelectTab={setActiveTab}
        currentUser={user}
        onLogout={logout}
        isBackendConnected={isBackendConnected}
      />

      {/* Main Content View Switcher */}
      <main className="main-content">
        {activeTab === "home" && <LandingPage onNavigate={setActiveTab} />}
        {activeTab === "auth" && <AuthPage onNavigate={setActiveTab} />}
        {activeTab === "coordinator" && <CoordinatorDashboard />}
        {activeTab === "hospital" && <HospitalPortal />}
        {activeTab === "simulation" && <SimulationPage />}
        {activeTab === "sandbox" && <DesignSandbox />}
      </main>

      {/* Footer */}
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
