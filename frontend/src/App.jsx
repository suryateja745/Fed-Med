import React, { useState, useEffect } from "react";
import { AuthProvider, useAuth } from "./context/AuthContext";
import { Navbar } from "./components/common/Navbar";
import { Footer } from "./components/common/Footer";
import { LandingPage } from "./pages/LandingPage";
import { AuthPage } from "./pages/AuthPage";
import { CoordinatorDashboard } from "./pages/CoordinatorDashboard";
import { HospitalPortal } from "./pages/HospitalPortal";
import { SimulationPage } from "./pages/SimulationPage";
import { AboutUsPage } from "./pages/AboutUsPage";
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

  // Dynamically update page title in browser tab based on current screen
  useEffect(() => {
    if (activeTab === "auth") {
      document.title = "Authentication Portal";
    } else if (activeTab === "coordinator") {
      document.title = "FedMed | Coordinator Dashboard";
    } else if (activeTab === "hospital") {
      document.title = "FedMed | Hospital Portal";
    } else if (activeTab === "about") {
      document.title = "FedMed | About Us";
    } else {
      document.title = "FedMed | Decentralized 3D MRI Federated Learning Platform";
    }
  }, [activeTab]);

  const isAuthPage = activeTab === "auth";

  return (
    <div className={`app-container ${isAuthPage ? "app-auth-mode" : ""}`}>
      {/* Top Navigation - hidden on Authentication Portal */}
      {!isAuthPage && (
        <Navbar
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          currentUser={user}
          onLogout={logout}
          isBackendConnected={isBackendConnected}
        />
      )}

      {/* Main Content View Switcher */}
      <main className={isAuthPage ? "auth-main-content" : "main-content"}>
        {activeTab === "home" && <LandingPage onNavigate={setActiveTab} />}
        {activeTab === "about" && <AboutUsPage onNavigate={setActiveTab} />}
        {activeTab === "auth" && <AuthPage onNavigate={setActiveTab} />}
        {activeTab === "coordinator" && <CoordinatorDashboard />}
        {activeTab === "hospital" && <HospitalPortal />}
        {activeTab === "simulation" && <SimulationPage />}
      </main>

      {/* Footer - hidden on Authentication Portal */}
      {!isAuthPage && <Footer onNavigate={setActiveTab} />}
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
