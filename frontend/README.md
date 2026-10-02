# 🌸 FedMed Frontend – React 19 Clinical AI Platform

A modern, publication-grade web application for decentralized 3D Brain Tumor MRI Federated Learning. Built with **React 19**, **Vite**, and a bespoke **Vanilla CSS medical dark theme design system**.

---

## 🎨 Design System & Color Palette

The user interface follows a specialized medical AI aesthetic engineered for clinical researchers, chief medical officers, and hospital ML engineers:

* **Obsidian & Midnight Navy:** `#0A0F1D` (Canvas) and `#0F172A` (Secondary Surfaces)
* **Card Containers:** Glassmorphism `rgba(22, 32, 50, 0.75)` with `backdrop-filter: blur(16px)`
* **Neon Cyan (`#06B6D4`):** Primary medical technology and coordinator accents
* **Bio Emerald (`#10B981`):** High Dice scores, active hospital nodes, and validation metrics
* **Glowing Violet (`#8B5CF6`):** Federated learning rounds and non-IID simulations
* **Danger Crimson (`#EF4444`):** Cryptographic tamper flags and emergency stops
* **Typography:**
  * **Headings:** Google Font **Outfit** (500, 600, 700, 800)
  * **Body & UI:** Google Font **Inter** (300, 400, 500, 600)
  * **Code & Telemetry:** Google Font **JetBrains Mono** (400, 500)

---

## 📁 Directory Structure

```
frontend/
├── index.html                  # HTML5 entry with Google Fonts & dark theme meta
├── package.json                # React 19 & Vite configuration
├── vite.config.js              # Vite server & backend proxy configuration
├── src/
│   ├── assets/                 # Brand assets and graphics
│   ├── components/
│   │   ├── common/             # Reusable UI primitives
│   │   │   ├── Badge.jsx       # Status badges with pulse glow
│   │   │   ├── Button.jsx      # Primary, secondary, glow, outline, danger
│   │   │   ├── Card.jsx        # Glassmorphic container with glow borders
│   │   │   ├── Footer.jsx      # Clinical disclaimer & compliance badges
│   │   │   ├── Icons.jsx       # Self-contained SVG medical AI icons
│   │   │   ├── Input.jsx       # Floating label, icon prefix, error states
│   │   │   ├── Modal.jsx       # Accessible blur backdrop dialog
│   │   │   └── Navbar.jsx      # Sticky navigation, status pill, role badge
│   │   ├── charts/             # Multi-region Dice & loss curves (Commit 25)
│   │   ├── coordinator/        # Cluster matrix & model registry (Commit 24/26)
│   │   └── hospital/           # MRI slice viewer & local monitor (Commit 27/28)
│   ├── context/
│   │   └── AuthContext.jsx     # JWT session persistence & RBAC provider
│   ├── pages/
│   │   ├── LandingPage.jsx     # Page 1: Showcase home & 3D MRI preview
│   │   ├── AuthPage.jsx        # Page 2: Role-based auth & key provisioning
│   │   ├── CoordinatorDashboard.jsx # Page 3: Flower central command center
│   │   ├── HospitalPortal.jsx  # Page 4: Local private node workspace
│   │   ├── SimulationPage.jsx  # Multi-hospital simulation testbed
│   │   └── DesignSandbox.jsx   # Interactive UI design tokens preview
│   ├── services/
│   │   └── api.js              # Fetch client with auto JWT bearer injection
│   ├── styles/
│   │   └── index.css           # Complete Vanilla CSS design system
│   ├── App.jsx                 # Dynamic multi-tab router & state hub
│   └── main.jsx                # React 19 root bootstrap
```

---

## 🚀 Running the Web Application

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start the Development Server
```bash
npm run dev
```
The application will launch at **`http://localhost:5173`**.

All backend API requests (`/api/*`) and real-time WebSocket connections (`/ws/*`) are automatically proxied to the FastAPI server at `http://127.0.0.1:8000`.

---

## 🔑 Pre-Seeded Institutional Demo Accounts

| Role | Username | Password | Linked Node / Facility |
| :--- | :--- | :--- | :--- |
| **Coordinator Admin** | `admin` | `Admin@FedMed2026!` | FedMed Central Command |
| **Hospital A Clinician** | `hosp_a_lead` | `HospA@FedMed2026!` | `NODE-HOSP-A` (Mount Sinai) |
| **Hospital B Clinician** | `hosp_b_lead` | `HospB@FedMed2026!` | `NODE-HOSP-B` (Johns Hopkins) |
| **Hospital C Clinician** | `hosp_c_lead` | `HospC@FedMed2026!` | `NODE-HOSP-C` (Mayo Clinic) |
| **Clinical Auditor** | `auditor` | `Audit@FedMed2026!` | Global Clinical Ethics Board |
