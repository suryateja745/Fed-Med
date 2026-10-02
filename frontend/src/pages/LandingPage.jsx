import React, { useRef } from "react";
import {
  IconBrain,
  IconShieldCheck,
  IconLock,
  IconHospital,
  IconActivity,
  IconServer,
  IconCheckCircle,
} from "../components/common/Icons";

export function LandingPage({ onNavigate }) {
  const problemRef = useRef(null);
  const solutionRef = useRef(null);

  const scrollToProblem = () => {
    if (problemRef.current) {
      problemRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  const scrollToSolution = () => {
    if (solutionRef.current) {
      solutionRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", position: "relative" }}>
      {/* ====================================================================
          1. HERO SECTION
          Clean, minimal white hero with centered display headline and pill CTAs
          ==================================================================== */}
      <section className="hero-antigravity-section">
        {/* Centered Content Container */}
        <div className="hero-content-wrapper">
          {/* Centered Logo / Feature Pill */}
          <div className="hero-pill-badge">
            <span
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: "var(--brand-blue)",
                display: "inline-block",
              }}
            />
            <span style={{ color: "var(--brand-blue)", fontWeight: 700 }}>FedMed AI</span>
            <span style={{ color: "var(--border-medium)" }}>•</span>
            <span>Next-Gen Decentralized Medicine</span>
          </div>

          {/* Main Display Headline (Google Antigravity Style) */}
          <h1 className="hero-main-title">
            Decentralized 3D Brain Tumor MRI Federated AI Platform
          </h1>

          {/* Narrative Subtitle */}
          <p className="hero-subtitle-desc">
            Train world-class volumetric segmentation models across global hospital networks
            without transferring a single patient record. Verifiable zero raw data egress,
            zero-trust homomorphic encryption, and consensus-driven clinical intelligence.
          </p>

          {/* Action CTAs */}
          <div className="hero-cta-group">
            <button
              className="btn-pill-primary"
              onClick={() => onNavigate("auth")}
            >
              <span>Start with FedMed</span>
              <span style={{ fontSize: "1.1rem" }}>&rarr;</span>
            </button>

            <button
              className="btn-pill-secondary"
              onClick={scrollToProblem}
            >
              Explore Architecture &darr;
            </button>
          </div>

          {/* Quick Real-Time Metrics Strip */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "2rem",
              flexWrap: "wrap",
              justifyContent: "center",
              paddingTop: "1.5rem",
              marginTop: "1rem",
              fontSize: "0.85rem",
              fontFamily: "var(--font-mono)",
              color: "var(--text-muted)",
            }}
          >
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--emerald-primary)" }} />
              <strong>0.00%</strong> Raw Patient Egress
            </span>
            <span style={{ color: "var(--border-subtle)" }}>|</span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--brand-blue)" }} />
              <strong>88.42%</strong> BraTS Mean Dice
            </span>
            <span style={{ color: "var(--border-subtle)" }}>|</span>
            <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "var(--violet-primary)" }} />
              <strong>AES-256-GCM</strong> Homomorphic Shards
            </span>
          </div>
        </div>
      </section>

      {/* ====================================================================
          2. THE PROBLEM WITH TRADITIONAL HEALTHCARE AI
          High-end illustration, structured clinical flaw cards, regulatory failure breakdown
          ==================================================================== */}
      <section ref={problemRef} className="section-problem-wrapper">
        <div className="section-header-centered">
          <div className="section-tag-pill danger">
            <span>The Healthcare AI Dilemma</span>
          </div>
          <h2 className="section-title">
            The Fatal Flaw of Centralized Healthcare AI
          </h2>
          <p className="section-description">
            Traditional machine learning forces clinical institutions to consolidate sensitive
            patient scans into a single cloud repository — creating immense regulatory, privacy,
            and institutional liabilities.
          </p>
        </div>

        {/* Workflow Card with Generated Diagram */}
        <div className="workflow-diagram-card">
          <div className="workflow-image-frame">
            <img
              src="/@fs/C:/Users/ASUS/.gemini/antigravity-ide/brain/935aba3a-aaa0-4174-95e7-849c96726697/traditional_architecture_flaw_1790436334389.jpg"
              alt="The Flaws of Traditional Centralized Healthcare AI Workflow"
              loading="lazy"
            />
          </div>

          {/* Three Core Problem Breakdown Cards */}
          <div className="problem-breakdown-grid">
            <div className="problem-point-card">
              <div className="problem-icon-wrapper">
                <IconShieldCheck size={20} />
              </div>
              <h3 className="problem-point-title">HIPAA & GDPR Violations</h3>
              <p className="problem-point-desc">
                Clinical data protection regulations strictly forbid transferring raw patient DICOM/NIfTI
                imaging across hospital boundaries without burdensome consent and legal exposure.
              </p>
            </div>

            <div className="problem-point-card">
              <div className="problem-icon-wrapper">
                <IconLock size={20} />
              </div>
              <h3 className="problem-point-title">Catastrophic Egress Leaks</h3>
              <p className="problem-point-desc">
                Centralized databases create a single high-value target for ransomware and state-sponsored
                breaches. Once raw scans leave hospital storage, chain of custody is permanently broken.
              </p>
            </div>

            <div className="problem-point-card">
              <div className="problem-icon-wrapper">
                <IconHospital size={20} />
              </div>
              <h3 className="problem-point-title">Siloed Data Starvation</h3>
              <p className="problem-point-desc">
                To prevent leaks, hospitals lock their clinical datasets in isolated PACS silos. AI models
                trained on single-hospital data suffer catastrophic generalization errors on external patients.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ====================================================================
          3. THE FEDMED SOLUTION: PRIVACY-PRESERVING FEDERATED LEARNING
          3 Google-style flashcards with clean white aesthetic, badges & generated illustrations
          ==================================================================== */}
      <section ref={solutionRef} className="section-solution-wrapper">
        <div className="section-header-centered">
          <div className="section-tag-pill success">
            <span>The Federated Breakthrough</span>
          </div>
          <h2 className="section-title">
            How FedMed Solves the Problem
          </h2>
          <p className="section-description">
            Instead of bringing private patient data to the model, FedMed dispatches the AI model
            directly to the data. Local hospital training, encrypted gradient transmission, and global consensus.
          </p>
        </div>

        {/* Google-Style Flash Cards Grid */}
        <div className="flashcards-grid">
          {/* Card 1: In-Situ Local Training */}
          <div className="google-flash-card">
            <div className="flash-card-image-box">
              <img
                src="/@fs/C:/Users/ASUS/.gemini/antigravity-ide/brain/935aba3a-aaa0-4174-95e7-849c96726697/card_local_training_1790436353507.jpg"
                alt="Local in-situ hospital AI training"
                loading="lazy"
              />
            </div>
            <div className="flash-card-body">
              <div className="section-tag-pill success" style={{ width: "fit-content" }}>
                <span>In-Situ Training</span>
              </div>
              <h3 className="flash-card-title">
                Zero Patient Data Egress
              </h3>
              <p className="flash-card-text">
                Every 3D brain tumor MRI scan remains strictly locked on local hospital PACS storage.
                The 3D U-Net trains locally on native hospital GPU hardware, computing weight deltas
                without exposing patient identities.
              </p>
              <div className="flash-card-metrics-row">
                <span>Data Egress Rate</span>
                <strong style={{ color: "var(--emerald-light)" }}>0.00% Transferred</strong>
              </div>
            </div>
          </div>

          {/* Card 2: Homomorphic Encryption */}
          <div className="google-flash-card">
            <div className="flash-card-image-box">
              <img
                src="/@fs/C:/Users/ASUS/.gemini/antigravity-ide/brain/935aba3a-aaa0-4174-95e7-849c96726697/card_encrypted_aggregation_1790436378879.jpg"
                alt="Homomorphic encrypted model weight updates"
                loading="lazy"
              />
            </div>
            <div className="flash-card-body">
              <div className="section-tag-pill brand" style={{ width: "fit-content" }}>
                <span>Cryptographic Security</span>
              </div>
              <h3 className="flash-card-title">
                Encrypted Weight Transmission
              </h3>
              <p className="flash-card-text">
                Only mathematical gradient parameters leave the hospital. Every weight vector is
                sealed with AES-256-GCM authenticated ciphertexts and differential privacy noise,
                mathematically thwarting reconstruction attacks.
              </p>
              <div className="flash-card-metrics-row">
                <span>Encryption Standard</span>
                <strong style={{ color: "var(--brand-blue)" }}>AES-256-GCM + DP</strong>
              </div>
            </div>
          </div>

          {/* Card 3: Central Consensus Aggregation */}
          <div className="google-flash-card">
            <div className="flash-card-image-box">
              <img
                src="/@fs/C:/Users/ASUS/.gemini/antigravity-ide/brain/935aba3a-aaa0-4174-95e7-849c96726697/card_global_consensus_1790436401762.jpg"
                alt="Central federated learning coordinator consensus"
                loading="lazy"
              />
            </div>
            <div className="flash-card-body">
              <div className="section-tag-pill danger" style={{ width: "fit-content", background: "var(--violet-subtle)", color: "var(--violet-primary)", borderColor: "var(--violet-border)" }}>
                <span>Global Consensus</span>
              </div>
              <h3 className="flash-card-title">
                Consortium Coordinator FedAvg
              </h3>
              <p className="flash-card-text">
                The central coordinator securely aggregates encrypted updates from Wockhardt Hospital,
                Mayo Clinic, KIMS Kingsway Hospital, and AIIMS (AIMS) using Federated Averaging (FedAvg), yielding a
                globally superior 3D segmentation model for all members.
              </p>
              <div className="flash-card-metrics-row">
                <span>BraTS Benchmark</span>
                <strong style={{ color: "var(--violet-primary)" }}>88.42% Mean Dice</strong>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ====================================================================
          4. CALL TO ACTION / WORKSPACE LAUNCH BANNER
          Clean, minimal card inviting clinical researchers to test or deploy
          ==================================================================== */}
      <section
        style={{
          margin: "4rem 0 2rem",
          background: "linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-2xl)",
          padding: "3.5rem 2rem",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "1.25rem",
        }}
      >
        <div className="section-tag-pill brand">
          <span>Get Started Today</span>
        </div>
        <h2 className="section-title" style={{ fontSize: "2.25rem" }}>
          Ready to Train Collaborative Medical AI?
        </h2>
        <p className="section-description" style={{ maxWidth: "600px" }}>
          Connect your hospital workstation or deploy the central coordinator engine
          in less than five minutes. Zero configuration, complete data protection.
        </p>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", justifyContent: "center", marginTop: "0.5rem" }}>
          <button
            className="btn-pill-primary"
            onClick={() => onNavigate("hospital")}
          >
            Deploy Hospital Node -&gt;
          </button>
          <button
            className="btn-pill-secondary"
            onClick={() => onNavigate("coordinator")}
          >
            Launch Coordinator Command
          </button>
        </div>
      </section>
    </div>
  );
}
