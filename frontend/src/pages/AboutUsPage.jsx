import React from "react";
import {
  IconBrain,
  IconShieldCheck,
  IconLock,
  IconHospital,
  IconActivity,
  IconServer,
  IconCheckCircle,
} from "../components/common/Icons";

export function AboutUsPage({ onNavigate }) {
  const partners = [
    {
      name: "Wockhardt Hospital",
      location: "Nagpur / Mumbai, India",
      specialty: "Neuro-Oncology & Volumetric Surgery",
      cohort: "1,180 Patients",
    },
    {
      name: "Mayo Clinic",
      location: "Rochester, MN, USA",
      specialty: "Neuro-Oncology & Volumetric Imaging",
      cohort: "1,420 Patients",
    },
    {
      name: "KIMS Kingsway Hospital",
      location: "Nagpur, India",
      specialty: "Multi-Modal Brain Tumor AI Translation",
      cohort: "1,050 Patients",
    },
    {
      name: "AIIMS (AIMS)",
      location: "Nagpur / New Delhi, India",
      specialty: "Clinical BraTS Oncology Benchmarking",
      cohort: "1,260 Patients",
    },
  ];

  const pillars = [
    {
      icon: <IconShieldCheck size={28} style={{ color: "var(--brand-blue)" }} />,
      title: "Zero Patient Data Egress",
      description:
        "Every patient DICOM or NIfTI MRI scan remains strictly resident on local hospital PACS storage. Raw medical records never traverse external internet channels.",
    },
    {
      icon: <IconLock size={28} style={{ color: "var(--emerald-light)" }} />,
      title: "Cryptographic Privacy",
      description:
        "Weight updates are shielded using homomorphic masking, differential privacy noise, and authenticated AES-256-GCM encryption with HMAC-SHA256 signatures.",
    },
    {
      icon: <IconActivity size={28} style={{ color: "var(--violet-primary)" }} />,
      title: "Clinical-Grade Performance",
      description:
        "Benchmarked on the international BraTS dataset with 88.42% Mean Dice across Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) sub-regions.",
    },
  ];

  return (
    <div className="about-page-container">
      {/* 1. HERO HEADER */}
      <section className="about-hero">
        <div className="section-tag-pill brand">
          <span>Our Clinical Mission</span>
        </div>
        <h1 className="hero-main-title" style={{ fontSize: "3rem" }}>
          Democratizing Medical AI Through Zero-Trust Collaboration
        </h1>
        <p className="hero-subtitle-desc">
          FedMed was founded on a simple medical conviction: clinical artificial intelligence
          should be trained on the world's most diverse patient populations without compromising
          a single patient's privacy or institutional trust.
        </p>

        <div style={{ display: "flex", gap: "1rem", marginTop: "1rem" }}>
          <button
            className="btn-pill-primary"
            onClick={() => onNavigate("hospital")}
          >
            Join the Consortium -&gt;
          </button>
          <button
            className="btn-pill-secondary"
            onClick={() => onNavigate("home")}
          >
            Back to Overview
          </button>
        </div>
      </section>

      {/* 2. CORE ARCHITECTURAL PILLARS */}
      <section>
        <div style={{ textAlign: "center", marginBottom: "2.5rem" }}>
          <h2 className="section-title" style={{ fontSize: "2rem" }}>
            The Three Pillars of FedMed
          </h2>
          <p className="section-description" style={{ maxWidth: "600px", margin: "0.5rem auto 0" }}>
            Rigorous mathematical guarantees combined with open-source clinical medical imaging standards.
          </p>
        </div>

        <div className="about-values-grid">
          {pillars.map((pillar, idx) => (
            <div key={idx} className="about-value-card">
              <div
                style={{
                  width: "56px",
                  height: "56px",
                  borderRadius: "var(--radius-lg)",
                  background: "var(--bg-secondary)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid var(--border-subtle)",
                }}
              >
                {pillar.icon}
              </div>
              <h3 className="problem-point-title" style={{ fontSize: "1.25rem" }}>
                {pillar.title}
              </h3>
              <p className="problem-point-desc" style={{ fontSize: "0.95rem" }}>
                {pillar.description}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* 3. CONSORTIUM PARTICIPANTS */}
      <section>
        <div style={{ textAlign: "center", marginBottom: "2.5rem" }}>
          <div className="section-tag-pill success" style={{ marginBottom: "0.5rem" }}>
            <span>Global Medical Network</span>
          </div>
          <h2 className="section-title" style={{ fontSize: "2rem" }}>
            Connected Hospital Consortia
          </h2>
          <p className="section-description" style={{ maxWidth: "600px", margin: "0.5rem auto 0" }}>
            Leading academic health systems collaboratively refining the global 3D U-Net segmentation weights.
          </p>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "1.5rem" }}>
          {partners.map((partner, idx) => (
            <div
              key={idx}
              style={{
                background: "#ffffff",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-xl)",
                padding: "1.75rem",
                boxShadow: "var(--shadow-card)",
                display: "flex",
                flexDirection: "column",
                gap: "0.75rem",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                  <IconHospital size={20} style={{ color: "var(--brand-blue)" }} />
                  <span style={{ fontWeight: 700, fontSize: "1.1rem", color: "var(--text-primary)" }}>
                    {partner.name}
                  </span>
                </div>
                <span
                  style={{
                    fontSize: "0.7rem",
                    fontFamily: "var(--font-mono)",
                    color: "var(--emerald-light)",
                    background: "var(--emerald-subtle)",
                    padding: "0.2rem 0.5rem",
                    borderRadius: "var(--radius-full)",
                    border: "1px solid var(--emerald-border)",
                    fontWeight: 600,
                  }}
                >
                  ONLINE
                </span>
              </div>

              <div style={{ fontSize: "0.82rem", color: "var(--text-muted)" }}>
                {partner.location}
              </div>

              <div style={{ fontSize: "0.88rem", color: "var(--text-secondary)" }}>
                {partner.specialty}
              </div>

              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  paddingTop: "0.75rem",
                  borderTop: "1px solid var(--border-subtle)",
                  marginTop: "auto",
                  fontSize: "0.8rem",
                  fontFamily: "var(--font-mono)",
                }}
              >
                <span style={{ color: "var(--text-muted)" }}>Verified Cohort:</span>
                <span style={{ fontWeight: 600, color: "var(--text-primary)" }}>{partner.cohort}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* 4. OPEN SCIENTIFIC STACK */}
      <section
        style={{
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-subtle)",
          borderRadius: "var(--radius-2xl)",
          padding: "3rem 2rem",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: "1.5rem",
        }}
      >
        <div className="section-tag-pill brand">
          <span>Standards & Frameworks</span>
        </div>
        <h2 className="section-title" style={{ fontSize: "2rem" }}>
          Built Upon Proven Healthcare Standards
        </h2>
        <p className="section-description" style={{ maxWidth: "680px" }}>
          FedMed leverages the best-of-breed open-source clinical software ecosystem:
          <strong> MONAI Core 1.4</strong> for 3D medical tensor pipelines,
          <strong> Flower FL 1.11</strong> for resilient federated orchestration, and
          <strong> PyTorch</strong> for GPU acceleration.
        </p>

        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "center",
            gap: "2rem",
            paddingTop: "1rem",
            color: "var(--text-secondary)",
            fontSize: "0.95rem",
            fontWeight: 600,
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <IconCheckCircle size={18} style={{ color: "var(--emerald-light)" }} /> MONAI 3D U-Net Architecture
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <IconCheckCircle size={18} style={{ color: "var(--emerald-light)" }} /> Flower FL Protocol Engine
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <IconCheckCircle size={18} style={{ color: "var(--emerald-light)" }} /> BraTS 2024 Benchmark Standard
          </span>
          <span style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
            <IconCheckCircle size={18} style={{ color: "var(--emerald-light)" }} /> HIPAA & GDPR In-Situ Compliance
          </span>
        </div>
      </section>
    </div>
  );
}
