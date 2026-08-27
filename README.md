# 🏥 FedMed: Cross-Silo Federated Learning Engine

> **Privacy-Preserving Machine Learning (PPML) for Collaborative Healthcare AI**

FedMed is a **privacy-first cross-silo federated learning framework** designed to enable multiple hospitals to collaboratively train a **brain-tumor segmentation model** without sharing raw patient MRI data with a centralized server.

Instead of transferring sensitive medical data to a central location, each hospital performs model training locally. Only model updates are transmitted to the federated server, where they are aggregated to improve the global model.

FedMed combines **Federated Learning, 3D medical-image segmentation, Homomorphic Encryption, Differential Privacy, secure communication, and real-time monitoring** into a unified healthcare AI architecture.

---

# 🎯 Problem Statement

Training highly accurate machine-learning models for rare diseases requires access to large and diverse patient datasets.

However, hospitals cannot freely share raw patient data because of strict privacy and data-protection requirements.

FedMed addresses this challenge by enabling three geographically distributed hospitals to collaboratively train a brain-tumor segmentation model while keeping their raw MRI datasets within their respective hospital environments.

### Traditional Approach

```text
Hospital 1 ──┐
Hospital 2 ──┼──> Central Server
Hospital 3 ──┘

       Raw Patient MRI Data
              ↓
       Privacy Risk
```

### FedMed Approach

```text
Hospital 1 ──> Local Training ──┐
                                │
Hospital 2 ──> Local Training ──┼──> Secure Aggregation
                                │
Hospital 3 ──> Local Training ──┘
                                      ↓
                              Global Model
                                      ↓
                           Updated Model to Nodes
```

**Raw MRI data remains within the hospital nodes.**

---

# 🚀 Key Objectives

* Enable collaborative machine-learning training across multiple hospitals.
* Keep raw patient MRI data within local hospital environments.
* Train a 3D U-Net model for brain-tumor segmentation.
* Orchestrate distributed training using Federated Learning.
* Protect model updates using Homomorphic Encryption.
* Apply Differential Privacy to strengthen privacy protection.
* Establish secure server-to-node communication using gRPC and TLS.
* Monitor distributed training through a real-time React dashboard.
* Compare centralized and federated model performance.
* Demonstrate resilience when a hospital node becomes unavailable.

---

# 🧠 Key Modules

## 1. Federated Learning Framework

FedMed uses a federated learning framework such as **Flower** to orchestrate decentralized model training.

Responsibilities include:

* Central server orchestration
* Hospital client management
* Local model training
* Model update exchange
* FedAvg aggregation
* Multi-round federated training
* Client participation management
* Hospital-node failure handling

---

## 2. Computer Vision & Medical AI

The machine-learning component uses **PyTorch and MONAI** to develop a 3D medical-image segmentation pipeline.

### Model

**3D U-Net**

The model is designed for segmenting medical imagery such as MRI scans.

The ML pipeline includes:

```text
MRI Dataset
     ↓
Preprocessing
     ↓
Normalization
     ↓
Augmentation
     ↓
3D U-Net
     ↓
Local Training
     ↓
Segmentation Mask
     ↓
Dice Score / Performance Evaluation
```

---

## 3. Privacy & Encryption

FedMed incorporates privacy-preserving techniques to protect model updates.

### TenSEAL

TenSEAL is used to explore **Homomorphic Encryption**, enabling mathematical operations to be performed on encrypted model updates.

```text
Local Model Update
        ↓
Encryption
        ↓
Encrypted Update
        ↓
Federated Server
        ↓
Encrypted Aggregation
        ↓
Updated Global Model
```

### Differential Privacy

Differential Privacy is introduced to add controlled statistical noise to model updates and strengthen protection against potential model-inversion attacks.

---

## 4. Distributed Systems

FedMed uses distributed communication technologies to connect the central federated server with hospital nodes.

### Technologies

* gRPC
* TLS
* WebSocket
* Secure server-node communication

The distributed layer handles:

* Hospital-node registration
* Model communication
* Secure requests/responses
* Node status
* Retry mechanisms
* Timeout handling
* Node failure recovery
* Real-time metrics streaming

---

## 5. Training Dashboard

A React-based dashboard provides visibility into the federated training process.

The dashboard is designed to display:

* Global training progress
* Local hospital-node status
* Training loss
* Accuracy metrics
* Federated rounds
* Global vs local performance
* Privacy/security status
* MRI segmentation masks
* Experiment history
* Training convergence

---

# 🏗️ System Architecture

```text
                         FEDMED ARCHITECTURE

              ┌─────────────────────────────────┐
              │       Federated Server           │
              │                                 │
              │  Flower + Aggregation + APIs    │
              └───────────────┬─────────────────┘
                              │
                 Secure Communication
                    gRPC + TLS
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
 ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
 │   Hospital 1   │  │   Hospital 2   │  │   Hospital 3   │
 │                │  │                │  │                │
 │ Private MRI    │  │ Private MRI    │  │ Private MRI    │
 │      ↓         │  │      ↓         │  │      ↓         │
 │  Local Model   │  │  Local Model   │  │  Local Model   │
 │      ↓         │  │      ↓         │  │      ↓         │
 │ Model Update   │  │ Model Update   │  │ Model Update   │
 └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
                       Secure Updates
                              ↓
                    ┌──────────────────┐
                    │ Privacy Layer    │
                    │                  │
                    │ TenSEAL + DP     │
                    └────────┬─────────┘
                             │
                             ▼
                    Secure Aggregation
                             │
                             ▼
                       Global Model
                             │
                             ▼
                    Hospital Nodes
                             │
                             ▼
                 ┌──────────────────────┐
                 │   React Dashboard    │
                 │                      │
                 │ Loss • Accuracy      │
                 │ Rounds • Node Status │
                 │ Segmentation Results │
                 └──────────────────────┘
```

---

# 🔄 Federated Training Workflow

```text
                  Global Model
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
         Hospital 1 Hospital 2 Hospital 3
             │         │         │
             ▼         ▼         ▼
       Local Training at Each Hospital
             │         │         │
             ▼         ▼         ▼
       Local Model Updates
             │         │         │
             └─────────┼─────────┘
                       ▼
                 Encryption / DP
                       │
                       ▼
                Secure Aggregation
                       │
                       ▼
                  FedAvg Update
                       │
                       ▼
                  Global Model
                       │
                       └───────────────┐
                                       │
                                       ▼
                              Next Training Round
```

---

# 📊 Centralized vs Federated Learning

FedMed evaluates the federated approach against a centralized baseline.

| Approach         | Data Location   | Training                         | Privacy                     |
| ---------------- | --------------- | -------------------------------- | --------------------------- |
| Centralized      | Central server  | Centralized                      | Raw data is transferred     |
| Federated        | Local hospitals | Distributed                      | Raw data remains local      |
| FedMed + Privacy | Local hospitals | Distributed + Secure Aggregation | Enhanced privacy protection |

The project aims to demonstrate that the federated model can approach the performance of a centralized baseline while avoiding the transfer of raw MRI datasets to the central server.

---

# 🛡️ Privacy Architecture

FedMed uses multiple privacy-preserving mechanisms:

```text
             Patient MRI Data
                     │
                     ▼
             Hospital Environment
                     │
              Local Training
                     │
                     ▼
             Model Weight Update
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   Homomorphic             Differential
    Encryption                Privacy
          │                     │
          └──────────┬──────────┘
                     ▼
              Secure Update
                     │
                     ▼
             Federated Server
                     │
                     ▼
              Aggregation
                     │
                     ▼
               Global Model
```

The architecture is designed around the principle that **raw patient MRI data should not be transmitted to the central federated server**.

---

# 🖥️ Technology Stack

## Machine Learning

* Python
* PyTorch
* MONAI
* 3D U-Net
* BraTS / public MRI datasets

## Federated Learning

* Flower
* FedAvg
* Federated Server
* Hospital Client Nodes

## Privacy & Security

* TenSEAL
* Homomorphic Encryption
* Differential Privacy
* Secure Aggregation
* TLS

## Distributed Systems

* gRPC
* WebSocket
* REST APIs

## Frontend

* React
* Recharts
* Data visualization

## DevOps & Testing

* Docker
* Docker Compose
* GitHub Actions
* Pytest
* CI/CD

## Version Control

* Git
* GitHub
* Feature Branch Workflow
* Pull Requests

---

# 📁 Project Structure

```text
FedMed/
│
├── backend/
│   ├── api/
│   ├── grpc/
│   ├── services/
│   └── websocket/
│
├── frontend/
│   ├── components/
│   ├── pages/
│   └── dashboard/
│
├── ml/
│   ├── models/
│   │   └── unet3d.py
│   ├── preprocessing/
│   ├── training/
│   └── evaluation/
│
├── federated/
│   ├── server/
│   ├── clients/
│   ├── strategies/
│   └── aggregation/
│
├── security/
│   ├── encryption/
│   ├── differential_privacy/
│   └── certificates/
│
├── tests/
│
├── docs/
│
├── docker/
│
├── requirements.txt
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

# 📅 Development Roadmap

## Week 1 — Foundation

### PPML Engineering

* Establish centralized 3D U-Net baseline.
* Configure PyTorch and MONAI.
* Prepare public MRI dataset.
* Begin medical-image preprocessing.
* Establish baseline performance metrics.

### Distributed Systems

* Set up Flower framework.
* Create federated server.
* Configure three mock hospital nodes.
* Establish initial node communication.

---

## Week 2 — Federated Training

### PPML Engineering

* Partition the dataset across three hospital nodes.
* Implement local training.
* Implement model broadcasting.
* Implement FedAvg aggregation.
* Run multiple federated training rounds.

### Distributed Systems

* Implement gRPC communication.
* Configure TLS certificates.
* Secure server-to-hospital communication.
* Add node timeout and resilience mechanisms.

### Mid-Project Review

The system should demonstrate:

* Federated model convergence.
* Comparison with the centralized baseline.
* Raw data remaining within hospital nodes.
* Recovery when a hospital node becomes unavailable.

---

## Week 3 — Privacy & Live Monitoring

### Privacy

* Integrate TenSEAL.
* Encrypt model updates at the client.
* Perform aggregation on encrypted updates.
* Validate encrypted communication and aggregation.

### Monitoring

* Stream training metrics.
* Implement WebSocket metrics endpoint.
* Connect metrics to the React dashboard.
* Display loss and accuracy progression.

---

## Week 4 — Differential Privacy & Finalization

### Privacy

* Implement Differential Privacy.
* Add controlled noise to model updates.
* Validate privacy-preserving training behavior.

### Dashboard

* Training loss visualization.
* Accuracy/convergence visualization.
* Federated round monitoring.
* Hospital-node status.
* MRI tumor segmentation visualization.
* Final experiment comparison.

### Final Review

Demonstrate:

* Three-hospital federated training.
* Secure model-update aggregation.
* Differential Privacy.
* MRI tumor segmentation.
* Centralized vs federated comparison.
* Real-time training dashboard.

---

# 🧪 Testing Strategy

FedMed will use automated and integration testing across all major components.

### Machine Learning Tests

* Dataset loading
* MRI preprocessing
* Model initialization
* Local training
* Segmentation output
* Dice-score evaluation

### Federated Learning Tests

* Client initialization
* Server-client communication
* Model broadcasting
* Local model updates
* FedAvg aggregation
* Multi-round training
* Client dropout handling

### Security Tests

* Encryption/decryption
* Encrypted tensor handling
* Secure aggregation
* Differential Privacy
* TLS communication

### Backend Tests

* API endpoints
* gRPC communication
* WebSocket metrics
* Retry mechanisms
* Timeout handling

### Frontend Tests

* Dashboard rendering
* Metrics visualization
* Node status
* Training progress
* Segmentation visualization

---

# 🐳 Docker

FedMed is designed to support containerized development and deployment.

### Build

```bash
docker build -t fedmed .
```

### Run

```bash
docker compose up --build
```

Docker Compose can be used to simulate the federated environment with multiple services representing the central server and hospital nodes.

---

# 🔧 Installation

## Clone Repository

```bash
git clone <repository-url>

cd FedMed
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

The exact startup commands will be finalized as the individual services are integrated.

The expected execution flow is:

```text
1. Start Federated Server
          ↓
2. Start Hospital Node 1
          ↓
3. Start Hospital Node 2
          ↓
4. Start Hospital Node 3
          ↓
5. Start Training
          ↓
6. Aggregate Model Updates
          ↓
7. Stream Metrics
          ↓
8. Open React Dashboard
```

---

# 🌐 Expected Dashboard

The FedMed dashboard is designed to provide a centralized view of the distributed training environment.

### Dashboard Components

```text
┌────────────────────────────────────────────────────┐
│                  FedMed Dashboard                  │
├────────────────────────────────────────────────────┤
│                                                    │
│  Global Accuracy       Training Round              │
│      92.4%                   15                     │
│                                                    │
├────────────────────────────────────────────────────┤
│                                                    │
│       Global Loss / Accuracy Chart                 │
│                                                    │
├────────────────────────────────────────────────────┤
│                                                    │
│ Hospital 1     🟢 Online     Training              │
│ Hospital 2     🟢 Online     Training              │
│ Hospital 3     🟢 Online     Training              │
│                                                    │
├────────────────────────────────────────────────────┤
│                                                    │
│          MRI Tumor Segmentation Mask               │
│                                                    │
└────────────────────────────────────────────────────┘
``'


# 🔀 Development Workflow

FedMed follows a structured feature-branch and pull-request workflow.

```text
                    main
                     │
                     │
                  develop
                     │
       ┌─────────────┼─────────────┐
       │             │             │
       ▼             ▼             ▼
   feature/      feature/      feature/
   frontend      backend       ml
       │             │             │
       └─────────────┼─────────────┘
                     ▼
                Pull Request
                     │
                     ▼
                Code Review
                     │
                     ▼
                  develop
                     │
                  Testing
                     │
                     ▼
                   main
```

### Development Principles

* No direct development on `main`.
* Use feature branches.
* Create Pull Requests for integration.
* Review code before merging.
* Run tests before merging.
* Keep commits focused and meaningful.
* Maintain documentation alongside development.

---

# 📈 Project Milestones

### Phase 1 — Foundation

* [ ] Repository and architecture setup
* [ ] PyTorch + MONAI environment
* [ ] 3D U-Net baseline
* [ ] Flower server/client skeleton
* [ ] Three hospital-node setup
* [ ] TenSEAL environment
* [ ] gRPC project structure
* [ ] React dashboard skeleton

### Phase 2 — Federated Training

* [ ] Local training
* [ ] Dataset partitioning
* [ ] FedAvg aggregation
* [ ] Multi-round training
* [ ] Hospital-node communication
* [ ] Node failure handling

### Phase 3 — Privacy

* [ ] Homomorphic Encryption
* [ ] Encrypted model updates
* [ ] Secure aggregation
* [ ] Differential Privacy
* [ ] Privacy validation

### Phase 4 — Integration

* [ ] gRPC + Federated Learning integration
* [ ] WebSocket metrics
* [ ] React dashboard integration
* [ ] MRI segmentation visualization
* [ ] Centralized vs federated comparison

### Phase 5 — Final Validation

* [ ] Full three-hospital simulation
* [ ] Security audit
* [ ] Regression testing
* [ ] User Acceptance Testing
* [ ] Final deployment
* [ ] Final demonstration

---

# 📊 Success Criteria

FedMed will be considered successfully integrated when the system can demonstrate:

```text
✓ Three hospital nodes
✓ Local MRI training
✓ No raw MRI transfer to central server
✓ Federated model aggregation
✓ Multiple training rounds
✓ Encrypted model updates
✓ Differential Privacy
✓ Secure server-node communication
✓ Node failure recovery
✓ Global training metrics
✓ MRI tumor segmentation
✓ Centralized vs federated comparison
✓ React monitoring dashboard
```

---

# 🔐 Privacy-First Design

FedMed is built around a privacy-first architecture in which sensitive medical datasets remain within their originating hospital environments.

The system focuses on protecting both:

* **Patient data**
* **Model updates**

through a combination of:

**Federated Learning + Homomorphic Encryption + Differential Privacy + Secure Communication**

> **Train together. Keep data private. Improve healthcare AI.**

---

# 📌 Project Status

## 🚧 Under Active Development

FedMed is being developed as a multi-stage federated healthcare AI system.

The final system is intended to demonstrate a complete privacy-preserving workflow combining:

* Cross-silo Federated Learning
* 3D medical-image segmentation
* Secure aggregation
* Homomorphic Encryption
* Differential Privacy
* gRPC communication
* Real-time metrics
* React monitoring
* Centralized vs federated evaluation

---

# 🏁 Final Deliverable

The final FedMed demonstration will showcase a **three-hospital federated learning workflow** for brain-tumor MRI segmentation.

The demonstration will cover:

```text
Three Hospital Nodes
        ↓
Local Model Training
        ↓
Encrypted Model Updates
        ↓
Secure Aggregation
        ↓
Global Model
        ↓
Multiple Federated Rounds
        ↓
Performance Evaluation
        ↓
MRI Tumor Segmentation
        ↓
React Dashboard
```

### FedMed

**A privacy-first approach to collaborative healthcare AI.**

---

## 📜 License

This project is intended for educational and research purposes.

License information will be finalized by the project team.
