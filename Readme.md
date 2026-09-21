# FedMed – Federated Learning with Flower

## Overview

FedMed is a privacy-preserving healthcare machine learning platform designed to allow multiple hospitals to collaboratively train a machine learning model without sharing their private patient data.

The project uses a **3D U-Net model** for brain tumor segmentation from MRI scans.

Instead of collecting all MRI data in one central location, each hospital trains the model using its own local data. The learned model parameters are then shared with a central server, where they are combined to improve a common global model.

This approach is called **Federated Learning**.

---

# What is Machine Learning?

In a normal machine learning system, a model learns from data.

For example, in FedMed:

```text
MRI Scan
   ↓
3D U-Net Model
   ↓
Predicted Tumor Segmentation
```

During training, the model compares its prediction with the correct tumor segmentation mask and adjusts its internal values to improve its performance.

These internal values mainly include:

* Weights
* Biases

After training on more data, the model's weights and biases are updated, allowing the model to make better predictions.

---

# The Problem with Centralized Learning

In traditional machine learning, data from all sources is collected in one central server.

For example:

```text
Hospital A MRI Data ─┐
                     │
Hospital B MRI Data ─┼──→ Central Server → Train Model
                     │
Hospital C MRI Data ─┘
```

The central server receives all patient MRI data and uses the combined dataset to train the model.

However, in healthcare, this creates a major privacy problem.

Hospitals cannot freely share sensitive patient information because medical data is private and may be protected by privacy regulations.

Therefore, sending all patient MRI scans to a single server is not always possible.

---

# What is Federated Learning?

Federated Learning is a machine learning approach where multiple clients collaboratively train a single model without sending their raw data to a central server.

In FedMed, each hospital acts as a **client**.

```text
                    Central Server
                         │
                    Global Model
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     Hospital A      Hospital B      Hospital C
```

The central server sends a copy of the global model to each hospital.

Each hospital then trains that model using its own private MRI data.

```text
Hospital A
Global Model
     ↓
Train using Hospital A's private MRI data
     ↓
Updated Model Parameters


Hospital B
Global Model
     ↓
Train using Hospital B's private MRI data
     ↓
Updated Model Parameters


Hospital C
Global Model
     ↓
Train using Hospital C's private MRI data
     ↓
Updated Model Parameters
```

The important point is:

> **The MRI data remains inside the hospital and is never sent to the central server.**

Only the learned model parameters or model updates are shared.

---

# How Federated Learning Works in FedMed

The complete training process happens in multiple rounds.

## Step 1: Create the Global Model

The central server starts with a global **3D U-Net model**.

Initially, the model contains its starting weights and biases.

```text
Central Server
       │
       ▼
Global 3D U-Net Model
```

---

## Step 2: Send the Model to Hospitals

The central server sends the current global model parameters to the hospital nodes.

```text
                    Global Model
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
     Hospital A      Hospital B      Hospital C
```

Each hospital receives the same starting version of the model.

---

## Step 3: Local Training

Each hospital trains the model locally using its own MRI dataset.

```text
Hospital A:
Global Model → Local Training → Updated Parameters A

Hospital B:
Global Model → Local Training → Updated Parameters B

Hospital C:
Global Model → Local Training → Updated Parameters C
```

At no point are the actual MRI scans transferred to the central server.

---

## Step 4: Send Model Updates

After local training, each hospital has updated the model's weights and biases.

```text
Hospital A → Updated Weights and Biases A
Hospital B → Updated Weights and Biases B
Hospital C → Updated Weights and Biases C
```

These updates are sent back to the central server.

In the FedMed architecture, additional privacy mechanisms such as encryption can be applied before the updates are transmitted.

---

## Step 5: Federated Averaging

The central server combines the updates received from the hospitals.

This process is called **Federated Averaging**, commonly known as **FedAvg**.

For a simple example, imagine one model weight has the following values after local training:

```text
Hospital A → 10
Hospital B → 20
Hospital C → 30
```

A simple average would be:

```text
(10 + 20 + 30) / 3 = 20
```

Therefore, the new global value becomes:

```text
20
```

In a real machine learning model, this process is performed across a very large number of model parameters.

The new global model therefore combines knowledge learned from all participating hospitals.

---

# The Federated Learning Cycle

The complete process can be represented as:

```text
                    CENTRAL SERVER
                          │
                    Global Model
                          │
                          ▼
             Send Model to Hospitals
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        Hospital A   Hospital B   Hospital C
             │            │            │
             ▼            ▼            ▼
        Local Train   Local Train   Local Train
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
                 Model Parameter Updates
                          │
                          ▼
                    Federated Averaging
                          │
                          ▼
                   Updated Global Model
                          │
                          ▼
                    Next Training Round
                          │
                          └──────→ Repeat
```

This cycle continues for multiple federated training rounds until the global model achieves the desired performance.

---

# What is Flower?

**Flower** is the Federated Learning framework used in FedMed.

Flower helps manage and coordinate the Federated Learning process.

Instead of manually writing all the communication and coordination logic between the central server and hospitals, Flower provides the framework required to organize this process.

In FedMed, Flower is responsible for coordinating the following workflow:

```text
Flower Server
      │
      │ Sends Global Model Parameters
      ▼
Flower Clients
(Hospital A, B, C)
      │
      │ Perform Local Training
      ▼
Updated Model Parameters
      │
      │ Send Updates Back
      ▼
Flower Server
      │
      │ Applies Federated Averaging
      ▼
New Global Model
```

---

# Flower Architecture in FedMed

FedMed will use one Flower server and multiple Flower clients.

```text
                        ┌──────────────────┐
                        │  Flower Server   │
                        │                  │
                        │  Global 3D U-Net │
                        │                  │
                        │     FedAvg       │
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │Hospital A│ │Hospital B│ │Hospital C│
              │Flower    │ │Flower    │ │Flower    │
              │Client    │ │Client    │ │Client    │
              └────┬─────┘ └────┬─────┘ └────┬─────┘
                   │            │            │
                   ▼            ▼            ▼
              Private MRI   Private MRI   Private MRI
                 Data          Data          Data
```

## Flower Server

The Flower server acts as the central coordinator.

Its responsibilities include:

1. Maintaining the global model.
2. Sending the current global model to hospital clients.
3. Starting federated training rounds.
4. Receiving model updates from participating clients.
5. Aggregating the updates using FedAvg.
6. Creating the updated global model.
7. Repeating the process for multiple rounds.

---

## Flower Clients

Each hospital runs a Flower client.

For example:

```text
Hospital A → Flower Client A
Hospital B → Flower Client B
Hospital C → Flower Client C
```

Each Flower client is responsible for:

1. Receiving the global model parameters.
2. Loading those parameters into its local 3D U-Net model.
3. Training the model using its own private MRI data.
4. Returning the updated model parameters.
5. Optionally evaluating the updated model on local validation data.

---

# Flower Training Process

A simplified Flower workflow looks like this:

```text
1. Server starts
        ↓
2. Hospital clients connect
        ↓
3. Server sends global model parameters
        ↓
4. Each hospital trains locally
        ↓
5. Clients return updated parameters
        ↓
6. Server aggregates updates using FedAvg
        ↓
7. New global model is created
        ↓
8. Next federated round starts
```

This process continues until the required number of training rounds is completed.

---

# Why Flower is Important for FedMed

Without a Federated Learning framework, we would need to manually implement:

* Server-client coordination
* Model parameter exchange
* Training round management
* Client participation
* Aggregation logic
* Federated training workflow

Flower provides a framework for managing these federated learning tasks.

Therefore, in FedMed, Flower is the main technology responsible for connecting the global model with the distributed hospital nodes.

```text
                 PYTORCH + MONAI
                       │
                       ▼
                    3D U-NET
                       │
                       ▼
                     FLOWER
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Hospital A   Hospital B   Hospital C
                       │
                       ▼
                     FedAvg
                       │
                       ▼
                 Global Model
```

---

# Role of Flower in the Complete FedMed System

FedMed consists of several technologies working together.

```text
┌─────────────────────────────────────────────┐
│                   FEDMED                    │
├─────────────────────────────────────────────┤
│                                             │
│ PyTorch + MONAI                             │
│        ↓                                    │
│  Builds and trains the 3D U-Net model       │
│                                             │
│ Flower                                      │
│        ↓                                    │
│ Coordinates Federated Learning between      │
│ the central server and hospital nodes       │
│                                             │
│ TenSEAL                                     │
│        ↓                                    │
│ Provides encryption for model updates       │
│                                             │
│ Differential Privacy                        │
│        ↓                                    │
│ Adds additional privacy protection          │
│                                             │
│ React + Recharts                            │
│        ↓                                    │
│ Displays training and model metrics         │
│                                             │
└─────────────────────────────────────────────┘
```

---

# My Contribution to FedMed

My primary responsibility in the FedMed project is the **Flower Federated Learning component**.

This includes:

* Setting up the Flower central server.
* Creating and managing three mock hospital clients.
* Connecting the global PyTorch 3D U-Net model with Flower.
* Sending global model parameters to hospital nodes.
* Performing local training on each hospital's private dataset partition.
* Returning updated model parameters to the server.
* Configuring Federated Averaging (FedAvg).
* Running multiple federated training rounds.
* Ensuring that raw MRI data remains local to each hospital.

The main objective of this component is:

> **To enable multiple hospital nodes to collaboratively train a single global 3D U-Net model while keeping their raw patient MRI data within their own local environments.**

---

# Simplified FedMed Flow

```text
                         START
                           │
                           ▼
                 Create Global 3D U-Net
                           │
                           ▼
                Flower Server Starts
                           │
                           ▼
            Hospital A, B, C Connect
                           │
                           ▼
              Send Global Model to All
                           │
                           ▼
                 Local Model Training
                           │
                           ▼
                 Get Updated Parameters
                           │
                           ▼
             Encrypt Updates (FedMed Layer)
                           │
                           ▼
              Send Updates to Server
                           │
                           ▼
                  FedAvg Aggregation
                           │
                           ▼
                 New Global Model
                           │
                           ▼
                More Training Rounds
                           │
                           ▼
                          END
```

## Summary

FedMed uses Federated Learning to solve a major problem in healthcare machine learning: how multiple hospitals can collaboratively train an AI model without directly sharing sensitive patient data.

The global model is distributed to hospital nodes, trained locally on private MRI data, and updated through the aggregation of model parameters.

**Flower is the technology responsible for orchestrating this Federated Learning workflow.** It manages the interaction between the central server and hospital clients, allowing the global model to learn from multiple distributed datasets.

In short:

```text
Private Hospital Data
        ↓
Local Training
        ↓
Flower Client
        ↓
Model Updates
        ↓
Flower Server
        ↓
FedAvg
        ↓
Improved Global Model
        ↓
Repeat
```
