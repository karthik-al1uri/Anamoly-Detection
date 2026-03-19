# Cold Start Streaming Defect Detector

An end-to-end anomaly detection project for manufacturing environments where **defect data is limited or unavailable** during a new product launch.

This project is designed to help you learn how a modern industrial AI system is built from the ground up:

- **PyTorch** for unsupervised visual anomaly detection
- **Databricks + Spark Structured Streaming** for real-time ingestion and processing
- **FastAPI** for serving model and orchestration APIs
- **LangGraph + FAISS** for LLM-driven defect diagnosis and SOP retrieval
- **Next.js** for the factory manager dashboard
- **MongoDB** for ticket and audit storage

The core idea is simple:

Instead of training a classifier on many examples of bad parts, the system learns what a **normal part** looks like. When a live image cannot be reconstructed well by the autoencoder, the frame is flagged as anomalous. That anomaly is then passed through an LLM pipeline to generate a diagnosis and a maintenance ticket.

---

## Problem Statement

In real factories, new products often begin production before there is enough historical defect data to train a traditional supervised vision classifier.

This project addresses that **cold start anomaly detection** problem by:

- learning the visual representation of good components
- detecting abnormal frames using reconstruction error
- describing the defect with a vision-language model
- retrieving the relevant SOP or repair guidance
- generating an actionable maintenance ticket

---

## Project Goals

- Build a real-time anomaly detection pipeline for manufacturing
- Learn unsupervised defect detection with autoencoders
- Understand how streaming systems work before moving into Databricks
- Learn how Databricks fits into ML + streaming workflows
- Add LLM-based diagnosis and retrieval over repair manuals
- Expose results through a backend API and dashboard

---

## Key Features

- Unsupervised defect detection for low-data manufacturing environments
- Real-time frame ingestion and anomaly scoring pipeline
- Delta-style anomaly event logging for downstream processing
- LLM-assisted defect description and SOP retrieval workflow
- Automated maintenance ticket generation
- Web dashboard for monitoring anomalies and system health

---

## System Architecture

```text
Image Stream / Simulator
        |
        v
Databricks / Spark Structured Streaming
        |
        v
PyTorch Autoencoder Inference
        |
        v
Anomaly Event Log (Delta Table)
        |
        v
FastAPI Backend
        |
        v
LangGraph + VLM + FAISS
        |
        v
Maintenance Ticket + Dashboard + MongoDB
```

---

## Repository Structure

```text
.
├── apps/
│   ├── api/                # FastAPI backend
│   └── dashboard/          # Next.js frontend dashboard
├── data/
│   ├── processed/          # Processed data artifacts
│   └── raw/                # Raw dataset / streamed frames
├── docs/
│   └── architecture.md     # High-level architecture notes
├── ml/
│   ├── inference/          # Frame scoring / anomaly scoring
│   ├── models/             # Autoencoder model definitions
│   └── training/           # Training scripts
├── orchestration/
│   ├── rag/                # FAISS indexing/query placeholders
│   ├── vector_store/       # Vector index storage
│   ├── agents.py           # LLM/VLM/RAG agent placeholders
│   └── graph.py            # LangGraph flow placeholder
├── streaming/
│   ├── databricks/         # Databricks Structured Streaming jobs
│   └── simulator/          # Local frame producer / stream simulator
├── .env.example
├── docker-compose.yml
└── README.md
```

---

## Tech Stack

### Machine Learning

- **PyTorch**
- **Torchvision**
- **MVTec AD dataset**

### Streaming and Data Infrastructure

- **Databricks**
- **Apache Spark Structured Streaming**
- **Delta Lake**
- **MLflow**

### LLM and Retrieval

- **LangGraph**
- **FAISS**
- **Gemini API** or an open-source VLM such as **LLaVA**

### Application Layer

- **FastAPI**
- **Next.js**
- **MongoDB**

---

## Implementation Phases

The project is organized into four progressive implementation phases.

### Phase 1: Local Anomaly Detection Baseline

**Objective**

- establish an unsupervised defect-detection baseline using the MVTec AD dataset

**Scope**

- dataset loading and preprocessing
- convolutional autoencoder training in PyTorch
- reconstruction error scoring with MSE
- threshold selection for anomaly classification
- offline validation on normal and defective samples

**Primary folders**

- `ml/models/`
- `ml/training/`
- `ml/inference/`

---

### Phase 2: Streaming Inference Pipeline

**Objective**

- convert offline scoring into a real-time anomaly detection workflow

**Scope**

- frame producer for simulated camera input
- continuous frame scoring and anomaly event generation
- local event logging for validation
- Spark Structured Streaming job design
- Databricks ingestion and Delta table integration

**Primary folders**

- `streaming/simulator/`
- `streaming/databricks/`
- `ml/inference/`

---

### Phase 3: Defect Diagnosis and Retrieval-Augmented Guidance

**Objective**

- enrich anomaly events with defect descriptions and repair guidance

**Scope**

- vision-language defect description step
- SOP and repair manual ingestion
- FAISS index construction and querying
- LangGraph workflow orchestration
- maintenance ticket draft generation

**Primary folders**

- `orchestration/`
- `apps/api/`

---

### Phase 4: Application Layer and Productization

**Objective**

- expose the system through service APIs, persistence, and a monitoring dashboard

**Scope**

- FastAPI endpoints for health, anomalies, and tickets
- MongoDB persistence for events and ticket history
- Next.js dashboard for monitoring and review
- auditability, operator workflows, and deployment readiness

**Primary folders**

- `apps/api/`
- `apps/dashboard/`
- `orchestration/`

---

## Roadmap

- [x] Monorepo scaffold
- [x] FastAPI service skeleton
- [x] Next.js dashboard skeleton
- [x] Autoencoder model placeholder
- [x] Streaming simulator placeholder
- [x] LangGraph and FAISS placeholders
- [ ] MVTec AD dataset loader
- [ ] Autoencoder training pipeline
- [ ] Threshold calibration and evaluation workflow
- [ ] Real-time anomaly scoring loop
- [ ] Databricks Structured Streaming integration
- [ ] Delta table anomaly logging
- [ ] VLM-powered defect description
- [ ] FAISS indexing and retrieval pipeline
- [ ] Ticket persistence in MongoDB
- [ ] Live anomaly dashboard and polling

---

## Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Anamoly-Detection
```

### 2. Create environment variables

```bash
cp .env.example .env
```

Fill in values as needed, especially if you later connect:

- Databricks
- Gemini API
- MongoDB

### 3. Start MongoDB locally

```bash
docker compose up -d mongo
```

### 4. Start the FastAPI backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn app.main:app --reload
```

Run this from the `apps/api` directory.

### 5. Start the Next.js dashboard

```bash
npm install
npm run dev
```

Run this from the `apps/dashboard` directory.

---

## Current Status

This repository is currently a **project skeleton**.

What already exists:

- base repo structure
- FastAPI scaffold
- Next.js scaffold
- autoencoder placeholder
- streaming simulator placeholder
- LangGraph / FAISS placeholders

What still needs to be implemented:

- real dataset loaders
- training and evaluation pipeline
- real-time scoring pipeline
- Databricks integration
- Delta table reads/writes
- VLM integration
- FAISS indexing pipeline
- MongoDB persistence logic
- dashboard polling and visualization

---

## Future Enhancements

- MLflow experiment tracking
- model registry integration
- Kafka-based ingestion
- alerting via Slack, email, or ticketing systems
- operator feedback loop for threshold tuning
- defect heatmaps and localization

---


---


