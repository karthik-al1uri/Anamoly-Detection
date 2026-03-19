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

## Learning Roadmap

If you are new to tools like Databricks, do **not** start by trying to build the full system at once.

Build the project in the following **4 phases**.

### Phase 1: Local Anomaly Detection Baseline

**Goal:** Learn the core ML idea first.

In this phase, ignore Databricks, LangGraph, and the full dashboard.

Build only:

- dataset loading from MVTec AD
- a simple convolutional autoencoder in PyTorch
- reconstruction error scoring using MSE
- anomaly threshold selection
- offline evaluation on normal vs defective samples

**What you will learn:**

- how unsupervised anomaly detection works
- why reconstruction error can detect defects
- how to train and validate an autoencoder
- how to define an anomaly threshold

**Suggested output:**

- train a model locally
- save checkpoints
- score sample images
- print anomaly predictions in a notebook or Python script

**Focus folders:**

- `ml/models/`
- `ml/training/`
- `ml/inference/`

---

### Phase 2: Simulated Real-Time Streaming

**Goal:** Learn streaming concepts locally before using Databricks.

In this phase, simulate a factory camera feed by sending images from a folder one by one.

Build:

- a local producer that emits image frames at a fixed FPS
- a local consumer or scoring loop that reads and scores those frames
- anomaly event logging to a local file, SQLite, or JSON output

After that works, move the same idea into:

- Spark Structured Streaming
- Databricks file ingestion or Kafka ingestion
- Delta table logging for anomalies

**What you will learn:**

- the basics of event streams
- how batch ML logic becomes streaming ML logic
- how Databricks can simplify ingestion, compute, and storage
- how Delta tables fit into streaming pipelines

**Suggested output:**

- frames processed continuously
- anomaly scores written per frame
- anomaly events stored in a structured log

**Focus folders:**

- `streaming/simulator/`
- `streaming/databricks/`
- `ml/inference/`

---

### Phase 3: LLM Defect Diagnosis and SOP Retrieval

**Goal:** Turn an anomaly event into an explanation and action.

Once anomaly events exist, connect them to an LLM workflow.

Build:

- a Vision-Language Model step that describes the visual defect
- document chunking for SOPs and repair manuals
- FAISS indexing for retrieval
- a LangGraph workflow that routes:
  - anomaly image
  - defect description
  - SOP retrieval
  - ticket synthesis

**What you will learn:**

- how VLMs convert images into structured text descriptions
- how RAG systems retrieve operational knowledge
- how LangGraph coordinates multi-step agent workflows
- how LLM outputs can be transformed into actionable tickets

**Suggested output:**

- text description of the defect
- retrieved SOP snippet
- generated maintenance ticket draft

**Focus folders:**

- `orchestration/`
- `apps/api/`

---

### Phase 4: Full-Stack Productization

**Goal:** Expose the system like a real internal factory tool.

Now connect everything end to end.

Build:

- FastAPI endpoints for anomalies, tickets, and system health
- MongoDB persistence for tickets and audit logs
- Next.js dashboard for live status and recent events
- optional auth, role-based access, and deployment workflows

**What you will learn:**

- how backend APIs connect ML systems to frontend apps
- how dashboards present streaming inference results
- how to store audit trails and operator actions
- how to structure an AI project like a production system

**Suggested output:**

- live dashboard cards
- recent anomaly table
- ticket generation workflow
- full demo across backend, model, and UI

**Focus folders:**

- `apps/api/`
- `apps/dashboard/`
- `orchestration/`

---

## Recommended Build Order

If your goal is to learn the tools properly, follow this order:

1. **Train the anomaly detector locally**
2. **Simulate streaming locally**
3. **Move streaming into Databricks**
4. **Add LLM diagnosis and retrieval**
5. **Expose everything through API + dashboard**

This way, Databricks becomes easier to understand because you will already know the logic it is orchestrating.

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

## Notes

- If you use **Gemini API**, you will need an API key.
- Do **not** hardcode secrets into the repository.
- If you are learning Databricks for the first time, treat it as a deployment and orchestration layer after your local prototype works.

---

## License

Add your preferred license here.
