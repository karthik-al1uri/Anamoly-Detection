# Architecture Overview

## Phase 1: Ingestion
- `streaming/simulator/producer.py` simulates image-frame streaming.
- `streaming/databricks/structured_streaming_job.py` is the Databricks entry point.

## Phase 2: Inference
- `ml/models/autoencoder.py` defines the baseline autoencoder.
- `ml/inference/score_frame.py` computes reconstruction error.
- `ml/training/train_autoencoder.py` is the training scaffold.

## Phase 3: LLM Diagnostics
- `orchestration/graph.py` defines the diagnostic flow.
- `orchestration/rag/` contains FAISS index placeholders.

## Phase 4: Applications
- `apps/api/` exposes FastAPI endpoints.
- `apps/dashboard/` contains the manager dashboard.
