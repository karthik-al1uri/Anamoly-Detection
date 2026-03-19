# Cold Start Streaming Defect Detector

Monorepo skeleton for a real-time manufacturing anomaly detection system.

## Structure

```text
apps/
  api/        FastAPI backend
  dashboard/  Next.js manager dashboard
ml/           Autoencoder training and inference
streaming/    Frame simulator and Databricks streaming jobs
orchestration/LangGraph and FAISS pipeline
```

## Quick Start

### API

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
uvicorn app.main:app --reload
```

Run the API from `apps/api`.

### Dashboard

```bash
npm install
npm run dev
```

Run the dashboard from `apps/dashboard`.

### Local MongoDB

```bash
docker compose up -d mongo
```
