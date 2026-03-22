from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import anomalies, diagnostics, health, tickets
from app.core.config import settings

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health.router, prefix=settings.api_prefix)
app.include_router(anomalies.router, prefix=settings.api_prefix)
app.include_router(diagnostics.router, prefix=settings.api_prefix)
app.include_router(tickets.router, prefix=settings.api_prefix)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "ok"}
