from fastapi import APIRouter

from app.services.databricks import fetch_recent_anomalies

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("/recent")
async def recent_anomalies() -> dict[str, list[dict[str, object]]]:
    return {"items": fetch_recent_anomalies()}
