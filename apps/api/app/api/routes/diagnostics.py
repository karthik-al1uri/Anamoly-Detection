from fastapi import APIRouter

from app.schemas.anomaly import AnomalyEventPayload
from app.schemas.diagnostic import DiagnosticReport
from app.services.diagnostics import generate_diagnostic_report

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.post("/from-anomaly", response_model=DiagnosticReport)
async def diagnose_from_anomaly(payload: AnomalyEventPayload) -> DiagnosticReport:
    return generate_diagnostic_report(payload.model_dump())
