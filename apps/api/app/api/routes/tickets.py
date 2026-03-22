from fastapi import APIRouter

from app.schemas.anomaly import AnomalyEventPayload
from app.schemas.ticket import TicketDraft
from app.services.langgraph_pipeline import generate_ticket_draft, generate_ticket_draft_from_anomaly

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("/preview", response_model=TicketDraft)
async def preview_ticket(payload: dict[str, str]) -> TicketDraft:
    defect_description = payload.get("defect_description", "Unknown visual defect")
    return generate_ticket_draft(defect_description)


@router.post("/from-anomaly", response_model=TicketDraft)
async def preview_ticket_from_anomaly(payload: AnomalyEventPayload) -> TicketDraft:
    return generate_ticket_draft_from_anomaly(payload.model_dump())
