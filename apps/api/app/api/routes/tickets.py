from fastapi import APIRouter

from app.schemas.ticket import TicketDraft
from app.services.langgraph_pipeline import generate_ticket_draft

router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("/preview", response_model=TicketDraft)
async def preview_ticket(payload: dict[str, str]) -> TicketDraft:
    defect_description = payload.get("defect_description", "Unknown visual defect")
    return generate_ticket_draft(defect_description)
