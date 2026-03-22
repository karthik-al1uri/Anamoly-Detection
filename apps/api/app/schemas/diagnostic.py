from pydantic import BaseModel

from app.schemas.ticket import TicketDraft


class RetrievedContext(BaseModel):
    document_id: str | None = None
    source: str | None = None
    title: str | None = None
    content: str
    score: float


class DiagnosticReport(BaseModel):
    image_path: str
    source_label: str | None = None
    category: str | None = None
    status: str
    priority: str
    anomaly_score: float | None = None
    threshold: float | None = None
    threshold_gap: float | None = None
    defect_description: str
    analysis_summary: str
    analysis_source: str
    recommended_action: str
    retrieved_context: str
    retrieved_matches: list[RetrievedContext]
    ticket_preview: TicketDraft
