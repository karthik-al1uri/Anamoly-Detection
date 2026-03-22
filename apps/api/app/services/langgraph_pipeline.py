from app.schemas.ticket import TicketDraft
from app.services.diagnostics import generate_diagnostic_report


def generate_ticket_draft(defect_description: str) -> TicketDraft:
    return TicketDraft(
        title="Investigate detected manufacturing defect",
        priority="high",
        defect_description=defect_description,
        recommended_action="Review the flagged component, inspect the station tooling, and follow the matched SOP once retrieval is connected.",
    )


def generate_ticket_draft_from_anomaly(anomaly_event: dict[str, object]) -> TicketDraft:
    return generate_diagnostic_report(anomaly_event).ticket_preview
