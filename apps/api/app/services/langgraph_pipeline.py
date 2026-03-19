from app.schemas.ticket import TicketDraft


def generate_ticket_draft(defect_description: str) -> TicketDraft:
    return TicketDraft(
        title="Investigate detected manufacturing defect",
        priority="high",
        defect_description=defect_description,
        recommended_action="Review the flagged component, inspect the station tooling, and follow the matched SOP once retrieval is connected.",
    )
