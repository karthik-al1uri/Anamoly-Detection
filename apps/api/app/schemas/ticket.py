from pydantic import BaseModel


class TicketDraft(BaseModel):
    title: str
    priority: str
    defect_description: str
    recommended_action: str
