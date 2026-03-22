from pydantic import BaseModel


class AnomalyEventPayload(BaseModel):
    image_path: str
    source_label: str
    mse_score: float
    threshold: float
    status: str
    is_anomaly: bool
    event_ts: str | None = None
    category: str | None = None
    threshold_strategy: str | None = None
