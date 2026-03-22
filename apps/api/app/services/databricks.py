import json
from pathlib import Path

from app.core.config import settings


def _load_anomaly_events() -> list[dict[str, object]]:
    events_path = Path(settings.anomaly_events_path)
    if not events_path.exists():
        return []

    events: list[dict[str, object]] = []
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def fetch_recent_anomalies(limit: int = 20) -> list[dict[str, object]]:
    events = _load_anomaly_events()
    events.sort(key=lambda event: str(event.get("event_ts") or event.get("timestamp") or ""), reverse=True)
    return events[: max(limit, 0)]
