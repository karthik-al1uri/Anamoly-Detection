from orchestration.rag.query_index import query_index


def describe_defect(
    image_path: str,
    source_label: str | None = None,
    mse_score: float | None = None,
    threshold: float | None = None,
) -> str:
    parts = [f"Visual anomaly detected for {image_path}"]
    if source_label:
        parts.append(f"observed defect bucket: {source_label}")
    if mse_score is not None and threshold is not None:
        parts.append(f"reconstruction error {mse_score:.6f} against threshold {threshold:.6f}")
    return "; ".join(parts)


def retrieve_sop(defect_description: str, source_dir: str = "docs") -> dict[str, object]:
    return query_index(defect_description, source_dir=source_dir, top_k=3)


def determine_priority(status: str, threshold_gap: float | None = None) -> str:
    normalized_status = status.lower().strip()
    if normalized_status != "anomaly":
        return "medium"
    if threshold_gap is not None and threshold_gap >= 0.001:
        return "high"
    return "medium"


def recommend_action(source_label: str | None, retrieved_context: str, priority: str) -> str:
    label = (source_label or "unknown").lower()
    if "contamination" in label:
        base = "Quarantine the affected units, clean the station, and inspect upstream contamination sources."
    elif "broken" in label:
        base = "Inspect tooling alignment, grippers, and mechanical impact points before resuming production."
    else:
        base = "Review the flagged component, verify image conditions, and inspect the station for repeat issues."
    escalation = "Escalate to the maintenance lead immediately." if priority == "high" else "Monitor the next production run after inspection."
    return f"{base} {retrieved_context} {escalation}".strip()


def build_ticket(defect_description: str, recommended_action: str, priority: str = "high") -> dict[str, str]:
    return {
        "title": "Generated maintenance ticket" if priority == "high" else "Review detected manufacturing anomaly",
        "priority": priority,
        "defect_description": defect_description,
        "recommended_action": recommended_action,
    }
