from orchestration.agents import build_ticket, describe_defect, determine_priority, recommend_action, retrieve_sop


def _to_optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def diagnose_anomaly_event(anomaly_event: dict[str, object]) -> dict[str, object]:
    image_path = str(anomaly_event.get("image_path") or "unknown-image")
    source_label = str(anomaly_event.get("source_label")) if anomaly_event.get("source_label") is not None else None
    category = str(anomaly_event.get("category")) if anomaly_event.get("category") is not None else None
    status = str(anomaly_event.get("status") or "normal")
    mse_score = _to_optional_float(anomaly_event.get("mse_score"))
    threshold = _to_optional_float(anomaly_event.get("threshold"))
    threshold_gap = (mse_score - threshold) if mse_score is not None and threshold is not None else None

    defect_description = describe_defect(
        image_path=image_path,
        source_label=source_label,
        mse_score=mse_score,
        threshold=threshold,
    )
    retrieval = retrieve_sop(defect_description)
    priority = determine_priority(status, threshold_gap=threshold_gap)
    retrieved_context = str(retrieval.get("match") or "No relevant SOP context found.")
    recommended_action = recommend_action(source_label, retrieved_context, priority)
    ticket_preview = build_ticket(defect_description, recommended_action, priority=priority)

    return {
        "image_path": image_path,
        "source_label": source_label,
        "category": category,
        "status": status,
        "priority": priority,
        "anomaly_score": mse_score,
        "threshold": threshold,
        "threshold_gap": threshold_gap,
        "defect_description": defect_description,
        "recommended_action": recommended_action,
        "retrieved_context": retrieved_context,
        "retrieved_matches": retrieval.get("matches", []),
        "ticket_preview": ticket_preview,
    }


def run_diagnostic_flow(image_path: str) -> dict[str, str]:
    report = diagnose_anomaly_event({"image_path": image_path, "status": "anomaly"})
    return dict(report["ticket_preview"])


def run_diagnostic_flow_for_event(anomaly_event: dict[str, object]) -> dict[str, str]:
    report = diagnose_anomaly_event(anomaly_event)
    return dict(report["ticket_preview"])
