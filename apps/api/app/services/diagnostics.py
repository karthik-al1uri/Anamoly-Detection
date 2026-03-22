import json

from app.core.config import settings
from app.schemas.diagnostic import DiagnosticReport
from app.schemas.ticket import TicketDraft
from orchestration.graph import diagnose_anomaly_event


def _build_fallback_analysis(result: dict[str, object]) -> dict[str, str]:
    defect_description = str(result["defect_description"])
    recommended_action = str(result["recommended_action"])
    priority = str(result["priority"])
    return {
        "analysis_summary": f"{defect_description}. Priority is {priority}. {recommended_action}",
        "recommended_action": recommended_action,
        "ticket_title": str(result["ticket_preview"]["title"]),
        "ticket_priority": priority,
        "analysis_source": "local_rag",
    }


def _generate_llm_analysis(result: dict[str, object]) -> dict[str, str] | None:
    if not settings.openai_api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        prompt_payload = {
            "image_path": result["image_path"],
            "source_label": result.get("source_label"),
            "category": result.get("category"),
            "status": result["status"],
            "priority": result["priority"],
            "anomaly_score": result.get("anomaly_score"),
            "threshold": result.get("threshold"),
            "threshold_gap": result.get("threshold_gap"),
            "defect_description": result["defect_description"],
            "retrieved_context": result["retrieved_context"],
        }

        response = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an industrial defect diagnostics assistant. "
                        "Return valid JSON with keys analysis_summary, recommended_action, ticket_title, and ticket_priority. "
                        "Ground the answer in the supplied anomaly metadata and retrieved context."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(prompt_payload),
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return {
            "analysis_summary": str(parsed.get("analysis_summary") or result["defect_description"]),
            "recommended_action": str(parsed.get("recommended_action") or result["recommended_action"]),
            "ticket_title": str(parsed.get("ticket_title") or result["ticket_preview"]["title"]),
            "ticket_priority": str(parsed.get("ticket_priority") or result["priority"]),
            "analysis_source": "openai",
        }
    except Exception:
        return None


def generate_diagnostic_report(anomaly_event: dict[str, object]) -> DiagnosticReport:
    result = diagnose_anomaly_event(anomaly_event)
    llm_result = _generate_llm_analysis(result) or _build_fallback_analysis(result)
    ticket = TicketDraft(
        title=llm_result["ticket_title"],
        priority=llm_result["ticket_priority"],
        defect_description=str(result["defect_description"]),
        recommended_action=llm_result["recommended_action"],
    )
    return DiagnosticReport(
        image_path=str(result["image_path"]),
        source_label=result.get("source_label"),
        category=result.get("category"),
        status=str(result["status"]),
        priority=str(ticket.priority),
        anomaly_score=float(result["anomaly_score"]) if result.get("anomaly_score") is not None else None,
        threshold=float(result["threshold"]) if result.get("threshold") is not None else None,
        threshold_gap=float(result["threshold_gap"]) if result.get("threshold_gap") is not None else None,
        defect_description=str(result["defect_description"]),
        analysis_summary=llm_result["analysis_summary"],
        analysis_source=llm_result["analysis_source"],
        recommended_action=llm_result["recommended_action"],
        retrieved_context=str(result["retrieved_context"]),
        retrieved_matches=result.get("retrieved_matches", []),
        ticket_preview=ticket,
    )
