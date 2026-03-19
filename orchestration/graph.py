from orchestration.agents import build_ticket, describe_defect, retrieve_sop


def run_diagnostic_flow(image_path: str) -> dict[str, str]:
    defect_description = describe_defect(image_path)
    sop = retrieve_sop(defect_description)
    return build_ticket(defect_description, sop)
