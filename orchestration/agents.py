def describe_defect(image_path: str) -> str:
    return f'Defect analysis placeholder for {image_path}'


def retrieve_sop(defect_description: str) -> str:
    return f'SOP retrieval placeholder for: {defect_description}'


def build_ticket(defect_description: str, sop: str) -> dict[str, str]:
    return {
        'title': 'Generated maintenance ticket',
        'defect_description': defect_description,
        'recommended_action': sop,
    }
