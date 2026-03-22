import re
from pathlib import Path


STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "once",
    "then",
    "have",
    "your",
    "their",
    "are",
    "was",
    "were",
    "been",
    "over",
    "when",
    "will",
    "also",
    "than",
    "each",
    "only",
    "used",
    "using",
}


def tokenize_text(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOP_WORDS]


def _chunk_text(text: str, max_chars: int = 500) -> list[str]:
    chunks: list[str] = []
    current = ""
    for paragraph in [part.strip() for part in text.split("\n\n") if part.strip()]:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        start = 0
        while start < len(paragraph):
            chunks.append(paragraph[start : start + max_chars].strip())
            start += max_chars
        current = ""
    if current:
        chunks.append(current)
    return chunks or [text.strip()] if text.strip() else []


def _knowledge_documents() -> list[dict[str, str]]:
    return [
        {
            "source": "knowledge://contamination-response",
            "title": "Contamination response procedure",
            "content": "When contamination is detected, isolate the affected units, inspect conveyors and station surfaces for residue, clean the work area, verify upstream material handling, and restart only after a clean inspection pass.",
        },
        {
            "source": "knowledge://surface-damage-response",
            "title": "Surface damage and breakage procedure",
            "content": "When cracked, broken, or chipped parts are detected, inspect tooling alignment, grippers, and impact points, review recent maintenance logs, quarantine affected components, and escalate recurring mechanical damage to maintenance.",
        },
        {
            "source": "knowledge://false-positive-review",
            "title": "False positive review guidance",
            "content": "If a normal unit is flagged as anomalous, compare the reconstruction score against the threshold margin, review image quality, lighting consistency, and camera framing, and consider recalibration or additional validation samples before changing the threshold.",
        },
        {
            "source": "knowledge://anomaly-triage",
            "title": "Anomaly triage checklist",
            "content": "Prioritize anomalies that exceed the threshold by a clear margin, identify the defect bucket, preserve evidence, notify operations for repeated occurrences, and create a ticket with the observed failure mode and recommended station checks.",
        },
    ]


def _file_documents(source_dir: str | Path) -> list[dict[str, str]]:
    root = Path(source_dir)
    if not root.exists():
        return []

    documents: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue
        documents.append(
            {
                "source": str(path),
                "title": path.stem.replace("_", " ").replace("-", " "),
                "content": text,
            }
        )
    return documents


def build_index(source_dir: str = "docs") -> dict[str, object]:
    documents = _knowledge_documents() + _file_documents(source_dir)
    entries: list[dict[str, object]] = []

    for document in documents:
        for chunk_number, chunk in enumerate(_chunk_text(document["content"]), start=1):
            entries.append(
                {
                    "document_id": f"{document['source']}#chunk-{chunk_number}",
                    "source": document["source"],
                    "title": document["title"],
                    "content": chunk,
                    "tokens": tokenize_text(f"{document['title']} {chunk}"),
                }
            )

    return {
        "documents_indexed": len(documents),
        "chunks_indexed": len(entries),
        "entries": entries,
    }


if __name__ == "__main__":
    print(build_index("docs"))
