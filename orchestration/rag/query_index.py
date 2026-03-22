from functools import lru_cache
from orchestration.rag.build_index import build_index, tokenize_text


@lru_cache(maxsize=4)
def _cached_index(source_dir: str) -> dict[str, object]:
    return build_index(source_dir)


def _score_match(query_tokens: list[str], candidate_tokens: list[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    query_set = set(query_tokens)
    candidate_set = set(candidate_tokens)
    overlap = query_set & candidate_set
    if not overlap:
        return 0.0
    precision = len(overlap) / len(query_set)
    recall = len(overlap) / len(candidate_set)
    return (2 * precision * recall) / max(precision + recall, 1e-8)


def query_index(query: str, source_dir: str = "docs", top_k: int = 3) -> dict[str, object]:
    query_tokens = tokenize_text(query)
    index = _cached_index(source_dir)
    matches: list[dict[str, object]] = []

    for entry in index.get("entries", []):
        score = _score_match(query_tokens, list(entry.get("tokens", [])))
        if score <= 0:
            continue
        matches.append(
            {
                "document_id": entry.get("document_id"),
                "source": entry.get("source"),
                "title": entry.get("title"),
                "content": entry.get("content"),
                "score": float(score),
            }
        )

    matches.sort(key=lambda match: (float(match["score"]), str(match["title"])), reverse=True)
    top_matches = matches[: max(top_k, 0)]
    return {
        "query": query,
        "match": top_matches[0]["content"] if top_matches else "No relevant SOP context found.",
        "matches": top_matches,
        "documents_indexed": int(index.get("documents_indexed", 0)),
        "chunks_indexed": int(index.get("chunks_indexed", 0)),
    }


if __name__ == "__main__":
    print(query_index("deep scratch on aluminum casing"))
