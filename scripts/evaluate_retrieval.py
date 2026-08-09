import argparse
import json
import sys
from pathlib import Path

# Allow running this script directly from the project root without installing it
# as a Python package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.embeddings import BGEEmbedder
from app.vector_store import FaissKnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate whether expected sources appear in retrieval results.")
    parser.add_argument("--cases", default="evaluation/questions.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Embedding device for offline evaluation (default: cpu to avoid competing with Qwen).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Override the configured acceptance threshold for an evaluation run.",
    )
    args = parser.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    settings = get_settings()
    store = FaissKnowledgeBase(settings.rag_data_dir)
    embedder = BGEEmbedder(settings.embedding_model, args.device)
    threshold = args.min_score if args.min_score is not None else settings.min_similarity_score

    passed = 0
    for case in cases:
        retrieved = store.search(embedder.encode_query(case["question"]), args.top_k)
        sources = [item.source_name for item in retrieved if item.score >= threshold]
        ranked = [
            {"source": item.source_name, "score": round(item.score, 4)}
            for item in retrieved
        ]
        expected_source = case.get("expected_source")
        hit = not sources if expected_source is None else expected_source in sources
        passed += int(hit)
        print(
            json.dumps(
                {
                    "question": case["question"],
                    "expected": expected_source,
                    "accepted_sources": sources,
                    "ranked_sources": ranked,
                    "hit": hit,
                },
                ensure_ascii=False,
            )
        )
    print(
        f"Pass@{args.top_k} (threshold={threshold:.2f}): "
        f"{passed}/{len(cases)} = {passed / len(cases):.1%}"
    )


if __name__ == "__main__":
    main()
