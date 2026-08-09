import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running this script directly from the project root without installing it
# as a Python package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.rag import RagService


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
        "--report-path",
        default=None,
        help="Path for the JSON summary consumed by the monitoring page.",
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
    # Exercise the real online retrieval path: dense recall, lexical fallback,
    # and cross-encoder reranking.  Keep inference on CPU so it never competes
    # with the Qwen service on the RTX 4090.
    settings.embedding_device = args.device
    settings.reranker_device = args.device
    if args.min_score is not None:
        settings.min_similarity_score = args.min_score
    service = RagService(settings)

    passed = 0
    reciprocal_ranks: list[float] = []
    details = []
    for case in cases:
        retrieved = service.retrieve_text(case["question"], args.top_k)
        sources = [item.source_name for item in retrieved]
        ranked = [
            {"source": item.source_name, "score": round(item.score, 4)}
            for item in retrieved
        ]
        expected_source = case.get("expected_source")
        hit = not sources if expected_source is None else expected_source in sources
        rank = (
            sources.index(expected_source) + 1
            if expected_source is not None and expected_source in sources
            else None
        )
        passed += int(hit)
        reciprocal_ranks.append(1 / rank if rank else 0)
        detail = {
            "question": case["question"],
            "expected": expected_source,
            "accepted_sources": sources,
            "ranked_sources": ranked,
            "rank": rank,
            "hit": hit,
        }
        details.append(detail)
        print(json.dumps(detail, ensure_ascii=False))
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "total_cases": len(cases),
        "passed": passed,
        "pass_at_k": round(passed / len(cases), 4) if cases else 0,
        "mrr": round(sum(reciprocal_ranks) / len(cases), 4) if cases else 0,
        "details": details,
    }
    report_path = Path(args.report_path) if args.report_path else settings.rag_data_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Hybrid Recall + Rerank Pass@{args.top_k}: {passed}/{len(cases)} = {passed / len(cases):.1%}")
    print(f"MRR: {report['mrr']:.4f}; report: {report_path}")


if __name__ == "__main__":
    main()
