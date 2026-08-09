import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking import Chunk
from app.config import get_settings
from app.milvus_store import MilvusKnowledgeBase
from app.vector_store import FaissKnowledgeBase


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate a local FAISS knowledge base to Milvus.")
    parser.add_argument("--replace", action="store_true", help="Drop existing Milvus data before migration.")
    args = parser.parse_args()
    settings = get_settings()
    source = FaissKnowledgeBase(settings.rag_data_dir)
    records, vectors = source.export_records_and_vectors()
    target = MilvusKnowledgeBase(settings.milvus_uri, settings.milvus_collection, settings.embedding_dimension)
    existing, _ = target.status()
    if existing and not args.replace:
        raise SystemExit("Milvus collection is not empty; rerun with --replace to overwrite it.")
    if args.replace:
        target.clear()
    if records:
        target.add([Chunk(**record) for record in records], vectors)
    count, dimension = target.status()
    print(f"Migrated {len(records)} chunks to Milvus: {count} rows, {dimension} dimensions.")


if __name__ == "__main__":
    main()
