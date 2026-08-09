import json
import os
import threading
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk
from .metadata import normalize_metadata


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source_name: str
    location: str
    text: str
    score: float
    department: str = "未分类"
    document_type: str = "其他"
    classification: str = "内部"
    effective_date: str = "未标注"
    tags: list[str] | None = None


class FaissKnowledgeBase:
    def __init__(self, data_dir: Path) -> None:
        self.index_path = data_dir / "faiss.index"
        self.records_path = data_dir / "chunks.json"
        self._index: faiss.Index | None = None
        self._records: list[dict] = []
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() != self.records_path.exists():
            raise RuntimeError("FAISS 索引与元数据文件不完整，请先清空知识库后重新导入。")
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._records = json.loads(self.records_path.read_text(encoding="utf-8"))
            if self._index.ntotal != len(self._records):
                raise RuntimeError("FAISS 索引与文本块数量不一致，请先清空知识库后重新导入。")
            migrated = False
            for record in self._records:
                metadata = normalize_metadata(record, record["source_name"])
                for key, value in metadata.items():
                    if record.get(key) != value:
                        record[key] = value
                        migrated = True
            if migrated:
                self._persist()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("向量数量与文本块数量不一致。")
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            if vectors.shape[1] != self._index.d:
                raise ValueError("Embedding 维度与现有知识库不一致，请清空知识库后重试。")
            self._index.add(vectors)
            self._records.extend(asdict(chunk) for chunk in chunks)
            self._persist()

    def search(self, vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            scores, indices = self._index.search(vector, min(top_k, self._index.ntotal))
            return [
                RetrievedChunk(score=float(score), **self._records[index])
                for score, index in zip(scores[0], indices[0])
                if index >= 0
            ]

    def all_chunks(self) -> list[RetrievedChunk]:
        """Return a snapshot for lightweight lexical recall before reranking."""
        with self._lock:
            return [RetrievedChunk(score=0.0, **record) for record in self._records]

    def export_records_and_vectors(self) -> tuple[list[dict], np.ndarray]:
        """Export the current local index for a one-off Milvus migration."""
        with self._lock:
            if self._index is None:
                return [], np.empty((0, 0), dtype=np.float32)
            return list(self._records), self._index.reconstruct_n(0, self._index.ntotal)

    def status(self) -> tuple[int, int | None]:
        with self._lock:
            return len(self._records), self._index.d if self._index is not None else None

    def list_sources(self) -> list[dict]:
        with self._lock:
            counts = Counter(record["source_name"] for record in self._records)
            sources = []
            for source_name, chunks in counts.items():
                record = next(record for record in self._records if record["source_name"] == source_name)
                metadata = normalize_metadata(record, source_name)
                sources.append({"source_name": source_name, "chunks": chunks, **metadata})
            return sorted(sources, key=lambda item: item["source_name"].lower())

    def delete_source(self, source_name: str) -> int:
        with self._lock:
            remove_indices = [
                index
                for index, record in enumerate(self._records)
                if record["source_name"] == source_name
            ]
            if not remove_indices:
                return 0
            assert self._index is not None
            vectors = self._index.reconstruct_n(0, self._index.ntotal)
            keep_indices = [
                index for index in range(len(self._records)) if index not in set(remove_indices)
            ]
            rebuilt = faiss.IndexFlatIP(self._index.d)
            if keep_indices:
                rebuilt.add(vectors[keep_indices])
            self._index = rebuilt if keep_indices else None
            self._records = [self._records[index] for index in keep_indices]
            if self._index is None:
                for path in (self.index_path, self.records_path):
                    if path.exists():
                        path.unlink()
            else:
                self._persist()
            return len(remove_indices)

    def clear(self) -> None:
        with self._lock:
            self._index = None
            self._records = []
            for path in (self.index_path, self.records_path):
                if path.exists():
                    path.unlink()

    def _persist(self) -> None:
        assert self._index is not None
        index_tmp = self.index_path.with_suffix(".index.tmp")
        records_tmp = self.records_path.with_suffix(".json.tmp")
        faiss.write_index(self._index, str(index_tmp))
        records_tmp.write_text(json.dumps(self._records, ensure_ascii=False), encoding="utf-8")
        os.replace(index_tmp, self.index_path)
        os.replace(records_tmp, self.records_path)
