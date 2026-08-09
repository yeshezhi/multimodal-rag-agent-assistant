import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np


@dataclass(frozen=True)
class RetrievedImage:
    image_id: str
    source_name: str
    score: float


class ImageFaissStore:
    def __init__(self, data_dir: Path) -> None:
        self.index_path = data_dir / "image-faiss.index"
        self.records_path = data_dir / "images.json"
        self._index: faiss.Index | None = None
        self._records: list[dict[str, str]] = []
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() != self.records_path.exists():
            raise RuntimeError("Image index and metadata are incomplete.")
        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._records = json.loads(self.records_path.read_text(encoding="utf-8"))
            if self._index.ntotal != len(self._records):
                raise RuntimeError("Image index and metadata count do not match.")

    def add(self, records: list[dict[str, str]], vectors: np.ndarray) -> None:
        if not records:
            return
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexFlatIP(vectors.shape[1])
            if vectors.shape[1] != self._index.d:
                raise ValueError("Image embedding dimension does not match existing index.")
            self._index.add(vectors)
            self._records.extend(records)
            self._persist()

    def search(self, vector: np.ndarray, top_k: int) -> list[RetrievedImage]:
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            scores, indices = self._index.search(vector, min(top_k, self._index.ntotal))
            return [
                RetrievedImage(score=round(float(score), 4), **self._records[index])
                for score, index in zip(scores[0], indices[0])
                if index >= 0
            ]

    def status(self) -> tuple[int, int | None]:
        with self._lock:
            return len(self._records), self._index.d if self._index else None

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
