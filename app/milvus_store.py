import json
import threading
from collections import Counter

import numpy as np
from pymilvus import DataType, MilvusClient

from .chunking import Chunk
from .metadata import normalize_metadata
from .vector_store import RetrievedChunk


class MilvusKnowledgeBase:
    """Milvus implementation matching the local FAISS store interface."""

    def __init__(self, uri: str, collection_name: str, dimension: int) -> None:
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dimension = dimension
        self._lock = threading.RLock()

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if not chunks:
            return
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError("向量数量与文本块数量不一致。")
        if vectors.shape[1] != self.dimension:
            raise ValueError("Embedding 维度与 Milvus 集合配置不一致。")
        with self._lock:
            self._ensure_collection()
            data = []
            for chunk, vector in zip(chunks, vectors):
                metadata = normalize_metadata(chunk.__dict__, chunk.source_name)
                data.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "source_name": chunk.source_name,
                        "location": chunk.location,
                        "text": chunk.text,
                        "department": metadata["department"],
                        "document_type": metadata["document_type"],
                        "classification": metadata["classification"],
                        "effective_date": metadata["effective_date"],
                        "tags": metadata["tags"],
                        "vector": vector.astype(np.float32).tolist(),
                    }
                )
            self.client.insert(collection_name=self.collection_name, data=data)
            self.client.flush(collection_name=self.collection_name)

    def search(self, vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if not self._exists():
            return []
        output_fields = self._output_fields()
        with self._lock:
            results = self.client.search(
                collection_name=self.collection_name,
                data=vector.astype(np.float32).tolist(),
                limit=top_k,
                output_fields=output_fields,
                search_params={"metric_type": "IP"},
            )
        return [self._to_chunk(hit) for hit in results[0]]

    def all_chunks(self) -> list[RetrievedChunk]:
        if not self._exists():
            return []
        with self._lock:
            rows = self.client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=self._output_fields(),
                limit=16_384,
            )
        return [self._to_chunk({"entity": row, "distance": 0.0}) for row in rows]

    def status(self) -> tuple[int, int | None]:
        if not self._exists():
            return 0, None
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0)), self.dimension

    def list_sources(self) -> list[dict]:
        records = self.all_chunks()
        counts = Counter(record.source_name for record in records)
        sources = []
        for source_name, chunks in counts.items():
            record = next(item for item in records if item.source_name == source_name)
            sources.append(
                {
                    "source_name": source_name,
                    "chunks": chunks,
                    "department": record.department,
                    "document_type": record.document_type,
                    "classification": record.classification,
                    "effective_date": record.effective_date,
                    "tags": record.tags or [],
                }
            )
        return sorted(sources, key=lambda item: item["source_name"].lower())

    def delete_source(self, source_name: str) -> int:
        records = [item for item in self.all_chunks() if item.source_name == source_name]
        if not records:
            return 0
        with self._lock:
            self.client.delete(
                collection_name=self.collection_name,
                filter=f"source_name == {json.dumps(source_name, ensure_ascii=False)}",
            )
            self.client.flush(collection_name=self.collection_name)
        return len(records)

    def clear(self) -> None:
        if self._exists():
            with self._lock:
                self.client.drop_collection(self.collection_name)

    def _exists(self) -> bool:
        return self.client.has_collection(collection_name=self.collection_name)

    def _ensure_collection(self) -> None:
        if self._exists():
            return
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("source_name", DataType.VARCHAR, max_length=512)
        schema.add_field("location", DataType.VARCHAR, max_length=128)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("department", DataType.VARCHAR, max_length=128)
        schema.add_field("document_type", DataType.VARCHAR, max_length=128)
        schema.add_field("classification", DataType.VARCHAR, max_length=64)
        schema.add_field("effective_date", DataType.VARCHAR, max_length=32)
        schema.add_field("tags", DataType.JSON)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=self.dimension)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="IP")
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    @staticmethod
    def _output_fields() -> list[str]:
        return [
            "chunk_id", "source_name", "location", "text", "department", "document_type",
            "classification", "effective_date", "tags",
        ]

    @staticmethod
    def _to_chunk(hit: dict) -> RetrievedChunk:
        entity = hit.get("entity", hit)
        return RetrievedChunk(
            chunk_id=entity["chunk_id"],
            source_name=entity["source_name"],
            location=entity["location"],
            text=entity["text"],
            score=float(hit.get("distance", 0.0)),
            department=entity["department"],
            document_type=entity["document_type"],
            classification=entity["classification"],
            effective_date=entity["effective_date"],
            tags=entity.get("tags") or [],
        )
