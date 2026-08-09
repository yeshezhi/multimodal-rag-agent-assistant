from dataclasses import dataclass
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .documents import SourceDocument
from .metadata import normalize_metadata


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_name: str
    location: str
    text: str
    department: str = "未分类"
    document_type: str = "其他"
    classification: str = "内部"
    effective_date: str = "未标注"
    tags: list[str] | None = None


class ChineseTextChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ";", " ", ""],
        )

    def split(self, documents: list[SourceDocument]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            metadata = normalize_metadata(document.metadata, document.source_name)
            for text in self.splitter.split_text(document.text):
                if text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid4()),
                            source_name=document.source_name,
                            location=document.location,
                            text=text.strip(),
                            **metadata,
                        )
                    )
        return chunks
