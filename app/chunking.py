from dataclasses import dataclass
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .documents import SourceDocument


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_name: str
    location: str
    text: str


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
            for text in self.splitter.split_text(document.text):
                if text.strip():
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid4()),
                            source_name=document.source_name,
                            location=document.location,
                            text=text.strip(),
                        )
                    )
        return chunks
