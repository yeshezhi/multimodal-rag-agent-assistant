import re
from uuid import uuid4

from .chunking import ChineseTextChunker
from .config import Settings
from .documents import SourceDocument
from .embeddings import BGEEmbedder
from .llm import LocalQwenGenerator
from .schemas import ChatResponse, Citation
from .vector_store import FaissKnowledgeBase, RetrievedChunk
from .vision import DinoVisionEmbedder
from .vision_store import ImageFaissStore, RetrievedImage


SYSTEM_PROMPT = """你是企业知识库助手。只能依据给出的检索资料回答问题。
如果资料不足以支持结论，请明确回答“根据当前知识库资料，无法确定”。
不要编造资料中没有的事实；回答使用中文，并在句末标注引用编号，例如 [1]。"""


VISION_SYSTEM_PROMPT = """你是多模态企业知识库助手。请仅描述图片中能够直接观察到的内容；不确定时明确说明不确定。不要把视觉相似检索结果当成图片事实，也不要编造地点、设备型号或故障原因。使用简洁中文回答。"""


class RagService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.chunker = ChineseTextChunker()
        self.embedder = BGEEmbedder(settings.embedding_model, settings.embedding_device)
        self.store = FaissKnowledgeBase(settings.rag_data_dir)
        self.image_store = ImageFaissStore(settings.rag_data_dir)
        self.vision_embedder = DinoVisionEmbedder(
            settings.vision_repo_path,
            settings.vision_model_path,
            settings.vision_checkpoint_path,
            settings.vision_device,
        )
        self.generator = LocalQwenGenerator(settings.qwen_model_path, settings.max_new_tokens)

    def ingest(self, documents: list[SourceDocument]) -> int:
        chunks = self.chunker.split(documents)
        if chunks:
            self.store.add(chunks, self.embedder.encode_documents([chunk.text for chunk in chunks]))
        return len(chunks)

    def chat(self, question: str, top_k: int | None) -> ChatResponse:
        chunk_count, _ = self.store.status()
        if chunk_count == 0:
            return ChatResponse(
                answer="根据当前知识库资料，无法确定。请先上传相关文档。",
                citations=[],
            )
        retrieved = self.store.search(
            self.embedder.encode_query(question),
            top_k or self.settings.default_top_k,
        )
        retrieved = [
            chunk for chunk in retrieved if chunk.score >= self.settings.min_similarity_score
        ]
        if not retrieved:
            return ChatResponse(
                answer="根据当前知识库资料，无法确定。请先上传相关文档。",
                citations=[],
            )
        context = self._build_context(retrieved)
        answer = self.generator.generate(SYSTEM_PROMPT, f"检索资料：\n{context}\n\n用户问题：{question}")
        if "无法确定" in answer:
            # A weak semantic match may still be retrieved. If Qwen correctly
            # determines that it does not support an answer, do not present the
            # unrelated retrieval as evidence in the UI.
            answer = re.sub(r"\s*\[\d+\]", "", answer).strip()
            return ChatResponse(answer=answer, citations=[])
        return ChatResponse(answer=answer, citations=[self._citation(chunk) for chunk in retrieved])

    def status(self) -> tuple[int, int | None]:
        return self.store.status()

    def list_documents(self) -> list[tuple[str, int]]:
        return self.store.list_sources()

    def delete_document(self, source_name: str) -> int:
        return self.store.delete_source(source_name)

    def ingest_images(self, images: list[tuple[str, bytes]]) -> int:
        records = []
        vectors = []
        for source_name, content in images:
            vectors.append(self.vision_embedder.encode_bytes(content)[0])
            records.append({"image_id": str(uuid4()), "source_name": source_name})
        if vectors:
            import numpy as np

            self.image_store.add(records, np.stack(vectors).astype(np.float32))
        return len(records)

    def search_images(self, content: bytes, top_k: int) -> list[RetrievedImage]:
        return self.image_store.search(self.vision_embedder.encode_bytes(content), top_k)

    def image_status(self) -> tuple[int, int | None]:
        return self.image_store.status()

    def analyze_image(
        self, content: bytes, prompt: str, top_k: int
    ) -> tuple[str, list[RetrievedImage]]:
        matches = self.search_images(content, top_k)
        analysis = self.generator.generate_with_image(
            VISION_SYSTEM_PROMPT, prompt, content, max_new_tokens=256
        )
        return analysis, matches

    def multimodal_rag(
        self, content: bytes, question: str, top_k: int
    ) -> tuple[str, list[RetrievedChunk], list[RetrievedImage]]:
        matches = self.search_images(content, top_k)
        retrieved = []
        chunk_count, _ = self.store.status()
        if chunk_count:
            retrieved = self.store.search(self.embedder.encode_query(question), top_k)
            retrieved = [
                chunk
                for chunk in retrieved
                if chunk.score >= self.settings.min_similarity_score
            ]
        text_context = self._build_context(retrieved) if retrieved else "无可靠文本检索结果。"
        visual_context = "\n".join(
            f"- {item.source_name}（视觉相似度 {item.score:.4f}）" for item in matches
        ) or "视觉图库为空。"
        prompt = (
            f"用户问题：{question}\n\n"
            f"可用文本知识：\n{text_context}\n\n"
            f"视觉相似检索结果（仅作相似性线索，不是图片事实）：\n{visual_context}\n\n"
            "请结合图片可见内容和文本知识回答用户问题。文本知识支持的事实可标注 [编号]；"
            "没有依据时请明确说明无法确定，不要根据相似图片名称编造事实。"
        )
        answer = self.generator.generate_with_image(
            VISION_SYSTEM_PROMPT, prompt, content, max_new_tokens=160
        )
        if "无法确定" in answer:
            answer = re.sub(r"\s*\[\d+\]", "", answer).strip()
            retrieved = []
        else:
            cited_numbers = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
            retrieved = [
                chunk
                for number, chunk in enumerate(retrieved, start=1)
                if number in cited_numbers
            ]
        return answer, retrieved, matches

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        used = 0
        for number, chunk in enumerate(chunks, start=1):
            part = f"[{number}] 来源：{chunk.source_name}，{chunk.location}\n{chunk.text}\n"
            if used + len(part) > self.settings.max_context_chars:
                break
            parts.append(part)
            used += len(part)
        return "\n".join(parts)

    @staticmethod
    def _citation(chunk: RetrievedChunk) -> Citation:
        excerpt = chunk.text[:220] + ("…" if len(chunk.text) > 220 else "")
        return Citation(
            chunk_id=chunk.chunk_id,
            source=chunk.source_name,
            location=chunk.location,
            score=round(chunk.score, 4),
            excerpt=excerpt,
        )
