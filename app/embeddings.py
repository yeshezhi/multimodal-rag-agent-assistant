import numpy as np
from sentence_transformers import SentenceTransformer


class BGEEmbedder:
    """Lazy BGE embedder with normalized vectors for inner-product retrieval."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True,
                )
            except Exception as error:
                raise RuntimeError(
                    f"无法加载 BGE Embedding 模型 {self.model_name}。"
                    "请检查模型目录或服务器到模型仓库的网络连接。"
                ) from error
        return self._model

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([query])

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=16,
        )
        return np.asarray(vectors, dtype=np.float32)
