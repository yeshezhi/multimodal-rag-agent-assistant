from sentence_transformers import CrossEncoder


class LocalReranker:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_path, device=self.device, trust_remote_code=True)
        return self._model

    def rerank(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(score) for score in self.model.predict([(question, text) for text in texts], batch_size=4, show_progress_bar=False)]
