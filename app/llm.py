import threading
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path

from PIL import Image


class AnswerGenerator(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class LocalQwenGenerator(AnswerGenerator):
    def __init__(self, model_path: Path, max_new_tokens: int) -> None:
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.model_path.exists():
            raise RuntimeError(f"找不到 Qwen 模型目录：{self.model_path}")
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self._load()
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            trimmed = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            return self._processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

    def generate_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_content: bytes,
        max_new_tokens: int | None = None,
    ) -> str:
        try:
            image = Image.open(BytesIO(image_content)).convert("RGB")
        except Exception as error:
            raise ValueError("Unsupported or corrupted image file.") from error
        with self._lock:
            self._load()
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]},
            ]
            inputs = self._processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self._model.device)
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens
            )
            trimmed = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
            return self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
