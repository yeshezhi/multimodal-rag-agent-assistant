import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


class DinoVisionEmbedder:
    """Adapter for the user's trained University DINOv3 retrieval model."""

    def __init__(
        self,
        repository_path: Path,
        model_path: Path,
        checkpoint_path: Path,
        device: str,
    ) -> None:
        self.repository_path = repository_path
        self.model_path = model_path
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if not self.repository_path.exists() or not self.model_path.exists():
                raise RuntimeError("DINOv3 code repository or backbone model is unavailable.")
            if not self.checkpoint_path.exists():
                raise RuntimeError("DINOv3 retrieval checkpoint is unavailable.")
            if str(self.repository_path) not in sys.path:
                sys.path.insert(0, str(self.repository_path))
            from sample4geo.model_softgroup_rel import DinoV3Model

            model = DinoV3Model(
                model_path=str(self.model_path),
                img_size=384,
                output_dim=1024,
                num_groups=2,
            )
            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            compatible_state = {
                (
                    "base_model.model." + key[len("base_model.") :]
                    if key.startswith("base_model.layer.")
                    else key
                ): value
                for key, value in state.items()
            }
            missing, unexpected = model.load_state_dict(compatible_state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    "DINOv3 checkpoint is incompatible: "
                    f"missing={len(missing)}, unexpected={len(unexpected)}"
                )
            self._model = model.to(self.device).eval()
        return self._model

    def encode_bytes(self, content: bytes) -> np.ndarray:
        try:
            image = Image.open(BytesIO(content)).convert("RGB")
        except Exception as error:
            raise ValueError("Unsupported or corrupted image file.") from error
        return self.encode_images([image])

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        with torch.inference_mode():
            inputs = self.model.processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            embedding = functional.normalize(self.model(pixels), p=2, dim=1)
        return embedding.detach().cpu().numpy().astype(np.float32)
