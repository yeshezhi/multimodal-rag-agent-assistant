"""Build a DINOv3 FAISS gallery from all University-Release training satellite images."""
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.vision import DinoVisionEmbedder
from app.vision_store import ImageFaissStore


def batches(items: list[Path], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    gallery_root = Path("/home/cjy/project/Sample4Geo_copy/dataset/U1652/University-Release/train/satellite")
    paths = sorted(gallery_root.glob("*/*.jpg"))
    if not paths:
        raise RuntimeError(f"No satellite images found under {gallery_root}")
    settings = get_settings()
    store = ImageFaissStore(settings.rag_data_dir)
    embedder = DinoVisionEmbedder(settings.vision_repo_path, settings.vision_model_path, settings.vision_checkpoint_path, settings.vision_device)
    store.clear()
    for number, batch in enumerate(batches(paths, 8), start=1):
        images = [Image.open(path).convert("RGB") for path in batch]
        vectors = embedder.encode_images(images)
        records = [{"image_id": path.parent.name, "source_name": f"U1652 satellite location {path.parent.name}"} for path in batch]
        store.add(records, vectors)
        print(f"indexed {min(number * 8, len(paths))}/{len(paths)}", flush=True)
    print("gallery build complete", flush=True)


if __name__ == "__main__":
    main()
