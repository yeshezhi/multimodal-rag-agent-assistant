from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", extra="ignore")

    rag_data_dir: Path = PROJECT_DIR / "data"
    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cuda"
    qwen_model_path: Path = Path(
        "/home/cjy/robot_project/lingbot-vla-v2/lingbot-vla/Qwen3-VL-4B-Instruct"
    )
    qwen_device: str = "cuda"
    vision_repo_path: Path = Path("/home/cjy/project/Sample4Geo_copy")
    vision_model_path: Path = Path("/home/cjy/project/dinov3-vitb16-pretrain-lvd1689m")
    vision_checkpoint_path: Path = Path(
        "/home/cjy/project/Sample4Geo_copy/university_dinov3_softgroup_rel/"
        "dinov3_softgroup_g2/144030/weights_e4_0.9493.pth"
    )
    vision_device: str = "cpu"
    max_upload_mb: int = 25
    default_top_k: int = 5
    min_similarity_score: float = 0.50
    max_context_chars: int = 12_000
    max_new_tokens: int = 512


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.rag_data_dir.mkdir(parents=True, exist_ok=True)
    return settings
