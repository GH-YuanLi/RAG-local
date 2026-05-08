import logging
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_LOCAL_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
HF_MIRROR = "https://hf-mirror.com"


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu"):
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = HF_MIRROR

        local_path = _LOCAL_MODEL_DIR / model_name.split("/")[-1]
        load_path = str(local_path) if local_path.exists() else model_name

        logger.info("正在加载 embedding 模型: %s (设备: %s)", load_path, device)
        self.model = SentenceTransformer(load_path, device=device)
        self.model_name = model_name
        logger.info("Embedding 模型加载完成，维度: %d", self.model.get_embedding_dimension())

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()
