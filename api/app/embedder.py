import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings

_model: SentenceTransformer | None = None


def load() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embed_model, device="cpu")
    return _model


def _encode(text: str) -> np.ndarray:
    # e5 ailesi "query: " ön eki ister
    vec = load().encode(f"query: {text}", normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


async def encode(text: str) -> np.ndarray:
    """Event loop'u bloklamamak için thread'e taşı."""
    return await asyncio.to_thread(_encode, text)
