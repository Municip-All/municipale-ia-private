from __future__ import annotations

import threading
from typing import List

import numpy as np

from municipal.config import EMBEDDING_MODEL

_lock = threading.Lock()
_model = None


def get_model():
    global _model
    with _lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model



def embed_texts(texts: list[str], normalize: bool = True) -> np.ndarray:
    m = get_model()
    return m.encode(
        texts,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def embed_one(text: str) -> list[float]:
    v = embed_texts([text.strip() or " "])[0]
    return [float(x) for x in v.tolist()]
