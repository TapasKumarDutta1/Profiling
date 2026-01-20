import numpy as np
from sentence_transformers import SentenceTransformer

from model_paths import resolve_model_name

_EMBEDDER = None


def embed_texts(texts):
    texts = ["" if t is None else str(t) for t in texts]

    global _EMBEDDER
    if _EMBEDDER is None:
        model_name = resolve_model_name(
            "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"
        )
        _EMBEDDER = SentenceTransformer(model_name)
    return _EMBEDDER.encode(texts, normalize_embeddings=True)


def cosine_sim_matrix(vec, mat):
    vec = np.asarray(vec, dtype=np.float32)
    mat = np.asarray(mat, dtype=np.float32)
    vec_norm = np.linalg.norm(vec) + 1e-8
    mat_norm = np.linalg.norm(mat, axis=1) + 1e-8
    return (mat @ vec) / (mat_norm * vec_norm)
