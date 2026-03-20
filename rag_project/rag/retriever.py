import faiss
import pickle
import numpy as np
from rag.embedder import embed_text

_index = None
_chunks = None


def load_index(index_path: str = "vector.index", chunks_path: str = "chunks.pkl"):
    """Load FAISS index và chunks từ disk."""
    global _index, _chunks
    _index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        _chunks = pickle.load(f)
    print(f"✅ Đã load index: {_index.ntotal} vectors, {len(_chunks)} chunks")


def retrieve(query: str, k: int = 3) -> list[str]:
    """
    Tìm top-k chunk liên quan nhất với câu hỏi.
    Trả về list các đoạn text.
    """
    if _index is None or _chunks is None:
        raise RuntimeError("Chưa load index! Gọi load_index() trước.")

    query_vector = embed_text(query).reshape(1, -1)
    distances, indices = _index.search(query_vector, k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_chunks):
            results.append(_chunks[idx])
    return results
