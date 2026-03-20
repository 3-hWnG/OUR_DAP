from sentence_transformers import SentenceTransformer
import numpy as np

# Load model 1 lần duy nhất khi import
_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print("⏳ Đang tải Embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model sẵn sàng!")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Chuyển một đoạn text thành vector float32."""
    model = get_model()
    vector = model.encode(text)
    return np.array(vector).astype("float32")


def embed_batch(texts: list[str]) -> np.ndarray:
    """Embed nhiều đoạn text cùng lúc (nhanh hơn gọi từng cái)."""
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=True)
    return np.array(vectors).astype("float32")
