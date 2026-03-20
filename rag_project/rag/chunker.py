def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Chia văn bản thành các đoạn nhỏ có overlap.
    chunk_size : số ký tự mỗi chunk
    overlap    : số ký tự overlap giữa 2 chunk liên tiếp
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
