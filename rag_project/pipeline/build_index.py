"""
pipeline/build_index.py
Chạy 1 lần để build vector database từ PDF của nhóm.

Cách chạy:
    python -m pipeline.build_index
"""

import os
import faiss
import pickle
import numpy as np

from rag.loader import load_pdf
from rag.chunker import chunk_text
from rag.embedder import embed_batch

# ========================================================
# ⚙️  CẤU HÌNH – chỉnh đường dẫn PDF tại đây
# ========================================================
PDF_PATH    = r'data\DAP_Template_FINAL.pdf'
INDEX_PATH  = "vector.index"
CHUNKS_PATH = "chunks.pkl"
CHUNK_SIZE  = 1500
OVERLAP     = 300
# ========================================================


def build_index():
    # Bước 1: Load PDF
    print(f"📄 Đang đọc PDF: {PDF_PATH}")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Không tìm thấy file: {PDF_PATH}")
    text = load_pdf(PDF_PATH)
    print(f"✅ Đọc xong — {len(text)} ký tự")

    # Bước 2: Chunking
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    print(f"✅ Chunking xong — {len(chunks)} chunks")

    # Bước 3: Embedding (batch cho nhanh)
    print("⏳ Đang embedding các chunks...")
    embeddings = embed_batch(chunks)
    print(f"✅ Embedding xong — shape: {embeddings.shape}")

    # Bước 4: Build FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS index: {index.ntotal} vectors, dimension={dimension}")

    # Bước 5: Lưu xuống disk
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\n🎉 Hoàn tất!")
    print(f"   Index  → {INDEX_PATH}")
    print(f"   Chunks → {CHUNKS_PATH}")


if __name__ == "__main__":
    build_index()
