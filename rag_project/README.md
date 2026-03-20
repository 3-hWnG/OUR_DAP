# RAG Chatbot — Hệ thống Nhận diện Biển số Xe

## Cấu trúc project
```
rag_project/
├── data/
│   └── data_sample.pdf        ← PDF tài liệu của nhóm
├── pipeline/
│   ├── __init__.py
│   └── build_index.py         ← Build vector database
├── rag/
│   ├── __init__.py
│   ├── loader.py              ← Đọc PDF
│   ├── chunker.py             ← Chia text thành chunks
│   ├── embedder.py            ← Tạo vector embedding
│   ├── retriever.py           ← Tìm kiếm FAISS
│   └── rag_chatbot.py         ← Gọi Gemini sinh câu trả lời
├── chatbot_ui.py              ← Giao diện Tkinter
├── .env                       ← API key (tự tạo)
├── .env.example               ← Mẫu .env
└── requirements.txt
```

## Hướng dẫn chạy

### Bước 1: Cài thư viện
```bash
pip install -r requirements.txt
```

### Bước 2: Tạo file .env
Copy `.env.example` thành `.env` rồi điền API key Gemini vào:
```
GEMINI_API_KEY=your_api_key_here
```
Lấy API key tại: https://aistudio.google.com/app/apikey

### Bước 3: Đặt PDF vào thư mục data/
Đặt tài liệu PDF của nhóm vào `data/data_sample.pdf`
(hoặc chỉnh `PDF_PATH` trong `pipeline/build_index.py`)

### Bước 4: Build vector database (chạy 1 lần)
```bash
python -m pipeline.build_index
```
Sau khi chạy xong sẽ tạo ra: `vector.index` và `chunks.pkl`

### Bước 5: Chạy chatbot UI
```bash
python chatbot_ui.py
```
