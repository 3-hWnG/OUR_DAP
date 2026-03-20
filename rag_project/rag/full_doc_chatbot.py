"""
rag/full_doc_chatbot.py – Full Document QA mode.
Load toàn bộ text PDF vào context một lần, không dùng FAISS chunking.
Phù hợp với PDF ngắn (~5-10 trang), context window Groq 128k tokens.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from rag.loader import load_pdf

load_dotenv()

_client = None
_full_text = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Chưa có GROQ_API_KEY trong file .env!")
        _client = Groq(api_key=api_key)
    return _client


def load_full_doc(pdf_path: str):
    """Load toàn bộ text PDF vào bộ nhớ."""
    global _full_text
    print(f"⏳ Đang đọc toàn bộ PDF: {pdf_path}")
    _full_text = load_pdf(pdf_path)
    print(f"✅ Đã load {len(_full_text)} ký tự vào context")


def ask_full_doc(question: str) -> str:
    """
    Full Document QA pipeline:
      1. Nhét toàn bộ text PDF vào prompt
      2. Gửi thẳng lên Groq — không cần FAISS
      3. Trả về câu trả lời
    """
    if not _full_text:
        raise RuntimeError("Chưa load PDF! Gọi load_full_doc() trước.")

    prompt = f"""Bạn là trợ lý AI hỗ trợ hỏi đáp về hệ thống nhận diện biển số xe.
Dưới đây là toàn bộ nội dung tài liệu của nhóm. Hãy đọc kỹ và trả lời câu hỏi dựa trên tài liệu.
Nếu tài liệu không có thông tin liên quan, hãy nói rõ điều đó.
Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn.

=== NỘI DUNG TÀI LIỆU ===
{_full_text}
=== HẾT TÀI LIỆU ===

Câu hỏi: {question}

Trả lời:"""

    client = get_client()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content
