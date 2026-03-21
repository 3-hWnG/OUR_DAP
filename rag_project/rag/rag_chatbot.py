import os
from dotenv import load_dotenv
from groq import Groq
from rag.retriever import retrieve

load_dotenv()

_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Chưa có GROQ_API_KEY trong file .env!")
        _client = Groq(api_key=api_key)
    return _client


def ask_rag(question: str, k: int = 5, lang: str = "vi") -> str:
    """
    Pipeline RAG:
      1. Retrieve top-k chunk liên quan
      2. Gửi context + câu hỏi lên Groq
      3. Trả về câu trả lời
    """
    contexts = retrieve(question, k=k)
    context_text = "\n\n---\n\n".join(contexts)

    lang_instruction = "Trả lời bằng tiếng Việt, rõ ràng và ngắn gọn." if lang == "vi" else "Answer in English, clearly and concisely."

    prompt = f"""Bạn là trợ lý AI hỗ trợ hỏi đáp về hệ thống nhận diện biển số xe.
Hãy trả lời câu hỏi dựa trên tài liệu được cung cấp bên dưới.
Nếu tài liệu không có thông tin liên quan, hãy nói rõ điều đó.
{lang_instruction}

Tài liệu tham khảo:
{context_text}

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
