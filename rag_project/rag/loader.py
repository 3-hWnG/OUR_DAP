from pypdf import PdfReader


def load_pdf(path: str) -> str:
    """Đọc toàn bộ text từ file PDF."""
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text
