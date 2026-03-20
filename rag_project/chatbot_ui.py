"""
chatbot_ui.py – Giao diện Tkinter cho RAG Chatbot biển số xe.

Cách chạy:
    1. Build index trước:  python -m pipeline.build_index
    2. Chạy UI:            python chatbot_ui.py
"""

import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

from rag.retriever import load_index
from rag.rag_chatbot import ask_rag
from rag.full_doc_chatbot import load_full_doc, ask_full_doc


# ========================================================
# ⚙️  CẤU HÌNH
# ========================================================
INDEX_PATH  = "vector.index"
CHUNKS_PATH = "chunks.pkl"
PDF_PATH    = r"data/DAP_Template_FINAL.pdf"
WINDOW_TITLE = "🚗 Chatbot Nhận Diện Biển Số Xe"
# ========================================================


class RAGChatbotUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("700x640")
        self.root.resizable(False, False)

        # Mode: "rag" hoặc "full_doc"
        self._mode = tk.StringVar(value="rag")

        self._build_ui()
        self._load_all_async()

    # ----------------------------------------------------------
    # BUILD UI
    # ----------------------------------------------------------

    def _build_ui(self):
        # ── Tiêu đề ─────────────────────────────────────────────
        tk.Label(
            self.root,
            text="🚗 RAG Chatbot — Hệ thống Nhận diện Biển số Xe",
            font=("Arial", 13, "bold")
        ).pack(pady=(14, 4))

        tk.Label(
            self.root,
            text="Hỏi bất kỳ điều gì về hệ thống nhận diện biển số của nhóm",
            font=("Arial", 9), fg="#555"
        ).pack(pady=(0, 4))

        # ── Toggle mode ──────────────────────────────────────────
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=(0, 8))

        tk.Label(mode_frame, text="Chế độ:", font=("Arial", 9, "bold")).pack(side="left", padx=(0, 6))

        tk.Radiobutton(
            mode_frame, text="🔍 RAG (Chunk + FAISS)",
            variable=self._mode, value="rag",
            font=("Arial", 9), fg="#1a6ea8",
            command=self._on_mode_change
        ).pack(side="left", padx=4)

        tk.Radiobutton(
            mode_frame, text="📄 Full Doc (Đọc toàn bộ PDF)",
            variable=self._mode, value="full_doc",
            font=("Arial", 9), fg="#1a7a1a",
            command=self._on_mode_change
        ).pack(side="left", padx=4)

        # ── Khung chat ──────────────────────────────────────────
        chat_frame = tk.Frame(self.root)
        chat_frame.pack(fill="both", expand=True, padx=16)

        self._chat_box = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=("Arial", 10),
            state="disabled",
            bg="#f9f9f9",
            relief="groove",
            bd=2,
            height=24,
        )
        self._chat_box.pack(fill="both", expand=True)

        self._chat_box.tag_config("user",   foreground="#1a6ea8", font=("Arial", 10, "bold"))
        self._chat_box.tag_config("bot",    foreground="#1a7a1a", font=("Arial", 10))
        self._chat_box.tag_config("system", foreground="#888",    font=("Arial", 9, "italic"))
        self._chat_box.tag_config("error",  foreground="#cc0000", font=("Arial", 10))

        # ── Input ────────────────────────────────────────────────
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill="x", padx=16, pady=10)

        self._input_var = tk.StringVar()
        self._input_entry = tk.Entry(
            input_frame,
            textvariable=self._input_var,
            font=("Arial", 11),
            relief="groove", bd=2
        )
        self._input_entry.pack(side="left", fill="x", expand=True, ipady=6)
        self._input_entry.bind("<Return>", lambda e: self._on_send())

        self._send_btn = tk.Button(
            input_frame,
            text="Gửi ➤",
            command=self._on_send,
            width=8, height=1,
            bg="#1a6ea8", fg="white",
            font=("Arial", 10, "bold"),
            relief="flat", cursor="hand2"
        )
        self._send_btn.pack(side="left", padx=(8, 0))

        # ── Status bar ───────────────────────────────────────────
        self._status_var = tk.StringVar(value="⏳ Đang tải...")
        tk.Label(
            self.root,
            textvariable=self._status_var,
            font=("Arial", 8), fg="#888", anchor="w"
        ).pack(fill="x", padx=16, pady=(0, 6))

    # ----------------------------------------------------------
    # LOAD
    # ----------------------------------------------------------

    def _load_all_async(self):
        """Load cả FAISS index và full doc text song song."""
        self._set_input_state("disabled")
        self._rag_ready = False
        self._fulldoc_ready = False

        def _load_rag():
            try:
                load_index(INDEX_PATH, CHUNKS_PATH)
                self._rag_ready = True
                self.root.after(0, self._check_ready)
            except Exception as e:
                self.root.after(0, lambda: self._on_load_error("RAG Index", str(e)))

        def _load_fulldoc():
            try:
                load_full_doc(PDF_PATH)
                self._fulldoc_ready = True
                self.root.after(0, self._check_ready)
            except Exception as e:
                self.root.after(0, lambda: self._on_load_error("Full Doc", str(e)))

        threading.Thread(target=_load_rag, daemon=True).start()
        threading.Thread(target=_load_fulldoc, daemon=True).start()

    def _check_ready(self):
        if self._rag_ready and self._fulldoc_ready:
            self._status_var.set("✅ Sẵn sàng — Hãy đặt câu hỏi!")
            self._set_input_state("normal")
            self._append_message(
                "🤖 Xin chào! Tôi là chatbot hỗ trợ về hệ thống nhận diện biển số xe.\n"
                "Bạn có thể chọn chế độ:\n"
                "  🔍 RAG — tìm kiếm theo chunk (nhanh hơn)\n"
                "  📄 Full Doc — đọc toàn bộ PDF (chính xác hơn)\n",
                tag="system"
            )
            self._input_entry.focus()
        elif self._rag_ready:
            self._status_var.set("⏳ Đang load Full Doc...")
        elif self._fulldoc_ready:
            self._status_var.set("⏳ Đang load RAG Index...")

    def _on_load_error(self, name: str, error: str):
        self._append_message(f"⚠️ Không load được {name}: {error}\n", tag="error")

    # ----------------------------------------------------------
    # MODE CHANGE
    # ----------------------------------------------------------

    def _on_mode_change(self):
        mode = self._mode.get()
        if mode == "rag":
            self._append_message("🔍 Đã chuyển sang chế độ RAG (Chunk + FAISS)\n", tag="system")
        else:
            self._append_message("📄 Đã chuyển sang chế độ Full Doc (Đọc toàn bộ PDF)\n", tag="system")

    # ----------------------------------------------------------
    # CHAT
    # ----------------------------------------------------------

    def _on_send(self):
        question = self._input_var.get().strip()
        if not question:
            return

        self._input_var.set("")
        self._set_input_state("disabled")

        mode = self._mode.get()
        if mode == "rag":
            self._status_var.set("🔍 RAG: Đang tìm kiếm chunk liên quan...")
        else:
            self._status_var.set("📄 Full Doc: Đang đọc toàn bộ tài liệu...")

        self._append_message(f"🧑 Bạn: {question}\n", tag="user")

        def _worker():
            try:
                if mode == "rag":
                    answer = ask_rag(question)
                else:
                    answer = ask_full_doc(question)
                self.root.after(0, lambda: self._on_answer(answer))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._on_error(err))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_answer(self, answer: str):
        self._append_message(f"🤖 Bot: {answer}\n\n", tag="bot")
        self._status_var.set("✅ Sẵn sàng — Hãy đặt câu hỏi!")
        self._set_input_state("normal")
        self._input_entry.focus()

    def _on_error(self, error: str):
        self._append_message(f"❌ Lỗi: {error}\n\n", tag="error")
        self._status_var.set("❌ Có lỗi xảy ra")
        self._set_input_state("normal")

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def _append_message(self, text: str, tag: str = "bot"):
        self._chat_box.config(state="normal")
        self._chat_box.insert(tk.END, text, tag)
        self._chat_box.see(tk.END)
        self._chat_box.config(state="disabled")

    def _set_input_state(self, state: str):
        self._input_entry.config(state=state)
        self._send_btn.config(state=state)


# ========================================================
# 🚀 ENTRY POINT
# ========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGChatbotUI(root)
    root.mainloop()