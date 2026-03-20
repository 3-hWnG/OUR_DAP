"""
main.py – Giao diện chính (Tkinter UI thuần tuý).
Toàn bộ logic xử lý được uỷ quyền cho:
  - export2.py  (VideoExporter)   : pipeline xuất video + CSV
  - utils.py                      : vẽ, crop, filter
  - ocr_engine.py (PlateOCR)      : đọc biển số
"""

import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
from ultralytics import YOLO

import utils
from ocr_engine import PlateOCR
from export2 import VideoExporter


# ========================================================
# ⚙️  CẤU HÌNH MẶC ĐỊNH
# ========================================================
DEFAULT_REALTIME_WEIGHTS = ''
DEFAULT_EXPORT_WEIGHTS   = ''

# ========================================================
# 🖥️  CỬA SỔ CÀI ĐẶT MODEL
# ========================================================

class ModelSettingsWindow(tk.Toplevel):
    """Cửa sổ popup để người dùng chọn / thay đổi model weights."""

    def __init__(self, parent, realtime_var: tk.StringVar, export_var: tk.StringVar,
                 gpu_var: tk.BooleanVar, on_apply):
        super().__init__(parent)
        self.title("⚙️Setting Model")
        self.geometry("540x290")
        self.resizable(False, False)
        self.grab_set()   # Modal

        self.on_apply = on_apply

        pad = {"padx": 12, "pady": 6}

        # ── Realtime model ──────────────────────────────────────────
        tk.Label(self, text="Model Realtime (Webcam / Video preview)",
                 font=("Arial", 9, "bold")).grid(row=0, column=0, columnspan=3,
                                                  sticky="w", **pad)

        self._rt_entry = tk.Entry(self, textvariable=realtime_var, width=48)
        self._rt_entry.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12)

        tk.Button(self, text="📂 Browse",
                  command=lambda: self._browse(realtime_var)).grid(row=1, column=2,
                                                                    padx=(4, 12))

        # ── Export model ────────────────────────────────────────────
        tk.Label(self, text="Model Export (export video + CSV)",
                 font=("Arial", 9, "bold")).grid(row=2, column=0, columnspan=3,
                                                  sticky="w", **pad)

        self._ex_entry = tk.Entry(self, textvariable=export_var, width=48)
        self._ex_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=12)

        tk.Button(self, text="📂 Browse",
                  command=lambda: self._browse(export_var)).grid(row=3, column=2,
                                                                  padx=(4, 12))

        # ── GPU option ───────────────────────────────────────────────
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_text  = "🟢  USE GPU (CUDA)"
            gpu_color = "#1a7a1a"
            gpu_state = "normal"
        else:
            gpu_text  = "🔴  USE GPU (CUDA)  — Not available"
            gpu_color = "#cc0000"
            gpu_state = "disabled"
            gpu_var.set(False)

        tk.Checkbutton(self, text=gpu_text, variable=gpu_var,
                       fg=gpu_color, state=gpu_state).grid(
            row=4, column=0, columnspan=3, sticky="w", padx=12, pady=8)

        # ── Nút ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=10)

        tk.Button(btn_frame, text="✅ Save & load Model",
                  command=self._apply, width=26, height=2,
                  bg="#2ecc71", fg="white", font=("Arial", 10, "bold")).pack(side="left", padx=8)

        tk.Button(btn_frame, text="✖  Cancel",
                  command=self.destroy, width=10, height=2).pack(side="left", padx=8)

        self.columnconfigure(0, weight=1)

    def _browse(self, var: tk.StringVar):
        path = filedialog.askopenfilename(
            title="Choose YOLO weights file",
            filetypes=[("PyTorch weights", "*.pt"), ("All files", "*.*")]
        )
        if path:
            var.set(path)

    def _apply(self):
        self.on_apply()
        self.destroy()


# ========================================================
# 📊 PANEL PROGRESS EXPORT
# ========================================================

class ExportProgressPanel(tk.Frame):
    """
    Panel nhỏ hiển thị tiến độ export gồm:
      - Label stage hiện tại
      - ProgressBar tổng thể (4 giai đoạn)
      - ProgressBar giai đoạn hiện tại
      - Label log dòng cuối
    """

    STAGE_WEIGHT = [15, 10, 70, 5]   # Tỉ trọng % của từng stage (tổng = 100)

    def __init__(self, parent, **kwargs):
        super().__init__(parent, relief="groove", bd=2, **kwargs)

        tk.Label(self, text="EXPORT VIDEO PROGRESS", font=("Arial", 9, "bold")).pack(pady=(8, 2))

        # Tổng thể
        tk.Label(self, text="Overall:", anchor="w").pack(fill="x", padx=10)
        self._total_bar = ttk.Progressbar(self, length=340, mode="determinate", maximum=100)
        self._total_bar.pack(padx=10, pady=(0, 4))

        # Giai đoạn hiện tại
        self._stage_label = tk.Label(self, text="Waiting...", anchor="w",
                                     font=("Arial", 8), fg="#555")
        self._stage_label.pack(fill="x", padx=10)
        self._stage_bar = ttk.Progressbar(self, length=340, mode="determinate", maximum=100)
        self._stage_bar.pack(padx=10, pady=(0, 4))

        # Log text
        self._log_label = tk.Label(self, text="", anchor="w", wraplength=330,
                                   font=("Arial", 8), fg="#333")
        self._log_label.pack(fill="x", padx=10, pady=(0, 8))

    def update_progress(self, stage: int, pct: float, label: str):
        """
        Được gọi từ thread worker qua widget.after() để an toàn với Tkinter.
        stage = 0 (init), 1–4 (pipeline stages)
        """
        # ── Cập nhật stage bar ──────────────────────────────────────
        self._stage_bar["value"] = pct
        self._stage_label["text"] = label
        self._log_label["text"] = label

        # ── Tính tổng thể ───────────────────────────────────────────
        if stage == 0:
            total = pct * 0.05          # Init chiếm 5% tổng
        else:
            offset = sum(self.STAGE_WEIGHT[:stage - 1])
            total = offset + (pct / 100) * self.STAGE_WEIGHT[stage - 1]
            # Cộng thêm 5% của init đã xong
            total = 5 + total * 0.95

        self._total_bar["value"] = min(total, 100)

    def reset(self):
        self._total_bar["value"] = 0
        self._stage_bar["value"] = 0
        self._stage_label["text"] = "Chờ bắt đầu..."
        self._log_label["text"] = ""


# ========================================================
# 🖥️  MAIN UI
# ========================================================

class PlateSystem:
    def __init__(self, window):
        self.window = window
        self.window.title("License Plate System")
        self.window.geometry("420x720")
        self.window.resizable(False, False)

        # --- State model paths (có thể thay đổi qua Settings) ---
        self._realtime_weights = tk.StringVar(value=DEFAULT_REALTIME_WEIGHTS)
        self._export_weights   = tk.StringVar(value=DEFAULT_EXPORT_WEIGHTS)
        self._use_gpu          = tk.BooleanVar(value=torch.cuda.is_available())

        # --- Cache kết quả realtime ---
        self.results_cache: dict[int, str] = {}
        self.is_ocr_busy = False

        # --- Khởi tạo models ---
        self.yolo_model: YOLO | None = None
        self.ocr_engine: PlateOCR | None = None
        self._load_models()

        # ── Tiêu đề ─────────────────────────────────────────────────
        tk.Label(window, text="LICENSE PLATE SYSTEM",
                 font=("Arial", 15, "bold")).pack(pady=(18, 4))

        # ── Model info bar ───────────────────────────────────────────
        self._model_info = tk.Label(
            window,
            text=self._short_model_label(),
            font=("Arial", 8), fg="#555", wraplength=380, justify="center"
        )
        self._model_info.pack(pady=(0, 10))

        # ── Nút chính ────────────────────────────────────────────────
        btn_cfg = {"width": 26, "height": 2}

        tk.Button(window, text="📸  WEBCAM REALTIME",
                  command=lambda: threading.Thread(
                      target=self._start_stream, args=("webcam",), daemon=True
                  ).start(), **btn_cfg).pack(pady=5)

        tk.Button(window, text="▶️   WATCH VIDEO REALTIME",
                  command=lambda: threading.Thread(
                      target=self._start_stream, args=("video",), daemon=True
                  ).start(), **btn_cfg).pack(pady=5)

        self._export_btn = tk.Button(
            window, text="📤  EXPORT VIDEO + CSV",
            command=self._run_export, **btn_cfg
        )
        self._export_btn.pack(pady=5)

        tk.Button(window, text="🖼️   DETECT STATIC IMAGE",
                  command=self._process_static_image, **btn_cfg).pack(pady=5)

        # ── Nút Settings ─────────────────────────────────────────────
        tk.Button(window, text="⚙️   SETTINGS MODELS",
                  command=self._open_settings,
                  width=26, height=1, fg="#1a6ea8",
                  font=("Arial", 9, "bold")).pack(pady=(10, 4))

        # ── Progress panel ───────────────────────────────────────────
        self._progress_panel = ExportProgressPanel(window)
        self._progress_panel.pack(fill="x", padx=18, pady=(10, 18))

    # ----------------------------------------------------------
    # MODEL LOADING
    # ----------------------------------------------------------

    def _get_device(self) -> str:
        """Trả về device dựa trên lựa chọn GPU của người dùng."""
        if self._use_gpu.get() and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_models(self):
        """Tải / tải lại YOLO realtime và PlateOCR."""
        rt_path = self._realtime_weights.get().strip()
        if not rt_path:
            print("⚠️  Choose realtime model.")
            self.yolo_model = None
            return 
        device = self._get_device()
        try:
            print(f"⏳ Loading YOLO from: {rt_path}  [{device.upper()}]")
            self.yolo_model = YOLO(rt_path).to(device)
            print(f"✅ YOLO realtime ready on [{device.upper()}]")
        except Exception as e:
            messagebox.showerror("Error loading YOLO model", str(e))
            self.yolo_model = None

        if self.ocr_engine is None:
            self.ocr_engine = PlateOCR(use_gpu=self._use_gpu.get())

    def _short_model_label(self) -> str:
        import os
        rt  = os.path.basename(self._realtime_weights.get()) or "(chưa chọn)"
        ex  = os.path.basename(self._export_weights.get())   or "(chưa chọn)"
        dev = self._get_device().upper()
        return f"Realtime: {rt}  |  Export: {ex}  |  Device: {dev}"

    # ----------------------------------------------------------
    # SETTINGS
    # ----------------------------------------------------------

    def _open_settings(self):
        ModelSettingsWindow(
            self.window,
            realtime_var=self._realtime_weights,
            export_var=self._export_weights,
            gpu_var=self._use_gpu,
            on_apply=self._on_settings_applied,
        )

    def _on_settings_applied(self):
        """Callback sau khi người dùng nhấn Áp dụng trong Settings."""
        self.results_cache.clear()
        self._load_models()
        self._model_info["text"] = self._short_model_label()
        messagebox.showinfo("Success", "✅ Load complete!")

    # ----------------------------------------------------------
    # REALTIME (webcam / video preview)
    # ----------------------------------------------------------

    def _ocr_worker(self, plate_img, track_id):
        """Chạy OCR trên luồng riêng, lưu kết quả vào cache."""
        self.is_ocr_busy = True
        text = self.ocr_engine.get_text(plate_img)
        if text and text != "Scanning...":
            self.results_cache[track_id] = text
        self.is_ocr_busy = False

    def _start_stream(self, mode: str):
        """Mở webcam hoặc video, chạy YOLO + OCR realtime."""
        if self.yolo_model is None:
            messagebox.showerror("Error", "Model cant be loaded. Please check settings.")
            return

        source = 0 if mode == "webcam" else filedialog.askopenfilename(
            title="Choose video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if source == "" or source is None:
            return

        cap = cv2.VideoCapture(source)
        win_name = "AI Traffic Monitor"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)

        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.resizeWindow(win_name, int(v_w * (720 / v_h)), 720)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if mode == "video":
            cv2.createTrackbar("Frame", win_name, 0, max(total_frames, 1), lambda x: None)

        self.results_cache.clear()
        paused = False
        current_results = None
        frame_count = 0
        frozen_display = None

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break


                if frame_count % 3 == 0 or current_results is None:
                    current_results = self.yolo_model.track(
                        frame, persist=True, imgsz=480, conf=0.45, verbose=False
                    )
                frame_count += 1

                if current_results[0].obb is not None and current_results[0].obb.id is not None:
                    boxes = current_results[0].obb.xyxyxyxy.cpu().numpy()
                    track_ids = current_results[0].obb.id.cpu().numpy().astype(int)
                    h_f, w_f = frame.shape[:2]

                    for box_pts, track_id in zip(boxes, track_ids):
                        pts = box_pts.astype(np.int32)
                        current_text = self.results_cache.get(track_id, "Reading...")

                        if track_id not in self.results_cache and not self.is_ocr_busy:
                            if utils.is_in_safe_zone(pts, w_f, h_f):
                                plate_crop = utils.warp_plate(frame, box_pts)
                                threading.Thread(
                                    target=self._ocr_worker,
                                    args=(plate_crop, track_id),
                                    daemon=True
                                ).start()

                        utils.draw_plate_video(frame, pts, f"ID:{track_id} | {current_text}")

                frozen_display = frame.copy()

                if mode == "video":
                    try:
                        cv2.setTrackbarPos("Frame", win_name,
                                           int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    except cv2.error:
                        pass

            if frozen_display is not None:
                cv2.imshow(win_name, frozen_display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
            elif key in [81, 83] and mode == "video":
                step = 30 if key == 83 else -30
                new_pos = max(0, min(total_frames - 1,
                                     cap.get(cv2.CAP_PROP_POS_FRAMES) + step))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                self.results_cache.clear()
                ret, frame = cap.read()
                if ret:
                    frozen_display = frame.copy()

        cap.release()
        cv2.destroyAllWindows()

    # ----------------------------------------------------------
    # EXPORT VIDEO + CSV
    # ----------------------------------------------------------

    def _run_export(self):
        """Chọn video → chạy VideoExporter pipeline 4 giai đoạn."""
        video_path = filedialog.askopenfilename(
            title="Choose video to export",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if not video_path:
            return

        # Khoá nút tránh bấm 2 lần
        self._export_btn.config(state="disabled", text="⏳ Exporting...")
        self._progress_panel.reset()

        def _on_progress(stage: int, pct: float, label: str):
            # Callback từ worker thread → phải dùng after() để update UI an toàn
            self.window.after(0, self._progress_panel.update_progress, stage, pct, label)

        def _worker():
            try:
                exporter = VideoExporter(
                    video_input=video_path,
                    yolo_model_path=self._export_weights.get(),
                    device=self._get_device(),
                    progress_callback=_on_progress,
                )
                result = exporter.run()
                self.window.after(0, lambda: messagebox.showinfo(
                    "Completed!",
                    f"✅ Export completed!\n\n"
                    f"📁 Folder : {result['output_dir']}\n"
                    f"🎬 Video   : {result['video_output']}\n"
                    f"📊 CSV     : {result['csv_output']}"
                ))
            except Exception as e:
                err_msg = str(e)
                self.window.after(0, lambda m=err_msg: messagebox.showerror("Error Export", m))
            finally:
                # Mở khoá nút sau khi xong
                self.window.after(0, lambda: self._export_btn.config(
                    state="normal", text="📤  EXPORT VIDEO + CSV"
                ))

        threading.Thread(target=_worker, daemon=True).start()

    # ----------------------------------------------------------
    # ẢNH TĨNH
    # ----------------------------------------------------------

    def _process_static_image(self):
        """Chọn ảnh → nhận diện biển số → hiển thị kết quả."""
        if self.yolo_model is None:
            messagebox.showerror("Error", "Model cant be loaded. Please check settings.")
            return

        path = filedialog.askopenfilename(
            title="Choose image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return

        img = cv2.imread(path)
        results = self.yolo_model.predict(img, conf=0.3)

        for r in results:
            if r.obb:
                for box in r.obb.xyxyxyxy:
                    pts = box.cpu().numpy().astype(int)
                    plate_crop = utils.warp_plate(img, box.cpu().numpy())
                    text = self.ocr_engine.get_text(plate_crop)
                    utils.draw_plate_static(img, pts, text)

        cv2.namedWindow("Result Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Result Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ========================================================
# 🚀 ENTRY POINT
# ========================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateSystem(root)
    root.mainloop()
