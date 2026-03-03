import torch
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import threading
from ultralytics import YOLO

# Nạp 2 file module bạn vừa tạo vào đây
import utils
from ocr_engine import PlateOCR

class PlateSystem:
    def __init__(self, window):
        self.window = window
        self.window.title("Hệ thống Nhận diện Biển số OBB")
        self.window.geometry("400x500")
        
        self.results_cache = {} 
        self.is_ocr_busy = False 
        
        # 1. Khởi tạo Models
        self.yolo_model = YOLO(r"D:\Study_FPTU\DAP391m\plate detection\runs\obb\runs\train\plate_model_v17\weights\best.pt").to("cuda" if torch.cuda.is_available() else "cpu")
        self.ocr_engine = PlateOCR() # Gọi class từ file ocr_engine.py
        
        device_name = self.yolo_model.device.type
        print(f"--- THÔNG BÁO: YOLO đang chạy trên: [{device_name.upper()}] ---")

        # 2. Giao diện HUD
        tk.Label(window, text="LICENSE PLATE SYSTEM", font=("Arial", 14, "bold")).pack(pady=20)
        tk.Button(window, text="📸 WEBCAM REALTIME", 
          command=lambda: threading.Thread(target=self.start, args=("webcam",), daemon=True).start(), 
          width=20, height=2).pack(pady=10)
        tk.Button(window, text="📁 CHỌN VIDEO", command=lambda: self.start("video"), width=20, height=2).pack(pady=10)
        tk.Button(window, text="🖼️ CHỌN ẢNH", command=self.process_static_image, width=20, height=2).pack(pady=10)

    def ocr_worker_with_id(self, plate_img, track_id):
        self.is_ocr_busy = True
        # Gọi hàm get_text từ engine
        text = self.ocr_engine.get_text(plate_img)
        if text and text != "Scanning...":
            self.results_cache[track_id] = text 
        self.is_ocr_busy = False

    def start(self, mode):
        source = 0 if mode == "webcam" else filedialog.askopenfilename()
        if not source: return
        
        cap = cv2.VideoCapture(source)
        window_name = "AI Traffic Monitor"
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        
        v_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.resizeWindow(window_name, int(v_w * (720/v_h)), 720)
        
        paused = False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.results_cache.clear() 
        current_results = None 
        frame_count = 0
        frozen_display = None
        
        if mode == "video":
            cv2.createTrackbar("Frame", window_name, 0, total_frames, lambda x: None)

        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1) if mode == "webcam" else cv2.flip(frame, -1)

                if frame_count % 3 == 0 or current_results is None:
                    results = self.yolo_model.track(frame, persist=True, imgsz=480, conf=0.45, verbose=False)
                    current_results = results
                else:
                    results = current_results
                
                frame_count += 1

                if results[0].obb is not None and results[0].obb.id is not None:
                    boxes = results[0].obb.xyxyxyxy.cpu().numpy()
                    track_ids = results[0].obb.id.cpu().numpy().astype(int)
                    h_f, w_f = frame.shape[:2]

                    for box_pts, track_id in zip(boxes, track_ids):
                        pts = box_pts.astype(np.int32)
                        current_text = self.results_cache.get(track_id, "Reading...")

                        if track_id not in self.results_cache and not self.is_ocr_busy:
                            # Gọi hàm từ utils
                            if utils.is_in_safe_zone(pts, w_f, h_f):
                                plate_crop = utils.warp_plate(frame, box_pts)
                                threading.Thread(target=self.ocr_worker_with_id, args=(plate_crop, track_id), daemon=True).start()

                        # Gọi hàm vẽ từ utils
                        utils.draw_plate_video(frame, pts, f"ID:{track_id} | {current_text}")

                frozen_display = frame.copy()

                if mode == "video":
                    cv2.setTrackbarPos("Frame", window_name, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))

            if frozen_display is not None:
                cv2.imshow(window_name, frozen_display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'): break
            elif key == ord(' '): paused = not paused
            elif key in [81, 83] and mode == "video":
                step = 30 if key == 83 else -30
                new_pos = max(0, min(total_frames - 1, cap.get(cv2.CAP_PROP_POS_FRAMES) + step))
                cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                self.results_cache.clear()
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, -1)
                    frozen_display = frame.copy()

        cap.release()
        cv2.destroyAllWindows()

    def process_static_image(self):
        path = filedialog.askopenfilename()
        if not path: return
        img = cv2.imread(path)
        
        results = self.yolo_model.predict(img, conf=0.3) 
        
        for r in results:
            if r.obb:
                for box in r.obb.xyxyxyxy:
                    pts = box.cpu().numpy().astype(int)
                    # Gọi hàm từ utils và ocr_engine
                    plate_crop = utils.warp_plate(img, box.cpu().numpy())
                    text = self.ocr_engine.get_text(plate_crop)
                    utils.draw_plate_static(img, pts, text)
        
        cv2.namedWindow("Result Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Result Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    root = tk.Tk()
    app = PlateSystem(root)
    root.mainloop()
