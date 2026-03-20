"""
export2.py – Module xử lý video theo pipeline 4 giai đoạn.
Sử dụng: khởi tạo VideoExporter rồi gọi .run()

progress_callback(stage: int, pct: float, label: str)
  - stage : 0 = khởi tạo, 1–4 = các giai đoạn
  - pct   : 0.0 – 100.0
  - label : chuỗi mô tả ngắn để hiển thị trên UI
"""

import cv2
import os
import glob
import csv
import numpy as np
from collections import defaultdict
from typing import Callable
from ultralytics import YOLO

import utils
from ocr_engine import PlateOCR


class VideoExporter:
    """
    Pipeline phát hiện + OCR biển số từ file video, xuất ra:
      - Thư mục crops/   : ảnh crop rõ nhất của từng biển
      - Video annotated  : video có vẽ khung + text
      - File CSV         : timeline xuất hiện từng biển số
    """

    def __init__(
        self,
        video_input: str,
        yolo_model_path: str,
        device: str = "cpu",
        output_dir: str | None = None,
        progress_callback: Callable[[int, float, str], None] | None = None,
    ):
        """
        Parameters
        ----------
        video_input         : Đường dẫn file video đầu vào.
        yolo_model_path     : Đường dẫn weights YOLO (.pt).
        device              : 'cuda' hoặc 'cpu' — do main.py truyền vào.
        output_dir          : Thư mục lưu kết quả. Mặc định tự sinh từ tên video.
        progress_callback   : Hàm callback(stage, pct, label) để UI cập nhật tiến độ.
        """
        self.video_input = video_input
        self.progress_cb = progress_callback or (lambda s, p, l: None)

        base_name = os.path.splitext(os.path.basename(video_input))[0]
        self.output_dir = output_dir or f"{base_name}_output"
        self.crop_folder = os.path.join(self.output_dir, "crops")
        self.video_output = os.path.join(self.output_dir, f"{base_name}_EnhancedOCR.mp4")
        self.csv_output = os.path.join(self.output_dir, f"{base_name}_results.csv")

        os.makedirs(self.crop_folder, exist_ok=True)

        print(f"🎥 Video đầu vào : {self.video_input}")
        print(f"📁 Kết quả tại   : {self.output_dir}/")
        print("=" * 60)

        self.progress_cb(0, 0.0, "Đang tải model YOLO...")
        self.yolo_model = YOLO(yolo_model_path).to(device)
        print(f"✅ YOLO export chạy trên [{device.upper()}]")
        self.progress_cb(0, 50.0, "Đang tải PaddleOCR...")
        self.ocr_engine = PlateOCR()
        self.progress_cb(0, 100.0, "Sẵn sàng xử lý!")

    # ------------------------------------------------------------------
    # PRIVATE – 4 giai đoạn
    # ------------------------------------------------------------------

    def _stage1_scan(self):
        """STAGE 1: Chạy YOLO tracking, lưu crop rõ nhất mỗi track."""
        print("\n[1/4] STAGE 1: Scanning plates YOLO...")
        self.progress_cb(1, 0.0, "Stage 1/4: Scanning YOLO...")

        cap = cv2.VideoCapture(self.video_input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_data = defaultdict(list)
        best_crops = {}
        margin = 15
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pct = (frame_count / max(total_frames, 1)) * 100
            self.progress_cb(1, pct,
                             f"Stage 1/4: Scanning frame {frame_count}/{total_frames}")
            print(f"\r⏳ Scanning: {frame_count}/{total_frames} ({pct:.1f}%)".ljust(60),
                  end="", flush=True)

            results = self.yolo_model.track(
                frame, persist=True, imgsz=1920, conf=0.35, verbose=False
            )

            if results[0].obb is not None and results[0].obb.id is not None:
                boxes = results[0].obb.xyxyxyxy.cpu().numpy()
                track_ids = results[0].obb.id.cpu().numpy().astype(int)

                for box, track_id in zip(boxes, track_ids):
                    frame_data[frame_count].append({'id': track_id, 'box': box})

                    x_coords, y_coords = box[:, 0], box[:, 1]
                    inside = (
                        (x_coords >= margin).all() and
                        (x_coords <= frame_w - margin).all() and
                        (y_coords >= margin).all() and
                        (y_coords <= frame_h - margin).all()
                    )

                    if inside:
                        plate_crop = utils.warp_plate(frame, box)
                        if plate_crop is None or plate_crop.size == 0:
                            continue

                        h_c, w_c = plate_crop.shape[:2]
                        sharpness = cv2.Laplacian(
                            cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                        ).var()
                        score = h_c * w_c * sharpness

                        if track_id not in best_crops or score > best_crops[track_id]['score']:
                            best_crops[track_id] = {'score': score, 'img': plate_crop.copy()}

            frame_count += 1

        cap.release()

        for track_id, data in best_crops.items():
            cv2.imwrite(os.path.join(self.crop_folder, f"track_{track_id}.jpg"), data['img'])

        self.progress_cb(1, 100.0, f"Stage 1/4 xong — {len(best_crops)} biển tìm được")
        print("\r✅ STAGE 1 Hoàn tất!".ljust(60))
        return frame_data, fps, frame_w, frame_h, total_frames

    def _stage2_ocr(self):
        """STAGE 2: Using OCR to read cropped images."""
        print("\n[2/4] STAGE 2: OCR's reading plates...")
        self.progress_cb(2, 0.0, "Stage 2/4: OCR is reading...")

        final_results = {}
        crop_images = glob.glob(os.path.join(self.crop_folder, "*.jpg"))
        total_crops = len(crop_images)

        for i, img_path in enumerate(crop_images):
            track_id = int(os.path.basename(img_path).split('_')[1].split('.')[0])
            plate_img = cv2.imread(img_path)

            if plate_img is not None:
                text = self.ocr_engine.get_text(plate_img)
                final_results[track_id] = text
                pct = (i + 1) / max(total_crops, 1) * 100
                self.progress_cb(2, pct,
                                 f"Stage 2/4: Plate {track_id} → {text}")
                print(f"\r⏳ OCR: {i+1}/{total_crops} ({pct:.1f}%) | Plate {track_id} → {text}".ljust(80),
                      end="", flush=True)

        self.progress_cb(2, 100.0, f"Stage 2/4 Done — {len(final_results)} read successfully")
        print("\r✅ STAGE 2 Completed!".ljust(80))
        return final_results

    def _stage3_render(self, frame_data, final_results, fps, frame_w, frame_h, total_frames):
        """STAGE 3: Export video annotated."""
        print(f"\n[3/4] STAGE 3: Rendering video → {self.video_output}...")
        self.progress_cb(3, 0.0, "Stage 3/4: Rendering video...")

        cap = cv2.VideoCapture(self.video_input)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.video_output, fourcc, int(fps), (frame_w, frame_h))

        if not out.isOpened():
            print("\r⚠️  Codec MP4 lỗi, chuyển sang AVI...".ljust(60))
            self.video_output = self.video_output.replace('.mp4', '.avi')
            out = cv2.VideoWriter(
                self.video_output, cv2.VideoWriter_fourcc(*'XVID'), int(fps), (frame_w, frame_h)
            )

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            pct = (frame_count / max(total_frames, 1)) * 100
            self.progress_cb(3, pct,
                             f"Stage 3/4: Render frame {frame_count}/{total_frames}")
            print(f"\r⏳ Render: {frame_count}/{total_frames} ({pct:.1f}%)".ljust(60),
                  end="", flush=True)

            if frame_count in frame_data:
                for obj in frame_data[frame_count]:
                    track_id = obj['id']
                    box = obj['box']
                    text = final_results.get(track_id, "Unknown")
                    cv2.polylines(frame, [box.astype(int)], True, (0, 255, 0), 2)
                    cv2.putText(frame, text,
                                (int(box[0][0]), int(box[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        self.progress_cb(3, 100.0, "Stage 3/4 Done — Video has been rendered")
        print("\r✅ STAGE 3 Completed!".ljust(60))

    def _stage4_csv(self, frame_data, final_results, fps):
        """STAGE 4: Collective timeline and exporting CSV."""
        print(f"\n[4/4] STAGE 4: Exporting CSV → {self.csv_output}...")
        self.progress_cb(4, 0.0, "Stage 4/4: Exporting CSV...")
 
        track_timings = defaultdict(lambda: {"start": float('inf'), "end": 0})
        for f_idx, objects in frame_data.items():
            for obj in objects:
                t_id = obj['id']
                track_timings[t_id]["start"] = min(track_timings[t_id]["start"], f_idx)
                track_timings[t_id]["end"] = max(track_timings[t_id]["end"], f_idx)
 
        with open(self.csv_output, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["Track ID", "Plate Number (OCR)",
                             "Appearance Time", "Disappearance Time"])
 
            for t_id, text in final_results.items():
                text_clean = utils.clean_plate(text)
                if "scanning" in text_clean.lower() or text_clean == "":
                    continue
                if t_id in track_timings:
                    writer.writerow([
                        t_id,
                        text_clean,
                        utils.format_time(track_timings[t_id]["start"], fps),
                        utils.format_time(track_timings[t_id]["end"], fps),
                    ])
 
        self.progress_cb(4, 100.0, "✅ Completed entire pipeline!")
        print("✅ STAGE 4 Completed! CSV has been saved.")

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def run(self):
        """Chạy toàn bộ pipeline 4 giai đoạn."""
        frame_data, fps, frame_w, frame_h, total_frames = self._stage1_scan()
        final_results = self._stage2_ocr()
        self._stage3_render(frame_data, final_results, fps, frame_w, frame_h, total_frames)
        self._stage4_csv(frame_data, final_results, fps)

        print(f"\n🎉 HOÀN THÀNH! Xem kết quả tại [{self.output_dir}]")
        return {
            "output_dir": self.output_dir,
            "video_output": self.video_output,
            "csv_output": self.csv_output,
            "results": final_results,
        }
