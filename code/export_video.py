import cv2
from collections import Counter
from ultralytics import YOLO
import utils
from ocr_engine import PlateOCR
import csv

VIDEO_INPUT = r"D:\Study_FPTU\DAP391m\plate detection\random data\12307593_2160_3840_60fps.mp4"
VIDEO_OUTPUT = r"Ket_qua_Export_Clean4.mp4"
CSV_OUTPUT = r"Danh_sach_bien_so.csv" # <--- File kết quả trả về

yolo_model = YOLO(r"D:\Study_FPTU\DAP391m\plate detection\runs\obb\runs\train\plate_model_v17\weights\best.pt").to("cuda")
ocr_engine = PlateOCR()

cap = cv2.VideoCapture(VIDEO_INPUT)
width, height = int(cap.get(3)), int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
\
ocr_history = {}       
final_results = {}     
csv_data = [] # Nơi lưu trữ tạm thời để xuất file
processed_ids = set() # Để đảm bảo mỗi xe chỉ ghi vào CSV đúng 1 lần khi đã chốt số

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
print(f"Bắt đầu xuất video MP4: Tổng cộng {total_frames} frames... ( đã chia 2 do tối ưu AI Tracking )")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count >= total_frames: break
    
    # Chạy tracking mỗi 2 frame để tăng tốc
    if frame_count % 2 == 0:
        results = yolo_model.track(frame, persist=True, imgsz=1920, conf=0.45, verbose=False)
        
        if results[0].obb is not None and results[0].obb.id is not None:
            boxes = results[0].obb.xyxyxyxy.cpu().numpy()
            track_ids = results[0].obb.id.cpu().numpy().astype(int)

            for box_pts, track_id in zip(boxes, track_ids):
                # 1. LOGIC BẦU CỬ OCR
                if track_id not in final_results:
                    if track_id not in ocr_history: ocr_history[track_id] = []
                    
                    if len(ocr_history[track_id]) < 7:
                        plate_crop = utils.warp_plate(frame, box_pts)
                        text = ocr_engine.get_text(plate_crop)
                        if len(text) > 4 and "SCANNING" not in text.upper():
                            ocr_history[track_id].append(text)
                    
                    # Khi đã đủ 7 frame để chốt số
                    if len(ocr_history[track_id]) >= 7:
                        best_text = Counter(ocr_history[track_id]).most_common(1)[0][0]
                        final_results[track_id] = best_text
                        
                        # 👉 GHI VÀO DANH SÁCH CSV: Tính giây dựa trên frame_count
                        if track_id not in processed_ids:
                            timestamp_seconds = round(frame_count / fps, 2)
                            csv_data.append([best_text, timestamp_seconds])
                            processed_ids.add(track_id)
                            print(f"📌 Đã chốt: {best_text} tại giây thứ {timestamp_seconds}")

                # 2. HIỂN THỊ LÊN VIDEO
                current_text = final_results.get(track_id, f"Scanning ({len(ocr_history.get(track_id, []))}/7)...")
                utils.draw_plate_video(frame, box_pts.astype(int), f"ID:{track_id} | {current_text}")

    out.write(frame)
    frame_count += 1
    if frame_count % 20 == 0:
        print(f"Đang render: {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)")

# --- XUẤT FILE CSV ---
with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Bien so', 'Time (s)']) # Ghi tiêu đề cột
    writer.writerows(csv_data)

cap.release()
out.release()
print(f"✅ HOÀN TẤT! Video lưu tại {VIDEO_OUTPUT} và Data lưu tại {CSV_OUTPUT}")
