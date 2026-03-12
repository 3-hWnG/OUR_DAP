import cv2
import os
import glob
import csv
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
import utils
from ocr_engine import PlateOCR

# ========================================================
# 🎯 TÙY CHỈNH ĐƯỜNG DẪN TẠI ĐÂY 
# ========================================================
VIDEO_INPUT = r"D:\Study_FPTU\DAP391m\plate detection\random data\video-20260309T034344Z-1-001\video\t6.mp4"
YOLO_MODEL_PATH = r"D:\Study_FPTU\DAP391m\plate detection\runs\obb\train9\weights\best.pt"
FORCE_ROTATE_180 = True

base_name = os.path.splitext(os.path.basename(VIDEO_INPUT))[0]

main_output_folder = f"{base_name}_output"
os.makedirs(main_output_folder, exist_ok=True)

crop_folder = os.path.join(main_output_folder, "crops")
VIDEO_OUTPUT = os.path.join(main_output_folder, f"{base_name}_EnhancedOCR.mp4")
CSV_OUTPUT = os.path.join(main_output_folder, f"{base_name}_results.csv")

os.makedirs(crop_folder, exist_ok=True)

print(f"🎥 Đang xử lý video: {VIDEO_INPUT}")
print(f"📁 Toàn bộ kết quả sẽ lưu tại: {main_output_folder}/")
print("=" * 60)

yolo_model = YOLO(YOLO_MODEL_PATH).to("cuda")

# ========================================================
# 🛠️ CÁC HÀM TIỆN ÍCH
# ========================================================

def format_time(frame_count, fps):
    total_seconds = int(frame_count / fps)
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

# 👉 HÀM MỚI: Tự động lật khung hình dựa trên Metadata của điện thoại
def auto_rotate_frame(frame, orientation):
    if orientation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


# ========================================================
# 🚀 STAGE 1: BÁM VẾT & TÌM ẢNH NÉT NHẤT
# ========================================================
print("\n[1/4] STAGE 1: Đang chạy YOLO quét biển số...")

cap = cv2.VideoCapture(VIDEO_INPUT)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Lấy kích thước gốc
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Lấy metadata xoay để lát nữa dùng lật 180 độ
orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)

frame_data = defaultdict(list) 
best_crops = {} 
margin = 15     
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 👉 LẬT KHUNG HÌNH (Nếu bị ngược/Ngang) TRƯỚC KHI ĐƯA VÀO YOLO
    if FORCE_ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    percent = (frame_count / total_frames) * 100
    print(f"\r⏳ Tiến độ quét: {frame_count}/{total_frames} frames ({percent:.1f}%)".ljust(60), end="", flush=True)
    
    results = yolo_model.track(frame, persist=True, imgsz=1920, conf=0.35, verbose=False)
    
    if results[0].obb is not None and results[0].obb.id is not None:
        boxes = results[0].obb.xyxyxyxy.cpu().numpy()
        track_ids = results[0].obb.id.cpu().numpy().astype(int)
        
        for box, track_id in zip(boxes, track_ids):
            frame_data[frame_count].append({'id': track_id, 'box': box})
            
            x_coords = box[:, 0]
            y_coords = box[:, 1]
            inside_frame = (x_coords >= margin).all() and (x_coords <= frame_w - margin).all() and \
                           (y_coords >= margin).all() and (y_coords <= frame_h - margin).all()
            
            if inside_frame:
                plate_crop = utils.warp_plate(frame, box)
                
                if plate_crop is None or plate_crop.size == 0:
                    continue
                
                h_crop, w_crop = plate_crop.shape[:2]
                area = h_crop * w_crop
                gray_crop = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                
                quality_score = area * sharpness
                
                if track_id not in best_crops or quality_score > best_crops[track_id]['score']:
                    best_crops[track_id] = {
                        'score': quality_score,
                        'img': plate_crop.copy() 
                    }
                    
    frame_count += 1
cap.release()

for track_id, data in best_crops.items():
    img_path = os.path.join(crop_folder, f"track_{track_id}.jpg")
    cv2.imwrite(img_path, data['img'])

print("\r✅ STAGE 1 Hoàn tất! Đã lưu các bức ảnh rõ nét nhất.".ljust(60))

# ========================================================
# 🔬 STAGE 2: TINH LUYỆN & ĐỌC OCR
# ========================================================
print("\n[2/4] STAGE 2: Xử lý làm nét ảnh và đọc OCR...")
ocr_engine = PlateOCR() 

final_results = {} 
crop_images = glob.glob(f"{crop_folder}/*.jpg")
total_crops = len(crop_images)

# export2.py (STAGE 2)
for i, img_path in enumerate(crop_images):
    track_id = int(os.path.basename(img_path).split('_')[1].split('.')[0])
    plate_img = cv2.imread(img_path)
    
    if plate_img is not None:
        # BỎ HÀM TĂNG CƯỜNG Ở ĐÂY, CHUYỂN THẲNG ẢNH GỐC VÀO OCR
        text = ocr_engine.get_text(plate_img) 
        final_results[track_id] = text
        
        percent = ((i + 1) / total_crops) * 100
        print(f"\r⏳ Tiến độ OCR: {i+1}/{total_crops} ({percent:.1f}%) | Xe {track_id} -> {text}".ljust(80), end="", flush=True)

print("\r✅ STAGE 2 Hoàn tất! Đã đọc xong nội dung biển số.".ljust(80))

# ========================================================
# 🎬 STAGE 3: ĐÓNG GÓI VIDEO
# ========================================================
print(f"\n[3/4] STAGE 3: Kết xuất video tại: {VIDEO_OUTPUT}...")

cap = cv2.VideoCapture(VIDEO_INPUT)

# Vẫn dùng w và h đã được tự động đảo ngược/lấy chuẩn ở Stage 1 để tạo VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(VIDEO_OUTPUT, fourcc, int(fps), (frame_w, frame_h))

if not out_video.isOpened():
    print("\r⚠️ Codec MP4 bị lỗi. Tự động chuyển qua chuẩn .AVI...".ljust(60))
    VIDEO_OUTPUT = VIDEO_OUTPUT.replace('.mp4', '.avi')
    out_video = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'XVID'), int(fps), (frame_w, frame_h))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    # 👉 LẬT KHUNG HÌNH Y NHƯ LÚC QUÉT TRƯỚC KHI VẼ TEXT
    if FORCE_ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    
    percent = (frame_count / total_frames) * 100
    print(f"\r⏳ Đang Render Video: {frame_count}/{total_frames} frames ({percent:.1f}%)".ljust(60), end="", flush=True)
    
    if frame_count in frame_data:
        for obj in frame_data[frame_count]:
            track_id = obj['id']
            box = obj['box']
            text = final_results.get(track_id, "Unknown")
            
            cv2.polylines(frame, [box.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(frame, text, (int(box[0][0]), int(box[0][1]) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
    out_video.write(frame)
    frame_count += 1

cap.release()
out_video.release()
print("\r✅ STAGE 3 Hoàn tất! Render video thành công.".ljust(60))

# ========================================================
# 📊 STAGE 4: TỔNG HỢP TIMELINE & XUẤT CSV (LỌC RÁC)
# ========================================================
print(f"\n[4/4] STAGE 4: Đang phân tích Timeline và xuất file {CSV_OUTPUT}...")

track_timings = defaultdict(lambda: {"start": float('inf'), "end": 0})

for f_idx, objects in frame_data.items():
    for obj in objects:
        t_id = obj['id']
        if f_idx < track_timings[t_id]["start"]:
            track_timings[t_id]["start"] = f_idx
        if f_idx > track_timings[t_id]["end"]:
            track_timings[t_id]["end"] = f_idx

with open(CSV_OUTPUT, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(["Track ID", "Biển Số (OCR)", "Thời điểm xuất hiện", "Thời điểm biến mất"])
    
    for t_id, text in final_results.items():
        text_clean = text.strip()
        if "scanning" in text_clean.lower() or text_clean == "":
            continue
            
        if t_id in track_timings:
            start_time = format_time(track_timings[t_id]["start"], fps)
            end_time = format_time(track_timings[t_id]["end"], fps)
            writer.writerow([t_id, text_clean, start_time, end_time])

print(f"✅ STAGE 4 Hoàn tất! Đã lưu kết quả sạch vào file CSV.")
print(f"\n🎉 HOÀN THÀNH! Hãy vào thư mục [{main_output_folder}] để xem toàn bộ kết quả nhé.")
