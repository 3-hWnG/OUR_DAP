import cv2
from collections import Counter
from ultralytics import YOLO
import utils
from ocr_engine import PlateOCR

VIDEO_INPUT = r"D:\Study_FPTU\DAP391m\plate detection\random data\14212904_3840_2160_60fps.mp4"
VIDEO_OUTPUT = r"Ket_qua_Export_Clean2.mp4"

yolo_model = YOLO(r"D:\Study_FPTU\DAP391m\plate detection\runs\obb\runs\train\plate_model_v17\weights\best.pt").to("cuda")
ocr_engine = PlateOCR()

cap = cv2.VideoCapture(VIDEO_INPUT)
width, height = int(cap.get(3)), int(cap.get(4))
fps = int(cap.get(5))
out = cv2.VideoWriter(VIDEO_OUTPUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

ocr_history = {}       
final_results = {}     

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_count = 0
print(f"Bắt đầu xuất video MP4: Tổng cộng {total_frames} frames... ( đã chia 2 do tối ưu AI Tracking )")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # TỐI ƯU: Chỉ chạy AI Tracking 2 frame 1 lần (Tăng tốc gấp đôi)
    if frame_count % 2 == 0 or current_results is None:
        results = yolo_model.track(frame, persist=True, imgsz=480, conf=0.4, verbose=False)
        current_results = results
    else:
        results = current_results
    
    frame_count += 1

    if results[0].obb is not None and results[0].obb.id is not None:
        boxes, track_ids = results[0].obb.xyxyxyxy.cpu().numpy(), results[0].obb.id.cpu().numpy().astype(int)

        for box_pts, track_id in zip(boxes, track_ids):
            pts = box_pts.astype(int)

            # Thuật toán Bầu Cử (Voting)
            if track_id not in final_results:
                if track_id not in ocr_history: ocr_history[track_id] = []
                
                if len(ocr_history[track_id]) < 7:
                    plate_crop = utils.warp_plate(frame, box_pts)
                    text = ocr_engine.get_text(plate_crop) # Đã tự động upscaling bên trong
                    if len(text) > 4 and "SCANNING" not in text.upper():
                        ocr_history[track_id].append(text)
                
                if len(ocr_history[track_id]) >= 7:
                    final_results[track_id] = Counter(ocr_history[track_id]).most_common(1)[0][0]

            current_text = final_results.get(track_id, f"Scanning ({len(ocr_history.get(track_id, []))}/7)...")
            
            # Hàm vẽ đã tự động tính toán chữ to/nhỏ
            utils.draw_plate_video(frame, pts, f"ID:{track_id} | {current_text}")

    out.write(frame)
    frame_count += 1
    if frame_count % 10 == 0:  # Cứ 10 frame thì báo cáo 1 lần
        percent = (frame_count / total_frames) * 100
        print(f"Đang render: Frame {frame_count}/{total_frames} ({percent:.1f}%)")

cap.release()
out.release()
print("Export hoàn tất!")
