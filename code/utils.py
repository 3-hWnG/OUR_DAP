import cv2
import numpy as np

def is_in_safe_zone(pts, frame_w, frame_h):
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    box_w = x_max - x_min
    box_h = y_max - y_min
    box_area = box_w * box_h
    frame_area = frame_w * frame_h

    # Không cho chạm mép frame
    edge_margin = 25
    if (x_min < edge_margin or y_min < edge_margin or 
        x_max > frame_w - edge_margin or y_max > frame_h - edge_margin):
        return False

    # Biển còn quá nhỏ (đang ở xa)
    area_ratio = box_area / frame_area
    if area_ratio < 0.008:
        return False

    return True

def warp_plate(img, box_coords, expand_ratio=0.13):
    """
    expand_ratio = 0.08 tức là nới rộng khung ra thêm 8%. 
    Bạn có thể tăng lên 0.1 (10%) nếu thấy viền vẫn bị sát.
    """
    pts = box_coords.reshape(4, 2).astype("float32")
    
    # --- THUẬT TOÁN NỞ KHUNG (BOX EXPANSION) ---
    # 1. Tìm điểm Tâm của biển số
    center = np.mean(pts, axis=0)
    
    # 2. Kéo dãn 4 đỉnh ra xa khỏi Tâm
    expanded_pts = np.zeros_like(pts)
    for i in range(4):
        vec = pts[i] - center # Vector hướng từ Tâm ra góc
        expanded_pts[i] = center + vec * (1.0 + expand_ratio) # Nhân bản vector lên
        
    # 3. Giới hạn tọa độ để không bị văng ra khỏi mép bức ảnh gốc gây lỗi
    h_img, w_img = img.shape[:2]
    expanded_pts[:, 0] = np.clip(expanded_pts[:, 0], 0, w_img - 1)
    expanded_pts[:, 1] = np.clip(expanded_pts[:, 1], 0, h_img - 1)
    
    pts = expanded_pts # Lấy tọa độ mới đã nở to để đi nắn phẳng
    
    # --- THUẬT TOÁN NẮN PHẲNG (GIỮ NGUYÊN) ---
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    
    w = int(max(np.linalg.norm(rect[0]-rect[1]), np.linalg.norm(rect[2]-rect[3])))
    h = int(max(np.linalg.norm(rect[0]-rect[3]), np.linalg.norm(rect[1]-rect[2])))
    
    dst = np.array([[0,0], [w-1,0], [w-1,h-1], [0,h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w, h))

def draw_plate_static(img, pts, text):
    h_img = img.shape[0]
    font_scale = max(0.5, min((h_img / 1000) * 0.8, 1.0))
    thickness = 2
    y_max = pts[:, 1].max()
    x_min = pts[:, 0].min()
    pos = (x_min, y_max + int(30 * font_scale))
    
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

def draw_plate_video(img, pts, text):
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    box_w = x_max - x_min

    # Vẽ khung OBB màu xanh
    cv2.polylines(img, [pts], True, (0, 255, 0), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 1. TÍNH TOÁN ÉP CHỮ VỪA KHÍT CHIỀU NGANG BIỂN SỐ
    # Lấy kích thước chữ ở scale = 1.0 làm mốc chuẩn
    (w_base, h_base), _ = cv2.getTextSize(text, font, 1.0, 2)
    
    # Tính font_scale để chữ nhét vừa chiều ngang box_w (trừ hao 4 pixel cho đỡ chạm viền)
    font_scale = (box_w - 4) / w_base 
    
    # Giới hạn scale để chữ không bị quá bé xíu hoặc to đùng vỡ nét
    font_scale = max(0.4, min(font_scale, 2.0))
    thickness = max(1, int(font_scale * 2))

    # Lấy lại kích thước thật của chữ sau khi đã ép scale
    (w_t, h_t), _ = cv2.getTextSize(text, font, font_scale, thickness)

    # 2. VẼ NỀN ĐEN (Bằng đúng chiều ngang khung xe: từ x_min đến x_max)
    cv2.rectangle(img, (x_min, y_max + 5), (x_max, y_max + h_t + 15), (0, 0, 0), -1)
    
    # 3. VẼ CHỮ TRẮNG (CĂN GIỮA NỀN ĐEN)
    # Tính tọa độ X sao cho chữ nằm ngay chính giữa cái nền đen
    text_x = x_min + int((box_w - w_t) / 2)
    # Nếu chữ bị to hơn khung do giới hạn min scale, ép nó bắt đầu từ lề trái
    text_x = max(x_min + 2, text_x) 
    
    cv2.putText(img, text, (text_x, y_max + h_t + 10), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
