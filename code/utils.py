import cv2
import numpy as np


# ========================================================
# 🛠️ TIỆN ÍCH CHUNG
# ========================================================

def format_time(frame_count, fps):
    """Chuyển số frame sang chuỗi thời gian HH:MM:SS hoặc MM:SS."""
    total_seconds = int(frame_count / fps)
    mins, secs = divmod(total_seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# ========================================================
# 🎯 FILTER & CROP
# ========================================================

def is_in_safe_zone(pts, frame_w, frame_h):
    """Kiểm tra biển số có nằm trong vùng an toàn (không chạm mép, đủ to) không."""
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

    box_w = x_max - x_min
    box_h = y_max - y_min
    box_area = box_w * box_h
    frame_area = frame_w * frame_h

    edge_margin = 25
    if (x_min < edge_margin or y_min < edge_margin or
            x_max > frame_w - edge_margin or y_max > frame_h - edge_margin):
        return False

    area_ratio = box_area / frame_area
    if area_ratio < 0.008:
        return False

    return True


def warp_plate(img, box_coords, expand_ratio=0.13):
    """
    Nắn phẳng (perspective warp) vùng biển số từ ảnh gốc.
    expand_ratio: tỉ lệ nới rộng khung trước khi warp (mặc định 13%).
    """
    pts = box_coords.reshape(4, 2).astype("float32")

    # --- NỞ KHUNG ---
    center = np.mean(pts, axis=0)
    expanded_pts = np.zeros_like(pts)
    for i in range(4):
        vec = pts[i] - center
        expanded_pts[i] = center + vec * (1.0 + expand_ratio)

    h_img, w_img = img.shape[:2]
    expanded_pts[:, 0] = np.clip(expanded_pts[:, 0], 0, w_img - 1)
    expanded_pts[:, 1] = np.clip(expanded_pts[:, 1], 0, h_img - 1)
    pts = expanded_pts

    # --- NẮN PHẲNG ---
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]

    w = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])))
    h = int(max(np.linalg.norm(rect[0] - rect[3]), np.linalg.norm(rect[1] - rect[2])))

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (w, h))


# ========================================================
# 🎨 VẼ KẾT QUẢ
# ========================================================

def draw_plate_static(img, pts, text):
    """Vẽ label biển số lên ảnh tĩnh."""
    h_img = img.shape[0]
    font_scale = max(0.5, min((h_img / 1000) * 0.8, 1.0))
    thickness = 2
    y_max = pts[:, 1].max()
    x_min = pts[:, 0].min()
    pos = (x_min, y_max + int(30 * font_scale))

    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 255, 0), thickness, cv2.LINE_AA)


def draw_plate_video(img, pts, text):
    """Vẽ khung OBB + label căn giữa dưới biển số lên frame video."""
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    box_w = x_max - x_min

    cv2.polylines(img, [pts], True, (0, 255, 0), 3)

    font = cv2.FONT_HERSHEY_SIMPLEX
    (w_base, _), _ = cv2.getTextSize(text, font, 1.0, 2)
    font_scale = (box_w - 4) / w_base
    font_scale = max(0.4, min(font_scale, 2.0))
    thickness = max(1, int(font_scale * 2))

    (w_t, h_t), _ = cv2.getTextSize(text, font, font_scale, thickness)

    cv2.rectangle(img, (x_min, y_max + 5), (x_max, y_max + h_t + 15), (0, 0, 0), -1)

    text_x = max(x_min + 2, x_min + int((box_w - w_t) / 2))
    cv2.putText(img, text, (text_x, y_max + h_t + 10), font, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
