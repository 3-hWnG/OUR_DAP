import os
import cv2
import numpy as np
import re

# Set môi trường cho PaddleOCR ở ngay đây để không rác file main
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from paddleocr import PaddleOCR

class PlateOCR:
    def __init__(self):
        print("Đang tải PaddleOCR VŨ KHÍ BÍ MẬT (CRNN)...")
        # 👉 CẬP NHẬT CẤU HÌNH KHỞI TẠO OCR
        self.ocr = PaddleOCR(use_angle_cls=True, 
                             lang='en', 
                             show_log=False, 
                             use_gpu=True, 
                             rec_algorithm='CRNN', # 👉 Chuyên trị méo
                             rec_image_shape='3, 48, 320', # 👉 Form dẹt dài
                             enable_mkldnn=True)

    def get_text(self, plate_img):
        try:
            # 1. BƯỚC CHUNG: Phóng to ảnh gấp 2 lần
            resized = cv2.resize(plate_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            
            # --- TẠO 2 PHIÊN BẢN ẢNH ---
            # Phiên bản 1: Sạch sẽ, nguyên bản (Chỉ thêm viền trắng chống cắt lẹm)
            img_raw = cv2.copyMakeBorder(resized, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            
            # Phiên bản 2: Dùng thuốc nặng CLAHE (Chống lóa, chống tối)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            img_clahe = cv2.copyMakeBorder(enhanced, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            # Hàm phụ để gọi OCR và dọn dẹp text
            def extract_and_clean(img):
                res = self.ocr.ocr(img, cls=True)
                if not res or res[0] is None: return ""
                text = "-".join([line[1][0].replace(" ", "").upper() for line in res[0]])
                # Chỉ giữ lại Chữ, Số và Dấu gạch ngang
                return re.sub(r'[^A-Z0-9\-]', '', text)

            # --- CHIẾN THUẬT ĐỌC ---
            
            # Lần 1: Thử đọc bằng ảnh nguyên bản (Nhanh, hiệu quả với 80% biển số)
            text_1 = extract_and_clean(img_raw)
            
            # Biển số VN thường có ít nhất 6-7 ký tự (ví dụ: 51F-12345).
            # Nếu đọc ra >= 6 ký tự, khả năng cao là chuẩn, trả về luôn!
            if len(text_1) >= 6:
                return text_1
                
            # Lần 2: Nếu ảnh gốc đọc quá tệ, tung "bài tẩy" CLAHE ra để cứu vớt
            text_2 = extract_and_clean(img_clahe)
            
            # So sánh xem cách nào vớt được nhiều chữ hơn thì lấy cách đó
            best_text = text_2 if len(text_2) > len(text_1) else text_1
            
            return best_text if best_text else "Scanning..."
        except Exception as e:
            print(f"❌ Chi tiết lỗi OCR: {e}")
            return "OCR Error"
