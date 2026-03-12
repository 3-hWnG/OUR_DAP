import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

class PlateOCR:
    def __init__(self):
        print("Đang tải PaddleOCR (CRNN - Chuyên trị chữ méo)...")
        # Khởi tạo OCR với cấu hình chuyên dụng cho biển số
        self.ocr = PaddleOCR(use_angle_cls=True, 
                             lang='en', 
                             show_log=False, 
                             use_gpu=False, # Đổi thành True nếu máy có GPU
                             rec_algorithm='CRNN', 
                             rec_image_shape='3, 48, 320', 
                             enable_mkldnn=True)

    def extract_and_clean(self, img):
        """Hàm con để xử lý việc đọc và ép định dạng ký tự"""
        res = self.ocr.ocr(img, cls=True)
        if not res or res[0] is None: 
            return ""
        
        # 1. Lấy text thô và viết hoa
        raw_text = "".join([line[1][0].replace(" ", "").upper() for line in res[0]])
        
        # 2. Loại bỏ ký tự đặc biệt (chỉ giữ chữ và số)
        clean_text = re.sub(r'[^A-Z0-9]', '', raw_text)
        
        # 3. Ép định dạng Biển số Việt Nam (VD: 63-B1-23456)
        if len(clean_text) >= 4:
            chars = list(clean_text)
            
            # Quy tắc: Vị trí index 2 phải là CHỮ (VD: 63-[B]-1...)
            num_to_char = {'8': 'B', '0': 'D', '5': 'S', '6': 'G', '2': 'Z', '4': 'A'}
            if chars[2] in num_to_char:
                chars[2] = num_to_char[chars[2]]
                
            # Quy tắc: Vị trí index 3 phải là SỐ (VD: 63-B-[1]...)
            char_to_num = {'B': '8', 'D': '0', 'S': '5', 'G': '6', 'Z': '2', 'A': '4'}
            if chars[3] in char_to_num:
                chars[3] = char_to_num[chars[3]]

            clean_text = "".join(chars)
            
        return clean_text

    def get_text(self, plate_img):
        """Hàm chính thực hiện chiến thuật đọc 3 lớp: Gốc -> CLAHE -> Morphology"""
        try:
            # Tiền xử lý: Resize vừa phải để không bị vỡ hạt (Scale 2.5x là đẹp)
            resized = cv2.resize(plate_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            pad_val = 30
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # --- LỚP 1: ẢNH NGUYÊN BẢN ---
            img_raw = cv2.copyMakeBorder(resized, pad_val, pad_val, pad_val, pad_val, 
                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_1 = self.extract_and_clean(img_raw)
            # Nếu biển đã đẹp và đọc đủ 7-8 số thì trả về luôn cho nhẹ máy
            if len(text_1) >= 7: return self.format_plate(text_1)

            # --- LỚP 2: ẢNH TĂNG CƯỜNG (CLAHE) - Trị lóa sáng/ Bóng râm ---
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            img_clahe = cv2.copyMakeBorder(enhanced, pad_val, pad_val, pad_val, pad_val, 
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_2 = self.extract_and_clean(cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR))
            if len(text_2) >= 7: return self.format_plate(text_2)

            # --- LỚP 3: MORPHOLOGICAL OPERATIONS - Trị đứt nét, mờ chữ ---
            # 1. Nhị phân hóa (Otsu) -> Ép ảnh thành 2 màu trắng/đen, chữ sẽ thành màu TRẮNG
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # 2. Phép Closing (Nối liền các nét chữ bị đứt do xước biển/bụi)
            # Kernel 2x2 là đủ cho nét chữ nhỏ, nếu chữ to bạn tăng lên (3x3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # 3. Đảo màu lại (Chữ đen, nền trắng) vì OCR thường nhạy với text đen hơn
            morph_img = cv2.bitwise_not(morph_img)
            
            img_morph = cv2.copyMakeBorder(morph_img, pad_val, pad_val, pad_val, pad_val, 
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_3 = self.extract_and_clean(cv2.cvtColor(img_morph, cv2.COLOR_GRAY2BGR))

            # --- CHỐT KẾT QUẢ ---
            # So sánh 3 lớp, lấy kết quả nào quét được nhiều ký tự hợp lệ nhất
            texts = [text_1, text_2, text_3]
            best_text = max(texts, key=len)
            
            return self.format_plate(best_text) if best_text else "Scanning..."
            
        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "OCR Error"

    # Tui tách riêng hàm format để code gọn gàng hơn
    def format_plate(self, text):
        if len(text) > 4:
            return f"{text[:2]}-{text[2:4]}-{text[4:]}"
        return text
