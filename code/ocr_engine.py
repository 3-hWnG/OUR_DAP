import cv2
import numpy as np
import re
from paddleocr import PaddleOCR


class PlateOCR:
    def __init__(self):
        print("Đang tải PaddleOCR (CRNN - Chuyên trị chữ méo)...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            use_gpu=False,          # Đổi thành True nếu máy có GPU
            rec_algorithm='CRNN',
            rec_image_shape='3, 48, 320',
            enable_mkldnn=True
        )

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _extract_and_clean(self, img):
        """Đọc OCR thô, viết hoa, lọc ký tự và ép định dạng biển VN."""
        res = self.ocr.ocr(img, cls=True)
        if not res or res[0] is None:
            return ""

        raw_text = "".join([line[1][0].replace(" ", "").upper() for line in res[0]])
        clean_text = re.sub(r'[^A-Z0-9]', '', raw_text)

        if len(clean_text) >= 4:
            chars = list(clean_text)

            # Vị trí index 2 phải là CHỮ (vd: 63-[B]-1...)
            num_to_char = {'8': 'B', '0': 'D', '5': 'S', '6': 'G', '2': 'Z', '4': 'A'}
            if chars[2] in num_to_char:
                chars[2] = num_to_char[chars[2]]

            # Vị trí index 3 phải là SỐ (vd: 63-B-[1]...)
            char_to_num = {'B': '8', 'D': '0', 'S': '5', 'G': '6', 'Z': '2', 'A': '4'}
            if chars[3] in char_to_num:
                chars[3] = char_to_num[chars[3]]

            clean_text = "".join(chars)

        return clean_text

    def _format_plate(self, text):
        """Định dạng XX-XX-XXXXX."""
        if len(text) > 4:
            return f"{text[:2]}-{text[2:4]}-{text[4:]}"
        return text

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def get_text(self, plate_img):
        """
        Chiến thuật đọc 3 lớp: Ảnh gốc → CLAHE → Morphology.
        Trả về chuỗi biển số đã định dạng, hoặc 'Scanning...' nếu thất bại.
        """
        try:
            resized = cv2.resize(plate_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            pad_val = 30
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Lớp 1: Ảnh nguyên bản
            img_raw = cv2.copyMakeBorder(resized, pad_val, pad_val, pad_val, pad_val,
                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_1 = self._extract_and_clean(img_raw)
            if len(text_1) >= 7:
                return self._format_plate(text_1)

            # Lớp 2: CLAHE – trị lóa sáng / bóng râm
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img_clahe = cv2.copyMakeBorder(enhanced, pad_val, pad_val, pad_val, pad_val,
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_2 = self._extract_and_clean(cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR))
            if len(text_2) >= 7:
                return self._format_plate(text_2)

            # Lớp 3: Morphological – trị đứt nét, mờ chữ
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph_img = cv2.bitwise_not(morph_img)
            img_morph = cv2.copyMakeBorder(morph_img, pad_val, pad_val, pad_val, pad_val,
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_3 = self._extract_and_clean(cv2.cvtColor(img_morph, cv2.COLOR_GRAY2BGR))

            best_text = max([text_1, text_2, text_3], key=len)
            return self._format_plate(best_text) if best_text else "Scanning..."

        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "OCR Error"
