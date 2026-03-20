import cv2
import numpy as np
import re
from paddleocr import PaddleOCR


class PlateOCR:
    def __init__(self, use_gpu: bool = False):
        print("Đang tải PaddleOCR (CRNN - Chuyên trị chữ méo)...")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            show_log=False,
            use_gpu=use_gpu,
            rec_algorithm='CRNN',
            rec_image_shape='3, 48, 320',
            enable_mkldnn=True
        )

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _deskew(self, img):
        """
        Phát hiện góc nghiêng và xoay về thẳng hàng ngang.
        Trả về ảnh gốc nếu không tìm được góc hợp lệ.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        edges = cv2.Canny(blurred, 30, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        plate_contour = None
        for c in contours:
            approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
            if len(approx) == 4:
                plate_contour = c
                break

        if plate_contour is None:
            plate_contour = contours[0]

        rect = cv2.minAreaRect(plate_contour)
        angle = rect[2]

        if angle < -45:
            angle = 90 + angle

        if abs(angle) < 1.0 or abs(angle) > 30.0:
            return img

        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    def _extract_and_clean(self, img) -> str:
        """Đọc OCR thô, viết hoa, chỉ giữ chữ và số."""
        res = self.ocr.ocr(img, cls=True)
        if not res or res[0] is None:
            return ""
        raw_text = "".join([line[1][0].replace(" ", "").upper() for line in res[0]])
        return re.sub(r'[^A-Z0-9]', '', raw_text)

    def _correct_chars(self, text: str) -> str:
        """
        Chỉ ép các vị trí chắc chắn là SỐ:
          - Vị trí 0, 1 : mã tỉnh → luôn là SỐ
          - Vị trí 4+   : số thứ tự → luôn là SỐ
          - Vị trí 2, 3 : KHÔNG ép (có thể là CHỮ+SỐ hoặc 2 CHỮ như LD, KT...)
        """
        char_to_num = {'B': '8', 'D': '0', 'S': '5', 'G': '6', 'Z': '2', 'A': '4',
                       'I': '1', 'O': '0', 'Q': '0'}

        chars = list(text)
        length = len(chars)

        if length < 4:
            return text

        for i in [0, 1]:
            if chars[i] in char_to_num:
                chars[i] = char_to_num[chars[i]]

        for i in range(4, length):
            if chars[i] in char_to_num:
                chars[i] = char_to_num[chars[i]]

        return "".join(chars)

    def _format_plate(self, text: str) -> str:
        """Định dạng XX-XX-XXXXX."""
        if len(text) > 4:
            return f"{text[:2]}-{text[2:4]}-{text[4:]}"
        return text

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def get_text(self, plate_img):
        """
        Pipeline đọc biển số:
          0. Deskew  – xoay ảnh về thẳng nếu bị nghiêng
          1. Raw     – ảnh gốc sau deskew
          2. CLAHE   – trị lóa sáng / bóng râm
          3. Morph   – trị đứt nét, mờ chữ
        Sau mỗi lớp: ép ký tự đúng vị trí rồi format.
        """
        try:
            resized = cv2.resize(plate_img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            pad_val = 30

            resized = self._deskew(resized)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

            # Lớp 1: Ảnh gốc
            img_raw = cv2.copyMakeBorder(resized, pad_val, pad_val, pad_val, pad_val,
                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_1 = self._correct_chars(self._extract_and_clean(img_raw))
            if len(text_1) >= 7:
                return self._format_plate(text_1)

            # Lớp 2: CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            img_clahe = cv2.copyMakeBorder(enhanced, pad_val, pad_val, pad_val, pad_val,
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_2 = self._correct_chars(
                self._extract_and_clean(cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)))
            if len(text_2) >= 7:
                return self._format_plate(text_2)

            # Lớp 3: Morphological
            _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph_img = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            morph_img = cv2.bitwise_not(morph_img)
            img_morph = cv2.copyMakeBorder(morph_img, pad_val, pad_val, pad_val, pad_val,
                                           cv2.BORDER_CONSTANT, value=[255, 255, 255])
            text_3 = self._correct_chars(
                self._extract_and_clean(cv2.cvtColor(img_morph, cv2.COLOR_GRAY2BGR)))

            best_text = max([text_1, text_2, text_3], key=len)
            return self._format_plate(best_text) if best_text else "Scanning..."

        except Exception as e:
            print(f"Lỗi OCR: {e}")
            return "OCR Error"
