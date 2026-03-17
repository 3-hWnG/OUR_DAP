"""
get_accuracy.py – Đánh giá độ chính xác pipeline YOLO + OCR trên folder ảnh test.

Flow:
  1. Đọc folder ảnh test + file ground truth CSV
  2. Chạy YOLO detect → crop từng biển → OCR
  3. Xuất CSV kết quả (predicted)
  4. So sánh với ground truth → in Plate Accuracy + Char Accuracy

Ground truth CSV format:
  image_name, ground_truth
  bien_1.jpg, 63B11234
  anh_nhieu_bien.jpg, 63B11234|51F12345|29A99999   ← nhiều biển dùng | ngăn cách

Chỉnh 3 biến ở phần CẤU HÌNH rồi chạy thẳng file này.
"""

import cv2
import os
import csv
import glob
from ultralytics import YOLO

import utils
from ocr_engine import PlateOCR
from utils import clean_plate


# ========================================================
# ⚙️  CẤU HÌNH – chỉnh tại đây
# ========================================================
TEST_IMAGE_FOLDER = r"D:\Study_FPTU\DAP391m\plate detection\dataset_split_old\test\images"
GROUND_TRUTH_CSV  = r"D:\Study_FPTU\DAP391m\plate detection\Hung.csv"
YOLO_MODEL_PATH   = r"D:\Study_FPTU\DAP391m\plate detection\runs\obb\v8n_9kimg\weights\best.pt"
DEVICE            = "cuda"   # "cuda" hoặc "cpu"

# Thư mục lưu kết quả
OUTPUT_DIR        = "accuracy_output"
PREDICTED_CSV     = os.path.join(OUTPUT_DIR, "predicted.csv")
FAILED_CSV        = os.path.join(OUTPUT_DIR, "failed.csv")
REPORT_TXT        = os.path.join(OUTPUT_DIR, "accuracy_report.txt")
# ========================================================


def run_pipeline(image_folder: str, yolo_model, ocr_engine: PlateOCR) -> dict[str, list[str]]:
    """
    Chạy YOLO + OCR trên toàn bộ ảnh trong folder.
    Trả về dict: { image_name: [plate1, plate2, ...] }
    """
    image_paths = (
        glob.glob(os.path.join(image_folder, "*.jpg")) +
        glob.glob(os.path.join(image_folder, "*.jpeg")) +
        glob.glob(os.path.join(image_folder, "*.png")) +
        glob.glob(os.path.join(image_folder, "*.bmp"))
    )

    if not image_paths:
        raise FileNotFoundError(f"Không tìm thấy ảnh nào trong: {image_folder}")

    results = {}
    total = len(image_paths)

    print(f"\n📂 Tìm thấy {total} ảnh. Bắt đầu chạy pipeline...\n")

    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️  Không đọc được ảnh: {img_name}, bỏ qua.")
            results[img_name] = []
            continue

        pct = (i + 1) / total * 100
        print(f"\r⏳ Xử lý: {i+1}/{total} ({pct:.1f}%) | {img_name}".ljust(80),
              end="", flush=True)

        # YOLO detect
        yolo_results = yolo_model.predict(img, conf=0.35, verbose=False)
        plates_in_image = []

        for r in yolo_results:
            if r.obb is None:
                continue

            # Sắp xếp theo confidence giảm dần để lấy biển rõ nhất trước
            boxes = r.obb.xyxyxyxy.cpu().numpy()
            confs = r.obb.conf.cpu().numpy()
            sorted_indices = confs.argsort()[::-1]

            for idx in sorted_indices:
                box = boxes[idx]
                plate_crop = utils.warp_plate(img, box)

                if plate_crop is None or plate_crop.size == 0:
                    continue

                text = ocr_engine.get_text(plate_crop)
                text_clean = clean_plate(text)

                # Bỏ qua kết quả rác
                if len(text_clean) >= 6 and "SCANNING" not in text_clean and "ERROR" not in text_clean:
                    plates_in_image.append(text_clean)

        results[img_name] = plates_in_image

    print(f"\r✅ Hoàn tất pipeline trên {total} ảnh.".ljust(80))
    return results


def load_ground_truth(gt_csv: str) -> dict[str, list[str]]:
    """
    Đọc file ground truth CSV.
    Trả về dict: { image_name: [plate1, plate2, ...] }
    """
    gt = {}
    with open(gt_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = row["image_name"].strip()
            plates = [clean_plate(p) for p in row["ground_truth"].split("|") if p.strip()]
            gt[img_name] = plates
    return gt


def save_predicted_csv(predicted: dict[str, list[str]], output_path: str):
    """Lưu kết quả OCR ra CSV để kiểm tra thủ công nếu cần."""
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "predicted"])
        for img_name, plates in predicted.items():
            writer.writerow([img_name, "|".join(plates) if plates else ""])


def save_failed_csv(predicted: dict[str, list[str]], output_path: str):
    """Xuất CSV các ảnh không đọc được biển số nào."""
    with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "reason"])
        for img_name, plates in predicted.items():
            if not plates:
                writer.writerow([img_name, "Không detect / đọc được biển số"])


def compute_accuracy(
    ground_truth: dict[str, list[str]],
    predicted: dict[str, list[str]]
) -> dict:
    """
    Tính Plate Accuracy và Char Accuracy.

    Plate Accuracy: biển số đọc đúng hoàn toàn / tổng biển số trong ground truth
    Char Accuracy : ký tự đúng / tổng ký tự trong ground truth
    """
    total_plates    = 0
    correct_plates  = 0
    total_chars     = 0
    correct_chars   = 0

    missed_images   = []   # ảnh YOLO không detect được gì
    wrong_plates    = []   # biển đọc sai để log ra

    for img_name, gt_plates in ground_truth.items():
        pred_plates = predicted.get(img_name, [])

        if not pred_plates:
            missed_images.append(img_name)

        for gt_plate in gt_plates:
            total_plates += 1
            total_chars  += len(gt_plate)

            # Tìm prediction khớp nhất với gt_plate này (theo số ký tự đúng)
            best_match     = ""
            best_char_hits = 0

            for pred in pred_plates:
                hits = sum(p == g for p, g in zip(pred, gt_plate))
                if hits > best_char_hits:
                    best_char_hits = hits
                    best_match = pred

            correct_chars += best_char_hits

            if best_match == gt_plate:
                correct_plates += 1
            else:
                wrong_plates.append({
                    "image"    : img_name,
                    "ground_truth": gt_plate,
                    "predicted": best_match or "(không detect được)",
                })

    plate_acc = correct_plates / total_plates * 100 if total_plates else 0
    char_acc  = correct_chars  / total_chars  * 100 if total_chars  else 0

    return {
        "total_plates"   : total_plates,
        "correct_plates" : correct_plates,
        "plate_accuracy" : plate_acc,
        "total_chars"    : total_chars,
        "correct_chars"  : correct_chars,
        "char_accuracy"  : char_acc,
        "missed_images"  : missed_images,
        "wrong_plates"   : wrong_plates,
    }


def print_and_save_report(metrics: dict, report_path: str):
    """In kết quả ra terminal và lưu vào file .txt."""
    lines = [
        "=" * 55,
        "         KẾT QUẢ ĐÁNH GIÁ ĐỘ CHÍNH XÁC OCR",
        "=" * 55,
        f"  Tổng biển số trong ground truth : {metrics['total_plates']}",
        f"  Biển đọc đúng hoàn toàn         : {metrics['correct_plates']}",
        f"  ✅ Plate Accuracy               : {metrics['plate_accuracy']:.2f}%",
        f"",
        f"  Tổng ký tự trong ground truth   : {metrics['total_chars']}",
        f"  Ký tự đọc đúng                  : {metrics['correct_chars']}",
        f"  ✅ Char Accuracy                : {metrics['char_accuracy']:.2f}%",
        "=" * 55,
    ]

    if metrics["missed_images"]:
        lines.append(f"\n⚠️  {len(metrics['missed_images'])} ảnh YOLO không detect được:")
        for img in metrics["missed_images"][:10]:   # chỉ in 10 cái đầu
            lines.append(f"   - {img}")
        if len(metrics["missed_images"]) > 10:
            lines.append(f"   ... và {len(metrics['missed_images']) - 10} ảnh khác")

    if metrics["wrong_plates"]:
        lines.append(f"\n❌ {len(metrics['wrong_plates'])} biển đọc sai (10 cái đầu):")
        lines.append(f"  {'Ảnh':<30} {'Ground Truth':<15} {'Predicted':<15}")
        lines.append(f"  {'-'*30} {'-'*15} {'-'*15}")
        for w in metrics["wrong_plates"][:10]:
            lines.append(f"  {w['image']:<30} {w['ground_truth']:<15} {w['predicted']:<15}")

    lines.append("")
    report = "\n".join(lines)

    print("\n" + report)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"📄 Báo cáo đã lưu tại: {report_path}")


# ========================================================
# 🚀 MAIN
# ========================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("⏳ Đang tải YOLO model...")
    yolo_model = YOLO(YOLO_MODEL_PATH).to(DEVICE)
    print(f"✅ YOLO sẵn sàng trên [{DEVICE.upper()}]")

    ocr_engine = PlateOCR()

    # Bước 1: Chạy pipeline trên folder ảnh test
    predicted = run_pipeline(TEST_IMAGE_FOLDER, yolo_model, ocr_engine)

    # Bước 2: Lưu CSV kết quả để kiểm tra thủ công nếu cần
    save_predicted_csv(predicted, PREDICTED_CSV)
    print(f"💾 Kết quả OCR đã lưu tại: {PREDICTED_CSV}")

    failed_count = sum(1 for plates in predicted.values() if not plates)
    save_failed_csv(predicted, FAILED_CSV)
    print(f"💾 Ảnh không đọc được ({failed_count} ảnh) đã lưu tại: {FAILED_CSV}")

    # Bước 3: Load ground truth và tính accuracy
    print("\n📊 Đang tính độ chính xác...")
    ground_truth = load_ground_truth(GROUND_TRUTH_CSV)
    metrics = compute_accuracy(ground_truth, predicted)

    # Bước 4: In report
    print_and_save_report(metrics, REPORT_TXT)