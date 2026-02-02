import os
import glob

def rename_dataset(img_folder, label_folder, prefix="oto", start_index=0):
    """
    Đổi tên file ảnh và label tương ứng theo format prefix + số thứ tự.
    """
    
    # Các đuôi ảnh hỗ trợ
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_files = []
    for ext in extensions:
        img_files.extend(glob.glob(os.path.join(img_folder, ext)))
    
    # Sắp xếp tên file cũ để đảm bảo thứ tự đổi tên ổn định
    img_files.sort() 
    
    print(f"Tìm thấy {len(img_files)} ảnh. Bắt đầu đổi tên từ số {start_index}...")

    count = 0
    
    for i, img_path in enumerate(img_files):
        # Tính chỉ số hiện tại: 1520 + 0, 1520 + 1,...
        current_index = start_index + i
        
        # Format :04d đảm bảo số 1520 giữ nguyên, nhưng số 1 sẽ thành 0001
        new_name_base = f"{prefix}{current_index:04d}"
        
        # --- Xử lý Ảnh ---
        img_dir, img_name = os.path.split(img_path)
        img_name_no_ext, img_ext = os.path.splitext(img_name)
        
        new_img_name = f"{new_name_base}{img_ext}"
        new_img_path = os.path.join(img_dir, new_img_name)
        
        # --- Xử lý Label ---
        # Mặc định là .txt (YOLO). Nếu dùng xml thì sửa thành ".xml"
        label_ext = ".txt" 
        old_label_path = os.path.join(label_folder, f"{img_name_no_ext}{label_ext}")
        
        has_label = os.path.exists(old_label_path)
        
        # --- Thực hiện Rename ---
        try:
            # Đổi tên ảnh
            os.rename(img_path, new_img_path)
            
            # Đổi tên label (nếu có)
            if has_label:
                new_label_path = os.path.join(label_folder, f"{new_name_base}{label_ext}")
                os.rename(old_label_path, new_label_path)
                print(f"[OK] {img_name} -> {new_img_name} | Label: Có")
            else:
                print(f"[OK] {img_name} -> {new_img_name} | Label: Không tìm thấy")
                
            count += 1
            
        except OSError as e:
            print(f"[LỖI] Không thể đổi tên {img_name}: {e}")

    print("---")
    print(f"Hoàn tất! Đã đổi tên {count} cặp file (Từ {prefix}{start_index} đến {prefix}{start_index + count - 1}).")

# --- CẤU HÌNH ---
if __name__ == "__main__":
    # 1. Điền đường dẫn folder của bạn vào đây:
    FOLDER_ANH = r"C:\Users\Asus\Downloads\vehicle_plate.v4i.yolov11\train\car_image"
    FOLDER_LABEL = r"C:\Users\Asus\Downloads\vehicle_plate.v4i.yolov11\train\car_label"
    
    # 2. Cấu hình số bắt đầu:
    SO_BAT_DAU = 0

    confirm = input(f"Chuẩn bị đổi tên file bắt đầu từ 'bienso{SO_BAT_DAU}'\nFolder: {FOLDER_ANH}\nNhấn 'y' để chạy: ")
    
    if confirm.lower() == 'y':
        rename_dataset(FOLDER_ANH, FOLDER_LABEL, prefix="car", start_index=SO_BAT_DAU)
    else:
        print("Đã hủy.")