from roboflow import Roboflow
import os
drive.mount('/content/drive')

# 1. Khởi tạo Roboflow
# Lưu ý: Nên bảo mật API Key, không nên share công khai
rf = Roboflow(api_key="3foez3ua5SAeYzVPYqa4")

# 2. Thiết lập Project
workspaceId = 'daplicenseplatemotorvn'
projectId = 'vehicle_plate-bddqu'
project = rf.workspace(workspaceId).project(projectId)

# 3. Cấu hình đường dẫn ảnh
# Nếu bạn dùng Google Colab, hãy chạy dòng này trước để mount drive:
# from google.colab import drive
# drive.mount('/content/drive')
# Sau đó đường dẫn thường sẽ là: "/content/drive/MyDrive/TÊN_FOLDER_ẢNH"

IMAGE_FOLDER_PATH = "/content/drive/MyDrive/bike_plates/plate_pics"
# Ví dụ: "C:/Users/Admin/Google Drive/Images" hoặc "/content/drive/MyDrive/BienSoXe"

# 4. Duyệt qua từng file và Upload
print(f"Đang quét ảnh trong thư mục: {IMAGE_FOLDER_PATH}...")

# Lấy danh sách tất cả các file trong folder
if os.path.exists(IMAGE_FOLDER_PATH):
    for filename in os.listdir(IMAGE_FOLDER_PATH):
        # Chỉ chọn các file ảnh (bạn có thể thêm đuôi file khác nếu cần)
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):

            # Tạo đường dẫn đầy đủ tới file ảnh
            full_file_path = os.path.join(IMAGE_FOLDER_PATH, filename)

            print(f"Đang upload: {filename}")

            try:
                # Upload ảnh
                # Roboflow mặc định sẽ dùng chính filename làm tên ảnh trên hệ thống
                project.upload(
                    image_path=full_file_path,
                    batch_name="pics_from_drive", # Đặt tên batch để dễ quản lý
                    split="train",               # Chọn tập dữ liệu: train, valid, hoặc test
                    num_retry_uploads=3          # Thử lại 3 lần nếu mạng lỗi
                )
            except Exception as e:
                print(f"--> Lỗi khi upload file {filename}: {e}")

    print("\nHoàn tất quá trình upload!")
else:
    print("Không tìm thấy đường dẫn thư mục. Vui lòng kiểm tra lại 'IMAGE_FOLDER_PATH'.")
