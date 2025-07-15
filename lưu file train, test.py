import os
import shutil
import random
from tqdm import tqdm

# --- CẤU HÌNH ---

# 1. Đường dẫn đến thư mục chứa dữ liệu đã được xử lý (ví dụ: các ảnh 112x112)
# Thư mục này nên có cấu trúc:
# /source_data_dir
#   /happy
#     /img1.jpg
#     /img2.jpg
#   /sad
#     /imgA.jpg
#     /imgB.jpg
#   ...
SOURCE_DATA_DIR = r"D:\môn AI\output" # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY

# 2. Đường dẫn đến thư mục output, nơi sẽ chứa hai thư mục 'train' và 'validation'
OUTPUT_DIR = r"D:\môn AI\EmotionDataset_split" # <<< THAY ĐỔI ĐƯỜNG DẪN NÀY

# 3. Tỷ lệ chia dữ liệu cho tập validation
# 0.2 có nghĩa là 20% cho validation, 80% cho training
VALIDATION_SPLIT_RATIO = 0.2

# 4. (Tùy chọn) Đặt một seed để kết quả chia luôn giống nhau mỗi lần chạy
RANDOM_SEED = 42

# --- KẾT THÚC CẤU HÌNH ---

def split_data(source_dir, output_dir, split_ratio, seed):
    """
    Chia dữ liệu từ source_dir vào hai thư mục train và validation trong output_dir.
    """
    # Thiết lập seed để có thể tái tạo kết quả
    if seed:
        random.seed(seed)

    # Tạo các thư mục output chính
    train_dir = os.path.join(output_dir, 'train')
    validation_dir = os.path.join(output_dir, 'validation')
    
    # Xóa thư mục output cũ nếu tồn tại để tránh lẫn lộn file
    if os.path.exists(output_dir):
        print(f"Thư mục output '{output_dir}' đã tồn tại. Xóa để tạo lại...")
        shutil.rmtree(output_dir)
        
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    print(f"Bắt đầu chia dữ liệu từ '{source_dir}'...")
    print(f"Thư mục training: '{train_dir}'")
    print(f"Thư mục validation: '{validation_dir}'")

    total_files_copied = 0

    # Lấy danh sách các thư mục class (ví dụ: 'happy', 'sad')
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    if not class_dirs:
        print(f"LỖI: Không tìm thấy thư mục class nào trong '{source_dir}'.")
        return

    # Duyệt qua từng thư mục class
    for class_name in tqdm(class_dirs, desc="Đang xử lý các class"):
        # Tạo thư mục class tương ứng trong train và validation
        train_class_dir = os.path.join(train_dir, class_name)
        validation_class_dir = os.path.join(validation_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(validation_class_dir, exist_ok=True)

        # Lấy danh sách tất cả các file ảnh trong thư mục class hiện tại
        class_source_dir = os.path.join(source_dir, class_name)
        image_files = os.listdir(class_source_dir)
        
        # Xáo trộn danh sách file một cách ngẫu nhiên
        random.shuffle(image_files)

        # Tính toán điểm chia
        split_point = int(len(image_files) * split_ratio)
        
        # Chia danh sách file thành 2 phần: validation và training
        validation_files = image_files[:split_point]
        training_files = image_files[split_point:]

        # Sao chép file vào thư mục validation
        for filename in validation_files:
            source_path = os.path.join(class_source_dir, filename)
            dest_path = os.path.join(validation_class_dir, filename)
            shutil.copy2(source_path, dest_path)
            total_files_copied += 1
            
        # Sao chép file vào thư mục training
        for filename in training_files:
            source_path = os.path.join(class_source_dir, filename)
            dest_path = os.path.join(train_class_dir, filename)
            shutil.copy2(source_path, dest_path)
            total_files_copied += 1

    print("-" * 40)
    print("HOÀN TẤT!".center(40))
    print(f"Tổng số file đã được sao chép: {total_files_copied}")

# Chạy hàm chia dữ liệu
if __name__ == "__main__":
    split_data(SOURCE_DATA_DIR, OUTPUT_DIR, VALIDATION_SPLIT_RATIO, RANDOM_SEED)