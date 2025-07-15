import os
from PIL import Image, ImageEnhance
from tqdm import tqdm 

# --- CẤU HÌNH ---
# <<< THAY ĐỔI 1: Cấu hình thư mục GỐC
# 1. Đường dẫn đến THƯ MỤC GỐC chứa các thư mục con cần xử lý (ví dụ: happy, sad, angry).
#    Script sẽ tự động tìm tất cả các thư mục bên trong đường dẫn này.
BASE_INPUT_DIR = r"D:\môn AI\input2"

# 2. Tên thư mục GỐC để lưu kết quả. Cấu trúc thư mục con sẽ được tự động tạo và giữ nguyên.
BASE_OUTPUT_DIR = r"D:\môn AI\output"

# 3. Kích thước mục tiêu mà model yêu cầu (ví dụ: 224x224)
TARGET_SIZE = (112, 112)

# 4. Hệ số tăng tương phản
CONTRAST_FACTOR = 1.5 

# 5. Chỉ xử lý các file có đuôi này
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

# --- KẾT THÚC CẤU HÌNH ---


def preprocess_images(input_dir, output_dir, size, contrast_factor):
    """
    (Hàm này giữ nguyên)
    Đọc ảnh từ một thư mục input_dir, xử lý và lưu vào output_dir.
    """
    # Tạo thư mục output nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Lọc ra các file ảnh hợp lệ
    try:
        image_files = [
            f for f in os.listdir(input_dir) 
            if os.path.isfile(os.path.join(input_dir, f)) and os.path.splitext(f)[1].lower() in ALLOWED_EXTENSIONS
        ]
    except FileNotFoundError:
        print(f"LỖI: Thư mục input '{input_dir}' không tồn tại hoặc không thể truy cập.")
        return
    
    if not image_files:
        print(f"Thông báo: Không tìm thấy file ảnh nào hợp lệ trong '{input_dir}'. Bỏ qua.")
        return

    print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu xử lý...")
    processed_count = 0
    failed_count = 0

    # Sử dụng tqdm để tạo thanh tiến trình
    for filename in tqdm(image_files, desc=f"Xử lý '{os.path.basename(input_dir)}'"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with Image.open(input_path) as img:
                resized_img = img.resize(size, Image.Resampling.LANCZOS)
                grayscale_img = resized_img.convert('L')
                enhancer = ImageEnhance.Contrast(grayscale_img)
                enhanced_img = enhancer.enhance(contrast_factor)
                enhanced_img.save(output_path)
                processed_count += 1
        except Exception as e:
            print(f"\nCẢNH BÁO: Lỗi khi xử lý file '{filename}': {e}. Bỏ qua file này.")
            failed_count += 1

    print(f"Hoàn tất xử lý thư mục '{os.path.basename(input_dir)}'.")
    print(f"-> Thành công: {processed_count} ảnh. Thất bại: {failed_count} ảnh.")


# <<< THAY ĐỔI 2: Hàm điều phối chính
def process_all_subdirectories(base_input, base_output, size, contrast):
    """
    Quét tất cả các thư mục con trong base_input, và gọi hàm preprocess_images
    cho mỗi thư mục con tìm thấy, lưu kết quả vào thư mục tương ứng trong base_output.
    """
    if not os.path.isdir(base_input):
        print(f"LỖI: Thư mục gốc '{base_input}' không tồn tại.")
        return

    print("=" * 60)
    print(f"Bắt đầu quét các thư mục con trong: '{base_input}'")
    
    # Tìm tất cả các mục trong thư mục gốc và chỉ giữ lại những mục là thư mục
    try:
        sub_dirs = [d for d in os.listdir(base_input) if os.path.isdir(os.path.join(base_input, d))]
    except Exception as e:
        print(f"LỖI: Không thể đọc danh sách thư mục từ '{base_input}': {e}")
        return

    if not sub_dirs:
        print("Không tìm thấy thư mục con nào để xử lý.")
        # Xử lý trường hợp không có thư mục con, nhưng có thể có ảnh ngay trong thư mục gốc
        print("Kiểm tra xem có ảnh trong thư mục gốc không...")
        preprocess_images(base_input, base_output, size, contrast)
        return

    print(f"Tìm thấy {len(sub_dirs)} thư mục con: {', '.join(sub_dirs)}")
    print("=" * 60)

    # Lặp qua từng thư mục con để xử lý
    for dir_name in sub_dirs:
        input_dir = os.path.join(base_input, dir_name)
        output_dir = os.path.join(base_output, dir_name)
        
        print(f"\n--- BẮT ĐẦU XỬ LÝ THƯ MỤC: [{dir_name}] ---")
        print(f"Input:  '{input_dir}'")
        print(f"Output: '{output_dir}'")
        
        preprocess_images(input_dir, output_dir, size, contrast)
        
        print(f"--- HOÀN TẤT THƯ MỤC: [{dir_name}] ---")

    print("\n" + "=" * 60)
    print("TẤT CẢ CÁC THƯ MỤC ĐÃ ĐƯỢC XỬ LÝ HOÀN TẤT!".center(60))
    print("=" * 60)


if __name__ == "__main__":
    # <<< THAY ĐỔI 3: Gọi hàm điều phối chính
    process_all_subdirectories(
        BASE_INPUT_DIR, 
        BASE_OUTPUT_DIR, 
        TARGET_SIZE, 
        CONTRAST_FACTOR
    )