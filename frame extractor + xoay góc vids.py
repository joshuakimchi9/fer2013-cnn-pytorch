import cv2
import os
import mediapipe as mp

# --- CẤU HÌNH ---
VIDEO_PATHS = [
    r"E:\DOWNLOAD\Compressed\data nhóm đã xử lý-20250621T035700Z-1-001\data nhóm đã xử lý\Cường\surprise.MOV",
    r"E:\DOWNLOAD\Compressed\data nhóm đã xử lý-20250621T035700Z-1-001\data nhóm đã xử lý\Minh\surprise.mp4",
    r"E:\DOWNLOAD\Compressed\data nhóm đã xử lý-20250621T035700Z-1-001\data nhóm đã xử lý\Ngọc\surprise.MOV",
    r"E:\DOWNLOAD\Compressed\data nhóm đã xử lý-20250621T035700Z-1-001\data nhóm đã xử lý\Nhi\surprise.mov",
    r"E:\DOWNLOAD\Compressed\data nhóm đã xử lý-20250621T035700Z-1-001\data nhóm đã xử lý\Trà Giang\surprise.MOV"    
]
OUTPUT_DIR = r'D:\input2\surprise'

# --- HÀM AI GIÁO SƯ ---
def find_true_rotation_with_ai_professor(video_path):
    """
    AI sẽ dùng quy luật bất biến: MẮT -> MŨI -> MIỆNG để tìm góc đúng.
    Phương pháp này không thể bị đánh lừa bởi biểu cảm.
    """
    print(f"Đang dùng AI 'Giáo Sư' để tìm góc xoay bất biến cho video: {os.path.basename(video_path)}...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    ret, frame = cap.read()
    cap.release()
    if not ret: return 0

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    rotations_to_test = {
        0: None, 90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }

    best_angle = 0
    best_upright_score = -100 # Điểm phải thật sự tốt mới được chọn

    for angle, rotation_code in rotations_to_test.items():
        test_frame = frame.copy()
        if rotation_code is not None:
            test_frame = cv2.rotate(test_frame, rotation_code)

        results = face_mesh.process(cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Các điểm mốc cốt lõi, không bị ảnh hưởng bởi biểu cảm
            eyes_bridge_y = landmarks[168].y  # Điểm giữa 2 mắt
            nose_tip_y = landmarks[1].y      # Chóp mũi
            mouth_center_y = (landmarks[13].y + landmarks[14].y) / 2 # Giữa môi trên và môi dưới

            # Tính điểm "thẳng đứng" dựa trên trật tự Mắt->Mũi->Miệng
            # Nếu đúng chiều, cả 2 vế đều dương -> điểm cao
            # Nếu ngược chiều, cả 2 vế đều âm -> điểm âm
            # Nếu nằm ngang, điểm gần bằng 0
            upright_score = (nose_tip_y - eyes_bridge_y) + (mouth_center_y - nose_tip_y)
            
            if upright_score > best_upright_score:
                best_upright_score = upright_score
                best_angle = angle

    face_mesh.close()
    
    if best_upright_score < 0.1: # Nếu điểm cao nhất vẫn quá thấp (không rõ mặt)
        print(f"⚠️ AI không thể xác định trật tự Mắt-Mũi-Miệng. Mặc định không xoay.")
        return 0
    
    print(f"✅ AI Giáo Sư đã xác định góc xoay bất biến là: {best_angle} độ.")
    return best_angle


# --- BẮT ĐẦU XỬ LÝ CHÍNH ---
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Đã tạo thư mục mới: {OUTPUT_DIR}")
else:
    print(f"Thư mục đã tồn tại: {OUTPUT_DIR}.")

global_frame_count = 0

for video_path in VIDEO_PATHS:
    print(f"\n--- Bắt đầu xử lý video: {os.path.basename(video_path)} ---")
    
    rotation_angle = find_true_rotation_with_ai_professor(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"LỖI: Không thể mở video tại đường dẫn: {video_path}")
        continue

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Đã xử lý xong video này.")
            break
        
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        filename = os.path.join(OUTPUT_DIR, f"frame_{global_frame_count:05d}.jpg")
        success = cv2.imwrite(filename, frame)
        if success:
            global_frame_count += 1
        else:
            print(f"Lỗi khi đang cố lưu frame tại: {filename}")
            break
    
    cap.release()

print(f"\n--- HOÀN TẤT ---")
print(f"Đã lưu tổng cộng {global_frame_count} frames vào thư mục: {OUTPUT_DIR}")