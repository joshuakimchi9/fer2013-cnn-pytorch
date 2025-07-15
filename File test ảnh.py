import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QFrame)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt

# --- 1. ĐỊNH NGHĨA MODEL (KHÔNG THAY ĐỔI) ---
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        # Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.25)
        # Block 3
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.25)
        # Block 4
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(512 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(256, 512)
        self.bn6 = nn.BatchNorm1d(512)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.bn4(self.conv4(x)))))
        x = torch.flatten(x, 1)
        x = self.dropout5(self.relu5(self.bn5(self.fc1(x))))
        x = self.dropout6(self.relu6(self.bn6(self.fc2(x))))
        x = self.fc3(x)
        return x

# --- 2. CÁC HÀM TIỆN ÍCH VÀ THÔNG SỐ (ĐÃ CẬP NHẬT) ---

# <<< THAY ĐỔI QUAN TRỌNG Ở ĐÂY >>>
# Danh sách nhãn đã được sắp xếp lại để khớp với thứ tự thư mục của bạn.
CLASS_NAMES = [
    'Tức giận (Angry)',      # 0
    'Ghê tởm (Disgust)',     # 1
    'Sợ hãi (Fear)',         # 2
    'Vui vẻ (Happy)',        # 3
    'Trung tính (Neutral)',  # 4
    'Buồn bã (Sad)',         # 5
    'Ngạc nhiên (Surprise)'  # 6
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

# --- 3. LỚP GIAO DIỆN NGƯỜI DÙNG (UI) (KHÔNG THAY ĐỔI) ---
class EmotionPredictorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = ""
        self.image_path = ""
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Ứng dụng Nhận diện Cảm xúc')
        self.setGeometry(200, 200, 700, 500)
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        
        # --- Cột trái: Các nút điều khiển ---
        self.btn_select_model = QPushButton('1. Chọn Model (.pth)', self)
        self.btn_select_model.clicked.connect(self.select_model_file)
        self.lbl_model_path = QLabel('Chưa chọn model nào', self)
        self.lbl_model_path.setStyleSheet("color: gray;")
        
        self.btn_select_image = QPushButton('2. Chọn Ảnh', self)
        self.btn_select_image.clicked.connect(self.select_image_file)
        self.lbl_image_path = QLabel('Chưa chọn ảnh nào', self)
        self.lbl_image_path.setStyleSheet("color: gray;")
        
        self.btn_predict = QPushButton('3. DỰ ĐOÁN CẢM XÚC', self)
        self.btn_predict.clicked.connect(self.run_prediction)
        self.btn_predict.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        
        self.lbl_result = QLabel('Kết quả:', self)
        self.lbl_result.setFont(QFont('Arial', 16, QFont.Bold))
        self.lbl_emotion = QLabel('', self)
        self.lbl_emotion.setFont(QFont('Arial', 14))
        self.lbl_emotion.setStyleSheet("color: #00008B;")
        self.lbl_confidence = QLabel('', self)
        self.lbl_confidence.setFont(QFont('Arial', 12))
        self.lbl_status = QLabel('Sẵn sàng', self)
        self.lbl_status.setStyleSheet("color: green;")

        left_layout.addWidget(self.btn_select_model)
        left_layout.addWidget(self.lbl_model_path)
        left_layout.addSpacing(20)
        left_layout.addWidget(self.btn_select_image)
        left_layout.addWidget(self.lbl_image_path)
        left_layout.addSpacing(40)
        left_layout.addWidget(self.btn_predict)
        left_layout.addSpacing(40)
        left_layout.addWidget(self.lbl_result)
        left_layout.addWidget(self.lbl_emotion)
        left_layout.addWidget(self.lbl_confidence)
        left_layout.addStretch(1)
        left_layout.addWidget(self.lbl_status)

        # --- Cột phải: Hiển thị ảnh ---
        self.lbl_image_display = QLabel('Vui lòng chọn ảnh để hiển thị', self)
        self.lbl_image_display.setFixedSize(350, 350)
        self.lbl_image_display.setAlignment(Qt.AlignCenter)
        self.lbl_image_display.setFrameShape(QFrame.Box)
        self.lbl_image_display.setStyleSheet("border: 2px solid #ccc;")
        right_layout.addWidget(self.lbl_image_display)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 1)
        self.setLayout(main_layout)

    def select_model_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn file model", "", "PyTorch Models (*.pth)", options=options)
        if file_name:
            self.model_path = file_name
            self.lbl_model_path.setText(f"Model: ...{self.model_path[-30:]}")
            self.lbl_model_path.setStyleSheet("color: black;")
            self.load_model()
    
    def load_model(self):
        try:
            self.lbl_status.setText('Đang tải model...')
            self.lbl_status.setStyleSheet("color: orange;")
            QApplication.processEvents()
            self.model = EmotionCNN(num_classes=7).to(DEVICE)
            self.model.load_state_dict(torch.load(self.model_path, map_location=DEVICE))
            self.model.eval()
            self.lbl_status.setText('Tải model thành công!')
            self.lbl_status.setStyleSheet("color: green;")
        except Exception as e:
            self.lbl_status.setText(f'Lỗi tải model: {e}')
            self.lbl_status.setStyleSheet("color: red;")
            self.model = None

    def select_image_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn file ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            self.image_path = file_name
            self.lbl_image_path.setText(f"Ảnh: ...{self.image_path[-30:]}")
            self.lbl_image_path.setStyleSheet("color: black;")
            pixmap = QPixmap(self.image_path)
            self.lbl_image_display.setPixmap(pixmap.scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_emotion.setText('')
            self.lbl_confidence.setText('')
            self.lbl_status.setText('Sẵn sàng dự đoán')
            self.lbl_status.setStyleSheet("color: green;")

    def run_prediction(self):
        if not self.model:
            self.lbl_status.setText('Lỗi: Vui lòng chọn và tải model trước.')
            self.lbl_status.setStyleSheet("color: red;")
            return
        if not self.image_path:
            self.lbl_status.setText('Lỗi: Vui lòng chọn ảnh.')
            self.lbl_status.setStyleSheet("color: red;")
            return
        
        try:
            self.lbl_status.setText('Đang dự đoán...')
            self.lbl_status.setStyleSheet("color: orange;")
            QApplication.processEvents()
            image = Image.open(self.image_path)
            image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            self.lbl_emotion.setText(f"{predicted_class}")
            self.lbl_confidence.setText(f"Độ tự tin: {confidence_score:.2f}%")
            self.lbl_status.setText('Dự đoán hoàn tất!')
            self.lbl_status.setStyleSheet("color: green;")
        except Exception as e:
            self.lbl_status.setText(f'Lỗi khi dự đoán: {e}')
            self.lbl_status.setStyleSheet("color: red;")

# --- 4. CHẠY ỨNG DỤNG ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = EmotionPredictorApp()
    ex.show()
    sys.exit(app.exec_())