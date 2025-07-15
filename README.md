# fer2013-cnn-pytorch Emotion Recognition with PyTorch & OpenCV

## 🚀 Project Overview

This project is an end-to-end system for real-time human emotion recognition. A custom Convolutional Neural Network (CNN) was built from scratch using PyTorch to classify 7 basic emotions (angry, disgust, fear, happy, neutral, sad, surprise). The trained model is then integrated with OpenCV to perform predictions on a live webcam feed.

This project serves as a practical demonstration of my skills in Deep Learning, computer vision, and model deployment, as highlighted in my resume.

## ✨ Key Features

*   **Real-time Detection:** Integrates with OpenCV to analyze faces and predict emotions live via webcam.
*   **Custom CNN Architecture:** The model is built from the ground up, demonstrating a strong understanding of deep learning fundamentals, including convolutional layers, batch normalization, and dropout for regularization.
*   **Data Augmentation:** Implements data augmentation techniques (random shifts, horizontal flips) to create a more robust and generalized model.
*   **Optimized Training:** Utilizes **Early Stopping** to prevent overfitting and save the best model based on validation loss, ensuring optimal performance.
*   **Detailed Analysis:** Includes Exploratory Data Analysis (EDA) to understand the dataset distribution and visualizes training history (loss/accuracy).

## 🛠️ Tech Stack

*   **Programming Language:** Python
*   **Deep Learning:** PyTorch
*   **Computer Vision:** OpenCV
*   **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
*   **Core Libraries:** Torchvision, PIL, Tqdm

## 📂 Project Structure

```
Emotion-Recognition-Project/
├── data/
│   ├── train/
│   └── test/
├── notebook4.ipynb         # Main notebook with EDA, model training, and evaluation.
├── best_model.pth          # Saved weights of the best performing model.
└── README.md
```

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Emotion-Recognition-Project.git
    cd Emotion-Recognition-Project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install torch torchvision opencv-python pandas matplotlib seaborn jupyterlab
    ```
4.  **Dataset:**
    Download the dataset (ví dụ: FER2013, hoặc dataset bạn dùng) and place the `train` and `test` folders inside the `data/` directory.

## 🏃‍♂️ Usage

1.  **Launch Jupyter Lab:**
    ```bash
    jupyter lab
    ```
2.  Open `notebook4.ipynb` and run the cells sequentially to see the data analysis, model training, and performance evaluation.
3.  *(Optional)* To run the live demo, you would typically have a separate `app.py` script that loads `best_model.pth` and uses OpenCV.

## 📊 Results

The model was trained for 100 epochs with an early stopping patience of 20. It achieved an impressive **~99.9% accuracy on the validation set**, demonstrating its effectiveness in classifying facial emotions.


*(Bạn hãy chụp lại ảnh đồ thị training/validation loss & accuracy và thay link vào đây)*

## 🔮 Future Improvements

*   Implement a standalone application (e.g., using Tkinter, PyQt, or Streamlit) for easier real-time demonstration.
*   Experiment with transfer learning using pre-trained models like ResNet or VGG to compare performance.
*   Deploy the model as a web service using Flask or FastAPI.
