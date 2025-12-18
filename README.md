# 🛰️ Hệ Thống Cảnh Báo Mưa Sớm - Tỉnh Long An

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Gradio](https://img.shields.io/badge/Gradio-6.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

**Ứng dụng Deep Learning (CNN) để dự báo mưa từ ảnh vệ tinh MODIS**

[🚀 Demo Trực Tiếp](https://huggingface.co/spaces/CaoPhu2305/DATTNT_CNN_NHOM08_2LABEL) • [📖 Hướng Dẫn](#-cài-đặt) • [📊 Kết Quả](#-kết-quả-thử-nghiệm)

</div>

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Tính Năng](#-tính-năng)
- [Kiến Trúc Hệ Thống](#-kiến-trúc-hệ-thống)
- [Công Nghệ Sử Dụng](#-công-nghệ-sử-dụng)
- [Cài Đặt](#-cài-đặt)
- [Cách Sử Dụng](#-cách-sử-dụng)
- [Dataset](#-dataset)
- [Huấn Luyện Model](#-huấn-luyện-model)
- [Kết Quả Thử Nghiệm](#-kết-quả-thử-nghiệm)
- [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
- [Thành Viên Nhóm](#-thành-viên-nhóm)
- [License](#-license)

---

## 🌟 Giới Thiệu

Dự án này xây dựng một **Hệ thống Cảnh báo Mưa Sớm** cho tỉnh Long An, sử dụng kỹ thuật Deep Learning để phân tích ảnh vệ tinh MODIS (NASA). Hệ thống có khả năng:

- 🔍 **Phân loại** điều kiện thời tiết thành 2 trạng thái: **An Toàn** (ít mây/không mưa) và **Nguy Cơ** (mây dày/có mưa)
- 🌐 **Tự động tải** ảnh vệ tinh real-time từ NASA GIBS API
- 💻 **Giao diện web** thân thiện, dễ sử dụng với Gradio

> **Ứng dụng thực tiễn:** Hỗ trợ nông dân, cơ quan quản lý nông nghiệp và người dân tỉnh Long An trong việc lên kế hoạch hoạt động dựa trên dự báo thời tiết.

---

## ✨ Tính Năng

| Tính năng | Mô tả |
|-----------|-------|
| 🛰️ **Ảnh vệ tinh Real-time** | Tự động lấy ảnh MODIS từ NASA GIBS API |
| 🤖 **Dự báo AI** | Sử dụng ResNet-18 đã được fine-tune |
| 📊 **Độ tin cậy** | Hiển thị % xác suất dự đoán |
| 🌍 **Khu vực cụ thể** | Tập trung vào tọa độ tỉnh Long An |
| ⏰ **Chọn thời điểm** | Hỗ trợ chọn ngày/tháng/năm và khung giờ |
| 🖥️ **Giao diện Web** | Deploy trên Hugging Face Spaces |

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   NASA GIBS API  │ ───▶ │   Tiền xử lý     │ ───▶ │   ResNet-18      │
│   (MODIS Image)  │      │   (Transform)    │      │   (CNN Model)    │
└──────────────────┘      └──────────────────┘      └────────┬─────────┘
                                                              │
                                                              ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   Kết quả        │ ◀─── │   Softmax        │ ◀─── │   Feature        │
│   Dự báo         │      │   Classification │      │   Extraction     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
```

### Quy trình hoạt động:
1. **Input:** Người dùng chọn ngày/giờ cần dự báo
2. **Fetch:** Hệ thống tự động tải ảnh vệ tinh MODIS từ NASA
3. **Preprocess:** Resize ảnh về 224x224, chuẩn hóa theo ImageNet
4. **Predict:** Model ResNet-18 phân loại ảnh
5. **Output:** Kết quả dự báo + độ tin cậy (%)

---

## 🛠️ Công Nghệ Sử Dụng

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| ![Python](https://img.shields.io/badge/Python-3.8+-blue) | 3.8+ | Ngôn ngữ lập trình chính |
| ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) | 2.0+ | Framework Deep Learning |
| ![Gradio](https://img.shields.io/badge/Gradio-6.0+-orange) | 6.0+ | Xây dựng giao diện web |
| ![Torchvision](https://img.shields.io/badge/Torchvision-0.15+-lightblue) | 0.15+ | Xử lý ảnh, pretrained models |
| ![PIL](https://img.shields.io/badge/Pillow-9.0+-yellow) | 9.0+ | Xử lý hình ảnh |
| ![NumPy](https://img.shields.io/badge/NumPy-1.24+-cyan) | 1.24+ | Tính toán số học |

---

## 📥 Cài Đặt

### Yêu cầu hệ thống
- Python 3.8 trở lên
- pip (Python package manager)
- GPU (khuyến nghị, nhưng không bắt buộc)

### Các bước cài đặt

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/longan-rain-prediction-cnn.git
cd longan-rain-prediction-cnn

# 2. Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate     # Windows

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Chạy ứng dụng
python app.py
```

### Yêu cầu file model
⚠️ **Lưu ý:** Bạn cần có file `best_model_binary_longan.pth` trong thư mục gốc. File này có thể tải từ:
- [Hugging Face Model Hub](https://huggingface.co/spaces/CaoPhu2305/DATTNT_CNN_NHOM08_2LABEL) 

---

## 💡 Cách Sử Dụng

### 1. Chạy local
```bash
python app.py
```
Sau đó mở trình duyệt và truy cập: `http://127.0.0.1:7860`

### 2. Demo online
Truy cập trực tiếp: [🔗 Hugging Face Space](https://huggingface.co/spaces/CaoPhu2305/DATTNT_CNN_NHOM08_2LABEL)

### 3. Hướng dẫn sử dụng
1. **Chọn ngày:** Nhập ngày, tháng, năm cần dự báo
2. **Chọn giờ:** Chọn khung giờ vệ tinh (khuyến nghị **13:30** để có ảnh rõ nhất)
3. **Nhấn "Phân Tích Ngay":** Hệ thống sẽ tải ảnh vệ tinh và đưa ra dự báo
4. **Xem kết quả:** Ảnh vệ tinh + Kết quả phân loại + Độ tin cậy

---

## 📊 Dataset

### Thông tin Dataset

| Thông số | Giá trị |
|----------|---------|
| **Tổng số ảnh** | 2,000 ảnh |
| **Lớp 0 - An Toàn** | 1,000 ảnh (mây ít, trời quang) |
| **Lớp 1 - Nguy Cơ** | 1,000 ảnh (mây dày, có mưa) |
| **Nguồn dữ liệu** | NASA MODIS Aqua TrueColor |
| **Khu vực** | Tỉnh Long An, Việt Nam |
| **Tọa độ** | `105.55°E - 107.05°E, 9.95°N - 11.45°N` |

### Cấu trúc Dataset
```
LongAn_Binary_Dataset_1k/
├── 0_AnToan/          # Ảnh thời tiết an toàn
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── 1_NguyCo/          # Ảnh thời tiết nguy cơ mưa
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

---

## 🎯 Huấn Luyện Model

### Kiến trúc Model
- **Base Model:** ResNet-18 (Pretrained trên ImageNet)
- **Transfer Learning:** Fine-tune toàn bộ model
- **Output Layer:** `Dropout(0.5) → Linear(512, 2)`

### Hyperparameters
```python
{
    "batch_size": 32,
    "image_size": 224,
    "epochs_freeze": 5,      # Freeze backbone
    "epochs_unfreeze": 20,   # Fine-tune toàn bộ
    "optimizer": "Adam",
    "learning_rate": 1e-4,   # Ban đầu
    "lr_after_unfreeze": 1e-5,
    "weight_decay": 1e-4,
    "cross_validation": "5-Fold"
}
```

### Data Augmentation
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## 📈 Kết Quả Thử Nghiệm

### Final Training với Cấu hình Trial 1

| Thông số | Giá trị |
|----------|---------|
| **Learning Rate** | 2.50e-04 |
| **Weight Decay** | 8.93e-04 |
| **Dropout** | 0.52 |
| **Epochs** | 45 (Freeze 5 + Unfreeze 40) |

### Kết quả Training

| Metric | Giá trị |
|--------|---------|
| **Best Test Accuracy** | **66.75%** |
| **Best Epoch** | 16 |
| **Final Train Loss** | 0.545 |
| **Final Test Loss** | 0.689 |

### Chi tiết đánh giá (Classification Report)

| Lớp | Precision | Recall | F1-Score | Support |
|-----|-----------|--------|----------|---------|
| 0_AnToan | 0.69 | 0.62 | 0.65 | 200 |
| 1_NguyCo | 0.65 | 0.71 | 0.68 | 200 |
| **Accuracy** | | | **0.67** | 400 |

### Nhận xét & Hạn chế

#### 📊 Kết quả huấn luyện:
- Model đạt độ chính xác **66.75%** trên Test Set - đây là kết quả **khiêm tốn**, chỉ nhỉnh hơn ngưỡng đoán ngẫu nhiên (50%)
- Model bắt đầu **overfit từ Epoch 16**: Train Loss tiếp tục giảm (0.773 → 0.545) trong khi Test Loss tăng lên (0.633 → 0.689)
- Precision và Recall không cân bằng giữa 2 lớp, cho thấy model gặp khó khăn trong việc phân biệt

#### ⚠️ Hạn chế của bài toán:
- **Dữ liệu ảnh vệ tinh MODIS có nhiều nhiễu**: Ảnh True Color chịu ảnh hưởng bởi góc chụp, thời điểm, khí quyển... làm giảm chất lượng đầu vào
- **Chỉ dựa vào ảnh RGB là chưa đủ**: Dự báo thời tiết thực tế cần kết hợp nhiều nguồn dữ liệu (nhiệt độ, độ ẩm, áp suất, dữ liệu radar...), không thể đạt độ chính xác cao chỉ với ảnh vệ tinh đơn thuần
- **Đặc điểm mây/mưa khó phân biệt trực quan**: Mây có thể xuất hiện mà không có mưa, hoặc mưa có thể đến từ hệ thống thời tiết ngoài vùng quan sát

#### 🔧 Kỹ thuật đã áp dụng:
- Transfer Learning từ ImageNet (ResNet-18)
- Data Augmentation đa dạng
- Regularization (Dropout 52% + Weight Decay)
- Freeze/Unfreeze Strategy
- CosineAnnealing LR Scheduler

> **Kết luận:** Dự án này mang tính chất **thử nghiệm/học thuật**, chứng minh khả năng áp dụng CNN vào phân loại ảnh vệ tinh. Tuy nhiên, để đạt độ chính xác cao hơn trong thực tế, cần bổ sung thêm các nguồn dữ liệu khí tượng khác.

---

## 📁 Cấu Trúc Thư Mục

```
📦 longan-rain-prediction-cnn/
├── 📄 app.py                        # Ứng dụng Gradio chính
├── 📄 train.ipynb                   # Notebook huấn luyện model
├── 📄 requirements.txt              # Danh sách thư viện
├── 📄 README.md                     # Tài liệu dự án
├── 📄 huongdansudung.txt           # Hướng dẫn sử dụng (Tiếng Việt)
├── 📄 .gitignore                    # Cấu hình Git
├── 🧠 best_model_binary_longan.pth  # Model đã huấn luyện (~45MB)
└── 📂 LongAn_Binary_Dataset_1k/     # Dataset (không push lên Git)
    ├── 📂 0_AnToan/
    └── 📂 1_NguyCo/
```

---

## 🙏 Lời Cảm Ơn

- **NASA** - Cung cấp dữ liệu ảnh vệ tinh MODIS qua GIBS API
- **PyTorch Team** - Framework Deep Learning mạnh mẽ
- **Hugging Face** - Nền tảng deploy ứng dụng ML miễn phí
- **Gradio** - Thư viện xây dựng giao diện web nhanh chóng

---

<div align="center">

**⭐ Nếu dự án hữu ích, hãy cho chúng mình một Star! ⭐**

Made with ❤️ by Nhóm 8

</div>
