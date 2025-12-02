import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from datetime import datetime, timedelta
import urllib3
import numpy as np
import os

# Tắt cảnh báo SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. CẤU HÌNH (Dành cho Model 2 Lớp)
# ==========================================
MODEL_PATH = 'best_model_binary_longan.pth' 

# Nhãn cho bài toán nhị phân (Binary)
CLASSES = ['✅ An Toàn (Mây Ít/Không Mưa)', '⚠️ Nguy Cơ (Mây Dày/Mưa)']

# Chạy trên CPU để ổn định trên Hugging Face Free Tier
DEVICE = torch.device("cpu") 

# Tọa độ Long An
LONG_AN_BBOX = "105.55,9.95,107.05,11.45"

# ==========================================
# 2. LOAD MODEL
# ==========================================
def load_model():
    print(f"⏳ Đang load model Binary từ {MODEL_PATH}...")
    try:
        # Khởi tạo ResNet18
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        
        # Cấu trúc lớp cuối khớp với lúc train (Dropout -> 2 Lớp)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.5), # Dropout không ảnh hưởng lúc eval, nhưng cần để khớp key
            nn.Linear(num_ftrs, 2) # QUAN TRỌNG: Output là 2
        )
        
        if os.path.exists(MODEL_PATH):
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval() # Chế độ dự báo
            print("✅ Load model thành công!")
            return model
        else:
            print(f"⚠️ Cảnh báo: Không tìm thấy file {MODEL_PATH}. Hãy upload file model lên Space.")
            return None
    except Exception as e:
        print(f"❌ Lỗi load model: {e}")
        return None

model = load_model()

# Transform ảnh (Giống hệt lúc train)
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 3. TẢI ẢNH VỆ TINH (NASA API)
# ==========================================
def fetch_modis_image(date_obj, time_str):
    try:
        # Chuyển giờ VN sang UTC để gọi API
        full_dt_vn = datetime.combine(date_obj, datetime.strptime(time_str, "%H:%M").time())
        full_dt_utc = full_dt_vn - timedelta(hours=7)
        time_param = full_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
        params = {
            "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
            "LAYERS": "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "STYLES": "", "FORMAT": "image/jpeg", "SRS": "EPSG:4326",
            "BBOX": LONG_AN_BBOX, 
            "WIDTH": "512", "HEIGHT": "512",
            "TIME": time_param
        }

        print(f"🔗 Tải ảnh Long An lúc: {time_param} UTC")
        response = requests.get(url, params=params, timeout=20, verify=False)
        
        if response.status_code == 200 and len(response.content) > 3000:
            img = Image.open(BytesIO(response.content))
            return img, "✅ Đã tải ảnh vệ tinh thành công."
        else:
            return None, "⚠️ Không tìm thấy dữ liệu ảnh. (Lỗi thường gặp: Vệ tinh chưa bay qua, hoặc trời tối)."

    except Exception as e:
        return None, f"Lỗi kết nối: {str(e)}"

# ==========================================
# 4. HÀM DỰ BÁO
# ==========================================
def predict_longan(day, month, year, time_input):
    if model is None: 
        return None, "❌ Lỗi: Chưa có file model (.pth) trên Server!"
    
    # Tạo đối tượng ngày từ 3 input riêng biệt
    try:
        date_input = datetime(int(year), int(month), int(day))
    except ValueError:
        return None, "⚠️ Ngày tháng năm không hợp lệ!"

    # Tải ảnh
    img, msg = fetch_modis_image(date_input, time_input)
    if img is None: 
        return None, msg
    
    # Dự báo
    try:
        img_t = val_transform(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(img_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, idx = torch.max(probs, 1)
            
        label = CLASSES[idx.item()]
        
        # Tạo thông báo kết quả
        res_text = (
            f"🎯 KẾT QUẢ: {label}\n"
            f"📊 Độ tin cậy: {conf.item()*100:.2f}%\n"
            f"🕒 Thời gian: {day}/{month}/{year} - {time_input}"
        )
        return img, res_text
    except Exception as e:
        return img, f"Lỗi khi chạy model: {str(e)}"

# ==========================================
# 5. GIAO DIỆN (3 INPUT NGÀY THÁNG NĂM)
# ==========================================
valid_times = ["13:00", "13:10", "13:20", "13:30", "13:40", "13:50"]

# Không dùng tham số theme/css để tránh lỗi version
with gr.Blocks() as demo:
    # Tiêu đề HTML căn giữa
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🛰️ HỆ THỐNG CẢNH BÁO MƯA SỚM - TỈNH LONG AN</h1>
            <p>Sử dụng công nghệ Deep Learning (ResNet-18) phân tích ảnh vệ tinh MODIS</p>
        </div>
        """
    )
    
    with gr.Row():
        # Cột trái: Nhập liệu
        with gr.Column():
            gr.Markdown("### 1️⃣ Chọn Thời Gian")
            
            # Hàng chứa 3 ô nhập: Ngày - Tháng - Năm
            with gr.Row():
                inp_day = gr.Number(label="Ngày", value=datetime.now().day, precision=0, minimum=1, maximum=31)
                inp_month = gr.Number(label="Tháng", value=datetime.now().month, precision=0, minimum=1, maximum=12)
                inp_year = gr.Number(label="Năm", value=datetime.now().year, precision=0, minimum=2000, maximum=2030)
            
            inp_time = gr.Dropdown(label="⏰ Giờ Vệ Tinh (Giờ VN)", choices=valid_times, value="13:30")
            
            btn = gr.Button("🔍 PHÂN TÍCH NGAY", variant="primary")
            
            gr.Markdown("ℹ️ *Khuyên dùng khung giờ **13:30** để có ảnh rõ nét nhất.*")
        
        # Cột phải: Kết quả
        with gr.Column():
            gr.Markdown("### 2️⃣ Kết Quả Dự Báo")
            out_img = gr.Image(label="Ảnh Vệ Tinh Thực Tế", type="pil")
            out_txt = gr.Textbox(label="Chi Tiết", lines=4)
            
    btn.click(predict_longan, inputs=[inp_day, inp_month, inp_year, inp_time], outputs=[out_img, out_txt])
demo.launch()
