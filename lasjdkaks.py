import sys
import os
import torch
import torchvision.transforms as transforms
import gradio as gr
import numpy as np
import cv2  
import mediapipe as mp # Bắt buộc phải có cho ST-GCN

# ---------------------------------------------------------
# 1. IMPORT KIẾN TRÚC MODEL
# ---------------------------------------------------------
sys.path.append('/kaggle/working/CS231_Video_Action_Recognition')

from tsm_resnet50_model import TSM_Network 
from model import ViTGRU
from stgcn_mediapipe_model import STGCN

# ---------------------------------------------------------
# 2. KHỞI TẠO 3 MODELS & LOAD TRỌNG SỐ (.pt)
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

num_classes = 51   
num_segments = 16  

# Khởi tạo model
model_tsm = TSM_Network(num_classes=num_classes, n_segment=num_segments).to(device)
model_vitgru = ViTGRU(num_classes=num_classes).to(device)
model_stgcn = STGCN(num_class=num_classes, in_channels=3).to(device)

# Load trọng số (LƯU Ý: Bạn cần đổi đúng link cho ViT-GRU và ST-GCN)
path_tsm = "/kaggle/input/models/midzid/tsm-resnet50-best/pytorch/default/1/tsm_resnet50_best.pt"
path_vitgru = "/kaggle/input/datasets/midzid/vit-gru-best/vit_gru_best.pt" 
path_stgcn = "/kaggle/input/datasets/midzid/stgcn-mediapipe-best/stgcn_mediapipe_best.pt" 

model_tsm.load_state_dict(torch.load(path_tsm, map_location=device))
# Bỏ comment 2 dòng dưới khi bạn đã có file trọng số thực tế của 2 model này
# model_vitgru.load_state_dict(torch.load(path_vitgru, map_location=device))
# model_stgcn.load_state_dict(torch.load(path_stgcn, map_location=device))

model_tsm.eval()
model_vitgru.eval()
model_stgcn.eval()

# ---------------------------------------------------------
# 3. TỪ ĐIỂN NHÃN & TIỀN XỬ LÝ (PRE-PROCESSING)
# ---------------------------------------------------------
LABELS = [
    "brush_hair", "cartwheel", "catch", "chew", "clap", 
    "climb", "climb_stairs", "dive", "draw_sword", "dribble", 
    "drink", "eat", "fall_floor", "fencing", "flic_flac", 
    "golf", "handstand", "hit", "hug", "jump", 
    "kick", "kick_ball", "kiss", "laugh", "pick", 
    "pour", "pullup", "punch", "push", "pushup", 
    "ride_bike", "ride_horse", "run", "shake_hands", "shoot_ball", 
    "shoot_bow", "shoot_gun", "sit", "situp", "smile", 
    "smoke", "somersault", "stand", "swing_baseball", "sword", 
    "sword_exercise", "talk", "throw", "turn", "walk", 
    "wave"
]

# A. Tiền xử lý Ảnh RGB cho TSM và ViT-GRU
transform = transforms.Compose([
    transforms.ToPILImage(),             
    transforms.Resize((224, 224)),       
    transforms.ToTensor(),               
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_video_rgb(video_path, num_frames=16): 
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return torch.zeros(1, num_frames, 3, 224, 224)
        
    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx - 1, 0))
            ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame_rgb))
        else:
            frames.append(frames[-1] if len(frames) > 0 else torch.zeros(3, 224, 224))
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if len(frames) > 0 else torch.zeros(3, 224, 224))
    
    video_tensor = torch.stack(frames[:num_frames]).unsqueeze(0) 
    return video_tensor

# B. Tiền xử lý Skeleton (MediaPipe) cho ST-GCN
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# B. Tiền xử lý Skeleton (MediaPipe) cho ST-GCN
from mediapipe.python.solutions import pose as mp_pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def process_video_skeleton(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return torch.zeros(1, 3, num_frames, 33, 1) # (N, C, T, V, M)

    frame_indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
    skeletons = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(idx - 1, 0))
            ret, frame = cap.read()
            
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks: # Nếu nhận diện được người
                landmarks = results.pose_landmarks.landmark
                frame_skeleton = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks]) # (33, 3)
            else:
                frame_skeleton = np.zeros((33, 3))
        else:
            frame_skeleton = skeletons[-1] if len(skeletons) > 0 else np.zeros((33, 3))
            
        skeletons.append(frame_skeleton)
    cap.release()
    
    while len(skeletons) < num_frames:
        skeletons.append(skeletons[-1] if len(skeletons) > 0 else np.zeros((33, 3)))
        
    skeletons = np.array(skeletons[:num_frames]) # Shape: (T, V, C) -> (16, 33, 3)
    skeletons = np.transpose(skeletons, (2, 0, 1)) # Shape: (C, T, V) -> (3, 16, 33)
    
    tensor_skel = torch.tensor(skeletons, dtype=torch.float32)
    # Thêm chiều Batch (N) và Số người (M) -> Shape cuối cùng (1, 3, 16, 33, 1)
    tensor_skel = tensor_skel.unsqueeze(0).unsqueeze(-1)
    
    return tensor_skel

# ---------------------------------------------------------
# 4. CÁC HÀM DỰ ĐOÁN ĐỘC LẬP
# ---------------------------------------------------------
def get_prediction_text(outputs):
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_idx].item()
    return f"Hành động: **{LABELS[predicted_idx]}**\nĐộ chính xác: {confidence*100:.2f}%"

def predict_tsm(video_filepath):
    if not video_filepath: return "Vui lòng tải video."
    try:
        input_tensor = process_video_rgb(video_filepath).to(device)
        with torch.no_grad():
            return get_prediction_text(model_tsm(input_tensor))
    except Exception as e: return f"Lỗi: {str(e)}"

def predict_vitgru(video_filepath):
    if not video_filepath: return "Vui lòng tải video."
    try:
        input_tensor = process_video_rgb(video_filepath).to(device)
        with torch.no_grad():
            return get_prediction_text(model_vitgru(input_tensor))
    except Exception as e: return f"Lỗi: {str(e)}"

def predict_stgcn(video_filepath):
    if not video_filepath: return "Vui lòng tải video."
    try:
        input_tensor = process_video_skeleton(video_filepath).to(device)
        with torch.no_grad():
            return get_prediction_text(model_stgcn(input_tensor))
    except Exception as e: return f"Lỗi: {str(e)}"

# ---------------------------------------------------------
# 5. XÂY DỰNG GIAO DIỆN GRADIO BLOCKS
# ---------------------------------------------------------
gr.close_all() 

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>🚀 Hệ Thống Nhận Diện Hành Động (Đa Mô Hình)</h1></center>")
    gr.Markdown("Tải video lên và trải nghiệm sức mạnh của 3 cấu trúc mạng khác nhau: CNN (TSM), Transformer + RNN (ViT-GRU), và Graph Neural Network (ST-GCN).")
    
    # --- TAB 1: TSM ---
    with gr.Tab("🎞️ TSM (ResNet50)"):
        with gr.Row():
            with gr.Column():
                vid_tsm = gr.Video(label="Upload Video Test")
                btn_tsm = gr.Button("Dự đoán bằng TSM", variant="primary")
            with gr.Column():
                out_tsm = gr.Markdown(label="Kết quả")
        btn_tsm.click(fn=predict_tsm, inputs=vid_tsm, outputs=out_tsm)

    # --- TAB 2: ViT-GRU ---
    with gr.Tab("👁️ ViT-GRU"):
        with gr.Row():
            with gr.Column():
                vid_vit = gr.Video(label="Upload Video Test")
                btn_vit = gr.Button("Dự đoán bằng ViT-GRU", variant="primary")
            with gr.Column():
                out_vit = gr.Markdown(label="Kết quả")
        btn_vit.click(fn=predict_vitgru, inputs=vid_vit, outputs=out_vit)

    # --- TAB 3: ST-GCN ---
    with gr.Tab("🦴 ST-GCN (Skeleton)"):
        with gr.Row():
            with gr.Column():
                vid_gcn = gr.Video(label="Upload Video Test")
                btn_gcn = gr.Button("Dự đoán bằng ST-GCN", variant="primary")
            with gr.Column():
                out_gcn = gr.Markdown(label="Kết quả")
        btn_gcn.click(fn=predict_stgcn, inputs=vid_gcn, outputs=out_gcn)

# ---------------------------------------------------------
# 6. KHỞI CHẠY APP
# ---------------------------------------------------------
demo.launch(server_name="0.0.0.0", share=True)