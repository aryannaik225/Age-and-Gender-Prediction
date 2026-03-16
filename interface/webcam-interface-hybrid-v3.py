import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import urllib.request
import time
from collections import Counter

# --- CONFIG ---
# Hardcoded paths to the centralized models folder
GENDER_MODEL_FILE = "models/swin_v2_0_gender.pth"
AGE_MODEL_FILE = "models/swin_ordinal_age_v2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VAULT_DIR = "secure_vault_v3_Temporal"

print(f"🧠 Static Load -> Gender: {os.path.basename(GENDER_MODEL_FILE)}")
print(f"🧠 Static Load -> Age: {os.path.basename(AGE_MODEL_FILE)}")

PROTO_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
MODEL_URL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
if not os.path.exists("deploy.prototxt"): urllib.request.urlretrieve(PROTO_URL, "deploy.prototxt")
if not os.path.exists("res10_300x300_ssd_iter_140000.caffemodel"): urllib.request.urlretrieve(MODEL_URL, "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

def process_crop_sota(crop_bgr):
    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        merged = cv2.merge((clahe.apply(l), a, b))
        contrast_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        bright_img = cv2.convertScaleAbs(contrast_img, alpha=1.1, beta=15)
        blurred_img = cv2.GaussianBlur(bright_img, (3, 3), 0)
        h, w = blurred_img.shape[:2]
        longest_edge = max(h, w)
        top, left = (longest_edge - h) // 2, (longest_edge - w) // 2
        padded = cv2.copyMakeBorder(blurred_img, top, longest_edge-h-top, left, longest_edge-w-left, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    except: return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

age_classes = ['0-12', '13-19', '20-35', '36-55', '56+']
gender_classes = ['Female', 'Male']
for gender in gender_classes: os.makedirs(os.path.join(VAULT_DIR, "Gender", gender), exist_ok=True)
for age in age_classes: os.makedirs(os.path.join(VAULT_DIR, "Age", age), exist_ok=True)

yolo_model = YOLO('yolov8n-seg.pt') 

gender_model = models.swin_t(weights=None); gender_model.head = nn.Linear(gender_model.head.in_features, 2)
gender_model.load_state_dict(torch.load(GENDER_MODEL_FILE, map_location=DEVICE))
gender_model.to(DEVICE).half().eval()

age_model = models.swin_t(weights=None); age_model.head = nn.Linear(age_model.head.in_features, 4)
age_model.load_state_dict(torch.load(AGE_MODEL_FILE, map_location=DEVICE))
age_model.to(DEVICE).half().eval() 

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# TEMPORAL MEMORY BUFFERS
age_memory = {} 
FRAMES_TO_WAIT = 3 

# UI STATE TRACKER (Holds the final answers for the overlay)
ui_state = {}

print(f"🚀 Live Webcam Demo Active on {DEVICE}... Press 'Q' to quit.")

# LINUX WEBCAM INIT
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Mirror the frame so it feels natural for the user
    frame = cv2.flip(frame, 1) 
    
    frame_count += 1
    
    # Run YOLO every frame for smooth UI tracking
    results = yolo_model.track(frame, classes=[0], persist=True, tracker="botsort.yaml", retina_masks=True, verbose=False)
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        # Only process heavy AI on every 3rd frame to save VRAM
        do_heavy_ai = (frame_count % 3 == 0) and (results[0].masks is not None)
        polygons = results[0].masks.xy if do_heavy_ai else None
        
        for i, (box, track_id) in enumerate(zip(boxes, track_ids)):
            x1, y1, x2, y2 = map(int, box)
            
            if track_id not in ui_state:
                ui_state[track_id] = {"gender": "Scanning...", "age": "Scanning..."}

            # --- HEAVY AI PROCESSING (Every 3rd frame) ---
            if do_heavy_ai:
                raw_crop = frame[max(0,y1):y2, max(0,x1):x2]
                if raw_crop.size > 0 and raw_crop.shape[0] >= 60:
                    
                    mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                    poly = np.int32(polygons[i])
                    if len(poly) > 0: cv2.fillPoly(mask_img, [poly], 255)
                    isolated_crop = cv2.bitwise_and(frame, frame, mask=mask_img)[max(0,y1):y2, max(0,x1):x2]

                    # GENDER
                    if ui_state[track_id]["gender"] == "Scanning...":
                        p_pil = process_crop_sota(isolated_crop)
                        with torch.no_grad():
                            g_out = gender_model(transform(p_pil).unsqueeze(0).to(DEVICE).half())
                            g_conf, g_idx = torch.max(torch.nn.functional.softmax(g_out[0], dim=0), dim=0)
                        if g_conf.item() >= 0.85:
                            ui_state[track_id]["gender"] = gender_classes[g_idx.item()]

                    # AGE
                    if ui_state[track_id]["age"] == "Scanning...":
                        h_c, w_c = isolated_crop.shape[:2]
                        face_search_zone = isolated_crop[0:int(h_c*0.45), :] 
                        if face_search_zone.size > 0:
                            blob = cv2.dnn.blobFromImage(cv2.resize(face_search_zone, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                            face_net.setInput(blob)
                            detections = face_net.forward()

                            for j in range(detections.shape[2]):
                                if detections[0, 0, j, 2] > 0.65: 
                                    zh, zw = face_search_zone.shape[:2]
                                    fx, fy, fx2, fy2 = (detections[0, 0, j, 3:7] * [zw, zh, zw, zh]).astype("int")
                                    face_crop = face_search_zone[max(0, fy):min(zh, fy2), max(0, fx):min(zw, fx2)]
                                    
                                    non_zero_pixels = cv2.countNonZero(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))
                                    if face_crop.size > 1500 and non_zero_pixels > (face_crop.size * 0.3):
                                        f_pil = process_crop_sota(face_crop)
                                        with torch.no_grad():
                                            a_out = age_model(transform(f_pil).unsqueeze(0).to(DEVICE).half())
                                            a_idx = (torch.sigmoid(a_out[0]) > 0.5).sum().item()
                                            raw_age_label = age_classes[a_idx]
                                        
                                        if track_id not in age_memory: age_memory[track_id] = []
                                        age_memory[track_id].append(raw_age_label)
                                        
                                        if len(age_memory[track_id]) >= FRAMES_TO_WAIT:
                                            ui_state[track_id]["age"] = Counter(age_memory[track_id]).most_common(1)[0][0]
                                        break

            # --- DRAWING THE SDE UI OVERLAY ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 2)
            
            display_text = f"{ui_state[track_id]['gender']} | {ui_state[track_id]['age']}"
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - 35), (x1 + text_w + 10, y1), (0, 255, 150), -1)
            
            cv2.putText(frame, display_text, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("SDE Demo - Age & Gender", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    if frame_count % 3 == 0:
        torch.cuda.empty_cache()

cap.release()
cv2.destroyAllWindows()