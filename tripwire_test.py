import cv2
import os
import glob
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import numpy as np
import datetime
import csv
from collections import Counter

# --- 1. CONFIG & PATHS ---
FRAME_FOLDER = "ChokePoint_DataSets/P1E_S1/P1E_S1_C1"
# FRAME_FOLDER = "ChokePoint_DataSets/P1E_S1/P1E_S1_C2"
CSV_FILE = "footfall_analytics.csv"

# Models
GENDER_MODEL_FILE = "models/swin_v2_0_gender.pth"
AGE_MODEL_FILE = "models/swin_ordinal_age_v2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tripwire Bounds
TRIPWIRE_Y = 450  
DOOR_LEFT_X = 80   
DOOR_RIGHT_X = 800  

print(f"🚀 Initializing Unified Sentinel on {DEVICE.upper()}...")

# --- 2. LOAD AI BRAINS ---
# YOLO Tracker
yolo_model = YOLO('yolov8n.pt')

# Swin Gender
gender_model = models.swin_t(weights=None)
gender_model.head = nn.Linear(gender_model.head.in_features, 2)
gender_model.load_state_dict(torch.load(GENDER_MODEL_FILE, map_location=DEVICE, weights_only=True))
gender_model.to(DEVICE).half().eval()

# Swin Age (Ordinal)
age_model = models.swin_t(weights=None)
age_model.head = nn.Linear(age_model.head.in_features, 4)
age_model.load_state_dict(torch.load(AGE_MODEL_FILE, map_location=DEVICE, weights_only=True))
age_model.to(DEVICE).half().eval()

# Caffe Face Detector (from your webcam script)
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gender_classes = ['Female', 'Male']
age_classes = ['0-12', '13-19', '20-35', '36-55', '56+']

# --- 3. STATE MEMORY ---
counted_ids = set()
track_history = {}
gender_memory = {}
age_memory = {}
demographics_cache = {} # THE PREDICTIVE BUFFER: Stores {id: {'gender': '...', 'age': '...'}}
total_footfall = 0
FRAMES_TO_WAIT = 5

# Initialize CSV File with headers if it doesn't exist
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Track_ID", "Gender", "Age", "Event"])

# --- 4. IMAGE PROCESSING HELPER ---
def process_crop(crop_bgr):
    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        merged = cv2.merge((cl, a, b))
        contrast_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        bright_img = cv2.convertScaleAbs(contrast_img, alpha=1.1, beta=15)
        blurred = cv2.GaussianBlur(bright_img, (3, 3), 0)
        h, w = blurred.shape[:2]
        longest = max(h, w)
        top, left = (longest - h) // 2, (longest - w) // 2
        padded = cv2.copyMakeBorder(blurred, top, longest-h-top, left, longest-w-left, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    except: 
        return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

# --- 5. MAIN LOOP ---
image_files = sorted(glob.glob(os.path.join(FRAME_FOLDER, "*.jpg")))

for frame_idx, img_path in enumerate(image_files):
    frame = cv2.imread(img_path)
    results = yolo_model.track(frame, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)
    
    cv2.line(frame, (DOOR_LEFT_X, TRIPWIRE_Y), (DOOR_RIGHT_X, TRIPWIRE_Y), (0, 0, 255), 3) 
    cv2.putText(frame, f"Store Footfall: {total_footfall}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 3)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            h_box = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # 1. EARLY DEMOGRAPHIC SCAN (Only run if we don't know who they are yet)
            if track_id not in demographics_cache and frame_idx % 2 == 0: 
                # Crop the top 45% of the body box to look for a face
                person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                h_c, w_c = person_crop.shape[:2]
                face_zone = person_crop[0:int(h_c*0.45), :]
                
                if face_zone.size > 0 and face_zone.shape[0] > 50:
                    blob = cv2.dnn.blobFromImage(cv2.resize(face_zone, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    face_net.setInput(blob)
                    detections = face_net.forward()

                    # If a face is found with high confidence
                    if detections[0, 0, 0, 2] > 0.65:
                        zh, zw = face_zone.shape[:2]
                        fx, fy, fx2, fy2 = (detections[0, 0, 0, 3:7] * [zw, zh, zw, zh]).astype("int")
                        face_crop = face_zone[max(0, fy):min(zh, fy2), max(0, fx):min(zw, fx2)]
                        
                        if face_crop.size > 1000:
                            f_pil = process_crop(face_crop)
                            with torch.no_grad():
                                # Predict Gender
                                g_out = gender_model(transform(f_pil).unsqueeze(0).to(DEVICE).half())
                                g_idx = torch.argmax(g_out[0]).item()
                                
                                # Predict Age (Ordinal)
                                a_out = age_model(transform(f_pil).unsqueeze(0).to(DEVICE).half())
                                a_idx = (torch.sigmoid(a_out[0]) > 0.5).sum().item()
                                
                            # Add to temporal buffers
                            if track_id not in gender_memory: gender_memory[track_id] = []
                            if track_id not in age_memory: age_memory[track_id] = []
                            
                            gender_memory[track_id].append(gender_classes[g_idx])
                            age_memory[track_id].append(age_classes[a_idx])
                            
                            # If we have collected enough good frames, lock in the majority vote!
                            if len(gender_memory[track_id]) >= FRAMES_TO_WAIT:
                                final_gender = Counter(gender_memory[track_id]).most_common(1)[0][0]
                                final_age = Counter(age_memory[track_id]).most_common(1)[0][0]
                                demographics_cache[track_id] = {'gender': final_gender, 'age': final_age}

            # 2. UI OVERLAY (Show cached data if it exists, else show 'Scanning')
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            if track_id in demographics_cache:
                label = f"ID:{track_id} | {demographics_cache[track_id]['gender']} {demographics_cache[track_id]['age']}"
                color = (0, 255, 0)
            else:
                collected = len(gender_memory.get(track_id, []))
                label = f"ID:{track_id} | Scanning... ({collected}/{FRAMES_TO_WAIT})"
                color = (0, 165, 255)
                
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 3. THE VECTOR TRIPWIRE & CSV LOGGER
            if track_id in track_history:
                prev_cy = track_history[track_id]
                
                if DOOR_LEFT_X < cx < DOOR_RIGHT_X:
                    if prev_cy < TRIPWIRE_Y and cy >= TRIPWIRE_Y:
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            total_footfall += 1
                            
                            # Fetch buffered data (fallback to 'Unknown' if they crossed too fast to scan)
                            demo_data = demographics_cache.get(track_id, {'gender': 'Unknown', 'age': 'Unknown'})
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Log to CSV
                            with open(CSV_FILE, mode='a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([timestamp, track_id, demo_data['gender'], demo_data['age'], "ENTERED"])
                            
                            print(f"✅ LOGGED: [{timestamp}] ID:{track_id} | {demo_data['gender']} | {demo_data['age']} | Total: {total_footfall}")
            
            track_history[track_id] = cy

    cv2.imshow("Unified Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"🛑 Feed terminated. Analytics saved to {CSV_FILE}.")