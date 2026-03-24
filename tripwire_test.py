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
import threading
import queue
import json

# --- 1. CONFIG & PATHS ---
FRAME_FOLDER = "ChokePoint_DataSets/P1E_S1/P1E_S1_C1"
CSV_FILE = "footfall_analytics.csv"
POLYGON_SAVE_FILE = "polygon_config.json"
HARVEST_DIR = "harvested_faces"
os.makedirs(HARVEST_DIR, exist_ok=True) 

GENDER_MODEL_FILE = "models/swin_v2_0_gender.pth"
AGE_MODEL_FILE = "models/swin_ordinal_age_v2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Initializing Asynchronous Sentinel on {DEVICE.upper()}...")

# --- 2. LOAD AI BRAINS ---
yolo_model = YOLO('yolov8n.pt')

gender_model = models.swin_t(weights=None)
gender_model.head = nn.Linear(gender_model.head.in_features, 2)
gender_model.load_state_dict(torch.load(GENDER_MODEL_FILE, map_location=DEVICE, weights_only=True))
gender_model.to(DEVICE).half().eval()

age_model = models.swin_t(weights=None)
age_model.head = nn.Linear(age_model.head.in_features, 4)
age_model.load_state_dict(torch.load(AGE_MODEL_FILE, map_location=DEVICE, weights_only=True))
age_model.to(DEVICE).half().eval()

face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gender_classes = ['Female', 'Male']
age_classes = ['0-12', '13-19', '20-35', '36-55', '56+']

# --- 3. STATE MEMORY & ASYNC QUEUES ---
counted_ids = set()
logged_ids = set() 
track_history = {} 
gender_memory = {}
age_memory = {}
demographics_cache = {} 
total_footfall = 0
FRAMES_TO_WAIT = 5

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Track_ID", "Gender", "Age", "Event"])

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

# --- 4. THE ASYNCHRONOUS BACKGROUND WORKER (THREAD 2) ---
ai_queue = queue.Queue(maxsize=100)

def background_ai_worker():
    global demographics_cache
    harvested_ids = set()
    
    print("⚙️ Background AI Worker is running...")
    while True:
        task = ai_queue.get()
        if task is None: break 
        
        track_id, frame_idx, person_crop = task
        h_c, w_c = person_crop.shape[:2]
        face_zone = person_crop[0:int(h_c*0.45), :]
        
        if face_zone.size > 0 and face_zone.shape[0] > 50:
            blob = cv2.dnn.blobFromImage(cv2.resize(face_zone, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()

            if detections[0, 0, 0, 2] > 0.65:
                zh, zw = face_zone.shape[:2]
                fx, fy, fx2, fy2 = (detections[0, 0, 0, 3:7] * [zw, zh, zw, zh]).astype("int")
                face_crop = face_zone[max(0, fy):min(zh, fy2), max(0, fx):min(zw, fx2)]
                
                if face_crop.size > 1000:
                    
                    if track_id not in harvested_ids:
                        cv2.imwrite(os.path.join(HARVEST_DIR, f"id_{track_id}_face.jpg"), face_crop)
                        cv2.imwrite(os.path.join(HARVEST_DIR, f"id_{track_id}_body.jpg"), person_crop)
                        harvested_ids.add(track_id)
                    
                    face_pil = process_crop(face_crop)
                    body_pil = process_crop(person_crop)
                    
                    with torch.no_grad():
                        g_out = gender_model(transform(body_pil).unsqueeze(0).to(DEVICE).half())
                        g_idx = torch.argmax(g_out[0]).item()
                        
                        a_out = age_model(transform(face_pil).unsqueeze(0).to(DEVICE).half())
                        a_idx = (torch.sigmoid(a_out[0]) > 0.5).sum().item()
                        
                    if track_id not in gender_memory: gender_memory[track_id] = []
                    if track_id not in age_memory: age_memory[track_id] = []
                    
                    gender_memory[track_id].append(gender_classes[g_idx])
                    age_memory[track_id].append(age_classes[a_idx])
                    
                    if len(gender_memory[track_id]) >= FRAMES_TO_WAIT:
                        final_gender = Counter(gender_memory[track_id]).most_common(1)[0][0]
                        final_age = Counter(age_memory[track_id]).most_common(1)[0][0]
                        demographics_cache[track_id] = {'gender': final_gender, 'age': final_age}
        
        ai_queue.task_done()

worker_thread = threading.Thread(target=background_ai_worker, daemon=True)
worker_thread.start()

# --- 5. DYNAMIC GUI (POLYGON BUILDER W/ PERSISTENCE) ---
image_files = sorted(glob.glob(os.path.join(FRAME_FOLDER, "*.jpg")))
first_frame = cv2.imread(image_files[0])
clone = first_frame.copy()

poly_pts = []
dragging_idx = -1
DRAG_RADIUS = 15 

if os.path.exists(POLYGON_SAVE_FILE):
    try:
        with open(POLYGON_SAVE_FILE, 'r') as f:
            saved_pts = json.load(f)
            poly_pts = [tuple(pt) for pt in saved_pts]
        print("💾 Loaded saved polygon configuration!")
    except Exception as e:
        print("⚠️ Could not load saved polygon, starting fresh.")

def draw_polygon(event, x, y, flags, param):
    global poly_pts, dragging_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        grabbed = False
        for i, pt in enumerate(poly_pts):
            if (pt[0] - x)**2 + (pt[1] - y)**2 < DRAG_RADIUS**2:
                dragging_idx = i
                grabbed = True
                break
        if not grabbed: poly_pts.append((x, y))
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_idx != -1: poly_pts[dragging_idx] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = -1
    elif event == cv2.EVENT_RBUTTONDOWN:
        poly_pts.clear()

cv2.namedWindow("Draw Your Polygon Zone")
cv2.setMouseCallback("Draw Your Polygon Zone", draw_polygon)

while True:
    disp = clone.copy()
    if len(poly_pts) > 0:
        pts_array = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
        overlay = disp.copy()
        cv2.fillPoly(overlay, [pts_array], (0, 0, 255))
        cv2.addWeighted(overlay, 0.3, disp, 0.7, 0, disp)
        cv2.polylines(disp, [pts_array], isClosed=True, color=(0, 0, 255), thickness=2)
        for i, pt in enumerate(poly_pts):
            cv2.circle(disp, pt, 5, (0, 255, 0) if i == dragging_idx else (0, 0, 255), -1)
            
    cv2.putText(disp, "ENTER: Start | RIGHT-CLICK: Erase & Redraw", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("Draw Your Polygon Zone", disp)
    if cv2.waitKey(30) & 0xFF == 13: break

cv2.destroyWindow("Draw Your Polygon Zone")

if len(poly_pts) < 3: 
    print("⚠️ Minimum 3 points required. Defaulting to safe box.")
    poly_pts = [(50, 400), (750, 400), (750, 550), (50, 550)]

with open(POLYGON_SAVE_FILE, 'w') as f:
    json.dump(poly_pts, f)
print("💾 Polygon configuration saved!")

polygon_array = np.array(poly_pts, np.int32)
poly_x, poly_top, poly_w, poly_h = cv2.boundingRect(polygon_array)
poly_bottom = poly_top + poly_h
poly_right = poly_x + poly_w


# --- 6. MAIN LOOP (THREAD 1 - THE EYES) ---
for frame_idx, img_path in enumerate(image_files):
    frame = cv2.imread(img_path)
    results = yolo_model.track(frame, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_array], (0, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [polygon_array], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.putText(frame, f"Store Footfall: {total_footfall}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 3)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            if track_id not in demographics_cache: 
                if not ai_queue.full():
                    person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    ai_queue.put((track_id, frame_idx, person_crop.copy()))

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

            is_inside_now = cv2.pointPolygonTest(polygon_array, (cx, cy), False) >= 0

            if track_id in track_history:
                was_inside_before = track_history[track_id]['in_zone']
                prev_cy = track_history[track_id]['y']
                
                standard_entry = not was_inside_before and is_inside_now
                teleported = (prev_cy < poly_top) and (cy > poly_bottom) and (poly_x < cx < poly_right)

                if standard_entry or teleported:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_footfall += 1
                        if teleported:
                            print(f"🚀 QUANTUM LEAP DETECTED! Caught fast walker ID: {track_id}")
            
            track_history[track_id] = {'in_zone': is_inside_now, 'y': cy}

            if track_id in counted_ids and track_id not in logged_ids and track_id in demographics_cache:
                demo_data = demographics_cache[track_id]
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, track_id, demo_data['gender'], demo_data['age'], "ENTERED ZONE"])
                
                logged_ids.add(track_id)
                print(f"✅ ASYNC LOGGED: [{timestamp}] ID:{track_id} | {demo_data['gender']} | {demo_data['age']} | Total: {total_footfall}")

    cv2.imshow("Unified Polygon Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

print("🛑 Feed terminated. Shutting down background workers...")
ai_queue.put(None) 
worker_thread.join()
print(f"💾 Faces harvested in /{HARVEST_DIR}. Analytics saved to {CSV_FILE}.")
cv2.destroyAllWindows()