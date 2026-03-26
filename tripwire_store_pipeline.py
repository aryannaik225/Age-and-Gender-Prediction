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
import shutil
import time
import chromadb

# --- 1. CONFIG & PATHS ---
VIDEO_PATH = "CCTV_Cameras_retail_store_720p.mp4"
CSV_FILE = "footfall_analytics_store.csv"
POLYGON_SAVE_FILE = "polygon_config_store.json"
HARVEST_DIR = "harvested_images_store"
DB_DIR = "./chroma_reid_db"
os.makedirs(HARVEST_DIR, exist_ok=True) 

RECORD_PEOPLE_INSIDE = True 

GENDER_MODEL_FILE = "models/swin_v2_0_gender.pth"
AGE_MODEL_FILE = "models/swin_ordinal_age_v2.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Initializing Asynchronous Sentinel on {DEVICE.upper()}...")
print("🗄️ Booting up ChromaDB Vector Storage...")
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="store_visitors",
    metadata={"hnsw:space": "cosine"} 
)

next_global_id = 1
existing_data = collection.get(include=["metadatas"])
if existing_data and existing_data["metadatas"]:
    max_id = max([meta["global_id"] for meta in existing_data["metadatas"]])
    next_global_id = max_id + 1
    print(f"💾 Past session found! Loaded {len(existing_data['metadatas'])} identities.")
    print(f"🔢 Starting new tracking at Global ID: {next_global_id}")

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

REID_SIMILARITY_THRESHOLD  = 0.82
REID_EMBED_UPDATE_INTERVAL = 30   
REID_MAX_GALLERY_SIZE      = 30    

print("🧠 Loading ReID Extractor...")
reid_backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
reid_backbone.fc = nn.Identity()   
reid_backbone.to(DEVICE).eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

reid_transform = transforms.Compose([
    transforms.Resize((256, 128)),          
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

gender_classes = ['Female', 'Male']
age_classes = ['0-12', '13-19', '20-35', '36-55', '56+']

# --- 3. STATE MEMORY & CSV UPDATER ---
reid_gallery: dict[int, list[np.ndarray]] = {}
yolo_to_canonical: dict[int, int] = {}
reid_lock = threading.Lock()  

def _extract_embedding(crop_bgr: np.ndarray) -> np.ndarray | None:
    if crop_bgr is None or crop_bgr.size == 0: return None
    h, w = crop_bgr.shape[:2]
    if h < 32 or w < 16: return None
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    tensor = reid_transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = reid_backbone(tensor).squeeze(0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm < 1e-6: return None
    return feat / norm

def _best_gallery_match(emb: np.ndarray) -> tuple[int | None, float]:
    best_id, best_sim = None, 0.0
    for cid, emb_list in reid_gallery.items():
        sims = [float(np.dot(emb, e)) for e in emb_list]
        top_sim = max(sims)
        if top_sim > best_sim:
            best_sim = top_sim
            best_id = cid
    return best_id, best_sim

def resolve_track_id(yolo_id: int, crop_bgr: np.ndarray) -> int:
    global next_global_id
    with reid_lock:
        if yolo_id in yolo_to_canonical:
            return yolo_to_canonical[yolo_id]

    emb = _extract_embedding(crop_bgr)

    with reid_lock:
        if emb is not None:
            # 1. Check RAM Gallery
            matched_id, sim = _best_gallery_match(emb)
            if matched_id is not None and sim >= REID_SIMILARITY_THRESHOLD:
                yolo_to_canonical[yolo_id] = matched_id
                print(f"🔁 RE-ID [RAM]: YOLO #{yolo_id} → canonical #{matched_id} (cosine={sim:.3f})")
                if len(reid_gallery[matched_id]) >= REID_MAX_GALLERY_SIZE:
                    reid_gallery[matched_id].pop(1)
                reid_gallery[matched_id].append(emb)
                return matched_id
            
            # 🗄️ 2. Check ChromaDB (Long-Term Memory)
            db_results = collection.query(
                query_embeddings=[emb.tolist()],
                n_results=1
            )
            if db_results['distances'] and len(db_results['distances'][0]) > 0:
                db_distance = db_results['distances'][0][0]
                db_similarity = 1.0 - db_distance 
                
                if db_similarity >= REID_SIMILARITY_THRESHOLD:
                    matched_meta = db_results['metadatas'][0][0]
                    matched_db_id = matched_meta['global_id']
                    
                    print(f"🗄️ RE-ID [DB]: YOLO #{yolo_id} → canonical #{matched_db_id} (cosine={db_similarity:.3f})")
                    yolo_to_canonical[yolo_id] = matched_db_id
                    reid_gallery[matched_db_id] = [emb]
                    
                    if matched_meta.get('gender') and matched_meta.get('age'):
                        demographics_cache[matched_db_id] = {
                            'gender': matched_meta['gender'],
                            'age': matched_meta['age'],
                            'gender_locked': True,
                            'age_locked': True
                        }
                        print(f"⚡ INSTANT PROFILE: Restored {matched_meta['gender']}, {matched_meta['age']} for ID {matched_db_id}")
                    
                    return matched_db_id

        # 3. Truly New Person
        new_id = next_global_id
        next_global_id += 1
        yolo_to_canonical[yolo_id] = new_id
        if emb is not None:
            reid_gallery[new_id] = [emb]
        return new_id

def update_gallery_embedding(canonical_id: int, crop_bgr: np.ndarray) -> None:
    emb = _extract_embedding(crop_bgr)
    if emb is None: return
    with reid_lock:
        if canonical_id not in reid_gallery: reid_gallery[canonical_id] = []
        if len(reid_gallery[canonical_id]) >= REID_MAX_GALLERY_SIZE:
            reid_gallery[canonical_id].pop(1)   
        reid_gallery[canonical_id].append(emb)

last_logged_event = {} 
people_inside_logged = set() 
track_history = {} 
gender_memory = {}
age_memory = {}
demographics_cache = {} 
total_footfall = 0
FRAMES_TO_WAIT = 5
TRIPWIRE_COOLDOWN_SECS = 2.0
last_event_time: dict[int, float] = {}  

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Track_ID", "Gender", "Age", "Event"])

def update_csv_demographics(target_id, new_gender, new_age):
    if not os.path.exists(CSV_FILE): return
    temp_rows = []
    with open(CSV_FILE, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader, None)
        if headers: temp_rows.append(headers)
        for row in reader:
            if len(row) >= 4 and row[1] == str(target_id):
                row[2] = new_gender
                row[3] = new_age
            temp_rows.append(row)
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(temp_rows)

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

# --- 4. THE DECOUPLED BACKGROUND WORKER (THREAD 2) ---
ai_queue = queue.Queue(maxsize=100)

def background_ai_worker():
    global demographics_cache
    harvested_gender_ids = set()
    harvested_age_ids = set()
    
    print("⚙️ Decoupled Background AI Worker is running...")
    while True:
        task = ai_queue.get()
        if task is None: break 
        
        track_id, frame_idx, person_crop = task
        
        if track_id not in demographics_cache:
            demographics_cache[track_id] = {'gender': '', 'age': '', 'gender_locked': False, 'age_locked': False}

        # 🧬 PATH 1: GENDER CALCULATION
        if not demographics_cache[track_id]['gender_locked']:
            body_pil = process_crop(person_crop)
            with torch.no_grad():
                g_out = gender_model(transform(body_pil).unsqueeze(0).to(DEVICE).half())
                g_idx = torch.argmax(g_out[0]).item()
            
            if track_id not in gender_memory: gender_memory[track_id] = []
            gender_memory[track_id].append(gender_classes[g_idx])
            
            if len(gender_memory[track_id]) == FRAMES_TO_WAIT:
                final_gender = Counter(gender_memory[track_id]).most_common(1)[0][0]
                demographics_cache[track_id]['gender'] = final_gender
                demographics_cache[track_id]['gender_locked'] = True
                print(f"🧬 GENDER LOCKED: ID {track_id} is {final_gender}")
                
                if track_id in last_logged_event or track_id in people_inside_logged:
                    update_csv_demographics(track_id, final_gender, demographics_cache[track_id]['age'])
                
                if track_id not in harvested_gender_ids:
                    gender_dir = os.path.join(HARVEST_DIR, "Gender", final_gender)
                    os.makedirs(gender_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(gender_dir, f"id_{track_id}_body.jpg"), person_crop)
                    harvested_gender_ids.add(track_id)

        # 🎂 PATH 2: AGE CALCULATION
        if not demographics_cache[track_id]['age_locked']:
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
                        face_pil = process_crop(face_crop)
                        with torch.no_grad():
                            a_out = age_model(transform(face_pil).unsqueeze(0).to(DEVICE).half())
                            a_idx = (torch.sigmoid(a_out[0]) > 0.5).sum().item()
                            
                        if track_id not in age_memory: age_memory[track_id] = []
                        age_memory[track_id].append(age_classes[a_idx])
                        
                        if len(age_memory[track_id]) == FRAMES_TO_WAIT:
                            final_age = Counter(age_memory[track_id]).most_common(1)[0][0]
                            demographics_cache[track_id]['age'] = final_age
                            demographics_cache[track_id]['age_locked'] = True
                            print(f"🎂 AGE LOCKED: ID {track_id} is {final_age}")
                            
                            if track_id in last_logged_event or track_id in people_inside_logged:
                                update_csv_demographics(track_id, demographics_cache[track_id]['gender'], final_age)
                            
                            if track_id not in harvested_age_ids:
                                age_dir = os.path.join(HARVEST_DIR, "Age", final_age)
                                os.makedirs(age_dir, exist_ok=True)
                                cv2.imwrite(os.path.join(age_dir, f"id_{track_id}_face.jpg"), face_crop)
                                cv2.imwrite(os.path.join(age_dir, f"id_{track_id}_body.jpg"), person_crop)
                                harvested_age_ids.add(track_id)
                                
        if demographics_cache[track_id]['gender_locked'] and demographics_cache[track_id]['age_locked']:
            with reid_lock:
                if track_id in reid_gallery and len(reid_gallery[track_id]) > 0:
                    embeds_to_save = []
                    metas_to_save = []
                    ids_to_save = []
                    
                    for i, embed in enumerate(reid_gallery[track_id]):
                        embeds_to_save.append(embed.tolist())
                        metas_to_save.append({
                            "global_id": track_id, 
                            "gender": demographics_cache[track_id]['gender'], 
                            "age": demographics_cache[track_id]['age']
                        })
                        ids_to_save.append(f"id_{track_id}_pose_{i}")
                        
                    collection.upsert(
                        embeddings=embeds_to_save,
                        metadatas=metas_to_save,
                        ids=ids_to_save
                    )
        
        ai_queue.task_done()

worker_thread = threading.Thread(target=background_ai_worker, daemon=True)
worker_thread.start()

# --- 5. DYNAMIC GUI (POLYGON BUILDER W/ ANCHOR) ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ ERROR: Could not open video file {VIDEO_PATH}. Check the path!")
    exit()

ret, first_frame = cap.read()
if not ret:
    print("❌ ERROR: Could not read the first frame of the video.")
    exit()
    
clone = first_frame.copy()

poly_pts = []
store_anchor = None
dragging_idx = -1
DRAG_RADIUS = 15 
ui_phase = "POLYGON"

if os.path.exists(POLYGON_SAVE_FILE):
    try:
        with open(POLYGON_SAVE_FILE, 'r') as f:
            data = json.load(f)
            poly_pts = [tuple(pt) for pt in data['polygon']]
            store_anchor = tuple(data['anchor'])
        print("💾 Loaded saved polygon & anchor!")
        ui_phase = "DONE"
    except Exception as e:
        print("⚠️ Old polygon format detected. Erasing and starting fresh.")
        poly_pts = []
        store_anchor = None

def draw_gui(event, x, y, flags, param):
    global poly_pts, dragging_idx, ui_phase, store_anchor
    if ui_phase == "POLYGON":
        if event == cv2.EVENT_LBUTTONDOWN:
            grabbed = False
            for i, pt in enumerate(poly_pts):
                if (pt[0] - x)**2 + (pt[1] - y)**2 < DRAG_RADIUS**2:
                    dragging_idx = i; grabbed = True; break
            if not grabbed: poly_pts.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE and dragging_idx != -1: poly_pts[dragging_idx] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP: dragging_idx = -1
        elif event == cv2.EVENT_RBUTTONDOWN: poly_pts.clear()
    elif ui_phase == "ANCHOR" and event == cv2.EVENT_LBUTTONDOWN:
        store_anchor = (x, y)

cv2.namedWindow("Setup Zone", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Setup Zone", 1280, 720)
cv2.setMouseCallback("Setup Zone", draw_gui)

while ui_phase != "DONE":
    disp = clone.copy()
    if len(poly_pts) > 0:
        pts_array = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(disp, [pts_array], isClosed=True, color=(0, 0, 255), thickness=2)
        for i, pt in enumerate(poly_pts):
            cv2.circle(disp, pt, 5, (0, 255, 0) if i == dragging_idx else (0, 0, 255), -1)
    if store_anchor:
        cv2.circle(disp, store_anchor, 8, (0, 255, 0), -1)
        cv2.putText(disp, "INSIDE STORE", (store_anchor[0] + 10, store_anchor[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    if ui_phase == "POLYGON":
        cv2.putText(disp, "1. Draw Polygon -> Press ENTER", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    elif ui_phase == "ANCHOR":
        cv2.putText(disp, "2. Click DEEP INSIDE the store -> Press ENTER", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Setup Zone", disp)
    if cv2.waitKey(30) & 0xFF == 13: 
        if ui_phase == "POLYGON" and len(poly_pts) >= 3: ui_phase = "ANCHOR"
        elif ui_phase == "ANCHOR" and store_anchor is not None: ui_phase = "DONE"

cv2.destroyWindow("Setup Zone")
with open(POLYGON_SAVE_FILE, 'w') as f:
    json.dump({'polygon': poly_pts, 'anchor': store_anchor}, f)

polygon_array = np.array(poly_pts, np.int32)
poly_x, poly_top, poly_w, poly_h = cv2.boundingRect(polygon_array)
poly_bottom = poly_top + poly_h
poly_right = poly_x + poly_w

poly_cx = int(np.mean(polygon_array[:, 0]))
poly_cy = int(np.mean(polygon_array[:, 1]))
inward_vec = np.array([store_anchor[0] - poly_cx, store_anchor[1] - poly_cy])

# --- 6. MAIN LOOP (THREAD 1 - THE EYES) ---
frame_idx = 0
cv2.namedWindow("Unified Polygon Pipeline", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Unified Polygon Pipeline", 1280, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame_idx += 1
    
    results = yolo_model.track(frame, classes=[0], persist=True, tracker="botsort.yaml", verbose=False)
    
    overlay = frame.copy()
    cv2.fillPoly(overlay, [polygon_array], (0, 0, 255))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [polygon_array], isClosed=True, color=(0, 0, 255), thickness=2)
    
    cv2.circle(frame, store_anchor, 6, (0, 255, 0), -1)
    cv2.putText(frame, f"Store Footfall: {total_footfall}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 3)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        for box, yolo_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            person_crop_reid = frame[max(0, y1):y2, max(0, x1):x2].copy()
            track_id = resolve_track_id(yolo_id, person_crop_reid)

            if frame_idx % REID_EMBED_UPDATE_INTERVAL == 0:
                update_gallery_embedding(track_id, person_crop_reid)

            cache = demographics_cache.get(track_id, {'gender_locked': False, 'age_locked': False})
            if not (cache.get('gender_locked') and cache.get('age_locked')): 
                if not ai_queue.full():
                    person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                    ai_queue.put((track_id, frame_idx, person_crop.copy()))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            
            if track_id in demographics_cache:
                demo = demographics_cache[track_id]
                disp_g = demo['gender'] if demo['gender'] != "" else "..."
                disp_a = demo['age'] if demo['age'] != "" else "..."
                label = f"ID:{track_id} | G:{disp_g} | A:{disp_a}"
                color = (0, 255, 0) if demo['gender_locked'] and demo['age_locked'] else (0, 165, 255)
            else:
                label = f"ID:{track_id} | Scanning..."
                color = (0, 165, 255)
                
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            is_inside_now = cv2.pointPolygonTest(polygon_array, (cx, cy), False) >= 0

            if track_id in track_history:
                was_inside_before = track_history[track_id]['in_zone']
                prev_cx, prev_cy = track_history[track_id]['x'], track_history[track_id]['y']
                
                crossed_boundary = (was_inside_before != is_inside_now)
                
                prev_person_vec = np.array([prev_cx - poly_cx, prev_cy - poly_cy])
                curr_person_vec = np.array([cx - poly_cx, cy - poly_cy])
                prev_inside_line = np.dot(prev_person_vec, inward_vec) > 0
                curr_inside_line = np.dot(curr_person_vec, inward_vec) > 0
                
                teleported = False
                if prev_inside_line != curr_inside_line and not is_inside_now and not was_inside_before:
                    if (poly_x - 50) < cx < (poly_right + 50) and (poly_top - 50) < cy < (poly_bottom + 50):
                        teleported = True

                now = time.monotonic()
                elapsed = now - last_event_time.get(track_id, 0.0)
                cooldown_clear = elapsed >= TRIPWIRE_COOLDOWN_SECS

                if (crossed_boundary or teleported) and cooldown_clear:
                    move_vec = np.array([cx - prev_cx, cy - prev_cy])
                    dot_product = np.dot(move_vec, inward_vec)
                    event_type = "ENTERED ZONE" if dot_product > 0 else "EXITED ZONE"
                    
                    if last_logged_event.get(track_id) != event_type:
                        if event_type == "ENTERED ZONE": total_footfall += 1
                        
                        demo_data = demographics_cache.get(track_id, {'gender': '', 'age': ''})
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        with open(CSV_FILE, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, track_id, demo_data['gender'], demo_data['age'], event_type])
                        
                        last_logged_event[track_id] = event_type
                        last_event_time[track_id] = now
                        print(f"✅ INSTANT LOG: [{timestamp}] ID:{track_id} | {event_type} | Total: {total_footfall}")

            track_history[track_id] = {'in_zone': is_inside_now, 'x': cx, 'y': cy}

            if RECORD_PEOPLE_INSIDE and track_id not in last_logged_event:
                if not is_inside_now:
                    person_vec = np.array([cx - poly_cx, cy - poly_cy])
                    is_physically_inside = np.dot(person_vec, inward_vec) > 0
                    
                    if is_physically_inside:
                        demo_data = demographics_cache.get(track_id, {'gender': '', 'age': ''})
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        with open(CSV_FILE, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([timestamp, track_id, demo_data['gender'], demo_data['age'], "ALREADY IN STORE"])
                        
                        last_logged_event[track_id] = "ALREADY IN STORE"
                        print(f"👀 WARM STATE LOG: [{timestamp}] ID:{track_id} | ALREADY IN STORE")

    cv2.imshow("Unified Polygon Pipeline", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

print("🛑 Feed terminated. Shutting down background workers...")
ai_queue.put(None) 
worker_thread.join()
cap.release()
cv2.destroyAllWindows()
print(f"💾 Faces harvested in /{HARVEST_DIR}. Analytics saved to {CSV_FILE}.")