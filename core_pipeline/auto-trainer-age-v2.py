import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import copy
import shutil
import time
import cv2
from PIL import Image

# --- CONFIG ---
# Hardcoded paths relative to the root Age-and-Gender-Prediction folder
LIVE_DATA_DIR = "age_face_staging"   
PROCESSED_LIVE_DIR = "temp_age_live_processed"

GOLDEN_TEST_DIR = "golden_datasets/age"     
PROCESSED_GOLDEN_DIR = "temp_age_golden_processed"
 
ARCHIVE_DIR = "archive_age_dataset"         

# The Kaggle Brain is now your base!
CURRENT_MODEL = "models/swin_ordinal_age_v2.pth"  
NEW_CANDIDATE = "models/swin_v2_1_ordinal_age.pth"       

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 
EPOCHS = 15    
LEARNING_RATE = 1e-5 

AGE_CLASSES = ['0-12', '13-19', '20-35', '36-55', '56+']

# --- 1. SOTA PREPROCESSOR ---
def enhance_age_prep(input_path, output_path):
    try:
        img = cv2.imread(input_path)
        if img is None: return
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l_channel)
        merged = cv2.merge((cl, a, b))
        contrast_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        bright_img = cv2.convertScaleAbs(contrast_img, alpha=1.1, beta=15)
        blurred_img = cv2.GaussianBlur(bright_img, (3, 3), 0)
        h, w = blurred_img.shape[:2]
        longest_edge = max(h, w)
        top = (longest_edge - h) // 2
        bottom = longest_edge - h - top
        left = (longest_edge - w) // 2
        right = longest_edge - w - left
        padded_img = cv2.copyMakeBorder(blurred_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv2.imwrite(output_path, padded_img)
    except Exception as e:
        pass

def process_folder(src_dir, dest_dir, desc):
    if not os.path.exists(src_dir): return False
    if desc == "Golden" and os.path.exists(dest_dir): return True
    if os.path.exists(dest_dir): shutil.rmtree(dest_dir)

    for age_group in AGE_CLASSES:
        in_folder = os.path.join(src_dir, age_group)
        out_folder = os.path.join(dest_dir, age_group)
        os.makedirs(out_folder, exist_ok=True)
        
        if os.path.exists(in_folder) and os.listdir(in_folder):
            for f in tqdm(os.listdir(in_folder), desc=f"Processing {desc} {age_group}", leave=False):
                enhance_age_prep(os.path.join(in_folder, f), os.path.join(out_folder, f))
    return True

# --- 2. LOADER & EVAL ---
def custom_opencv_loader(path):
    img = cv2.imread(path)
    if img is None: return Image.new('RGB', (224, 224), (0, 0, 0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Taking Golden Exam", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            
            # ORDINAL DECODE: Sum up the "Yes" answers to get the predicted bucket (0 to 4)
            preds = (torch.sigmoid(outputs) > 0.5).sum(dim=1)
            correct += torch.sum(preds == labels.data)
    return correct.double() / len(loader.dataset)

def archive_images():
    timestamp = int(time.time())
    for age_group in AGE_CLASSES:
        src_path = os.path.join(LIVE_DATA_DIR, age_group)
        dst_path = os.path.join(ARCHIVE_DIR, str(timestamp), age_group)
        if os.path.exists(src_path) and os.listdir(src_path):
            os.makedirs(dst_path, exist_ok=True)
            for f in os.listdir(src_path):
                shutil.move(os.path.join(src_path, f), os.path.join(dst_path, f))
    print(f"📦 Raw Live images archived to {ARCHIVE_DIR}/{timestamp}")
    if os.path.exists(PROCESSED_LIVE_DIR): shutil.rmtree(PROCESSED_LIVE_DIR)

def start_retraining():
    print(f"🕵️ Age Shadow Trainer active on {DEVICE.upper()}...")

    if not os.path.exists(LIVE_DATA_DIR):
        print("💤 Live dataset folder missing. Going to sleep.")
        return
    
    # --- ANCHOR INJECTION FIX ---
    for age_group in AGE_CLASSES:
        live_path = os.path.join(LIVE_DATA_DIR, age_group)
        os.makedirs(live_path, exist_ok=True)
        if len(os.listdir(live_path)) == 0:
            golden_path = os.path.join(GOLDEN_TEST_DIR, age_group)
            if os.path.exists(golden_path) and len(os.listdir(golden_path)) > 0:
                safe_img = os.listdir(golden_path)[0]
                shutil.copy(os.path.join(golden_path, safe_img), os.path.join(live_path, safe_img))
    
    class_counts = []
    for age_group in AGE_CLASSES:
        path = os.path.join(LIVE_DATA_DIR, age_group)
        count = len(os.listdir(path)) if os.path.exists(path) else 0
        class_counts.append(count)
    
    total_samples = sum(class_counts)
    if total_samples < 10:
        print(f"💤 Only {total_samples} total images found. Need more to train.")
        return

    print("✨ Preparing Datasets...")
    process_folder(GOLDEN_TEST_DIR, PROCESSED_GOLDEN_DIR, "Golden")
    process_folder(LIVE_DATA_DIR, PROCESSED_LIVE_DIR, "Live")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    golden_dataset = datasets.ImageFolder(PROCESSED_GOLDEN_DIR, transform=transform, loader=custom_opencv_loader)
    golden_loader = DataLoader(golden_dataset, batch_size=BATCH_SIZE, shuffle=False)

    train_dataset = datasets.ImageFolder(PROCESSED_LIVE_DIR, transform=transform, loader=custom_opencv_loader)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model_blue = models.swin_t(weights=None)
    # ORDINAL FIX: 4 Output Thresholds
    model_blue.head = nn.Linear(model_blue.head.in_features, 4)
    if os.path.exists(CURRENT_MODEL):
        model_blue.load_state_dict(torch.load(CURRENT_MODEL, map_location=DEVICE, weights_only=True))
    model_blue.to(DEVICE)
    
    old_acc = evaluate_model(model_blue, golden_loader)
    print(f"📍 Current Golden Score (Kaggle Base): {old_acc:.4f}")

    print(f"🏋️ Fine-tuning on {len(train_dataset)} Hard Negatives & New Data...")
    model_green = copy.deepcopy(model_blue)
    
    # ORDINAL FIX: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model_green.parameters(), lr=LEARNING_RATE)
    
    levels = torch.arange(4).to(DEVICE)

    for epoch in range(EPOCHS):
        model_green.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Map standard label (e.g., 2) to Ordinal Tensor [1, 1, 0, 0]
            ordinal_labels = (labels.unsqueeze(1) > levels).float()
            
            optimizer.zero_grad()
            outputs = model_green(inputs)
            loss = criterion(outputs, ordinal_labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    new_acc = evaluate_model(model_green, golden_loader)
    print(f"📍 Candidate Golden Score (V2.1): {new_acc:.4f}")

    if new_acc > old_acc: 
        print("🏆 SUCCESS: Model learned from its mistakes! Publishing V2.1...")
        torch.save(model_green.state_dict(), NEW_CANDIDATE)
    else:
        print("❌ REJECTED: Candidate did not beat the Golden Exam.")
    
    archive_images()

if __name__ == "__main__":
    start_retraining()