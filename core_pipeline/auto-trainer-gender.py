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
LIVE_DATA_DIR = "secure_vault_v3_Temporal/Gender"
PROCESSED_LIVE_DIR = "temp_gender_live_processed"

GOLDEN_TEST_DIR = "golden_datasets/gender"
PROCESSED_GOLDEN_DIR = "temp_gender_golden_processed"
 
ARCHIVE_DIR = "archive_gender_dataset"         
CURRENT_MODEL = "models/swin_v2_0_gender.pth"
NEW_CANDIDATE = "models/swin_v2_1_gender.pth"       

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # Optimized for GPU
EPOCHS = 2     
LEARNING_RATE = 5e-6 # Micro-dosing weights

# --- 1. V2.0 PREPROCESSOR ---
def enhance_v2_0_prep(input_path, output_path):
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
        
        padded_img = cv2.copyMakeBorder(
            blurred_img, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

        cv2.imwrite(output_path, padded_img)
    except Exception as e:
        pass

def process_folder(src_dir, dest_dir, desc):
    if not os.path.exists(src_dir): return False
    
    if desc == "Golden" and os.path.exists(dest_dir):
        return True

    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)

    for gender in ["Male", "Female"]:
        in_folder = os.path.join(src_dir, gender)
        out_folder = os.path.join(dest_dir, gender)
        os.makedirs(out_folder, exist_ok=True)
        
        if os.path.exists(in_folder) and os.listdir(in_folder):
            for f in tqdm(os.listdir(in_folder), desc=f"Processing {desc} {gender}", leave=False):
                enhance_v2_0_prep(os.path.join(in_folder, f), os.path.join(out_folder, f))
    return True

# --- 2. CUSTOM OPENCV LOADER ---
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
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
    return correct.double() / len(loader.dataset)

def archive_images():
    timestamp = int(time.time())
    for gender in ["Male", "Female"]:
        src_path = os.path.join(LIVE_DATA_DIR, gender)
        dst_path = os.path.join(ARCHIVE_DIR, str(timestamp), gender)
        if os.path.exists(src_path) and os.listdir(src_path):
            os.makedirs(dst_path, exist_ok=True)
            for f in os.listdir(src_path):
                shutil.move(os.path.join(src_path, f), os.path.join(dst_path, f))
    print(f"📦 Raw Live images archived to {ARCHIVE_DIR}/{timestamp}")
    
    if os.path.exists(PROCESSED_LIVE_DIR):
        shutil.rmtree(PROCESSED_LIVE_DIR)

def start_retraining():
    print(f"🕵️ Shadow Trainer active on {DEVICE.upper()}...")

    m_path = os.path.join(LIVE_DATA_DIR, "Male")
    f_path = os.path.join(LIVE_DATA_DIR, "Female")
    if not os.path.exists(m_path) or not os.path.exists(f_path):
        print("💤 Folders missing. Going to sleep.")
        return
    
    m_count = len(os.listdir(m_path))
    f_count = len(os.listdir(f_path))
    if m_count < 5 or f_count < 5:
        print(f"💤 Only {m_count}M/{f_count}F found. Need more to train.")
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
    model_blue.head = nn.Linear(model_blue.head.in_features, 2)
    if os.path.exists(CURRENT_MODEL):
        model_blue.load_state_dict(torch.load(CURRENT_MODEL, map_location=DEVICE, weights_only=True))
    model_blue.to(DEVICE)
    
    old_acc = evaluate_model(model_blue, golden_loader)
    print(f"📍 Current Golden Score (V2.0): {old_acc:.4f}")

    print(f"🏋️ Fine-tuning on {len(train_dataset)} Hard Negatives & New Data...")
    model_green = copy.deepcopy(model_blue)
    
    # --- THE DYNAMIC AUTO-BALANCER ---
    total_samples = m_count + f_count
    weight_female = total_samples / (2.0 * f_count)
    weight_male = total_samples / (2.0 * m_count)
    
    class_weights = torch.tensor([weight_female, weight_male], dtype=torch.float).to(DEVICE)
    print(f"⚖️ Applied Dynamic Weights -> Female: {weight_female:.2f}x penalty | Male: {weight_male:.2f}x penalty")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model_green.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model_green.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model_green(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
            torch.cuda.empty_cache() 

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