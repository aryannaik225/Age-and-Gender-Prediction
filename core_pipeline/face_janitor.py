import os
import shutil
from tqdm import tqdm

# --- CONFIG ---
VAULT_DIR = "secure_vault_v2_9_Premium/Age"
STAGING_DIR = "age_detection/age_face_staging"
AGE_CLASSES = ['0-12', '13-19', '20-35', '36-55', '56+']

print("🧹 Firing up the Face Janitor...")

# 1. Setup the clean staging area
for age in AGE_CLASSES:
    os.makedirs(os.path.join(STAGING_DIR, age), exist_ok=True)

moved_count = 0

# 2. Hunt and Extract
if os.path.exists(VAULT_DIR):
    for age_folder in AGE_CLASSES:
        source_age_path = os.path.join(VAULT_DIR, age_folder)
        
        if not os.path.exists(source_age_path): 
            continue
            
        # Scan all files in the specific age folder
        files = os.listdir(source_age_path)
        for filename in tqdm(files, desc=f"Sweeping {age_folder}", leave=False):
            
            # Identify ONLY the tight face crops
            if filename.startswith("FACE_") and filename.endswith(".jpg"):
                src_file = os.path.join(source_age_path, filename)
                dst_file = os.path.join(STAGING_DIR, age_folder, filename)
                
                # THE SDE RULE: Move, DO NOT Copy
                shutil.move(src_file, dst_file)
                moved_count += 1

print(f"\n✨ Janitor finished! Physically moved {moved_count} FACE crops to '{STAGING_DIR}'.")
print("Your main vault is now clean, and your training data is prepped!")