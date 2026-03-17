import torch
import torch.nn as nn
from torchvision import models
import os

# --- CONFIG ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GENDER_MODEL_PATH = os.path.join(SCRIPT_DIR, "swin_v2_0_gender.pth")
AGE_MODEL_PATH = os.path.join(SCRIPT_DIR, "swin_ordinal_age_v2.pth")

GENDER_ONNX_OUT = os.path.join(SCRIPT_DIR, "swin_gender_v2_0.onnx")
AGE_ONNX_OUT = os.path.join(SCRIPT_DIR, "swin_ordinal_age_v2.onnx")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Initializing ONNX Exporter...")

# 1. Prepare the Dummy Input (A standard 224x224 RGB face crop)
# Using batch size 1 for the trace
dummy_input = torch.randn(1, 3, 224, 224, device=DEVICE)

# --- EXPORT GENDER MODEL ---
print(f"📦 Loading Gender Model from {GENDER_MODEL_PATH}...")
gender_model = models.swin_t(weights=None)
gender_model.head = nn.Linear(gender_model.head.in_features, 2)
gender_model.load_state_dict(torch.load(GENDER_MODEL_PATH, map_location=DEVICE, weights_only=True))
gender_model.to(DEVICE)
gender_model.eval()

print("⚙️ Tracing and Exporting Gender ONNX...")
torch.onnx.export(
    gender_model,               
    dummy_input,                
    GENDER_ONNX_OUT,            
    export_params=True,         
    opset_version=14,           # Opset 14 is highly stable for TensorRT
    do_constant_folding=True,   # Optimizes the graph
    input_names=['input_face'], 
    output_names=['gender_logits'], 
    dynamic_axes={'input_face': {0: 'batch_size'}, 'gender_logits': {0: 'batch_size'}} # Allows batch processing
)
print(f"✅ Gender ONNX saved to {GENDER_ONNX_OUT}")

# --- EXPORT AGE MODEL ---
print(f"\n📦 Loading Age Model from {AGE_MODEL_PATH}...")
age_model = models.swin_t(weights=None)
# Ordinal math requires 4 outputs
age_model.head = nn.Linear(age_model.head.in_features, 4)
age_model.load_state_dict(torch.load(AGE_MODEL_PATH, map_location=DEVICE, weights_only=True))
age_model.to(DEVICE)
age_model.eval()

print("⚙️ Tracing and Exporting Age ONNX...")
torch.onnx.export(
    age_model,               
    dummy_input,                
    AGE_ONNX_OUT,            
    export_params=True,         
    opset_version=14,           
    do_constant_folding=True,   
    input_names=['input_face'], 
    output_names=['ordinal_age_logits'], 
    dynamic_axes={'input_face': {0: 'batch_size'}, 'ordinal_age_logits': {0: 'batch_size'}}
)
print(f"✅ Age ONNX saved to {AGE_ONNX_OUT}")

print("\n🎉 ONNX Export Complete! Ready for TensorRT compilation.")