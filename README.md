# Real-Time Age and Gender Detection with Auto Training

An end-to-end Machine Learning pipeline for real-time Age and Gender classification. Designed for enterprise domain adaptation, this architecture uses a cloud-to-local training strategy to solve the "Domain Gap" caused by specific webcam hardware and office lighting.

<br/>

## 🧠 Core Architecture Highlights
* **Temporal Smoothing (Inference):** Utilizes YOLO track-IDs and a multi-frame memory buffer to calculate the statistical mode of predictions. This completely eliminates UI flickering and stabilizes real-time webcam tracking.
* **Ordinal Regression (Age Math):** Replaces standard multi-class classification with a continuous threshold logic ($N-1$ binary questions using `BCEWithLogitsLoss`). This forces the Swin Transformer to learn the biological timeline of aging rather than treating age groups as isolated buckets.
* **Automated MLOps Pipeline:** A zero-touch orchestrator (`run-pipeline.py`) that sweeps isolated local face crops, benchmarks current models against a Golden Dataset, and dynamically fine-tunes the network to local lighting conditions.

---

<br/>

## 📂 Repository Structure
```text
/
├── README.md                          
├── run-pipeline.py                    # The Master CI/CD Orchestrator
│
├── inference/                         # Live Trackers & UI
│   ├── live-interface-hybrid-v3.py    # Screen-grab inference
│   └── webcam-interface-hybrid-v3.py  # Real-time webcam overlay
│
├── core_pipeline/                     # The Auto-Trainers
│   ├── face_janitor.py                # Sweeps and preps raw data
│   ├── auto-trainer-gender.py         # Gender fine-tuning logic
│   └── auto-trainer-age-v2.py         # Ordinal Age fine-tuning logic
│
├── cloud_training/                    # Kaggle GPU Base Builders
│   ├── age-and-gender-detection-v2.ipynb       
│   └── age-detection-v3.ipynb
│       
├── models/                            # Git LFS Tracked Weights
│   ├── swin_ordinal_age_v2.pth
│   └── swin_v2_0_gender.pth
│
└── secure_vault_v3_Temporal/          # Local Domain Data (Empty Structure)
    ├── Age/
    └── Gender/
```

<br/>

## Dataset used while training:

[Clean-Dataset-1000](https://www.kaggle.com/datasets/aryannaik225/clean-dataset-1000)

<br/>

## ⚙️ Universal Setup (Windows / macOS / Linux)
### 1. **Clone the Repository (Requires Git LFS)**

This repository uses Git Large File Storage (LFS) for the Swin Transformer `.pth` models. Ensure Git LFS is installed on your system before cloning:

- **Windows:** Download the Git LFS installer or run `winget install GitHub.GitLFS`

- **macOS:** `brew install git-lfs`

- **Linux:** `sudo apt install git-lfs`

Once installed, initialize LFS and clone:

```Bash
git lfs install
git clone https://github.com/aryannaik225/Age-and-Gender-Prediction
```
(The heavy `.pth` model files will automatically pull directly into the repository structure).

<br/>

### 2. **Create a Virtual Environment**

To avoid OS-level package conflicts, create and activate a Python virtual environment:

- **Windows:**

```DOS
python -m venv venv
venv\Scripts\activate
```

- **macOS & Linux:**

```Bash
python3 -m venv venv
source venv/bin/activate
```

<br/>

### 3. **Install Dependencies**

With the virtual environment active, install the required packages:

```Bash
pip install torch torchvision torchaudio ultralytics opencv-python pillow mss lapx
```

*Note: The `lapx` package handles YOLO's Temporal Smoothing matrix math. If it fails to install, ensure your OS has basic C++ build tools installed (Visual Studio Build Tools for Windows, Xcode CLI for Mac, or `build-essential` for Linux).*

---

## 🚀 Usage Guide
### **Phase 1: Real-Time Inference (The Demo)**

To run the live webcam overlay with Temporal Smoothing active:

```Bash
python interface/webcam-interface-hybrid-v3.py
```
Press `Q` to close the webcam feed.

<br/>

### **Phase 2: Domain Adaptation (The Pipeline)**

When new faces are captured and placed in the staging directories, run the Master Orchestrator to automatically clean the data, evaluate the Golden Score, and fine-tune the models:

```Bash
python run-pipeline.py
```
The script will safely abort training if the new data degrades the model's performance on the Golden Exam.
