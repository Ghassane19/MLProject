# PACTE ML — Classification d'Objets Routiers pour les Véhicules Autonomes

## 1. Clear Guide for the Dataset (What Your Data Should Look Like)

### A. What should a single image look like?
Because your academic specification requires training **Classical Machine Learning models (SVM, Random Forest, k-NN)** after feature extraction (HOG + Color Histograms), your dataset images inside `train/`, `validation/`, and `test/` must be **single-object cropped images** (one main road object per photo):
* **`car/` (`voiture`)**: Photos where a car is the main subject.
* **`pedestrian/` (`pieton`)**: Photos of a person walking or standing near/on a road.
* **`cyclist/` (`cycliste`)**: Photos of a bicycle or a person riding a bicycle.
* **`motorcycle/` (`moto`)**: Photos of a motorcycle, scooter, or rider on a motorbike.
* **`bus/` (`bus`)**: Photos of a city bus, coach, or school bus.

> **What about images with MULTIPLE objects (full street scenes)?**
> Keep 10 to 20 full street scene photos in a separate folder (`./demo_scenes/`) to test the **Multi-Object Detection mode** in your React web interface during your live presentation. Do **not** mix full street scenes into the `train/` folder of your classical SVM/Random Forest models.

---

### B. How many images do you need per class and per folder?

For classical Machine Learning (`scikit-learn`), **more is not always better**. If you put 50,000 images, SVM will take hours to train and freeze your laptop. The sweet spot is **500 to 1,000 images per class** (2,500 to 5,000 images total).

Here is the **ideal target distribution** (using an **70% Train / 15% Validation / 15% Test** split):

| Class (Folder Name) | `train/` (70%) | `validation/` (15%) | `test/` (15%) | **Total per Class** |
| :--- | :---: | :---: | :---: | :---: |
| **`car`** (Voiture) | 700 images | 150 images | 150 images | **1,000 images** |
| **`pedestrian`** (Piéton) | 700 images | 150 images | 150 images | **1,000 images** |
| **`cyclist`** (Cycliste) | 700 images | 150 images | 150 images | **1,000 images** |
| **`motorcycle`** (Moto) | 700 images | 150 images | 150 images | **1,000 images** |
| **`bus`** (Bus) | 700 images | 150 images | 150 images | **1,000 images** |
| **TOTAL** | **3,500 images** | **750 images** | **750 images** | **5,000 images** |

*(Minimum acceptable size if you have a small dataset: **200 Train / 50 Validation / 50 Test** per class = 1,500 images total).*

> **Note if you only have `train/` and `test/` right now:**
> If you don't have a `validation/` folder on your disk, aim for **800 images per class in `train/`** and **200 images per class in `test/`**. The Python script will automatically split 15% of `train/` into `validation/` during execution.

---

### C. Should `train` and `test` images be similar or completely different?

This is the most important rule in Machine Learning:

1. **Zero Exact Duplicates (100% Different Files):**
   An image file inside `test/bus/` must **NEVER** be the exact same photo as an image inside `train/bus/`. If the same photo is in both folders, the model is cheating (*Data Leakage*).
2. **Same Domain & Visual Style (Similar Context):**
   While the photos must be *different*, they must represent the **same real-world category and style**:
   * **Good:** `train/bus/` has 700 photos of real buses on streets (front, side, back, day, cloudy), and `test/bus/` has 150 **other** photos of real buses on streets.
   * **Bad:** `train/bus/` has real street photos of buses, and `test/bus/` has 3D cartoons, drawings, or toy buses.
3. **Avoid Video Frame Clones:**
   If you extract frames from a video, do not put Frame #10 in `train` and Frame #11 (taken 0.03 seconds later) in `test`, because they look 99% identical.

---

## 2. Project Context & Instructions for AI Coding Agent (Codex)

> **ATTENTION CODEX / AI AGENT:** Read this entire specification carefully before generating or modifying any files in this repository.

### Project Overview
This repository implements an academic and practical Machine Learning engineering project (**PACTE 4th Year Engineering — HESTIM**):
**"Classification d'Objets Routiers par Machine Learning pour les Véhicules Autonomes"** (Road Object Classification by Machine Learning for Autonomous Vehicles).

### Strict Academic & Technical Requirements
1. **Target Classes (5 Classes):**
   `car` (Voiture), `pedestrian` (Piéton), `cyclist` (Cycliste), `motorcycle` (Moto), `bus` (Bus).
   *(Map folder names dynamically if the dataset uses French or English labels).*
2. **Real Data Only (No Synthetic Noise in Production):**
   * Always point the training and preprocessing pipeline to the real dataset directory (`./dataset/train` and `./dataset/test`).
   * **NEVER** overwrite or mix synthetic random noise test fixtures (`np.random`) into the main artifact directories used for model training or evaluation.
3. **Step-by-Step ML Pipeline (Required for Deliverable 1 — Jupyter Notebook & Training Script):**
   * **Step 1 (Acquisition):** Load real images from `./dataset/train` and `./dataset/test` (and `./dataset/validation` if present; otherwise split `train` into 85% train / 15% validation using stratified sampling).
   * **Step 2 (Preprocessing):**
     * Remove corrupted/unreadable images.
     * Deduplicate images across `train`, `validation`, and `test` using MD5 file hashes so there is zero data leakage between splits.
     * Resize all images uniformly to `128x128` pixels.
     * Normalize pixel intensities (`[0, 1]` or standard scaling).
   * **Step 3 (Exploratory Data Analysis - EDA):**
     * Compute and save class counts across splits (`train`, `val`, `test`).
     * Plot class distribution bar charts to highlight any class imbalance.
     * Save a visual grid of sample real images from each class (never noise images).
   * **Step 4 (Feature Extraction & Classical ML Models):**
     * Extract **HOG (Histogram of Oriented Gradients)** features (`skimage.feature.hog`) concatenated with **HSV/RGB Color Histograms**.
     * Apply `StandardScaler`.
     * Train and compare at least three classical classifiers from `scikit-learn`:
       1. **SVM** (`SVC(kernel='rbf', probability=True, class_weight='balanced')`)
       2. **Random Forest** (`RandomForestClassifier(n_estimators=200, class_weight='balanced')`)
       3. **k-NN** (`KNeighborsClassifier(n_neighbors=5, weights='distance')`)
   * **Step 5 (Evaluation):**
     * Compute `Accuracy`, `Precision`, `Recall`, and `F1-Score` (weighted and per-class) on the `test` set.
     * Generate and save **Confusion Matrices** for all 3 models.
     * Automatically identify and log the most confused class pairs (e.g., `cyclist` vs `motorcycle`) and explain why.
   * **Step 6 (Model Export & Application):**
     * Save the best-performing classical ML pipeline (`scaler` + `feature_config` + `best_model` + `class_names`) to `./backend/models/classical_pipeline.joblib`.
     * Test inference on single unseen images and output structured predictions: `Image -> Voiture (92%)`.

---

### Dual-Mode Web Application Architecture (FastAPI + React)
To provide a live interactive demonstration on presentation day, the project includes a full-stack web dashboard with **two inference modes**:

1. **Mode A — Classical ML Classifier (Academic Requirement):**
   * Uses the trained `classical_pipeline.joblib` (HOG + Color Histogram + SVM/RF/k-NN).
   * Accepts one or multiple uploaded road object images.
   * Returns predicted class, obstacle category (`Vehicle`, `Heavy Vehicle`, `Vulnerable Road User`, `Two-Wheeler`), confidence percentage, full probability distribution across the 5 classes, and simulated autonomous driving action (`Maintain Speed`, `Slow Down`, `Emergency Brake`).
2. **Mode B — Multi-Object Street Scene Detection (Live Demo Bonus):**
   * Accepts a full street scene image containing multiple road objects.
   * Uses `ultralytics` YOLO (`yolo11n.pt`) to detect bounding boxes `[x1, y1, x2, y2]` of road obstacles (`car`, `person`, `bicycle`, `motorcycle`, `bus`).
   * Crops each detected bounding box and passes it through our trained Classical ML model (`classical_pipeline.joblib`) so both YOLO localization and our custom ML classification are showcased together.
   * Returns the annotated image (with bounding boxes drawn) and a telemetry list of all detected objects in the frame.

---

### Expected Repository Structure
```text
pacte-ml-autonomous/
├── README.md                          # Project specification & dataset guide (this file)
├── dataset/                           # Real image dataset (DO NOT fill with random noise)
│   ├── train/
│   │   ├── bus/
│   │   ├── car/
│   │   ├── cyclist/
│   │   ├── motorcycle/
│   │   └── pedestrian/
│   └── test/
│       ├── bus/
│       ├── car/
│       ├── cyclist/
│       ├── motorcycle/
│       └── pedestrian/
├── notebooks/
│   └── PACTE_ML_Road_Objects.ipynb    # Complete academic notebook (Steps 1 to 6)
├── backend/
│   ├── main.py                        # FastAPI server (/api/analyze, /api/metrics)
│   ├── ml_pipeline.py                 # Training, HOG feature extraction, evaluation script
│   ├── requirements.txt               # Python dependencies
│   └── models/
│       ├── classical_pipeline.joblib  # Exported best classical model
│       └── evaluation_metrics.json    # Saved Accuracy, F1, Confusion Matrix data for UI
└── frontend/                          # React + Vite + Tailwind CSS Autonomous Dashboard
    ├── package.json
    └── src/
        ├── App.jsx
        └── components/
```