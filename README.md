# AUTOMATED-BLADDER-CANCER-SCREENING-WITH-DEEP-LEARNING-ALGORITHMS

This project presents an end-to-end pipeline for segmenting urothelial cell images and computing the **nuclear-to-cytoplasmic (N/C) ratio**, a critical biomarker used in cytological screening for bladder cancer. The system uses a deep learning model based on **U-Net with ResNet34 and FPN**, trained to segment nucleus and cytoplasm regions from RGB cell images. The resulting N/C ratios are then used to perform statistical analysis and cancer risk prediction.

---

## 🔍 Project Objectives

- Automate cell segmentation in urine cytology images.
- Calculate the N/C ratio per cell image.
- Correlate N/C ratios with clinical diagnoses.
- Evaluate predictive potential via logistic regression.
- Compare deep learning segmentation against traditional methods.

---

## 🏗️ Project Structure

```
Project2/
│
├── 1_prepdata.py                 # Prepares image/mask data from raw pickle files
├── 1_load_data.ipynb             # Intensity-based segmentation (OpenCV)
├── 2_Intensity_KMeans.ipynb      # KMeans and traditional segmentation methods
├── 3_DL_image_segmentation.ipynb # Deep learning segmentation and model training
├── 4_SpecimenAnalysis.ipynb      # Apply model to patient data, risk scoring
├── semseg_functions.py           # Custom functions for segmentation, training, and evaluation
├── train/                        # Training images and labels
├── val/                          # Validation images and labels
├── specimens_toy_data.pkl        # Specimen dataset with patient cell groupings
├── urothelial_cell_toy_data.pkl  # Training/validation data for segmentation
```

---

## 🛠️ Installation & Environment Setup

### Requirements

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install opencv-python pandas numpy torch torchvision matplotlib seaborn scikit-image scipy tqdm Pillow segmentation_models_pytorch scikit-learn
```

---

## 🚀 How to Run

### Step 1: Data Download and Preparation
Urothelial Cell Dataset: https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/tree/main/Project2/imagedata

Specimen Cell Dataset: https://github.com/jlevy44/Cedars_AI_Campus_Tutorials/raw/main/Project2/specimens_toy_data.pkl

```bash
python 1_prepdata.py
```

Generates `urothelial_cell_toy_data.pkl` from source images.

### Step 2: Traditional Segmentation

Run `1_load_data.ipynb` and `2_Intensity_KMeans.ipynb` to try intensity thresholding, clustering, and handcrafted feature segmentation.

### Step 3: Deep Learning Training

Run `3_DL_image_segmentation.ipynb` to train the U-Net+FPN model.

```python
model = train_model(X_train, Y_train, X_val, Y_val, encoder_name="resnet34", model_key="fpn", n_epochs=25)
```

Model checkpoints will be saved to `./seg_models`.

### Step 4: Apply Model to Real Patient Data

Run `4_SpecimenAnalysis.ipynb` to:
- Predict segmentation masks
- Compute N/C ratios
- Perform statistical analysis and binary classification

---

## 📊 Results Summary

- Spearman correlation (N/C ratio vs. diagnosis): **ρ ≈ 0.52**
- Binary classification (low vs. high risk): **F1-score = 0.77**
- Deep learning vs. traditional N/C ratio correlation: **ρ ≈ 0.73**

---

## 📌 Key Features

- Fully automated image segmentation pipeline
- Statistical validation using diagnostic labels
- Transparent and interpretable N/C ratio–based scoring
- Traditional vs. DL-based segmentation benchmarking

---
