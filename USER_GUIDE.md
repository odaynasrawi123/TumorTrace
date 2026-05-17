# TumorTrace- User Guide

1. Overview

TumorTrace is a deep-learning system for bone tumor detection and segmentation from X-ray images.  
The system allows users to analyze medical images, generate segmentation masks, view model predictions, and review model performance through graphs and dashboard outputs.

The project includes:
- U-Net V17 model
- ResNet50 V18 model
- UNet++ V26 model
- ROC/AUC evaluation
- Dice and IoU segmentation metrics
- Dashboard visualizations
- Prediction overlay examples


2. How to run:
## How to Run the Project
### Step 1 — Install Requirements
Install all required Python libraries:
pip install -r requirements.txt

### Step 2— Download External Files
###Download the following resources from the links provided in the README.md file
models/
sample_data/
Annotations/
data/
graphs/

###Step 3 — Run Dataset Preprocessing
###Generate the training, validation, and test arrays:
python preprocessing/roi_dataset_builder.py

###step4 - Train the Models
U-Net V17:
python training/v17_unet_from_scratch.py
ResNet50-U-Net V18:
python training/v18_resnet50_unet.py
UNet++ V26:
python training/v23_unetpp_tta.py


###Step 5 — Generate Evaluation Graphs
Generate comparison graphs and evaluation metrics:
python evaluation/evaluation_3_models_generator.py
Generate ROC and AUC graphs:
python evaluation/generate_auc_roc.py

###Step 6 — Generate Dashboard Outputs
python dashboard/dashboard_app.py

3. Project Structure

```text
TumorTrace-AI/
│
├── training/
│   ├── v17_unet.py
│   ├── v18_resnet50.py
│   └── v26_unet++_precision.py
│
├── preprocessing/
│   └── roi_dataset_builder.py
│
├── evaluation/
│   ├── evaluation_3_models_generator.py
│   └── generate_auc_roc.py
│
├── dashboard/
│   └── dashboard_app.py
│
├── models/
│   └── README.md
│
├── data/
│   └── README.md
│
├── graphs/
│   └── README.md
│
├── requirements.txt
├── README.md
└── USER_GUIDE.md
