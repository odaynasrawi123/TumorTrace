# TumorTrace- User Guide

1. Overview

TumorTrace is a deep-learning system for bone tumor detection and segmentation from X-ray images.  
The system allows users to analyze medical images, generate segmentation masks, view model predictions, and review model performance through graphs and dashboard outputs.

The project includes:
- U-Net V17 model
- ResNet50-U-Net V18 model
- UNet++ V23/V26 model
- ROC/AUC evaluation
- Dice and IoU segmentation metrics
- Dashboard visualizations
- Prediction overlay examples

2. Project Structure

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
