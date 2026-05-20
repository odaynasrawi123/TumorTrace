# TumorTrace AI – User Guide
## Overview

TumorTrace AI is an interactive deep-learning dashboard developed for bone tumor detection and segmentation from X-ray images.
The system uses CNN-based medical image segmentation architectures, including:
- U-Net
- ResNet50
- UNet++

The dashboard allows users to:
- upload X-ray images,
- run tumor segmentation predictions,
- visualize prediction masks and overlays,
- compare model performance,
- explore ROC/AUC analysis,
- review threshold optimization results,
- and analyze generated graphs and evaluation metrics.

# System Requirements
To run the dashboard successfully, the following are required:

- Google account
- Google Colab access
- Google Drive storage
- Internet connection

# Project Structure
The project is hosted on GitHub:
https://github.com/odaynasrawi123/TumorTrace

Main project files:

| File | Description |
|------|-------------|
| app.py | Main Flask dashboard application |
| TumorTrace_Dashboard_Colab.ipynb | Google Colab notebook used to run the dashboard |
| requirements.txt | Required Python libraries |
| README.md | Main project documentation |
| USER_GUIDE.md | User manual |

Large files such as:
- trained models,
- graphs,
- datasets,
- annotations

are stored externally on Google Drive due to GitHub size limitations.

# How to Run the Dashboard
## Step 1 – Open the GitHub Repository

Open the repository:
https://github.com/odaynasrawi123/TumorTrace

## Step 2 – Open the Colab Notebook
Click:
"Open Dashboard in Google Colab"

or open directly:
https://colab.research.google.com/github/odaynasrawi123/TumorTrace/blob/main/TumorTrace_Dashboard_Colab.ipynb

## Step 3 – Run All Notebook Cells

Inside Colab:

- Click:
  Runtime → Run All

The notebook will automatically:
- install required libraries,
- mount Google Drive,
- download the Flask dashboard,
- launch the Flask server,
- generate a dashboard URL.

## Step 4 – Open the Dashboard Link
After all cells finish running, a dashboard URL will appear:

Example:
https://8501-xxxxxxxx.colab.dev

Open this link in the browser.

# Dashboard Features
## 1. Overview Page

Displays:
- project summary,
- selected final model,
- prediction thresholds,
- segmentation pipeline,
- system health check,
- model comparison summary.

## 2. Prediction Demo

Users can upload an X-ray image and receive:
- tumor detection result,
- prediction confidence,
- probability map,
- binary segmentation mask,
- segmentation overlay visualization.

Supported formats:
- PNG
- JPG
- JPEG

## 3. Model Comparison

Displays comparison graphs between:
- U-Net V17
- ResNet50-U-Net V18
- UNet++ V26

Includes:
- Dice score comparisons,
- IoU comparisons,
- Precision/Recall metrics,
- evaluation visualizations.

## 4. ROC / AUC Analysis

Displays:
- ROC curves,
- AUC scores,
- Precision-Recall curves,
- classification performance graphs.

## 5. Final Model Graphs

Displays graphs generated from the final selected model:
- training curves,
- segmentation examples,
- prediction overlays,
- validation/test visualizations.

## 6. Threshold Analysis

Displays:
- threshold optimization table,
- segmentation score analysis,
- prediction threshold tuning results.

## 7. Graph Gallery

Centralized gallery for all generated graphs:
- final model graphs,
- ROC/AUC graphs,
- evaluation graphs.

## 8. About Page

Provides:
- project background,
- technologies used,
- project disclaimer.

# Technologies Used

## Programming Languages
- Python

## Frameworks
- Flask
- TensorFlow / Keras

## Libraries
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Pillow

## Deployment
- Google Colab
- Google Drive
- GitHub

# Dataset Information
The project uses the BTXRD dataset:
"A Radiograph Dataset for the Classification, Localization and Segmentation of Primary Bone Tumors"

Dataset components:
- X-ray images,
- annotation JSON files,
- metadata tables.


# Important Notes

- The dashboard works only while the Google Colab runtime is active.
- The user must grant Google Drive access when prompted.
- Large model files are not stored directly on GitHub.
- The system is intended for academic and research purposes only.
- The dashboard is not a certified medical diagnostic system.
# Troubleshooting

## Dashboard does not open
- Ensure all notebook cells completed successfully.
- Verify that Google Drive was mounted.
- Re-run the notebook if the Colab session disconnected.


## Model loading error
Verify that the model exists in:

/content/drive/MyDrive/TumorDataset/models/


## Missing graphs
Verify that graph folders exist in:

/content/drive/MyDrive/TumorDataset/graphs/

# Authors
TumorTrace Project

Developed as part of an academic final project in Information Systems and Deep Learning.
