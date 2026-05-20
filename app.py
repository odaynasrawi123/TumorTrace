# ============================================================
# TumorTrace AI Flask Dashboard - One Colab Cell
# ============================================================

!pip install -q flask tensorflow opencv-python pandas numpy matplotlib pillow

from google.colab import drive, output
drive.mount('/content/drive')

app_code = r'''
from flask import Flask, request, render_template_string, send_file
from pathlib import Path
import base64
import io
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

app = Flask(__name__)

PROJECT_DIR = Path("/content/drive/MyDrive/TumorDataset")

MODEL_PATH = PROJECT_DIR / "models" / "best_unetpp_v26_unetpp_v25_100epoch_precision_tta.keras"
FINAL_MODEL_GRAPHS_DIR = PROJECT_DIR / "graphs" / "v26_unetpp_v25_100epoch_precision_tta"
ROC_AUC_GRAPHS_DIR = PROJECT_DIR / "graphs" / "roc_auc_3_models"
EVAL_GRAPHS_DIR = PROJECT_DIR / "graphs" / "evaluation_3_models_report"
THRESHOLD_CSV_PATH = PROJECT_DIR / "reports" / "threshold_search_v26_unetpp_v25_100epoch_precision_tta.csv"

IMG_SIZE = 160
MODEL = None


# ============================================================
# Custom objects
# ============================================================

class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, *args, **kwargs):
        kwargs.pop("renorm", None)
        kwargs.pop("renorm_clipping", None)
        kwargs.pop("renorm_momentum", None)
        super().__init__(*args, **kwargs)

@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

@tf.keras.utils.register_keras_serializable()
def soft_iou_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def jaccard_loss(y_true, y_pred):
    return 1.0 - soft_iou_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable()
def tversky_index(y_true, y_pred, alpha=0.40, beta=0.60, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

@tf.keras.utils.register_keras_serializable()
def focal_tversky_loss(y_true, y_pred, gamma=1.05):
    tv = tversky_index(y_true, y_pred)
    return tf.pow((1.0 - tv), gamma)

@tf.keras.utils.register_keras_serializable()
def final_segmentation_loss(y_true, y_pred):
    ft = focal_tversky_loss(y_true, y_pred)
    jl = jaccard_loss(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return 0.18 * ft + 0.44 * jl + 0.30 * dl + 0.08 * bce

@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

CUSTOM_OBJECTS = {
    "BatchNormalization": CompatibleBatchNormalization,
    "CompatibleBatchNormalization": CompatibleBatchNormalization,
    "dice_coef": dice_coef,
    "soft_iou_coef": soft_iou_coef,
    "dice_loss": dice_loss,
    "jaccard_loss": jaccard_loss,
    "tversky_index": tversky_index,
    "focal_tversky_loss": focal_tversky_loss,
    "final_segmentation_loss": final_segmentation_loss,
    "focal_tversky_iou_dice_bce_loss": final_segmentation_loss,
    "iou_metric": iou_metric,
}


# ============================================================
# Helpers
# ============================================================

def load_model_once():
    global MODEL
    if MODEL is None and MODEL_PATH.exists():
        MODEL = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False
        )
    return MODEL


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def np_image_to_base64(arr, cmap="gray"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(arr, cmap=cmap)
    ax.axis("off")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def overlay_to_base64(image, mask):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="gray")
    ax.imshow(mask, alpha=0.45)
    ax.axis("off")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def list_graphs(folder, limit=20):
    if not folder.exists():
        return []
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        files += list(folder.glob(ext))
    return sorted(files)[:limit]


def load_thresholds():
    if not THRESHOLD_CSV_PATH.exists():
        return 0.47, 1, 360

    try:
        df = pd.read_csv(THRESHOLD_CSV_PATH)
        if "seg_score" in df.columns:
            best = df.sort_values("seg_score", ascending=False).iloc[0]
        elif "combined_score" in df.columns:
            best = df.sort_values("combined_score", ascending=False).iloc[0]
        else:
            best = df.iloc[0]

        return (
            float(best.get("pred_threshold", 0.47)),
            int(best.get("pixel_threshold", 1)),
            int(best.get("min_area", 360))
        )
    except Exception:
        return 0.47, 1, 360


def postprocess(pred_probs, threshold, min_area):
    mask = (pred_probs.squeeze() > threshold).astype(np.uint8)

    mask[:8, :] = 0
    mask[-8:, :] = 0
    mask[:, :8] = 0
    mask[:, -8:] = 0

    open_k = np.ones((3, 3), np.uint8)
    close_k = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    best_label = None
    best_area = 0

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label = label_id
            best_area = area

    if best_label is not None:
        clean[labels == best_label] = 1

    return clean


def health_table():
    rows = [
        ("Project Directory", PROJECT_DIR.exists()),
        ("Final Model", MODEL_PATH.exists()),
        ("Final Model Graphs", FINAL_MODEL_GRAPHS_DIR.exists()),
        ("ROC/AUC Graphs", ROC_AUC_GRAPHS_DIR.exists()),
        ("Evaluation Graphs", EVAL_GRAPHS_DIR.exists()),
        ("Threshold CSV", THRESHOLD_CSV_PATH.exists()),
    ]
    html = "<table><tr><th>Component</th><th>Status</th></tr>"
    for name, ok in rows:
        status = "✅ Found" if ok else "❌ Missing"
        html += f"<tr><td>{name}</td><td>{status}</td></tr>"
    html += "</table>"
    return html


# ============================================================
# HTML Template
# ============================================================

BASE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>TumorTrace AI Dashboard</title>
    <style>
        body {
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #f5f8fc;
            color: #172033;
        }
        .sidebar {
            width: 260px;
            background: #0f172a;
            color: white;
            height: 100vh;
            position: fixed;
            padding: 28px 20px;
        }
        .sidebar h2 {
            margin-top: 0;
            font-size: 26px;
        }
        .sidebar a {
            display: block;
            color: #cbd5e1;
            text-decoration: none;
            margin: 14px 0;
            padding: 10px 12px;
            border-radius: 10px;
        }
        .sidebar a:hover {
            background: #1e293b;
            color: white;
        }
        .main {
            margin-left: 310px;
            padding: 35px;
        }
        .card {
            background: white;
            border-radius: 18px;
            padding: 24px;
            margin-bottom: 25px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 18px;
        }
        .metric {
            background: white;
            padding: 22px;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
        }
        .metric h3 {
            margin: 0;
            color: #64748b;
            font-size: 15px;
        }
        .metric p {
            font-size: 26px;
            font-weight: bold;
            margin: 8px 0 0;
        }
        img {
            max-width: 100%;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
        }
        .image-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 22px;
        }
        .triple-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 22px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }
        th, td {
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
            text-align: left;
        }
        th {
            background: #eaf1fb;
        }
        .btn {
            background: #2563eb;
            color: white;
            padding: 10px 18px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
        }
        input[type=file] {
            padding: 12px;
            background: white;
            border-radius: 12px;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <h2>TumorTrace AI</h2>
        <p>Bone Tumor Detection & Segmentation</p>
        <hr>
        <a href="/">Overview</a>
        <a href="/predict">Prediction Demo</a>
        <a href="/comparison">Model Comparison</a>
        <a href="/roc">ROC / AUC</a>
        <a href="/final">Final Model Graphs</a>
        <a href="/threshold">Threshold Analysis</a>
        <a href="/gallery">Graph Gallery</a>
        <a href="/about">About</a>
    </div>
    <div class="main">
        {{ content|safe }}
    </div>
</body>
</html>
"""


def render(content):
    return render_template_string(BASE_HTML, content=content)


# ============================================================
# Routes
# ============================================================

@app.route("/")
def overview():
    pred_th, pix_th, min_area = load_thresholds()

    content = f"""
    <h1>TumorTrace AI Dashboard</h1>
    <p>Deep-learning dashboard for bone tumor detection and segmentation from X-ray images.</p>

    <div class="grid">
        <div class="metric"><h3>Final Model</h3><p>UNet++ V26</p></div>
        <div class="metric"><h3>Input Size</h3><p>160×160</p></div>
        <div class="metric"><h3>Prediction Threshold</h3><p>{pred_th}</p></div>
        <div class="metric"><h3>Min Area</h3><p>{min_area}</p></div>
    </div>

    <div class="card">
        <h2>System Health Check</h2>
        {health_table()}
    </div>

    <div class="card">
        <h2>Project Pipeline</h2>
        <ol>
            <li>Load X-ray images and annotation files</li>
            <li>Generate binary tumor masks</li>
            <li>Apply ROI preprocessing</li>
            <li>Train U-Net, ResNet50-U-Net and UNet++ models</li>
            <li>Optimize thresholds and post-processing</li>
            <li>Evaluate using Dice, IoU, Precision, Recall, F1 and AUC</li>
            <li>Visualize predictions and performance graphs</li>
        </ol>
    </div>
    """
    return render(content)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result_html = ""

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            model = load_model_once()

            if model is None:
                result_html = "<p style='color:red;'>Model not found.</p>"
            else:
                pred_th, pix_th, min_area = load_thresholds()

                image = Image.open(file).convert("L")
                image_np = np.array(image)

                resized = cv2.resize(image_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
                resized_float = resized.astype(np.float32) / 255.0

                x = resized_float[np.newaxis, ..., np.newaxis]

                pred_probs = model.predict(x, verbose=0)[0]
                pred_mask = postprocess(pred_probs, pred_th, min_area)

                tumor_area = int(pred_mask.sum())
                detected = "Yes" if tumor_area >= pix_th else "No"
                confidence = float(pred_probs.max())

                original_b64 = np_image_to_base64(resized_float)
                prob_b64 = np_image_to_base64(pred_probs.squeeze())
                mask_b64 = np_image_to_base64(pred_mask)
                overlay_b64 = overlay_to_base64(resized_float, pred_mask)

                result_html = f"""
                <div class="grid">
                    <div class="metric"><h3>Tumor Detected</h3><p>{detected}</p></div>
                    <div class="metric"><h3>Predicted Area</h3><p>{tumor_area}</p></div>
                    <div class="metric"><h3>Max Confidence</h3><p>{confidence:.4f}</p></div>
                    <div class="metric"><h3>Threshold</h3><p>{pred_th}</p></div>
                </div>

                <div class="card">
                    <h2>Prediction Results</h2>
                    <div class="triple-grid">
                        <div><h3>Input Image</h3><img src="data:image/png;base64,{original_b64}"></div>
                        <div><h3>Probability Map</h3><img src="data:image/png;base64,{prob_b64}"></div>
                        <div><h3>Predicted Mask</h3><img src="data:image/png;base64,{mask_b64}"></div>
                    </div>
                    <br>
                    <h3>Segmentation Overlay</h3>
                    <img src="data:image/png;base64,{overlay_b64}" style="max-width:450px;">
                </div>
                """

    content = f"""
    <h1>Prediction Demo</h1>
    <div class="card">
        <p>Upload an X-ray image. Best results are expected for ROI-style images similar to the training data.</p>
        <form method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button class="btn" type="submit">Run Prediction</button>
        </form>
    </div>
    {result_html}
    """
    return render(content)


@app.route("/comparison")
def comparison():
    graphs = list_graphs(EVAL_GRAPHS_DIR, limit=20)
    items = ""

    for g in graphs:
        b64 = image_to_base64(g)
        items += f"<div><h3>{g.name}</h3><img src='data:image/png;base64,{b64}'></div>"

    content = f"""
    <h1>Model Comparison</h1>
    <div class="card">
        <p>Comparison between U-Net V17, ResNet50-U-Net V18 and UNet++ V26.</p>
    </div>
    <div class="image-grid">{items}</div>
    """
    return render(content)


@app.route("/roc")
def roc():
    graphs = list_graphs(ROC_AUC_GRAPHS_DIR, limit=20)
    items = ""

    for g in graphs:
        b64 = image_to_base64(g)
        items += f"<div><h3>{g.name}</h3><img src='data:image/png;base64,{b64}'></div>"

    content = f"""
    <h1>ROC / AUC Analysis</h1>
    <div class="card">
        <p>ROC curves, AUC comparison, Precision-Recall curves and probability analysis.</p>
    </div>
    <div class="image-grid">{items}</div>
    """
    return render(content)


@app.route("/final")
def final_graphs():
    graphs = list_graphs(FINAL_MODEL_GRAPHS_DIR, limit=20)
    items = ""

    for g in graphs:
        b64 = image_to_base64(g)
        items += f"<div><h3>{g.name}</h3><img src='data:image/png;base64,{b64}'></div>"

    content = f"""
    <h1>Final UNet++ V26 Graphs</h1>
    <div class="image-grid">{items}</div>
    """
    return render(content)


@app.route("/threshold")
def threshold():
    if THRESHOLD_CSV_PATH.exists():
        df = pd.read_csv(THRESHOLD_CSV_PATH).head(30)
        table = df.to_html(index=False)
    else:
        table = "<p>Threshold CSV not found.</p>"

    content = f"""
    <h1>Threshold Analysis</h1>
    <div class="card">
        {table}
    </div>
    """
    return render(content)


@app.route("/gallery")
def gallery():
    folders = [
        ("Final Model Graphs", FINAL_MODEL_GRAPHS_DIR),
        ("ROC / AUC Graphs", ROC_AUC_GRAPHS_DIR),
        ("3-Model Evaluation Graphs", EVAL_GRAPHS_DIR),
    ]

    sections = ""

    for title, folder in folders:
        graphs = list_graphs(folder, limit=8)
        items = ""
        for g in graphs:
            b64 = image_to_base64(g)
            items += f"<div><h3>{g.name}</h3><img src='data:image/png;base64,{b64}'></div>"

        sections += f"""
        <div class="card">
            <h2>{title}</h2>
            <div class="image-grid">{items}</div>
        </div>
        """

    return render(f"<h1>Graph Gallery</h1>{sections}")


@app.route("/about")
def about():
    content = """
    <h1>About TumorTrace AI</h1>
    <div class="card">
        <p>TumorTrace AI is an academic deep-learning project focused on medical image segmentation.</p>
        <ul>
            <li>ROI preprocessing</li>
            <li>U-Net baseline</li>
            <li>ResNet50-U-Net comparison</li>
            <li>UNet++ final model</li>
            <li>Threshold optimization</li>
            <li>Post-processing</li>
            <li>ROC/AUC evaluation</li>
            <li>Interactive dashboard</li>
        </ul>
        <p><b>Disclaimer:</b> This project is for academic and research purposes only and is not a certified medical diagnostic system.</p>
    </div>
    """
    return render(content)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=False)
'''

with open("/content/app_flask.py", "w", encoding="utf-8") as f:
    f.write(app_code)

print("Flask dashboard file created.")

# ----------------------------
# 4. Stop old servers
# ----------------------------
!pkill -f streamlit || true
!pkill -f flask || true
!pkill -f app_flask.py || true

# ----------------------------
# 5. Run Flask
# ----------------------------
!nohup python /content/app_flask.py > /content/flask_logs.txt 2>&1 &

import time
time.sleep(5)

print("Flask logs:")
!tail -20 /content/flask_logs.txt

# ----------------------------
# 6. Open with Colab proxy
# ----------------------------
url = output.eval_js("google.colab.kernel.proxyPort(8501)")

print("\n======================================")
print("OPEN THIS DASHBOARD LINK:")
print(url)
print("======================================")
