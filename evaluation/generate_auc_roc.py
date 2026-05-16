# ============================================================
# ROC / AUC Graph Generator from Saved Segmentation Models
# FIXED VERSION:
# - Handles old BatchNormalization renorm keys
# - Loads U-Net V17, ResNet50-U-Net V18, UNet++ V26
# - Uses X_test.npy and Y_test.npy
# - Creates ROC, AUC, Precision-Recall, Threshold Analysis
# ============================================================

import os
import json
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# 0. DRIVE
# ============================================================
from google.colab import drive
drive.mount("/content/drive")

print("TensorFlow:", tf.__version__)

# ============================================================
# 1. PATHS
# ============================================================
PROJECT_DIR = Path("/content/drive/MyDrive/TumorDataset")
ARRAYS_DIR = PROJECT_DIR / "arrays" / "v9_roi_polygon_only_160"
MODELS_DIR = PROJECT_DIR / "models"

OUTPUT_DIR = PROJECT_DIR / "graphs" / "roc_auc_3_models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATCHED_MODELS_DIR = MODELS_DIR / "patched_for_tf20"
PATCHED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "U-Net V17": MODELS_DIR / "best_v17_unet_baseline_160.keras",
    "ResNet50-U-Net V18": MODELS_DIR / "best_v18_resnet50_unet_colab_gpu_scratch_compare.keras",
    "UNet++ V26": MODELS_DIR / "best_unetpp_v26_unetpp_v25_100epoch_precision_tta.keras",
}

print("ARRAYS_DIR:", ARRAYS_DIR)
print("MODELS_DIR:", MODELS_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)

# ============================================================
# 2. LOAD TEST DATA
# ============================================================
X_test = np.load(ARRAYS_DIR / "X_test.npy").astype(np.float32)
Y_test = np.load(ARRAYS_DIR / "Y_test.npy").astype(np.float32)

if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]

if Y_test.ndim == 3:
    Y_test = Y_test[..., np.newaxis]

if X_test.max() > 1.5:
    X_test = X_test / 255.0

Y_test = (Y_test > 0.5).astype(np.float32)

y_true_img = (Y_test.reshape(len(Y_test), -1).sum(axis=1) > 0).astype(int)

print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
print("Image-level positives:", int(y_true_img.sum()))
print("Image-level negatives:", int(len(y_true_img) - y_true_img.sum()))

# ============================================================
# 3. CUSTOM OBJECTS
# ============================================================
class CompatibleBatchNormalization(tf.keras.layers.BatchNormalization):
    @classmethod
    def from_config(cls, config):
        config.pop("renorm", None)
        config.pop("renorm_clipping", None)
        config.pop("renorm_momentum", None)
        return super().from_config(config)


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
def focal_tversky_iou_dice_bce_loss(y_true, y_pred):
    return final_segmentation_loss(y_true, y_pred)


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
    "focal_tversky_iou_dice_bce_loss": focal_tversky_iou_dice_bce_loss,
    "iou_metric": iou_metric,
}

# ============================================================
# 4. PATCH OLD .KERAS FILES IF NEEDED
# ============================================================
def remove_bn_renorm_keys_from_config(obj):
    if isinstance(obj, dict):
        if obj.get("class_name") == "BatchNormalization":
            config = obj.get("config", {})
            if isinstance(config, dict):
                config.pop("renorm", None)
                config.pop("renorm_clipping", None)
                config.pop("renorm_momentum", None)

        for value in obj.values():
            remove_bn_renorm_keys_from_config(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_bn_renorm_keys_from_config(item)


def patch_keras_file_remove_bn_renorm(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)

    if output_path.exists():
        print("Using existing patched model:", output_path)
        return output_path

    print("Patching old Keras model:")
    print("Input :", input_path)
    print("Output:", output_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with zipfile.ZipFile(input_path, "r") as zin:
            zin.extractall(tmpdir)

        config_path = tmpdir / "config.json"

        if not config_path.exists():
            raise FileNotFoundError("config.json not found inside .keras file")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        remove_bn_renorm_keys_from_config(config)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for file in tmpdir.rglob("*"):
                if file.is_file():
                    zout.write(file, file.relative_to(tmpdir))

    print("Patched model saved:", output_path)
    return output_path


def load_model_compatible(model_path, model_name):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model for {model_name}: {model_path}")

    print("\nLoading:", model_name)
    print("Path:", model_path)

    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )
        print("Loaded directly:", model_name)
        return model

    except Exception as e:
        print("Direct load failed for:", model_name)
        print("Reason:", str(e)[:700])

        patched_path = PATCHED_MODELS_DIR / f"patched_{model_path.name}"
        patched_path = patch_keras_file_remove_bn_renorm(model_path, patched_path)

        model = tf.keras.models.load_model(
            patched_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
            safe_mode=False,
        )

        print("Loaded patched model:", model_name)
        return model

# ============================================================
# 5. PREDICTION HELPERS
# ============================================================
def image_probability_from_mask(pred_mask_probs, method="max"):
    flat = pred_mask_probs.reshape(pred_mask_probs.shape[0], -1)

    if method == "max":
        return flat.max(axis=1)

    if method == "mean_top_1_percent":
        k = max(1, int(flat.shape[1] * 0.01))
        sorted_vals = np.sort(flat, axis=1)
        return sorted_vals[:, -k:].mean(axis=1)

    if method == "mean_top_5_percent":
        k = max(1, int(flat.shape[1] * 0.05))
        sorted_vals = np.sort(flat, axis=1)
        return sorted_vals[:, -k:].mean(axis=1)

    if method == "mean":
        return flat.mean(axis=1)

    raise ValueError("Unknown method")


def safe_filename(name):
    return (
        name.replace(" ", "_")
        .replace("+", "pp")
        .replace("-", "_")
        .replace("/", "_")
    )


def predict_model_probs(model_path, model_name):
    model = load_model_compatible(model_path, model_name)

    print("Predicting:", model_name)
    pred_probs = model.predict(X_test, batch_size=8, verbose=1)

    pred_probs = np.asarray(pred_probs).astype(np.float32)

    if pred_probs.ndim == 3:
        pred_probs = pred_probs[..., np.newaxis]

    pred_probs = np.clip(pred_probs, 0.0, 1.0)

    y_prob_max = image_probability_from_mask(pred_probs, method="max")
    y_prob_top1 = image_probability_from_mask(pred_probs, method="mean_top_1_percent")
    y_prob_top5 = image_probability_from_mask(pred_probs, method="mean_top_5_percent")
    y_prob_mean = image_probability_from_mask(pred_probs, method="mean")

    np.save(
        OUTPUT_DIR / f"pred_mask_probs_{safe_filename(model_name)}.npy",
        pred_probs
    )

    pd.DataFrame({
        "y_true": y_true_img,
        "y_prob_max": y_prob_max,
        "y_prob_top1": y_prob_top1,
        "y_prob_top5": y_prob_top5,
        "y_prob_mean": y_prob_mean,
    }).to_csv(
        OUTPUT_DIR / f"image_level_probs_{safe_filename(model_name)}.csv",
        index=False
    )

    return {
        "model": model_name,
        "pred_probs": pred_probs,
        "y_prob_max": y_prob_max,
        "y_prob_top1": y_prob_top1,
        "y_prob_top5": y_prob_top5,
        "y_prob_mean": y_prob_mean,
    }

# ============================================================
# 6. RUN PREDICTIONS
# ============================================================
all_predictions = []

for model_name, model_path in MODEL_PATHS.items():
    result = predict_model_probs(model_path, model_name)
    all_predictions.append(result)

# ============================================================
# 7. ROC + AUC COMPARISON
# ============================================================
PROBABILITY_METHODS = [
    "y_prob_max",
    "y_prob_top1",
    "y_prob_top5",
    "y_prob_mean",
]

all_auc_rows = []

for prob_method in PROBABILITY_METHODS:
    plt.figure(figsize=(10, 8))

    method_rows = []

    for item in all_predictions:
        model_name = item["model"]
        y_prob = item[prob_method]

        fpr, tpr, thresholds = roc_curve(y_true_img, y_prob)
        roc_auc = auc(fpr, tpr)

        best_idx = np.argmax(tpr - fpr)
        best_threshold = thresholds[best_idx]

        y_pred_best = (y_prob >= best_threshold).astype(int)

        acc = accuracy_score(y_true_img, y_pred_best)
        prec = precision_score(y_true_img, y_pred_best, zero_division=0)
        rec = recall_score(y_true_img, y_pred_best, zero_division=0)
        f1 = f1_score(y_true_img, y_pred_best, zero_division=0)
        cm = confusion_matrix(y_true_img, y_pred_best, labels=[0, 1])

        row = {
            "model": model_name,
            "probability_method": prob_method,
            "auc": roc_auc,
            "best_roc_threshold": best_threshold,
            "accuracy_at_best_threshold": acc,
            "precision_at_best_threshold": prec,
            "recall_at_best_threshold": rec,
            "f1_at_best_threshold": f1,
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

        method_rows.append(row)
        all_auc_rows.append(row)

        plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} AUC={roc_auc:.4f}")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.title(f"ROC Curve Comparison - {prob_method}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    roc_path = OUTPUT_DIR / f"01_roc_curve_comparison_{prob_method}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", roc_path)

    method_df = pd.DataFrame(method_rows)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(method_df["model"], method_df["auc"])
    ax.set_title(f"AUC Comparison - {prob_method}")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    for i, value in enumerate(method_df["auc"]):
        ax.text(i, value + 0.01, f"{value:.4f}", ha="center")

    plt.tight_layout()

    auc_path = OUTPUT_DIR / f"02_auc_bar_comparison_{prob_method}.png"
    plt.savefig(auc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", auc_path)

auc_df = pd.DataFrame(all_auc_rows)
auc_df.to_csv(OUTPUT_DIR / "roc_auc_all_methods_summary.csv", index=False)

# ============================================================
# 8. PRECISION-RECALL CURVES
# ============================================================
all_pr_rows = []

for prob_method in PROBABILITY_METHODS:
    plt.figure(figsize=(10, 8))

    pr_method_rows = []

    for item in all_predictions:
        model_name = item["model"]
        y_prob = item[prob_method]

        precision, recall, thresholds = precision_recall_curve(y_true_img, y_prob)
        ap = average_precision_score(y_true_img, y_prob)

        row = {
            "model": model_name,
            "probability_method": prob_method,
            "average_precision": ap,
        }

        pr_method_rows.append(row)
        all_pr_rows.append(row)

        plt.plot(recall, precision, linewidth=2, label=f"{model_name} AP={ap:.4f}")

    plt.title(f"Precision-Recall Curve Comparison - {prob_method}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    pr_path = OUTPUT_DIR / f"03_precision_recall_curve_comparison_{prob_method}.png"
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", pr_path)

    pr_method_df = pd.DataFrame(pr_method_rows)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(pr_method_df["model"], pr_method_df["average_precision"])
    ax.set_title(f"Average Precision Comparison - {prob_method}")
    ax.set_ylabel("Average Precision")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    for i, value in enumerate(pr_method_df["average_precision"]):
        ax.text(i, value + 0.01, f"{value:.4f}", ha="center")

    plt.tight_layout()

    ap_path = OUTPUT_DIR / f"04_average_precision_bar_comparison_{prob_method}.png"
    plt.savefig(ap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", ap_path)

pr_df = pd.DataFrame(all_pr_rows)
pr_df.to_csv(OUTPUT_DIR / "precision_recall_all_methods_summary.csv", index=False)

# ============================================================
# 9. THRESHOLD ANALYSIS PER MODEL
# ============================================================
threshold_grid = np.linspace(0, 1, 101)

for prob_method in PROBABILITY_METHODS:
    for item in all_predictions:
        model_name = item["model"]
        y_prob = item[prob_method]

        rows = []

        for thr in threshold_grid:
            y_pred = (y_prob >= thr).astype(int)

            rows.append({
                "threshold": thr,
                "accuracy": accuracy_score(y_true_img, y_pred),
                "precision": precision_score(y_true_img, y_pred, zero_division=0),
                "recall": recall_score(y_true_img, y_pred, zero_division=0),
                "f1": f1_score(y_true_img, y_pred, zero_division=0),
            })

        thr_df = pd.DataFrame(rows)

        safe = safe_filename(model_name)

        thr_df.to_csv(
            OUTPUT_DIR / f"threshold_analysis_{safe}_{prob_method}.csv",
            index=False
        )

        plt.figure(figsize=(11, 6))
        plt.plot(thr_df["threshold"], thr_df["accuracy"], label="Accuracy")
        plt.plot(thr_df["threshold"], thr_df["precision"], label="Precision")
        plt.plot(thr_df["threshold"], thr_df["recall"], label="Recall")
        plt.plot(thr_df["threshold"], thr_df["f1"], label="F1")

        plt.title(f"Threshold Analysis - {model_name} - {prob_method}")
        plt.xlabel("Image-Level Probability Threshold")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        path = OUTPUT_DIR / f"05_threshold_analysis_{safe}_{prob_method}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", path)

# ============================================================
# 10. SUMMARY TABLE IMAGES
# ============================================================
def save_table_image(df_table, title, filename):
    display_df = df_table.copy()

    for col in display_df.columns:
        if col != "model" and col != "probability_method":
            display_df[col] = pd.to_numeric(display_df[col], errors="ignore")
            if pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].round(4)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    plt.title(title, pad=20)
    plt.tight_layout()

    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", path)


best_auc_summary = (
    auc_df.sort_values("auc", ascending=False)
    .groupby("model")
    .head(1)
    .reset_index(drop=True)
)

save_table_image(
    best_auc_summary[
        [
            "model",
            "probability_method",
            "auc",
            "best_roc_threshold",
            "accuracy_at_best_threshold",
            "precision_at_best_threshold",
            "recall_at_best_threshold",
            "f1_at_best_threshold",
            "tn",
            "fp",
            "fn",
            "tp",
        ]
    ],
    "Best ROC / AUC Summary per Model",
    "06_best_roc_auc_summary_table.png"
)

all_auc_table = auc_df[
    [
        "model",
        "probability_method",
        "auc",
        "accuracy_at_best_threshold",
        "precision_at_best_threshold",
        "recall_at_best_threshold",
        "f1_at_best_threshold",
    ]
].copy()

save_table_image(
    all_auc_table,
    "All ROC / AUC Methods Summary",
    "07_all_roc_auc_methods_table.png"
)

# ============================================================
# 11. FINAL PRINT
# ============================================================
print("\n================================================")
print("ROC / AUC analysis completed.")
print("Output folder:")
print(OUTPUT_DIR)
print("================================================")

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*")):
    print(file.name)