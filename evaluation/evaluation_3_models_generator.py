# ============================================================
# Evaluation Graphs Generator - FIXED VERSION
# 3 Model Comparison:
# 1) U-Net V17
# 2) ResNet50-U-Net V18
# 3) UNet++ V26 selected
#
# Output:
# /content/drive/MyDrive/TumorDataset/graphs/evaluation_3_models_report
# ============================================================

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. DRIVE MOUNT
# ============================================================
try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as e:
    print("Drive mount skipped or failed:", e)


# ============================================================
# 1. PATHS
# ============================================================
PROJECT_DIR = Path("/content/drive/MyDrive/TumorDataset")
REPORTS_DIR = PROJECT_DIR / "reports"

OUTPUT_DIR = PROJECT_DIR / "graphs" / "evaluation_3_models_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("PROJECT_DIR:", PROJECT_DIR)
print("REPORTS_DIR:", REPORTS_DIR)
print("OUTPUT_DIR:", OUTPUT_DIR)


# ============================================================
# 2. INPUT FILES
# ============================================================
THRESHOLD_FILES = {
    "U-Net V17": REPORTS_DIR / "threshold_search_v17_unet_baseline_160.csv",
    "ResNet50-U-Net V18": REPORTS_DIR / "threshold_search_v18_resnet50_unet_colab_gpu_scratch_compare.csv",
    "UNet++ V26": REPORTS_DIR / "threshold_search_v26_unetpp_v25_100epoch_precision_tta.csv",
}

HISTORY_FILES = {
    "U-Net V17": REPORTS_DIR / "history_v17_unet_baseline_160.csv",
    "ResNet50-U-Net V18": REPORTS_DIR / "history_v18_resnet50_unet_colab_gpu_scratch_compare.csv",
    "UNet++ V26": REPORTS_DIR / "history_v26_unetpp_v25_100epoch_precision_tta.csv",
}


# ============================================================
# 3. FINAL MODEL METRICS
# ============================================================
models = [
    {
        "model": "U-Net V17",
        "architecture": "U-Net",
        "run_version": "v17_unet_baseline_160",
        "img_size": 160,
        "batch_size": 4,
        "epochs": 25,
        "learning_rate": 0.0001,
        "training_type": "Resumed training",
        "loss": "focal_tversky_iou_dice_bce",

        "threshold": 0.7200000286102295,
        "pixel_threshold": 80,
        "min_area": 80,

        "raw_dice": 0.7280,
        "raw_iou": 0.6292,
        "post_dice": 0.8104,
        "post_iou": 0.6812,

        "pixel_precision": 0.7969,
        "pixel_recall": 0.8243,
        "pixel_f1": 0.8104,
        "pixel_accuracy": 0.9377,

        "image_precision": 0.8567,
        "image_recall": 0.9821,
        "image_f1": 0.9151,
        "image_accuracy": 0.9089,

        "tn": 234,
        "fp": 46,
        "fn": 5,
        "tp": 275,
    },
    {
        "model": "ResNet50-U-Net V18",
        "architecture": "ResNet50-U-Net",
        "run_version": "v18_resnet50_unet_colab_gpu_scratch_compare",
        "img_size": 160,
        "batch_size": 8,
        "epochs": 40,
        "learning_rate": 0.0001,
        "training_type": "From scratch",
        "loss": "Segmentation loss",

        "threshold": 0.44999998807907104,
        "pixel_threshold": 1,
        "min_area": 20,

        "raw_dice": 0.7836,
        "raw_iou": 0.6665,
        "post_dice": 0.8087,
        "post_iou": 0.6789,

        "pixel_precision": 0.8333,
        "pixel_recall": 0.7856,
        "pixel_f1": 0.8087,
        "pixel_accuracy": 0.9400,

        "image_precision": 0.9155,
        "image_recall": 0.9679,
        "image_f1": 0.9410,
        "image_accuracy": 0.9393,

        "tn": 255,
        "fp": 25,
        "fn": 9,
        "tp": 271,
    },
    {
        "model": "UNet++ V26",
        "architecture": "UNet++",
        "run_version": "v26_unetpp_v25_100epoch_precision_tta",
        "img_size": 160,
        "batch_size": 8,
        "epochs": 100,
        "learning_rate": 1.25e-7,
        "training_type": "Continued from V25",
        "loss": "final_segmentation_loss",

        # Official selected V26 result, not manual combo.
        "threshold": 0.4699999988079071,
        "pixel_threshold": 1,
        "min_area": 360,

        "raw_dice": 0.7843,
        "raw_iou": 0.6598,
        "post_dice": 0.8101,
        "post_iou": 0.6808,

        "pixel_precision": 0.7917,
        "pixel_recall": 0.8293,
        "pixel_f1": 0.8101,
        "pixel_accuracy": 0.9372,

        "image_precision": 0.8440,
        "image_recall": 0.9857,
        "image_f1": 0.9094,
        "image_accuracy": 0.9018,

        "tn": 229,
        "fp": 51,
        "fn": 4,
        "tp": 276,
    },
]

df = pd.DataFrame(models)

# FIXED: create error columns before using them in graphs
df["false_positives"] = df["fp"]
df["false_negatives"] = df["fn"]
df["total_errors"] = df["fp"] + df["fn"]

# Extra classification analysis metrics
df["specificity"] = df["tn"] / (df["tn"] + df["fp"])
df["fn_rate"] = df["fn"] / (df["fn"] + df["tp"])
df["fp_rate"] = df["fp"] / (df["fp"] + df["tn"])

summary_csv = OUTPUT_DIR / "evaluation_3_models_summary.csv"
df.to_csv(summary_csv, index=False)
print("Saved:", summary_csv)


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================
def save_current_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", path)


def add_value_labels(ax, fmt="{:.4f}"):
    for container in ax.containers:
        try:
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=8)
        except Exception:
            pass


def safe_name(name):
    return (
        name.replace(" ", "_")
        .replace("+", "pp")
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "")
    )


def bar_chart(y_cols, title, ylabel, filename, ylim=None):
    ax = df.set_index("model")[y_cols].plot(kind="bar", figsize=(12, 6))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.xticks(rotation=20, ha="right")
    add_value_labels(ax)
    save_current_fig(OUTPUT_DIR / filename)


def single_bar(metric, title, ylabel, filename, ylim=None, label_fmt="{:.4f}"):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    ax.bar(df["model"], df[metric])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    if ylim is not None:
        ax.set_ylim(*ylim)

    plt.xticks(rotation=20, ha="right")

    for i, value in enumerate(df[metric]):
        if isinstance(value, (float, np.floating)):
            label = label_fmt.format(value)
            offset = (ylim[1] - ylim[0]) * 0.02 if ylim else 0.005
            ax.text(i, value + offset, label, ha="center", fontsize=9)
        else:
            ax.text(i, value + 1, str(value), ha="center", fontsize=9)

    save_current_fig(OUTPUT_DIR / filename)


def confusion_matrix_plot(model_name, tn, fp, fn, tp):
    cm = np.array([[tn, fp], [fn, tp]])

    plt.figure(figsize=(6, 5))
    ax = plt.gca()
    im = ax.imshow(cm)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar(im)

    labels = ["No Tumor", "Tumor"]
    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    threshold = cm.max() / 2.0

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=14,
                fontweight="bold",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    save_current_fig(OUTPUT_DIR / f"16_confusion_matrix_{safe_name(model_name)}.png")


def table_image(dataframe, title, filename, round_digits=4, figsize=(16, 4)):
    display_df = dataframe.copy()

    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].round(round_digits)

    fig, ax = plt.subplots(figsize=figsize)
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
    save_current_fig(OUTPUT_DIR / filename)


# ============================================================
# 5. MAIN COMPARISON GRAPHS
# ============================================================
bar_chart(
    ["post_dice", "post_iou"],
    "Final Post-Processing Segmentation Comparison",
    "Score",
    "01_final_post_dice_iou_comparison.png",
    ylim=(0.675, 0.815),
)

bar_chart(
    ["raw_dice", "post_dice"],
    "Raw Dice vs Post-Processed Dice",
    "Dice Score",
    "02_raw_vs_post_dice.png",
    ylim=(0.70, 0.83),
)

bar_chart(
    ["raw_iou", "post_iou"],
    "Raw IoU vs Post-Processed IoU",
    "IoU Score",
    "03_raw_vs_post_iou.png",
    ylim=(0.60, 0.70),
)

bar_chart(
    ["image_accuracy", "image_precision", "image_recall", "image_f1"],
    "Image-Level Classification Metrics",
    "Score",
    "04_image_level_metrics.png",
    ylim=(0.82, 1.00),
)

bar_chart(
    ["pixel_precision", "pixel_recall", "pixel_f1", "pixel_accuracy"],
    "Pixel-Level Segmentation Metrics",
    "Score",
    "05_pixel_level_metrics.png",
    ylim=(0.75, 1.00),
)

single_bar(
    "post_dice",
    "Final Test Post Dice Ranking",
    "Dice Score",
    "06_post_dice_ranking.png",
    ylim=(0.807, 0.812),
)

single_bar(
    "post_iou",
    "Final Test Post IoU Ranking",
    "IoU Score",
    "07_post_iou_ranking.png",
    ylim=(0.677, 0.683),
)

single_bar(
    "image_accuracy",
    "Image-Level Accuracy Ranking",
    "Accuracy",
    "08_image_accuracy_ranking.png",
    ylim=(0.89, 0.95),
)

single_bar(
    "image_f1",
    "Image-Level F1 Ranking",
    "F1 Score",
    "09_image_f1_ranking.png",
    ylim=(0.89, 0.95),
)

bar_chart(
    ["image_precision", "image_recall"],
    "Image-Level Precision vs Recall",
    "Score",
    "10_precision_recall_comparison.png",
    ylim=(0.82, 1.00),
)

# FIXED: now these columns exist
bar_chart(
    ["false_positives", "false_negatives", "total_errors"],
    "Image-Level Error Comparison",
    "Number of Images",
    "11_fp_fn_total_errors.png",
    ylim=None,
)

single_bar(
    "total_errors",
    "Total Image-Level Errors",
    "Number of Errors",
    "12_total_errors_ranking.png",
    ylim=None,
    label_fmt="{:.0f}",
)

bar_chart(
    ["specificity", "image_recall"],
    "Specificity vs Recall",
    "Score",
    "13_specificity_vs_recall.png",
    ylim=(0.80, 1.00),
)

bar_chart(
    ["fp_rate", "fn_rate"],
    "False Positive Rate vs False Negative Rate",
    "Rate",
    "14_fp_rate_fn_rate.png",
    ylim=(0, 0.20),
)

bar_chart(
    ["threshold", "pixel_threshold", "min_area"],
    "Selected Threshold and Post-Processing Parameters",
    "Value",
    "15_threshold_parameters.png",
    ylim=None,
)


# ============================================================
# 6. CONFUSION MATRICES FOR ALL MODELS
# ============================================================
for _, row in df.iterrows():
    confusion_matrix_plot(
        model_name=row["model"],
        tn=int(row["tn"]),
        fp=int(row["fp"]),
        fn=int(row["fn"]),
        tp=int(row["tp"]),
    )


# ============================================================
# 7. TABLES AS IMAGES + CSV
# ============================================================
final_metrics_df = df[
    [
        "model",
        "raw_dice",
        "raw_iou",
        "post_dice",
        "post_iou",
        "pixel_precision",
        "pixel_recall",
        "pixel_f1",
        "pixel_accuracy",
        "image_precision",
        "image_recall",
        "image_f1",
        "image_accuracy",
    ]
].copy()

final_metrics_df.to_csv(OUTPUT_DIR / "final_metrics_table.csv", index=False)

table_image(
    final_metrics_df,
    "Final Metrics Comparison",
    "19_final_metrics_table.png",
    figsize=(22, 5),
)

hyper_df = df[
    [
        "model",
        "architecture",
        "img_size",
        "batch_size",
        "epochs",
        "learning_rate",
        "threshold",
        "pixel_threshold",
        "min_area",
        "training_type",
        "loss",
    ]
].copy()

hyper_df.to_csv(OUTPUT_DIR / "hyperparameters_table.csv", index=False)

table_image(
    hyper_df,
    "Hyperparameters and Selected Thresholds",
    "20_hyperparameters_table.png",
    figsize=(22, 5),
)

cm_df = df[
    [
        "model",
        "tn",
        "fp",
        "fn",
        "tp",
        "false_positives",
        "false_negatives",
        "total_errors",
        "specificity",
        "fp_rate",
        "fn_rate",
    ]
].copy()

cm_df.to_csv(OUTPUT_DIR / "confusion_error_table.csv", index=False)

table_image(
    cm_df,
    "Confusion Matrix and Error Summary",
    "21_confusion_error_table.png",
    figsize=(20, 5),
)


# ============================================================
# 8. MODEL RANKING GRAPH
# ============================================================
ranking_df = df.copy()

# Segmentation is more important for this project.
ranking_df["segmentation_score"] = (
    0.60 * ranking_df["post_iou"]
    + 0.40 * ranking_df["post_dice"]
)

ranking_df["classification_score"] = (
    0.40 * ranking_df["image_f1"]
    + 0.30 * ranking_df["image_accuracy"]
    + 0.30 * ranking_df["image_recall"]
)

ranking_df["overall_score"] = (
    0.70 * ranking_df["segmentation_score"]
    + 0.30 * ranking_df["classification_score"]
)

ranking_df = ranking_df.sort_values("overall_score", ascending=False)

ranking_df[
    [
        "model",
        "segmentation_score",
        "classification_score",
        "overall_score",
    ]
].to_csv(OUTPUT_DIR / "model_ranking_scores.csv", index=False)

plt.figure(figsize=(12, 6))
ax = plt.gca()
ranking_plot = ranking_df.set_index("model")[
    ["segmentation_score", "classification_score", "overall_score"]
]
ranking_plot.plot(kind="bar", ax=ax)

ax.set_title("Weighted Model Ranking Scores")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=20, ha="right")
add_value_labels(ax)

save_current_fig(OUTPUT_DIR / "22_weighted_model_ranking.png")

table_image(
    ranking_df[
        [
            "model",
            "post_dice",
            "post_iou",
            "image_accuracy",
            "image_recall",
            "image_f1",
            "segmentation_score",
            "classification_score",
            "overall_score",
        ]
    ],
    "Model Ranking Summary",
    "23_model_ranking_table.png",
    figsize=(18, 5),
)


# ============================================================
# 9. THRESHOLD SEARCH GRAPHS
# ============================================================
def find_column(cols, candidates):
    lower_map = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def plot_threshold_search(model_name, csv_path):
    if not csv_path.exists():
        print(f"Threshold CSV not found for {model_name}: {csv_path}")
        return

    data = pd.read_csv(csv_path)

    threshold_col = find_column(
        data.columns,
        ["pred_threshold", "prediction_threshold", "threshold", "best_pred_threshold"]
    )

    min_area_col = find_column(
        data.columns,
        ["min_area", "best_min_area"]
    )

    val_iou_col = find_column(
        data.columns,
        ["val_iou", "val_iou_score", "iou", "val_post_iou"]
    )

    val_dice_col = find_column(
        data.columns,
        ["val_dice", "val_dice_score", "dice", "val_post_dice"]
    )

    seg_score_col = find_column(
        data.columns,
        ["seg_score", "segmentation_score"]
    )

    balanced_score_col = find_column(
        data.columns,
        ["balanced_score"]
    )

    if threshold_col is None:
        print(f"No threshold column found in {csv_path}")
        print("Columns:", list(data.columns))
        return

    name = safe_name(model_name)

    if val_iou_col is not None:
        grouped = data.groupby(threshold_col)[val_iou_col].max().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(grouped[threshold_col], grouped[val_iou_col], marker="o")
        plt.title(f"Threshold vs Validation IoU - {model_name}")
        plt.xlabel("Prediction Threshold")
        plt.ylabel("Best Validation IoU")
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"24_threshold_vs_val_iou_{name}.png")

    if val_dice_col is not None:
        grouped = data.groupby(threshold_col)[val_dice_col].max().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(grouped[threshold_col], grouped[val_dice_col], marker="o")
        plt.title(f"Threshold vs Validation Dice - {model_name}")
        plt.xlabel("Prediction Threshold")
        plt.ylabel("Best Validation Dice")
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"25_threshold_vs_val_dice_{name}.png")

    if seg_score_col is not None:
        grouped = data.groupby(threshold_col)[seg_score_col].max().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(grouped[threshold_col], grouped[seg_score_col], marker="o")
        plt.title(f"Threshold vs Segmentation Score - {model_name}")
        plt.xlabel("Prediction Threshold")
        plt.ylabel("Best Segmentation Score")
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"26_threshold_vs_seg_score_{name}.png")

    if balanced_score_col is not None:
        grouped = data.groupby(threshold_col)[balanced_score_col].max().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(grouped[threshold_col], grouped[balanced_score_col], marker="o")
        plt.title(f"Threshold vs Balanced Score - {model_name}")
        plt.xlabel("Prediction Threshold")
        plt.ylabel("Best Balanced Score")
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"27_threshold_vs_balanced_score_{name}.png")

    if min_area_col is not None and val_iou_col is not None:
        grouped = data.groupby(min_area_col)[val_iou_col].max().reset_index()
        plt.figure(figsize=(10, 5))
        plt.plot(grouped[min_area_col], grouped[val_iou_col], marker="o")
        plt.title(f"Min Area vs Validation IoU - {model_name}")
        plt.xlabel("Minimum Area")
        plt.ylabel("Best Validation IoU")
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"28_min_area_vs_val_iou_{name}.png")

    if threshold_col is not None and min_area_col is not None and val_iou_col is not None:
        pivot = data.pivot_table(
            index=min_area_col,
            columns=threshold_col,
            values=val_iou_col,
            aggfunc="max"
        )

        plt.figure(figsize=(12, 7))
        plt.imshow(pivot.values, aspect="auto", origin="lower")
        plt.colorbar(label="Validation IoU")
        plt.title(f"Threshold-MinArea Validation IoU Heatmap - {model_name}")
        plt.xlabel("Prediction Threshold")
        plt.ylabel("Minimum Area")

        plt.xticks(
            ticks=np.arange(len(pivot.columns)),
            labels=[f"{float(x):.2f}" for x in pivot.columns],
            rotation=45,
            ha="right"
        )
        plt.yticks(
            ticks=np.arange(len(pivot.index)),
            labels=pivot.index
        )

        save_current_fig(OUTPUT_DIR / f"29_threshold_min_area_heatmap_{name}.png")


for model_name, path in THRESHOLD_FILES.items():
    plot_threshold_search(model_name, path)


# ============================================================
# 10. TRAINING CURVES
# ============================================================
def plot_history(model_name, csv_path):
    if not csv_path.exists():
        print(f"History CSV not found for {model_name}: {csv_path}")
        return

    hist = pd.read_csv(csv_path)
    name = safe_name(model_name)

    if "loss" in hist.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(hist["loss"], label="Train Loss")
        if "val_loss" in hist.columns:
            plt.plot(hist["val_loss"], label="Validation Loss")
        plt.title(f"Training Loss - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"30_training_loss_{name}.png")

    dice_cols = [c for c in hist.columns if "dice" in c.lower()]
    if dice_cols:
        plt.figure(figsize=(10, 5))
        for col in dice_cols:
            plt.plot(hist[col], label=col)
        plt.title(f"Dice Curve - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"31_training_dice_{name}.png")

    iou_cols = [c for c in hist.columns if "iou" in c.lower()]
    if iou_cols:
        plt.figure(figsize=(10, 5))
        for col in iou_cols:
            plt.plot(hist[col], label=col)
        plt.title(f"IoU Curve - {model_name}")
        plt.xlabel("Epoch")
        plt.ylabel("IoU")
        plt.legend()
        plt.grid(alpha=0.3)
        save_current_fig(OUTPUT_DIR / f"32_training_iou_{name}.png")


for model_name, path in HISTORY_FILES.items():
    plot_history(model_name, path)


# ============================================================
# 11. COMBINED TRAINING CURVES
# ============================================================
def combined_curve(metric_type, title, ylabel, filename):
    plt.figure(figsize=(11, 6))
    found = False

    for model_name, path in HISTORY_FILES.items():
        if not path.exists():
            continue

        hist = pd.read_csv(path)

        if metric_type == "loss":
            preferred = ["val_loss", "loss"]
        elif metric_type == "dice":
            preferred = ["val_dice_coef", "dice_coef", "val_dice", "dice"]
        elif metric_type == "iou":
            preferred = ["val_iou_metric", "iou_metric", "val_iou", "iou"]
        else:
            preferred = []

        selected_col = None
        for col in preferred:
            if col in hist.columns:
                selected_col = col
                break

        if selected_col is None:
            candidates = [c for c in hist.columns if metric_type in c.lower()]
            if candidates:
                selected_col = candidates[0]

        if selected_col is not None:
            plt.plot(hist[selected_col].values, label=f"{model_name} ({selected_col})")
            found = True

    if not found:
        plt.close()
        print("No combined data found for:", title)
        return

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(alpha=0.3)
    save_current_fig(OUTPUT_DIR / filename)


combined_curve(
    metric_type="loss",
    title="Combined Validation/Training Loss Curves",
    ylabel="Loss",
    filename="33_combined_loss_curves.png",
)

combined_curve(
    metric_type="dice",
    title="Combined Dice Curves",
    ylabel="Dice",
    filename="34_combined_dice_curves.png",
)

combined_curve(
    metric_type="iou",
    title="Combined IoU Curves",
    ylabel="IoU",
    filename="35_combined_iou_curves.png",
)


# ============================================================
# 12. FINAL INDEX FILE
# ============================================================
index_path = OUTPUT_DIR / "README_generated_graphs.txt"

with open(index_path, "w", encoding="utf-8") as f:
    f.write("Evaluation Graphs Generated\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Output directory:\n{OUTPUT_DIR}\n\n")
    f.write("Models compared:\n")
    for m in models:
        f.write(f"- {m['model']} ({m['run_version']})\n")

    f.write("\nImportant final selected result:\n")
    f.write("UNet++ V26 selected combo only:\n")
    f.write("threshold=0.47, pixel_threshold=1, min_area=360\n")
    f.write("Test Post Dice=0.8101, Test Post IoU=0.6808\n")
    f.write("Image Accuracy=0.9018, Recall=0.9857, F1=0.9094\n\n")

    f.write("Files generated:\n")
    for file in sorted(OUTPUT_DIR.glob("*")):
        f.write(f"- {file.name}\n")

print("Saved:", index_path)


# ============================================================
# 13. FINAL PRINT
# ============================================================
print("\n================================================")
print("DONE - all graphs and tables were generated.")
print("Output folder:")
print(OUTPUT_DIR)
print("================================================")

print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.glob("*")):
    print(file.name)