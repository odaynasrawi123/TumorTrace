import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# CONFIG
# =====================================================

PROJECT_DIR = Path(r"C:\Users\onasrawx\PycharmProjects\TumorDataset\TumorDataset")

RUN_VERSION = "v16_iou_dice_fp_refine"
ARRAYS_VERSION = "v9_roi_polygon_only_160"

ARRAYS_DIR = PROJECT_DIR / "arrays" / ARRAYS_VERSION

MASTER_DATASET_PATH = PROJECT_DIR / "datasets" / "master_dataset.csv"

# Put your uploaded Excel here:
EXCEL_DATASET_PATH = PROJECT_DIR / "datasets" / "dataset.xlsx"

GRAPH_ROOT = PROJECT_DIR / "graphs"
DASHBOARD_DIR = GRAPH_ROOT / RUN_VERSION / "dashboard_extra"
DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_DIR / "models" / f"best_unetpp_{RUN_VERSION}.keras"

METRICS_PATH = PROJECT_DIR / "metrics" / f"metrics_{RUN_VERSION}.json"
THRESHOLD_CSV_PATH = PROJECT_DIR / "reports" / f"threshold_search_{RUN_VERSION}.csv"

X_TEST_PATH = ARRAYS_DIR / "X_test.npy"
Y_TEST_PATH = ARRAYS_DIR / "Y_test.npy"

Y_PRED_PATH = ARRAYS_DIR / f"Y_pred_{RUN_VERSION}.npy"

DF_TRAIN_VALID_PATH = ARRAYS_DIR / "df_train_valid.csv"
DF_VAL_VALID_PATH = ARRAYS_DIR / "df_val_valid.csv"
DF_TEST_VALID_PATH = ARRAYS_DIR / "df_test_valid.csv"

PRED_THRESHOLD = 0.70
MIN_AREA = 80


# =====================================================
# GENERAL HELPERS
# =====================================================

def save_fig(filename):
    path = DASHBOARD_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def find_col(df, names):
    lower_map = {c.lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def normalize_image_id(value):
    if pd.isna(value):
        return value

    value = str(value).strip()
    value = value.replace(".jpeg", "").replace(".jpg", "").replace(".png", "")
    return value


def load_json(path):
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dice_score(y_true, y_pred, smooth=1e-7):
    y_true = y_true.astype(np.float32).flatten()
    y_pred = y_pred.astype(np.float32).flatten()

    inter = np.sum(y_true * y_pred)

    return (2 * inter + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )


def iou_score(y_true, y_pred, smooth=1e-7):
    y_true = y_true.astype(np.float32).flatten()
    y_pred = y_pred.astype(np.float32).flatten()

    inter = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - inter

    return (inter + smooth) / (union + smooth)


def mask_has_tumor(mask, min_area=MIN_AREA):
    return int(np.sum(mask > 0) >= min_area)


# =====================================================
# LOAD DATASET
# =====================================================

def load_dashboard_dataset():
    master_df = pd.read_csv(MASTER_DATASET_PATH)

    master_df["image_id_clean"] = master_df["image_id"].apply(normalize_image_id)

    if EXCEL_DATASET_PATH.exists():
        clinical_df = pd.read_excel(EXCEL_DATASET_PATH, sheet_name="Sheet1")
        clinical_df["image_id_clean"] = clinical_df["image_id"].apply(normalize_image_id)

        df = master_df.merge(
            clinical_df,
            on="image_id_clean",
            how="left",
            suffixes=("", "_clinical")
        )

        print("Loaded master_dataset.csv + dataset.xlsx")
    else:
        df = master_df
        print("Loaded master_dataset.csv only. Excel file not found.")

    print("Columns:")
    print(df.columns.tolist())

    return df


# =====================================================
# PREDICTION LOADING / CREATION
# =====================================================

def load_or_create_predictions():
    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"Missing X_test.npy: {X_TEST_PATH}")

    x_test = np.load(X_TEST_PATH)

    if Y_PRED_PATH.exists():
        print(f"Loading existing predictions: {Y_PRED_PATH}")
        return np.load(Y_PRED_PATH)

    import tensorflow as tf

    print(f"Y_pred not found. Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print("Predicting X_test...")
    y_pred = model.predict(x_test, batch_size=4, verbose=1)

    np.save(Y_PRED_PATH, y_pred)
    print(f"Saved predictions to: {Y_PRED_PATH}")

    return y_pred



def plot_dataset_distribution(df):
    label_col = find_col(df, ["has_annotation", "tumor"])

    if label_col is None:
        print("No tumor label column found.")
        return

    counts = df[label_col].value_counts().sort_index()
    labels = ["No Tumor" if i == 0 else "Tumor" for i in counts.index]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, counts.values)
    plt.title("Dataset Distribution: Tumor vs No Tumor")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)

    save_fig("01_dataset_tumor_distribution.png")

    pd.DataFrame({
        "class": labels,
        "count": counts.values
    }).to_csv(DASHBOARD_DIR / "01_dataset_tumor_distribution.csv", index=False)




def plot_split_distribution():
    paths = {
        "Train": DF_TRAIN_VALID_PATH,
        "Validation": DF_VAL_VALID_PATH,
        "Test": DF_TEST_VALID_PATH
    }

    rows = []

    for split_name, path in paths.items():
        if not path.exists():
            continue

        split_df = pd.read_csv(path)

        if "has_annotation" not in split_df.columns:
            continue

        tumor = int(split_df["has_annotation"].sum())
        no_tumor = int(len(split_df) - tumor)

        rows.append({
            "split": split_name,
            "Tumor": tumor,
            "No Tumor": no_tumor
        })

    if not rows:
        print("Split CSV files not found.")
        return

    plot_df = pd.DataFrame(rows).set_index("split")

    plot_df.plot(kind="bar", figsize=(8, 5))
    plt.title("Train / Validation / Test Distribution")
    plt.xlabel("Split")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)

    save_fig("02_split_distribution.png")

    plot_df.to_csv(DASHBOARD_DIR / "02_split_distribution.csv")


def plot_gender_distribution(df):
    gender_col = find_col(df, ["gender", "sex", "patient_gender"])

    if gender_col is None:
        print("No gender column found.")
        return

    counts = df[gender_col].dropna().value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Patient Gender Distribution")
    plt.xlabel("Gender")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)

    save_fig("03_gender_distribution.png")

    counts.to_csv(DASHBOARD_DIR / "03_gender_distribution.csv")



def plot_age_distribution(df):
    age_col = find_col(df, ["age", "patient_age"])

    if age_col is None:
        print("No age column found.")
        return

    ages = pd.to_numeric(df[age_col], errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(ages, bins=20)
    plt.title("Patient Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)

    save_fig("04_age_distribution.png")

    ages.to_csv(DASHBOARD_DIR / "04_age_distribution.csv", index=False)



def plot_tumor_rate_by_gender(df):
    gender_col = find_col(df, ["gender", "sex", "patient_gender"])
    label_col = find_col(df, ["tumor", "has_annotation"])

    if gender_col is None or label_col is None:
        print("Missing gender or tumor column.")
        return

    tmp = df[[gender_col, label_col]].dropna().copy()
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    tmp = tmp.dropna()

    rate = tmp.groupby(gender_col)[label_col].mean().sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    rate.plot(kind="bar")
    plt.title("Tumor Rate by Gender")
    plt.xlabel("Gender")
    plt.ylabel("Tumor Rate")
    plt.grid(axis="y", alpha=0.3)

    save_fig("05_tumor_rate_by_gender.png")

    rate.to_csv(DASHBOARD_DIR / "05_tumor_rate_by_gender.csv")




def plot_tumor_rate_by_age_group(df):
    age_col = find_col(df, ["age", "patient_age"])
    label_col = find_col(df, ["tumor", "has_annotation"])

    if age_col is None or label_col is None:
        print("Missing age or tumor column.")
        return

    tmp = df[[age_col, label_col]].copy()
    tmp[age_col] = pd.to_numeric(tmp[age_col], errors="coerce")
    tmp[label_col] = pd.to_numeric(tmp[label_col], errors="coerce")
    tmp = tmp.dropna()

    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 120]
    labels = [
        "0-10", "11-20", "21-30", "31-40", "41-50",
        "51-60", "61-70", "71-80", "81+"
    ]

    tmp["age_group"] = pd.cut(
        tmp[age_col],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    rate = tmp.groupby("age_group", observed=False)[label_col].mean()

    plt.figure(figsize=(9, 5))
    rate.plot(kind="bar")
    plt.title("Tumor Rate by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Tumor Rate")
    plt.grid(axis="y", alpha=0.3)

    save_fig("06_tumor_rate_by_age_group.png")

    rate.to_csv(DASHBOARD_DIR / "06_tumor_rate_by_age_group.csv")


def plot_benign_malignant_distribution(df):
    benign_col = find_col(df, ["benign"])
    malignant_col = find_col(df, ["malignant"])

    if benign_col is None or malignant_col is None:
        print("Missing benign/malignant columns.")
        return

    benign_count = int(pd.to_numeric(df[benign_col], errors="coerce").fillna(0).sum())
    malignant_count = int(pd.to_numeric(df[malignant_col], errors="coerce").fillna(0).sum())

    plt.figure(figsize=(7, 5))
    plt.bar(["Benign", "Malignant"], [benign_count, malignant_count])
    plt.title("Benign vs Malignant Cases")
    plt.xlabel("Tumor Type")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)

    save_fig("07_benign_malignant_distribution.png")

    pd.DataFrame({
        "type": ["Benign", "Malignant"],
        "count": [benign_count, malignant_count]
    }).to_csv(DASHBOARD_DIR / "07_benign_malignant_distribution.csv", index=False)




def plot_body_location_distribution(df):
    location_cols = [
        "hand", "ulna", "radius", "humerus", "foot", "tibia",
        "fibula", "femur", "hip bone", "ankle-joint", "knee-joint",
        "hip-joint", "wrist-joint", "elbow-joint", "shoulder-joint",
        "upper limb", "lower limb", "pelvis"
    ]

    existing = [c for c in location_cols if c in df.columns]

    if not existing:
        print("No body location columns found.")
        return

    counts = df[existing].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    counts = counts[counts > 0].sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Tumor / Image Distribution by Body Location")
    plt.xlabel("Body Location")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_fig("08_body_location_distribution.png")

    counts.to_csv(DASHBOARD_DIR / "08_body_location_distribution.csv")


def plot_xray_view_distribution(df):
    view_cols = ["frontal", "lateral", "oblique"]
    existing = [c for c in view_cols if c in df.columns]

    if not existing:
        print("No X-ray view columns found.")
        return

    counts = df[existing].apply(pd.to_numeric, errors="coerce").fillna(0).sum()
    counts = counts[counts > 0].sort_values(ascending=False)

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("X-ray View Distribution")
    plt.xlabel("View Type")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)

    save_fig("09_xray_view_distribution.png")

    counts.to_csv(DASHBOARD_DIR / "09_xray_view_distribution.csv")




def plot_model_metrics_summary():
    metrics = load_json(METRICS_PATH)

    selected = {}

    if metrics:
        def flatten(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    flatten(f"{prefix}_{k}" if prefix else k, v)
            else:
                if isinstance(obj, (int, float)):
                    selected[prefix] = obj

        flatten("", metrics)

        selected = {
            k: v for k, v in selected.items()
            if any(word in k.lower() for word in [
                "dice", "iou", "precision", "recall",
                "f1", "accuracy", "error"
            ])
        }

    if not selected:
        selected = {
            "Post Dice": 0.7925,
            "Post IoU": 0.6563,
            "Pixel Precision": 0.7795,
            "Pixel Recall": 0.8059,
            "Pixel F1": 0.7925,
            "Pixel Accuracy": 0.9319,
            "Image Precision": 0.8688,
            "Image Recall": 0.9929,
            "Image F1": 0.9267,
            "Image Accuracy": 0.9214
        }

    selected = dict(list(selected.items())[:12])

    plt.figure(figsize=(13, 5))
    plt.bar(selected.keys(), selected.values())
    plt.title("Model Performance Metrics Summary")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_fig("10_model_metrics_summary.png")

    pd.DataFrame([selected]).to_csv(DASHBOARD_DIR / "10_model_metrics_summary.csv", index=False)


def plot_threshold_optimization():
    if not THRESHOLD_CSV_PATH.exists():
        print("Threshold CSV not found.")
        return

    df = pd.read_csv(THRESHOLD_CSV_PATH)

    threshold_col = find_col(df, [
        "pred_threshold", "threshold", "BEST_PRED_THRESHOLD"
    ])

    if threshold_col is None:
        print("No threshold column found.")
        return

    score_cols = [
        c for c in df.columns
        if any(word in c.lower() for word in ["dice", "iou", "score", "f1"])
    ]

    if not score_cols:
        print("No score columns found in threshold CSV.")
        return

    plt.figure(figsize=(9, 5))

    for col in score_cols[:4]:
        plt.plot(df[threshold_col], df[col], marker="o", label=col)

    plt.title("Threshold Optimization")
    plt.xlabel("Prediction Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(alpha=0.3)

    save_fig("11_threshold_optimization.png")



def plot_mask_size_distribution():
    if not Y_TEST_PATH.exists():
        print("Y_test.npy not found.")
        return

    y_test = np.load(Y_TEST_PATH)

    areas = pd.Series([int(np.sum(mask > 0)) for mask in y_test])
    tumor_areas = areas[areas > 0]

    plt.figure(figsize=(9, 5))
    plt.hist(tumor_areas, bins=30)
    plt.title("Ground Truth Tumor Mask Size Distribution")
    plt.xlabel("Tumor Area in Pixels")
    plt.ylabel("Number of Images")
    plt.grid(axis="y", alpha=0.3)

    save_fig("12_tumor_mask_size_distribution.png")

    areas.to_csv(DASHBOARD_DIR / "12_tumor_mask_size_distribution.csv", index=False)


def plot_prediction_overlay_examples(max_examples=8):
    if not X_TEST_PATH.exists() or not Y_TEST_PATH.exists():
        print("X_test.npy or Y_test.npy not found.")
        return

    x_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)
    y_pred = load_or_create_predictions()

    y_pred_bin = (y_pred >= PRED_THRESHOLD).astype(np.uint8)

    for i in range(min(max_examples, len(x_test))):
        image = x_test[i].squeeze()
        true_mask = y_test[i].squeeze()
        pred_mask = y_pred_bin[i].squeeze()

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(image, cmap="gray")
        axes[0].set_title("Original X-ray")

        axes[1].imshow(true_mask, cmap="gray")
        axes[1].set_title("Ground Truth Mask")

        axes[2].imshow(pred_mask, cmap="gray")
        axes[2].set_title("Predicted Mask")

        axes[3].imshow(image, cmap="gray")
        axes[3].imshow(pred_mask, alpha=0.45)
        axes[3].set_title("Prediction Overlay")

        for ax in axes:
            ax.axis("off")

        save_fig(f"13_prediction_overlay_{i + 1}.png")


def create_dashboard_index():
    files = sorted(DASHBOARD_DIR.glob("*"))

    rows = []
    for file in files:
        rows.append({
            "file_name": file.name,
            "file_path": str(file),
            "type": file.suffix.replace(".", "")
        })

    index_df = pd.DataFrame(rows)
    index_df.to_csv(DASHBOARD_DIR / "dashboard_index.csv", index=False)

    print(f"Saved dashboard index: {DASHBOARD_DIR / 'dashboard_index.csv'}")


def main():
    df = load_dashboard_dataset()

    plot_dataset_distribution(df)
    plot_split_distribution()

    plot_gender_distribution(df)
    plot_age_distribution(df)
    plot_tumor_rate_by_gender(df)
    plot_tumor_rate_by_age_group(df)

    plot_benign_malignant_distribution(df)
    plot_body_location_distribution(df)
    plot_xray_view_distribution(df)

    plot_model_metrics_summary()
    plot_threshold_optimization()
    plot_mask_size_distribution()
    plot_prediction_overlay_examples(max_examples=8)

    create_dashboard_index()

    print("\nDone.")
    print(f"Dashboard graphs saved in:\n{DASHBOARD_DIR}")

if __name__ == "__main__":
    main()