import os
import json
import cv2
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split


# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# 1. USER SETTINGS
# ============================================================
RUN_VERSION = "v9_roi_polygon_only_160"
IMG_SIZE = 160
BORDER_SUPPRESSION = 8
NUM_DEBUG_SAMPLES = 20

# ROI SETTINGS
ROI_MARGIN = 30               # margin around tumor bbox
NEGATIVE_CROP_RATIO = 0.50    # central crop size for non-tumor images (50% of min(H,W))
MIN_CROP_SIZE = 96            # minimum crop side before resize

# True = reuse split csv files if they already exist in splits/RUN_VERSION
REUSE_EXISTING_SPLITS = False


# ============================================================
# 2. PROJECT PATHS
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

ANNOTATIONS_DIR = os.path.join(PROJECT_DIR, "Annotations")
SAMPLE_DATA_DIR = os.path.join(PROJECT_DIR, "sample_data")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")

ARRAYS_DIR = os.path.join(PROJECT_DIR, "arrays", RUN_VERSION)
SPLITS_DIR = os.path.join(PROJECT_DIR, "splits", RUN_VERSION)
DEBUG_DIR = os.path.join(PROJECT_DIR, "debug_masks", RUN_VERSION)

MASTER_CSV_PATH = os.path.join(DATASETS_DIR, "master_dataset.csv")
DATASET_XLSX_PATH = os.path.join(DATASETS_DIR, "dataset.xlsx")

os.makedirs(ARRAYS_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

print("PROJECT_DIR      :", PROJECT_DIR)
print("ANNOTATIONS_DIR  :", ANNOTATIONS_DIR)
print("SAMPLE_DATA_DIR  :", SAMPLE_DATA_DIR)
print("DATASETS_DIR     :", DATASETS_DIR)
print("ARRAYS_DIR       :", ARRAYS_DIR)
print("SPLITS_DIR       :", SPLITS_DIR)
print("DEBUG_DIR        :", DEBUG_DIR)
print("MASTER_CSV_PATH  :", MASTER_CSV_PATH)
print("DATASET_XLSX_PATH:", DATASET_XLSX_PATH)
print("RUN_VERSION      :", RUN_VERSION)
print("IMG_SIZE         :", IMG_SIZE)
print("ROI_MARGIN       :", ROI_MARGIN)
print("NEGATIVE_CROP_RATIO:", NEGATIVE_CROP_RATIO)
print("MIN_CROP_SIZE    :", MIN_CROP_SIZE)


# ============================================================
# 3. LOAD MASTER DATASET
# ============================================================
df_master = pd.read_csv(MASTER_CSV_PATH)

print("\nOriginal shape:", df_master.shape)
print("Columns:", df_master.columns.tolist())

required_cols = ["image_id", "has_annotation"]
for col in required_cols:
    if col not in df_master.columns:
        raise ValueError(f"Missing required column in master_dataset.csv: {col}")

if "tumor" not in df_master.columns:
    df_master["tumor"] = df_master["has_annotation"].astype(int)

df_master["image_id"] = df_master["image_id"].astype(str).str.strip()
df_master["has_annotation"] = df_master["has_annotation"].astype(int)
df_master["tumor"] = df_master["tumor"].astype(int)

df_master["image_path"] = df_master["image_id"].apply(
    lambda x: os.path.join(SAMPLE_DATA_DIR, f"{x}.jpeg")
)
df_master["annotation_path"] = df_master["image_id"].apply(
    lambda x: os.path.join(ANNOTATIONS_DIR, f"{x}.json")
)

df_all = df_master.copy()

print("\nTumor distribution:")
print(df_all["tumor"].value_counts(dropna=False))

image_exists = df_all["image_path"].apply(os.path.exists)
ann_exists = df_all["annotation_path"].apply(os.path.exists)

print("\nImage files found :", int(image_exists.sum()), "/", len(df_all))
print("Annotation files found :", int(ann_exists.sum()), "/", len(df_all))

print("\nSample path check:")
for i in range(min(5, len(df_all))):
    row = df_all.iloc[i]
    print({
        "image_id": row["image_id"],
        "has_annotation": int(row["has_annotation"]),
        "image_exists": os.path.exists(row["image_path"]),
        "annotation_exists": os.path.exists(row["annotation_path"]),
        "image_path": row["image_path"],
        "annotation_path": row["annotation_path"],
    })


# ============================================================
# 4. TRAIN / VAL / TEST SPLITS
# ============================================================
train_split_path = os.path.join(SPLITS_DIR, "train_split.csv")
val_split_path = os.path.join(SPLITS_DIR, "val_split.csv")
test_split_path = os.path.join(SPLITS_DIR, "test_split.csv")

if (
    REUSE_EXISTING_SPLITS
    and os.path.exists(train_split_path)
    and os.path.exists(val_split_path)
    and os.path.exists(test_split_path)
):
    print("\nLoading existing splits...")
    df_train = pd.read_csv(train_split_path)
    df_val = pd.read_csv(val_split_path)
    df_test = pd.read_csv(test_split_path)
else:
    print("\nCreating fresh splits...")
    df_train, df_temp = train_test_split(
        df_all,
        test_size=0.30,
        random_state=SEED,
        stratify=df_all["tumor"]
    )

    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=df_temp["tumor"]
    )

    df_train.to_csv(train_split_path, index=False)
    df_val.to_csv(val_split_path, index=False)
    df_test.to_csv(test_split_path, index=False)

print("\nSplit sizes:")
print("Train:", len(df_train))
print("Val  :", len(df_val))
print("Test :", len(df_test))

print("\nTrain class balance:")
print(df_train["tumor"].value_counts())
print("\nVal class balance:")
print(df_val["tumor"].value_counts())
print("\nTest class balance:")
print(df_test["tumor"].value_counts())


# ============================================================
# 5. IMAGE / MASK HELPERS
# ============================================================
def suppress_border_markers(img, border=8):
    img = img.copy()
    if border > 0:
        img[:border, :] = 0
        img[-border:, :] = 0
        img[:, :border] = 0
        img[:, -border:] = 0
    return img


def create_empty_mask(height, width):
    return np.zeros((height, width), dtype=np.uint8)


def create_mask_from_json_polygon_only(json_path, height=None, width=None):
    """
    Create mask using ONLY polygon annotations.
    Rectangle annotations are intentionally ignored.
    """
    if json_path is None or pd.isna(json_path) or not os.path.exists(json_path):
        if height is None or width is None:
            raise ValueError("Missing json_path and no height/width provided.")
        return create_empty_mask(height, width)

    with open(json_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    h = ann["imageHeight"]
    w = ann["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in ann.get("shapes", []):
        shape_type = shape.get("shape_type", "")
        points = shape.get("points", [])

        if len(points) == 0:
            continue

        if shape_type == "polygon":
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


def get_bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, y_min, x_max, y_max


def expand_bbox(x_min, y_min, x_max, y_max, img_w, img_h, margin=30, min_crop_size=96):
    x1 = max(0, x_min - margin)
    y1 = max(0, y_min - margin)
    x2 = min(img_w, x_max + margin)
    y2 = min(img_h, y_max + margin)

    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w < min_crop_size:
        pad = (min_crop_size - crop_w) // 2 + 1
        x1 = max(0, x1 - pad)
        x2 = min(img_w, x2 + pad)

    if crop_h < min_crop_size:
        pad = (min_crop_size - crop_h) // 2 + 1
        y1 = max(0, y1 - pad)
        y2 = min(img_h, y2 + pad)

    return x1, y1, x2, y2


def get_center_crop_bbox(img_h, img_w, crop_ratio=0.5, min_crop_size=96):
    crop_size = int(min(img_h, img_w) * crop_ratio)
    crop_size = max(crop_size, min_crop_size)
    crop_size = min(crop_size, min(img_h, img_w))

    cy = img_h // 2
    cx = img_w // 2
    half = crop_size // 2

    y1 = max(0, cy - half)
    y2 = min(img_h, cy + half)
    x1 = max(0, cx - half)
    x2 = min(img_w, cx + half)

    return x1, y1, x2, y2


def crop_image_and_mask(img, mask, has_annotation, roi_margin=30, negative_crop_ratio=0.5, min_crop_size=96):
    img_h, img_w = img.shape[:2]

    if has_annotation == 1:
        bbox = get_bbox_from_mask(mask)
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            x1, y1, x2, y2 = expand_bbox(
                x_min, y_min, x_max, y_max,
                img_w=img_w, img_h=img_h,
                margin=roi_margin,
                min_crop_size=min_crop_size
            )
        else:
            x1, y1, x2, y2 = get_center_crop_bbox(
                img_h, img_w,
                crop_ratio=negative_crop_ratio,
                min_crop_size=min_crop_size
            )
    else:
        x1, y1, x2, y2 = get_center_crop_bbox(
            img_h, img_w,
            crop_ratio=negative_crop_ratio,
            min_crop_size=min_crop_size
        )

    cropped_img = img[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]

    return cropped_img, cropped_mask, (x1, y1, x2, y2)


def load_image_and_mask_polygon_roi(
    image_path,
    json_path=None,
    has_annotation=0,
    img_size=160,
    border=8,
    roi_margin=30,
    negative_crop_ratio=0.5,
    min_crop_size=96
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("L")
    img = np.array(img)

    img = suppress_border_markers(img, border=border)
    orig_h, orig_w = img.shape[:2]

    if has_annotation == 1 and os.path.exists(json_path):
        raw_mask = create_mask_from_json_polygon_only(
            json_path=json_path,
            height=orig_h,
            width=orig_w
        )
    else:
        raw_mask = create_empty_mask(orig_h, orig_w)

    cropped_img, cropped_mask, crop_box = crop_image_and_mask(
        img=img,
        mask=raw_mask,
        has_annotation=has_annotation,
        roi_margin=roi_margin,
        negative_crop_ratio=negative_crop_ratio,
        min_crop_size=min_crop_size
    )

    img_resized = cv2.resize(cropped_img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img_resized = img_resized.astype(np.float32) / 255.0
    img_resized = np.expand_dims(img_resized, axis=-1)

    mask_resized = cv2.resize(cropped_mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    mask_resized = (mask_resized > 0).astype(np.float32)
    mask_resized = np.expand_dims(mask_resized, axis=-1)

    return img_resized, mask_resized, img, raw_mask, cropped_img, cropped_mask, crop_box


# ============================================================
# 6. DEBUG VISUALIZATION
# ============================================================
def save_debug_example(
    image_id,
    raw_img,
    raw_mask,
    cropped_img,
    cropped_mask,
    resized_img,
    resized_mask,
    crop_box,
    save_dir
):
    x1, y1, x2, y2 = crop_box

    overlay = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 3)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(raw_img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(raw_mask, cmap="gray")
    plt.title("Raw Polygon Mask")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(overlay)
    plt.title("ROI Box on Original")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(cropped_img, cmap="gray")
    plt.title("Cropped ROI Image")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(cropped_mask, cmap="gray")
    plt.title("Cropped ROI Mask")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.imshow(resized_img.squeeze(), cmap="gray")
    plt.imshow(resized_mask.squeeze(), cmap="Reds", alpha=0.35)
    plt.title("Final Resized ROI + Mask")
    plt.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{image_id}_debug.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# 7. BUILD ARRAYS
# ============================================================
def build_arrays(df, split_name, img_size=160, border=8, debug_limit=0):
    X, Y = [], []
    valid_rows = []
    errors = []

    debug_saved = 0
    annotation_rows = 0
    non_empty_polygon_masks = 0
    roi_rows = 0

    for i, row in enumerate(df.itertuples(index=False)):
        try:
            image_id = getattr(row, "image_id")
            image_path = getattr(row, "image_path")
            annotation_path = getattr(row, "annotation_path")
            has_annotation = int(getattr(row, "has_annotation"))

            (
                img_resized,
                mask_resized,
                raw_img,
                raw_mask,
                cropped_img,
                cropped_mask,
                crop_box
            ) = load_image_and_mask_polygon_roi(
                image_path=image_path,
                json_path=annotation_path,
                has_annotation=has_annotation,
                img_size=img_size,
                border=border,
                roi_margin=ROI_MARGIN,
                negative_crop_ratio=NEGATIVE_CROP_RATIO,
                min_crop_size=MIN_CROP_SIZE
            )

            X.append(img_resized)
            Y.append(mask_resized)
            valid_rows.append(row)
            roi_rows += 1

            if has_annotation == 1:
                annotation_rows += 1
                if raw_mask.sum() > 0:
                    non_empty_polygon_masks += 1

            if debug_saved < debug_limit:
                save_debug_example(
                    image_id=image_id,
                    raw_img=raw_img,
                    raw_mask=raw_mask,
                    cropped_img=cropped_img,
                    cropped_mask=cropped_mask,
                    resized_img=img_resized,
                    resized_mask=mask_resized,
                    crop_box=crop_box,
                    save_dir=DEBUG_DIR
                )
                debug_saved += 1

        except Exception as e:
            image_id = getattr(row, "image_id", f"row_{i}")
            errors.append((image_id, str(e)))

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    valid_df = pd.DataFrame(valid_rows, columns=df.columns)

    print(f"\n[{split_name}] valid loaded:", len(valid_df))
    print(f"[{split_name}] errors:", len(errors))
    print(f"[{split_name}] ROI cropped rows:", roi_rows)
    print(f"[{split_name}] rows with annotation:", annotation_rows)
    print(f"[{split_name}] rows with non-empty polygon mask:", non_empty_polygon_masks)

    if errors:
        print(f"\n[{split_name}] first 10 errors:")
        for err in errors[:10]:
            print(err)

    return X, Y, valid_df, errors


# ============================================================
# 8. BUILD EVERYTHING
# ============================================================
print("\nBuilding ROI polygon-only arrays...")
build_start = time.time()

X_train, Y_train, df_train_valid, train_errors = build_arrays(
    df_train, "train", IMG_SIZE, BORDER_SUPPRESSION, debug_limit=NUM_DEBUG_SAMPLES
)
X_val, Y_val, df_val_valid, val_errors = build_arrays(
    df_val, "val", IMG_SIZE, BORDER_SUPPRESSION, debug_limit=NUM_DEBUG_SAMPLES
)
X_test, Y_test, df_test_valid, test_errors = build_arrays(
    df_test, "test", IMG_SIZE, BORDER_SUPPRESSION, debug_limit=NUM_DEBUG_SAMPLES
)

build_end = time.time()

print("\nShapes after loading:")
print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val  :", X_val.shape, "Y_val  :", Y_val.shape)
print("X_test :", X_test.shape, "Y_test :", Y_test.shape)

print("\nErrors:")
print("Train:", len(train_errors))
print("Val  :", len(val_errors))
print("Test :", len(test_errors))

print("\nArray build time (sec):", round(build_end - build_start, 2))
print("Array build time (min):", round((build_end - build_start) / 60, 2))

if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
    raise RuntimeError("Array build failed: one or more splits are empty. Check printed errors above.")


# ============================================================
# 9. SAVE ARRAYS + METADATA
# ============================================================
np.save(os.path.join(ARRAYS_DIR, "X_train.npy"), X_train)
np.save(os.path.join(ARRAYS_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(ARRAYS_DIR, "X_val.npy"), X_val)
np.save(os.path.join(ARRAYS_DIR, "Y_val.npy"), Y_val)
np.save(os.path.join(ARRAYS_DIR, "X_test.npy"), X_test)
np.save(os.path.join(ARRAYS_DIR, "Y_test.npy"), Y_test)

df_train_valid.to_csv(os.path.join(ARRAYS_DIR, "df_train_valid.csv"), index=False)
df_val_valid.to_csv(os.path.join(ARRAYS_DIR, "df_val_valid.csv"), index=False)
df_test_valid.to_csv(os.path.join(ARRAYS_DIR, "df_test_valid.csv"), index=False)

pd.DataFrame(train_errors, columns=["image_id", "error"]).to_csv(
    os.path.join(ARRAYS_DIR, "train_errors.csv"), index=False
)
pd.DataFrame(val_errors, columns=["image_id", "error"]).to_csv(
    os.path.join(ARRAYS_DIR, "val_errors.csv"), index=False
)
pd.DataFrame(test_errors, columns=["image_id", "error"]).to_csv(
    os.path.join(ARRAYS_DIR, "test_errors.csv"), index=False
)

summary = {
    "run_version": RUN_VERSION,
    "img_size": IMG_SIZE,
    "border_suppression": BORDER_SUPPRESSION,
    "num_debug_samples": NUM_DEBUG_SAMPLES,
    "reuse_existing_splits": REUSE_EXISTING_SPLITS,
    "roi_margin": ROI_MARGIN,
    "negative_crop_ratio": NEGATIVE_CROP_RATIO,
    "min_crop_size": MIN_CROP_SIZE,
    "train_shape": list(X_train.shape),
    "val_shape": list(X_val.shape),
    "test_shape": list(X_test.shape),
    "train_errors": len(train_errors),
    "val_errors": len(val_errors),
    "test_errors": len(test_errors),
    "build_seconds": round(build_end - build_start, 2),
    "notes": "ROI crop around polygon-only tumor masks; central crop for non-tumor images."
}

with open(os.path.join(ARRAYS_DIR, "build_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("\nSaved arrays and metadata successfully.")
print("Saved build summary:", os.path.join(ARRAYS_DIR, "build_summary.json"))
print("Done.")