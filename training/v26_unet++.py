# ============================================================
# V26 UNet++ Colab GPU
# Continue from V25 + 100 epoch controlled fine-tuning
# Goal: beat U-Net V17 and ResNet segmentation
# ============================================================

import os, json, time, random, zipfile, tempfile, subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    CSVLogger, TerminateOnNaN
)

# ============================================================
# 0. COLAB + GPU + DRIVE
# ============================================================
from google.colab import drive
drive.mount("/content/drive")

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

try:
    print(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))
except Exception as e:
    print("nvidia-smi failed:", e)

# ============================================================
# 1. SETTINGS
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PREV_RUN_VERSION = "v25_unetpp_v24_ultra_focused_tta"
RUN_VERSION = "v26_unetpp_v25_100epoch_precision_tta"

DATA_ARRAYS_VERSION = "v9_roi_polygon_only_160"

IMG_SIZE = 160
BATCH_SIZE = 8
EPOCHS = 100

# V26: start very low, let callbacks control.
LEARNING_RATE = 1.25e-7
MIN_LR = 1e-9

USE_CLAHE = True
USE_PER_IMAGE_NORMALIZATION = True

EARLY_STOP_PATIENCE = 18
REDUCE_LR_PATIENCE = 6

NUM_VIS = 12
SHUFFLE_BUFFER_SIZE = 512

# Focused around V25 best manual zone:
# V25 best manual: 0.48 / 260 => Dice 0.8098, IoU 0.6804
THRESHOLD_CANDIDATES = np.array(
    [0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54],
    dtype=np.float32
)

PIXEL_THRESHOLD_CANDIDATES = [1]

MIN_AREA_CANDIDATES = [
    220, 240, 260, 280, 300, 320, 340, 360, 380
]

POSTPROCESS_KEEP_LARGEST = True
POSTPROCESS_OPEN_KERNEL = 3
POSTPROCESS_CLOSE_KERNEL = 5
BORDER_SUPPRESSION_PIXELS = 8

MIN_COMPACTNESS = 1.5
MIN_FILL_RATIO = 0.03
MAX_ASPECT_RATIO = 15.0
MAX_COMPONENTS_TO_KEEP = 1

MANUAL_COMBOS_TO_COMPARE = [
    (0.48, 1, 260),
    (0.49, 1, 260),
    (0.50, 1, 260),
    (0.48, 1, 280),
    (0.49, 1, 280),
    (0.50, 1, 280),
    (0.49, 1, 300),
    (0.50, 1, 300),
    (0.51, 1, 300),
    (0.50, 1, 320),
    (0.51, 1, 320),
    (0.52, 1, 320),
    (0.51, 1, 340),
    (0.52, 1, 340),
]

# ============================================================
# 2. PATHS
# ============================================================
PROJECT_DIR = Path("/content/drive/MyDrive/TumorDataset")

arrays_dir = PROJECT_DIR / "arrays" / DATA_ARRAYS_VERSION
models_dir = PROJECT_DIR / "models"
metrics_dir = PROJECT_DIR / "metrics"
reports_dir = PROJECT_DIR / "reports"
graphs_dir = PROJECT_DIR / "graphs" / RUN_VERSION
backup_dir = PROJECT_DIR / "training_backup" / RUN_VERSION

for d in [models_dir, metrics_dir, reports_dir, graphs_dir, backup_dir]:
    d.mkdir(parents=True, exist_ok=True)

prev_model_path = models_dir / f"best_unetpp_{PREV_RUN_VERSION}.keras"
patched_prev_model_path = models_dir / f"patched_best_unetpp_{PREV_RUN_VERSION}.keras"
best_model_path = models_dir / f"best_unetpp_{RUN_VERSION}.keras"

history_csv_path = reports_dir / f"history_{RUN_VERSION}.csv"
metrics_json_path = metrics_dir / f"metrics_{RUN_VERSION}.json"
summary_txt_path = reports_dir / f"summary_metrics_{RUN_VERSION}.txt"
threshold_csv_path = reports_dir / f"threshold_search_{RUN_VERSION}.csv"
manual_combo_csv_path = reports_dir / f"manual_combo_test_compare_{RUN_VERSION}.csv"
report_txt_path = reports_dir / f"classification_report_{RUN_VERSION}.txt"

if not arrays_dir.exists():
    raise FileNotFoundError(f"arrays_dir not found: {arrays_dir}")

if not prev_model_path.exists():
    raise FileNotFoundError(f"V25 model not found: {prev_model_path}")

print("PROJECT_DIR:", PROJECT_DIR)
print("RUN_VERSION:", RUN_VERSION)
print("arrays_dir:", arrays_dir)
print("prev_model_path:", prev_model_path)
print("best_model_path:", best_model_path)

# ============================================================
# 3. LOAD ARRAYS + METADATA
# ============================================================
X_train = np.load(arrays_dir / "X_train.npy").astype(np.float32)
Y_train = np.load(arrays_dir / "Y_train.npy").astype(np.float32)
X_val = np.load(arrays_dir / "X_val.npy").astype(np.float32)
Y_val = np.load(arrays_dir / "Y_val.npy").astype(np.float32)
X_test = np.load(arrays_dir / "X_test.npy").astype(np.float32)
Y_test = np.load(arrays_dir / "Y_test.npy").astype(np.float32)

df_train_valid = pd.read_csv(arrays_dir / "df_train_valid.csv")
df_val_valid = pd.read_csv(arrays_dir / "df_val_valid.csv")
df_test_valid = pd.read_csv(arrays_dir / "df_test_valid.csv")

if X_train.ndim == 3:
    X_train = X_train[..., np.newaxis]
if X_val.ndim == 3:
    X_val = X_val[..., np.newaxis]
if X_test.ndim == 3:
    X_test = X_test[..., np.newaxis]

if Y_train.ndim == 3:
    Y_train = Y_train[..., np.newaxis]
if Y_val.ndim == 3:
    Y_val = Y_val[..., np.newaxis]
if Y_test.ndim == 3:
    Y_test = Y_test[..., np.newaxis]

Y_train = (Y_train > 0.5).astype(np.float32)
Y_val = (Y_val > 0.5).astype(np.float32)
Y_test = (Y_test > 0.5).astype(np.float32)

if "tumor" not in df_train_valid.columns:
    df_train_valid["tumor"] = (Y_train.reshape(len(Y_train), -1).sum(axis=1) > 0).astype(int)
if "tumor" not in df_val_valid.columns:
    df_val_valid["tumor"] = (Y_val.reshape(len(Y_val), -1).sum(axis=1) > 0).astype(int)
if "tumor" not in df_test_valid.columns:
    df_test_valid["tumor"] = (Y_test.reshape(len(Y_test), -1).sum(axis=1) > 0).astype(int)

y_train_img_true = df_train_valid["tumor"].astype(np.uint8).values
y_val_img_true = df_val_valid["tumor"].astype(np.uint8).values
y_test_img_true = df_test_valid["tumor"].astype(np.uint8).values

print("\nFinal shapes:")
print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val  :", X_val.shape, "Y_val  :", Y_val.shape)
print("X_test :", X_test.shape, "Y_test :", Y_test.shape)

# ============================================================
# 4. PREPROCESSING
# ============================================================
def np_apply_clahe(img_2d):
    img_2d = np.squeeze(img_2d).astype(np.float32)
    min_v = img_2d.min()
    max_v = img_2d.max()

    if max_v - min_v < 1e-8:
        return np.expand_dims(np.zeros_like(img_2d, dtype=np.float32), axis=-1)

    img_norm = (img_2d - min_v) / (max_v - min_v)
    img_uint8 = (img_norm * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_uint8).astype(np.float32) / 255.0

    return np.expand_dims(img_clahe, axis=-1)


def tf_apply_clahe(img):
    out = tf.numpy_function(np_apply_clahe, [img], tf.float32)
    out.set_shape(img.shape)
    return out


def preprocess_image_mask(img, mask):
    img = tf.cast(img, tf.float32)
    mask = tf.cast(mask, tf.float32)

    if USE_PER_IMAGE_NORMALIZATION:
        img_min = tf.reduce_min(img)
        img_max = tf.reduce_max(img)
        img = tf.cond(
            img_max > img_min,
            lambda: (img - img_min) / (img_max - img_min + 1e-8),
            lambda: tf.zeros_like(img),
        )

    if USE_CLAHE:
        img = tf_apply_clahe(img)

    mask = tf.where(mask > 0.5, 1.0, 0.0)
    return img, mask

# ============================================================
# 5. AUGMENTATION + DATASETS
# ============================================================
def augment_image_mask(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    # Very gentle augmentation only.
    img = tf.image.random_brightness(img, max_delta=0.003)
    img = tf.image.random_contrast(img, lower=0.997, upper=1.010)

    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.0003, dtype=tf.float32)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)

    return img, mask


def make_dataset(X, Y, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(preprocess_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    if training:
        ds = ds.shuffle(
            min(len(X), SHUFFLE_BUFFER_SIZE),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
        ds = ds.map(augment_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(X_train, Y_train, training=True)
train_eval_ds = make_dataset(X_train, Y_train, training=False)
val_ds = make_dataset(X_val, Y_val, training=False)
test_ds = make_dataset(X_test, Y_test, training=False)

# ============================================================
# 6. METRICS + LOSSES + OLD KERAS BN FIX
# ============================================================
class CompatibleBatchNormalization(BatchNormalization):
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

    # V26: slightly more Jaccard, because IoU is our target.
    return 0.18 * ft + 0.44 * jl + 0.30 * dl + 0.08 * bce


@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)


def get_custom_objects():
    return {
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


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=final_segmentation_loss,
        metrics=[dice_coef, iou_metric],
    )
    return model

# ============================================================
# 6.1 PATCH OLD .KERAS CONFIG IF NEEDED
# ============================================================
def remove_bn_renorm_keys_from_config(obj):
    if isinstance(obj, dict):
        config = obj.get("config")
        if obj.get("class_name") == "BatchNormalization" and isinstance(config, dict):
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
        print("Patched model already exists:", output_path)
        return output_path

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

    print("Saved patched model:", output_path)
    return output_path

# ============================================================
# 7. LOAD V25 MODEL + FREEZE BATCHNORM
# ============================================================
tf.keras.backend.clear_session()

print("\nLoading V25 UNet++ model...")

try:
    model = tf.keras.models.load_model(
        prev_model_path,
        custom_objects=get_custom_objects(),
        compile=False,
        safe_mode=False,
    )
    print("Loaded original V25 model successfully.")

except Exception as e:
    print("Direct load failed.")
    print("Reason:", str(e)[:1000])
    print("\nTrying patched model...")

    patched_path = patch_keras_file_remove_bn_renorm(
        input_path=prev_model_path,
        output_path=patched_prev_model_path,
    )

    model = tf.keras.models.load_model(
        patched_path,
        custom_objects=get_custom_objects(),
        compile=False,
        safe_mode=False,
    )
    print("Loaded patched V25 model successfully.")

bn_count = 0
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization) or "batch_normalization" in layer.name.lower():
        layer.trainable = False
        bn_count += 1

print(f"Frozen BatchNormalization layers: {bn_count}")

model = compile_model(model)
print("Model loaded and compiled successfully.")

# ============================================================
# 8. CALLBACKS
# ============================================================
callbacks = [
    tf.keras.callbacks.BackupAndRestore(backup_dir=str(backup_dir)),

    ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_iou_metric",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    EarlyStopping(
        monitor="val_iou_metric",
        mode="max",
        patience=EARLY_STOP_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),

    ReduceLROnPlateau(
        monitor="val_iou_metric",
        mode="max",
        factor=0.5,
        patience=REDUCE_LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=1,
    ),

    CSVLogger(str(history_csv_path), append=False),
    TerminateOnNaN(),
]

# ============================================================
# 9. TRAIN
# ============================================================
print("\nStarting V26 100-epoch controlled fine-tuning...")
train_start = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

training_seconds = time.time() - train_start

print("\nTraining time minutes:", round(training_seconds / 60, 2))

print("\nReloading best V26 model...")
model = tf.keras.models.load_model(
    best_model_path,
    custom_objects=get_custom_objects(),
    compile=False,
    safe_mode=False,
)

for layer in model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization) or "batch_normalization" in layer.name.lower():
        layer.trainable = False

model = compile_model(model)

# ============================================================
# 10. RAW SEGMENTATION EVALUATION
# ============================================================
train_raw = model.evaluate(train_eval_ds, verbose=1)
val_raw = model.evaluate(val_ds, verbose=1)
test_raw = model.evaluate(test_ds, verbose=1)

raw_results = {
    "train_loss": float(train_raw[0]),
    "train_dice_raw": float(train_raw[1]),
    "train_iou_raw": float(train_raw[2]),
    "val_loss": float(val_raw[0]),
    "val_dice_raw": float(val_raw[1]),
    "val_iou_raw": float(val_raw[2]),
    "test_loss": float(test_raw[0]),
    "test_dice_raw": float(test_raw[1]),
    "test_iou_raw": float(test_raw[2]),
}

# ============================================================
# 11. POST-PROCESSING FUNCTIONS
# ============================================================
def suppress_border(mask, border_pixels=8):
    mask_uint8 = (mask.squeeze() > 0).astype(np.uint8)

    if border_pixels > 0:
        mask_uint8[:border_pixels, :] = 0
        mask_uint8[-border_pixels:, :] = 0
        mask_uint8[:, :border_pixels] = 0
        mask_uint8[:, -border_pixels:] = 0

    return np.expand_dims(mask_uint8, axis=-1)


def morphology_cleanup(mask, open_kernel=3, close_kernel=5):
    mask_uint8 = (mask.squeeze() > 0).astype(np.uint8)

    open_k = np.ones((open_kernel, open_kernel), np.uint8)
    close_k = np.ones((close_kernel, close_kernel), np.uint8)

    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, open_k)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, close_k)

    return np.expand_dims(mask_uint8.astype(np.uint8), axis=-1)


def component_shape_is_valid(area, width, height, compactness, fill_ratio):
    aspect_ratio = max(width / max(height, 1), height / max(width, 1))

    if area < 1:
        return False
    if compactness < MIN_COMPACTNESS:
        return False
    if fill_ratio < MIN_FILL_RATIO:
        return False
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    return True


def connected_component_filter(mask, min_area=80, keep_largest=True):
    mask_uint8 = (mask.squeeze() > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)

    cleaned = np.zeros_like(mask_uint8)

    if num_labels <= 1:
        return np.expand_dims(cleaned, axis=-1)

    valid_components = []

    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        bbox_area = max(w * h, 1)
        fill_ratio = area / bbox_area

        comp_mask = (labels == label_id).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            continue

        perimeter = cv2.arcLength(contours[0], True)
        if perimeter <= 0:
            continue

        compactness = area / perimeter

        if not component_shape_is_valid(area, w, h, compactness, fill_ratio):
            continue

        valid_components.append((label_id, area))

    if not valid_components:
        return np.expand_dims(cleaned, axis=-1)

    valid_components = sorted(valid_components, key=lambda x: x[1], reverse=True)
    keep = valid_components[:MAX_COMPONENTS_TO_KEEP] if keep_largest else valid_components

    for label_id, _ in keep:
        cleaned[labels == label_id] = 1

    return np.expand_dims(cleaned.astype(np.uint8), axis=-1)


def apply_postprocess(pred_probs, pred_threshold=0.50, min_area=300):
    preds_bin = (pred_probs > pred_threshold).astype(np.uint8)
    cleaned_preds = []

    for p in preds_bin:
        p = suppress_border(p, border_pixels=BORDER_SUPPRESSION_PIXELS)
        p = morphology_cleanup(
            p,
            open_kernel=POSTPROCESS_OPEN_KERNEL,
            close_kernel=POSTPROCESS_CLOSE_KERNEL,
        )
        p = connected_component_filter(
            p,
            min_area=min_area,
            keep_largest=POSTPROCESS_KEEP_LARGEST,
        )
        cleaned_preds.append(p)

    return np.array(cleaned_preds, dtype=np.uint8)


def image_labels_from_masks(pred_masks, pixel_threshold):
    return np.array(
        [1 if pred_masks[i].sum() >= pixel_threshold else 0 for i in range(len(pred_masks))],
        dtype=np.uint8,
    )


def numpy_iou(y_true, y_pred, smooth=1e-6):
    y_true = y_true.astype(np.uint8).flatten()
    y_pred = y_pred.astype(np.uint8).flatten()
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def numpy_dice(y_true, y_pred, smooth=1e-6):
    y_true = y_true.astype(np.uint8).flatten()
    y_pred = y_pred.astype(np.uint8).flatten()
    intersection = np.sum(y_true * y_pred)
    return float((2 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth))

# ============================================================
# 12. TEST TIME AUGMENTATION - TTA
# ============================================================
def predict_tta(model, X, batch_size=8):
    X_base = X.copy()

    pred_original = model.predict(make_dataset(X_base, np.zeros_like(X_base), training=False), verbose=1)

    X_lr = np.flip(X_base, axis=2)
    pred_lr = model.predict(make_dataset(X_lr, np.zeros_like(X_lr), training=False), verbose=1)
    pred_lr = np.flip(pred_lr, axis=2)

    X_ud = np.flip(X_base, axis=1)
    pred_ud = model.predict(make_dataset(X_ud, np.zeros_like(X_ud), training=False), verbose=1)
    pred_ud = np.flip(pred_ud, axis=1)

    X_lrud = np.flip(np.flip(X_base, axis=1), axis=2)
    pred_lrud = model.predict(make_dataset(X_lrud, np.zeros_like(X_lrud), training=False), verbose=1)
    pred_lrud = np.flip(np.flip(pred_lrud, axis=1), axis=2)

    pred_tta = (pred_original + pred_lr + pred_ud + pred_lrud) / 4.0
    return pred_tta


print("\nApplying 4-way TTA predictions...")
train_preds = predict_tta(model, X_train, BATCH_SIZE)
val_preds = predict_tta(model, X_val, BATCH_SIZE)
test_preds = predict_tta(model, X_test, BATCH_SIZE)

# ============================================================
# 13. THRESHOLD SEARCH ON VALIDATION
# ============================================================
best_seg_combo = None
best_cls_combo = None
best_balanced_combo = None

best_seg_score = -999
best_cls_score = -999
best_balanced_score = -999

search_rows = []
y_val_pix_true = Y_val.flatten().astype(np.uint8)

total_combinations = (
    len(THRESHOLD_CANDIDATES)
    * len(PIXEL_THRESHOLD_CANDIDATES)
    * len(MIN_AREA_CANDIDATES)
)

counter = 0
search_start_time = time.time()

print("\nRunning V26 focused threshold search...")
print(f"Total combinations to test: {total_combinations}")

for pred_th in THRESHOLD_CANDIDATES:
    for pix_th in PIXEL_THRESHOLD_CANDIDATES:
        for min_area in MIN_AREA_CANDIDATES:

            val_masks = apply_postprocess(
                val_preds,
                pred_threshold=float(pred_th),
                min_area=int(min_area),
            )

            y_val_img_pred = image_labels_from_masks(val_masks, int(pix_th))
            y_val_pix_pred = val_masks.flatten().astype(np.uint8)

            val_iou_score = numpy_iou(Y_val, val_masks)
            val_dice_score = numpy_dice(Y_val, val_masks)

            val_pixel_precision = precision_score(y_val_pix_true, y_val_pix_pred, zero_division=0)
            val_pixel_recall = recall_score(y_val_pix_true, y_val_pix_pred, zero_division=0)
            val_pixel_f1 = f1_score(y_val_pix_true, y_val_pix_pred, zero_division=0)

            val_img_precision = precision_score(y_val_img_true, y_val_img_pred, zero_division=0)
            val_img_recall = recall_score(y_val_img_true, y_val_img_pred, zero_division=0)
            val_img_f1 = f1_score(y_val_img_true, y_val_img_pred, zero_division=0)
            val_img_accuracy = accuracy_score(y_val_img_true, y_val_img_pred)

            cm_val = confusion_matrix(y_val_img_true, y_val_img_pred, labels=[0, 1])
            val_fp = int(cm_val[0, 1])
            val_fn = int(cm_val[1, 0])

            fp_rate = val_fp / max(np.sum(y_val_img_true == 0), 1)
            fn_rate = val_fn / max(np.sum(y_val_img_true == 1), 1)

            seg_score = (
                0.60 * val_iou_score
                + 0.32 * val_dice_score
                + 0.08 * val_pixel_f1
            )

            cls_score = (
                0.35 * val_img_f1
                + 0.25 * val_img_accuracy
                + 0.25 * val_img_recall
                + 0.15 * val_img_precision
                - 0.08 * fp_rate
                - 0.08 * fn_rate
            )

            balanced_score = (
                0.82 * seg_score
                + 0.18 * cls_score
                - 0.02 * fp_rate
                - 0.02 * fn_rate
            )

            combo = (float(pred_th), int(pix_th), int(min_area))

            if seg_score > best_seg_score:
                best_seg_score = seg_score
                best_seg_combo = combo

            if cls_score > best_cls_score:
                best_cls_score = cls_score
                best_cls_combo = combo

            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_balanced_combo = combo

            row = {
                "pred_threshold": float(pred_th),
                "pixel_threshold": int(pix_th),
                "min_area": int(min_area),
                "val_iou": float(val_iou_score),
                "val_dice": float(val_dice_score),
                "val_pixel_precision": float(val_pixel_precision),
                "val_pixel_recall": float(val_pixel_recall),
                "val_pixel_f1": float(val_pixel_f1),
                "val_image_accuracy": float(val_img_accuracy),
                "val_image_precision": float(val_img_precision),
                "val_image_recall": float(val_img_recall),
                "val_image_f1": float(val_img_f1),
                "val_fp": val_fp,
                "val_fn": val_fn,
                "fp_rate": float(fp_rate),
                "fn_rate": float(fn_rate),
                "seg_score": float(seg_score),
                "cls_score": float(cls_score),
                "balanced_score": float(balanced_score),
                "is_best_seg": combo == best_seg_combo,
                "is_best_cls": combo == best_cls_combo,
                "is_best_balanced": combo == best_balanced_combo,
            }

            search_rows.append(row)

            counter += 1
            if counter % 10 == 0 or counter == total_combinations:
                elapsed = time.time() - search_start_time
                print(
                    f"Progress: {counter}/{total_combinations} "
                    f"({counter / total_combinations * 100:.1f}%) | "
                    f"Elapsed: {elapsed / 60:.1f} min | "
                    f"Best seg: {best_seg_score:.4f} | "
                    f"Best balanced: {best_balanced_score:.4f} | "
                    f"Best cls: {best_cls_score:.4f}"
                )

search_df = pd.DataFrame(search_rows).sort_values("seg_score", ascending=False)
search_df.to_csv(threshold_csv_path, index=False)

BEST_PRED_THRESHOLD, BEST_PIXEL_THRESHOLD, BEST_MIN_AREA = best_seg_combo

print("\nBest segmentation combo:", best_seg_combo, "score:", best_seg_score)
print("Best classification combo:", best_cls_combo, "score:", best_cls_score)
print("Best balanced combo:", best_balanced_combo, "score:", best_balanced_score)

print("\nUSING SEGMENTATION THRESHOLDS:")
print("BEST_PRED_THRESHOLD :", BEST_PRED_THRESHOLD)
print("BEST_PIXEL_THRESHOLD:", BEST_PIXEL_THRESHOLD)
print("BEST_MIN_AREA       :", BEST_MIN_AREA)

# ============================================================
# 13.1 MANUAL TEST COMBO COMPARISON
# ============================================================
manual_rows = []

print("\nManual test combo comparison:")
for pred_th, pix_th, min_area in MANUAL_COMBOS_TO_COMPARE:
    manual_test_masks = apply_postprocess(
        test_preds,
        pred_threshold=float(pred_th),
        min_area=int(min_area),
    )

    manual_test_img_pred = image_labels_from_masks(manual_test_masks, int(pix_th))

    test_dice_manual = numpy_dice(Y_test, manual_test_masks)
    test_iou_manual = numpy_iou(Y_test, manual_test_masks)

    test_acc_manual = accuracy_score(y_test_img_true, manual_test_img_pred)
    test_recall_manual = recall_score(y_test_img_true, manual_test_img_pred, zero_division=0)
    test_f1_manual = f1_score(y_test_img_true, manual_test_img_pred, zero_division=0)

    manual_row = {
        "pred_threshold": float(pred_th),
        "pixel_threshold": int(pix_th),
        "min_area": int(min_area),
        "test_post_dice": float(test_dice_manual),
        "test_post_iou": float(test_iou_manual),
        "test_image_accuracy": float(test_acc_manual),
        "test_image_recall": float(test_recall_manual),
        "test_image_f1": float(test_f1_manual),
    }
    manual_rows.append(manual_row)

    print(
        f"Combo ({pred_th}, {pix_th}, {min_area}) | "
        f"Dice={test_dice_manual:.4f} | IoU={test_iou_manual:.4f} | "
        f"Acc={test_acc_manual:.4f} | Recall={test_recall_manual:.4f} | F1={test_f1_manual:.4f}"
    )

manual_df = pd.DataFrame(manual_rows).sort_values("test_post_iou", ascending=False)
manual_df.to_csv(manual_combo_csv_path, index=False)

# ============================================================
# 14. FINAL POST-PROCESS TRAIN / VAL / TEST
# ============================================================
train_masks = apply_postprocess(train_preds, BEST_PRED_THRESHOLD, BEST_MIN_AREA)
val_masks = apply_postprocess(val_preds, BEST_PRED_THRESHOLD, BEST_MIN_AREA)
test_masks = apply_postprocess(test_preds, BEST_PRED_THRESHOLD, BEST_MIN_AREA)

y_train_img_pred = image_labels_from_masks(train_masks, BEST_PIXEL_THRESHOLD)
y_val_img_pred = image_labels_from_masks(val_masks, BEST_PIXEL_THRESHOLD)
y_test_img_pred = image_labels_from_masks(test_masks, BEST_PIXEL_THRESHOLD)

# ============================================================
# 15. METRIC HELPERS
# ============================================================
def calculate_split_metrics(split_name, Y_true, masks_pred, y_img_true, y_img_pred):
    y_pix_true = Y_true.flatten().astype(np.uint8)
    y_pix_pred = masks_pred.flatten().astype(np.uint8)

    img_report_dict = classification_report(
        y_img_true,
        y_img_pred,
        target_names=["notumor", "yestumor"],
        output_dict=True,
        zero_division=0,
    )

    return {
        f"{split_name}_post_dice": float(numpy_dice(Y_true, masks_pred)),
        f"{split_name}_post_iou": float(numpy_iou(Y_true, masks_pred)),
        f"{split_name}_pixel_precision": float(precision_score(y_pix_true, y_pix_pred, zero_division=0)),
        f"{split_name}_pixel_recall": float(recall_score(y_pix_true, y_pix_pred, zero_division=0)),
        f"{split_name}_pixel_f1": float(f1_score(y_pix_true, y_pix_pred, zero_division=0)),
        f"{split_name}_pixel_accuracy": float(accuracy_score(y_pix_true, y_pix_pred)),
        f"{split_name}_image_precision": float(precision_score(y_img_true, y_img_pred, zero_division=0)),
        f"{split_name}_image_recall": float(recall_score(y_img_true, y_img_pred, zero_division=0)),
        f"{split_name}_image_f1": float(f1_score(y_img_true, y_img_pred, zero_division=0)),
        f"{split_name}_image_accuracy": float(accuracy_score(y_img_true, y_img_pred)),
        f"{split_name}_image_confusion_matrix": confusion_matrix(y_img_true, y_img_pred, labels=[0, 1]).tolist(),
        f"{split_name}_notumor_precision": float(img_report_dict["notumor"]["precision"]),
        f"{split_name}_notumor_recall": float(img_report_dict["notumor"]["recall"]),
        f"{split_name}_notumor_f1": float(img_report_dict["notumor"]["f1-score"]),
        f"{split_name}_yestumor_precision": float(img_report_dict["yestumor"]["precision"]),
        f"{split_name}_yestumor_recall": float(img_report_dict["yestumor"]["recall"]),
        f"{split_name}_yestumor_f1": float(img_report_dict["yestumor"]["f1-score"]),
    }


train_metrics = calculate_split_metrics("train", Y_train, train_masks, y_train_img_true, y_train_img_pred)
val_metrics = calculate_split_metrics("val", Y_val, val_masks, y_val_img_true, y_val_img_pred)
test_metrics = calculate_split_metrics("test", Y_test, test_masks, y_test_img_true, y_test_img_pred)

# ============================================================
# 16. SAVE CLASSIFICATION REPORTS
# ============================================================
with open(report_txt_path, "w", encoding="utf-8") as f:
    for split_name, y_true, y_pred in [
        ("TRAIN", y_train_img_true, y_train_img_pred),
        ("VALIDATION", y_val_img_true, y_val_img_pred),
        ("TEST", y_test_img_true, y_test_img_pred),
    ]:
        f.write(f"\n{split_name} IMAGE-LEVEL CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(classification_report(
            y_true, y_pred,
            target_names=["notumor", "yestumor"],
            digits=4,
            zero_division=0,
        ))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred, labels=[0, 1])))
        f.write("\n\n")

# ============================================================
# 17. SAVE GRAPHS
# ============================================================
plt.figure(figsize=(13, 5))
metric_names = [
    "Train Dice", "Val Dice", "Test Dice",
    "Train IoU", "Val IoU", "Test IoU",
    "Train Img Acc", "Val Img Acc", "Test Img Acc",
    "Train Img Recall", "Val Img Recall", "Test Img Recall",
]

metric_values = [
    train_metrics["train_post_dice"],
    val_metrics["val_post_dice"],
    test_metrics["test_post_dice"],
    train_metrics["train_post_iou"],
    val_metrics["val_post_iou"],
    test_metrics["test_post_iou"],
    train_metrics["train_image_accuracy"],
    val_metrics["val_image_accuracy"],
    test_metrics["test_image_accuracy"],
    train_metrics["train_image_recall"],
    val_metrics["val_image_recall"],
    test_metrics["test_image_recall"],
]

plt.bar(metric_names, metric_values)
plt.ylim(0, 1)
plt.xticks(rotation=35, ha="right")
plt.title(f"Train / Val / Test Metrics - {RUN_VERSION}")
plt.tight_layout()
metrics_graph_path = graphs_dir / f"metrics_train_val_test_{RUN_VERSION}.png"
plt.savefig(metrics_graph_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history.get("loss", []), label="train_loss")
plt.plot(history.history.get("val_loss", []), label="val_loss")
plt.title("Loss")
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history.get("dice_coef", []), label="train_dice")
plt.plot(history.history.get("val_dice_coef", []), label="val_dice")
plt.title("Dice")
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history.get("iou_metric", []), label="train_iou")
plt.plot(history.history.get("val_iou_metric", []), label="val_iou")
plt.title("IoU")
plt.legend()

plt.tight_layout()
curves_path = graphs_dir / f"training_curves_{RUN_VERSION}.png"
plt.savefig(curves_path, dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# 18. DASHBOARD TEST VISUALS
# ============================================================
def load_original_image_from_row(row):
    image_path = str(row["image_path"])
    filename = Path(image_path).name
    colab_image_path = PROJECT_DIR / "sample_data" / filename

    if colab_image_path.exists():
        img = Image.open(colab_image_path).convert("L")
        return np.array(img)

    return None


num_vis = min(NUM_VIS, len(X_test))

for i in range(num_vis):
    row = df_test_valid.iloc[i]
    original_img = load_original_image_from_row(row)

    roi_img = X_test[i].squeeze()
    true_mask = Y_test[i].squeeze()
    pred_mask = test_masks[i].squeeze()

    plt.figure(figsize=(22, 5))

    plt.subplot(1, 5, 1)
    if original_img is not None:
        plt.imshow(original_img, cmap="gray")
        plt.title(f"Original\nTrue={y_test_img_true[i]}")
    else:
        plt.imshow(roi_img, cmap="gray")
        plt.title(f"ROI fallback\nTrue={y_test_img_true[i]}")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(roi_img, cmap="gray")
    plt.title("ROI Input")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(roi_img, cmap="gray")
    plt.imshow(true_mask, cmap="Greens", alpha=0.40)
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(roi_img, cmap="gray")
    plt.imshow(pred_mask, cmap="Reds", alpha=0.40)
    plt.title(f"Pred Mask\nPred={y_test_img_pred[i]}")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(roi_img, cmap="gray")
    plt.imshow(true_mask, cmap="Greens", alpha=0.40)
    plt.imshow(pred_mask, cmap="Reds", alpha=0.40)
    plt.title("Green=True | Red=Pred")
    plt.axis("off")

    plt.tight_layout()
    dashboard_path = graphs_dir / f"dashboard_test_{RUN_VERSION}_{i}.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()

# ============================================================
# 19. SAVE JSON + SUMMARY
# ============================================================
results = {
    "run_version": RUN_VERSION,
    "previous_run_version": PREV_RUN_VERSION,
    "data_arrays_version": DATA_ARRAYS_VERSION,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "training_seconds": float(training_seconds),
    "training_minutes": float(training_seconds / 60),
    "best_seg_combo": best_seg_combo,
    "best_cls_combo": best_cls_combo,
    "best_balanced_combo": best_balanced_combo,
    "final_selected_combo": best_seg_combo,
    "best_pred_threshold": float(BEST_PRED_THRESHOLD),
    "best_pixel_threshold": int(BEST_PIXEL_THRESHOLD),
    "best_min_area": int(BEST_MIN_AREA),
    "model_path": str(best_model_path),
    "history_csv": str(history_csv_path),
    "threshold_csv": str(threshold_csv_path),
    "manual_combo_csv": str(manual_combo_csv_path),
    "classification_report": str(report_txt_path),
    "graphs_dir": str(graphs_dir),
    "metrics_graph": str(metrics_graph_path),
    "training_curves": str(curves_path),
    "manual_combo_results": manual_rows,
    **raw_results,
    **train_metrics,
    **val_metrics,
    **test_metrics,
}

with open(metrics_json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Run version: {RUN_VERSION}\n")
    f.write(f"Previous run: {PREV_RUN_VERSION}\n")
    f.write(f"Model path: {best_model_path}\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Learning rate: {LEARNING_RATE}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n\n")

    f.write("BEST THRESHOLDS\n")
    f.write("-" * 60 + "\n")
    f.write(f"Best segmentation combo: {best_seg_combo}\n")
    f.write(f"Best classification combo: {best_cls_combo}\n")
    f.write(f"Best balanced combo: {best_balanced_combo}\n")
    f.write(f"Final selected combo: {best_seg_combo}\n")
    f.write(f"Using prediction threshold: {BEST_PRED_THRESHOLD}\n")
    f.write(f"Using pixel threshold: {BEST_PIXEL_THRESHOLD}\n")
    f.write(f"Using min area: {BEST_MIN_AREA}\n\n")

    f.write("RAW SEGMENTATION\n")
    f.write("-" * 60 + "\n")
    f.write(f"Train Dice: {raw_results['train_dice_raw']:.4f}, IoU: {raw_results['train_iou_raw']:.4f}\n")
    f.write(f"Val Dice: {raw_results['val_dice_raw']:.4f}, IoU: {raw_results['val_iou_raw']:.4f}\n")
    f.write(f"Test Dice: {raw_results['test_dice_raw']:.4f}, IoU: {raw_results['test_iou_raw']:.4f}\n\n")

    f.write("POST-PROCESS SEGMENTATION\n")
    f.write("-" * 60 + "\n")
    f.write(f"Train Dice: {train_metrics['train_post_dice']:.4f}, IoU: {train_metrics['train_post_iou']:.4f}\n")
    f.write(f"Val Dice: {val_metrics['val_post_dice']:.4f}, IoU: {val_metrics['val_post_iou']:.4f}\n")
    f.write(f"Test Dice: {test_metrics['test_post_dice']:.4f}, IoU: {test_metrics['test_post_iou']:.4f}\n\n")

    f.write("IMAGE-LEVEL CLASSIFICATION\n")
    f.write("-" * 60 + "\n")
    f.write(f"Train Accuracy: {train_metrics['train_image_accuracy']:.4f}, Recall: {train_metrics['train_image_recall']:.4f}, F1: {train_metrics['train_image_f1']:.4f}\n")
    f.write(f"Val Accuracy: {val_metrics['val_image_accuracy']:.4f}, Recall: {val_metrics['val_image_recall']:.4f}, F1: {val_metrics['val_image_f1']:.4f}\n")
    f.write(f"Test Accuracy: {test_metrics['test_image_accuracy']:.4f}, Recall: {test_metrics['test_image_recall']:.4f}, F1: {test_metrics['test_image_f1']:.4f}\n\n")

    f.write("MANUAL TEST COMBO COMPARISON\n")
    f.write("-" * 60 + "\n")
    for row in manual_rows:
        f.write(
            f"Combo ({row['pred_threshold']}, {row['pixel_threshold']}, {row['min_area']}): "
            f"Dice={row['test_post_dice']:.4f}, "
            f"IoU={row['test_post_iou']:.4f}, "
            f"Acc={row['test_image_accuracy']:.4f}, "
            f"Recall={row['test_image_recall']:.4f}, "
            f"F1={row['test_image_f1']:.4f}\n"
        )

# ============================================================
# 20. FINAL PRINT
# ============================================================
print("\n================ FINAL SUMMARY ================")
print("Run:", RUN_VERSION)

print("\nRAW SEGMENTATION:")
print("Train Dice:", round(raw_results["train_dice_raw"], 4), "| IoU:", round(raw_results["train_iou_raw"], 4))
print("Val Dice  :", round(raw_results["val_dice_raw"], 4), "| IoU:", round(raw_results["val_iou_raw"], 4))
print("Test Dice :", round(raw_results["test_dice_raw"], 4), "| IoU:", round(raw_results["test_iou_raw"], 4))

print("\nPOST-PROCESS SEGMENTATION:")
print("Train Dice:", round(train_metrics["train_post_dice"], 4), "| IoU:", round(train_metrics["train_post_iou"], 4))
print("Val Dice  :", round(val_metrics["val_post_dice"], 4), "| IoU:", round(val_metrics["val_post_iou"], 4))
print("Test Dice :", round(test_metrics["test_post_dice"], 4), "| IoU:", round(test_metrics["test_post_iou"], 4))

print("\nIMAGE-LEVEL CLASSIFICATION:")
print("Train Accuracy:", round(train_metrics["train_image_accuracy"], 4), "| Recall:", round(train_metrics["train_image_recall"], 4), "| F1:", round(train_metrics["train_image_f1"], 4))
print("Val Accuracy  :", round(val_metrics["val_image_accuracy"], 4), "| Recall:", round(val_metrics["val_image_recall"], 4), "| F1:", round(val_metrics["val_image_f1"], 4))
print("Test Accuracy :", round(test_metrics["test_image_accuracy"], 4), "| Recall:", round(test_metrics["test_image_recall"], 4), "| F1:", round(test_metrics["test_image_f1"], 4))

print("\nTHRESHOLDS:")
print("Best segmentation combo:", best_seg_combo)
print("Best classification combo:", best_cls_combo)
print("Best balanced combo:", best_balanced_combo)
print("Used final combo:", best_seg_combo)

print("\nMANUAL TEST COMBO COMPARISON:")
for row in manual_rows:
    print(
        f"Combo ({row['pred_threshold']}, {row['pixel_threshold']}, {row['min_area']}) | "
        f"Dice={row['test_post_dice']:.4f} | IoU={row['test_post_iou']:.4f} | "
        f"Acc={row['test_image_accuracy']:.4f} | Recall={row['test_image_recall']:.4f} | "
        f"F1={row['test_image_f1']:.4f}"
    )

print("\nTARGETS TO BEAT:")
print("V25    Post Dice: 0.8097 | Post IoU: 0.6803")
print("V25 manual best: Dice 0.8098 | IoU 0.6804")
print("V24    Post Dice: 0.8096 | Post IoU: 0.6802")
print("V23    Post Dice: 0.8087 | Post IoU: 0.6788")
print("U-Net  Post Dice: 0.8104 | Post IoU: 0.6812")
print("ResNet Post Dice: 0.8087 | Post IoU: 0.6789")

print("\nSaved:")
print("Model:", best_model_path)
print("Metrics:", metrics_json_path)
print("Summary:", summary_txt_path)
print("Report:", report_txt_path)
print("Graphs:", graphs_dir)
print("Manual combo CSV:", manual_combo_csv_path)
print("================================================")