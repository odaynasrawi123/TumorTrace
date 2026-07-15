
import os
import json
import time
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TerminateOnNaN,
)

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


# ============================================================
# 1. SETTINGS
# ============================================================
RUN_VERSION = "v18_resnet50_unet_160_full"
DATA_ARRAYS_VERSION = "v9_roi_polygon_only_160"

IMG_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 25
LEARNING_RATE = 1e-4

USE_CLAHE = True
USE_PER_IMAGE_NORMALIZATION = True
NUM_VIS = 12

RESNET_WEIGHTS = None

THRESHOLD_CANDIDATES = np.array(
    [0.50, 0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.74],
    dtype=np.float32,
)

PIXEL_THRESHOLD_CANDIDATES = [60, 80, 100, 120, 140, 160]
MIN_AREA_CANDIDATES = [60, 80, 100, 120, 150, 180]

POSTPROCESS_KEEP_LARGEST = True
POSTPROCESS_OPEN_KERNEL = 3
POSTPROCESS_CLOSE_KERNEL = 5
BORDER_SUPPRESSION_PIXELS = 8

MIN_COMPACTNESS = 3.0
MIN_FILL_RATIO = 0.12
MAX_ASPECT_RATIO = 8.0
MAX_COMPONENTS_TO_KEEP = 1

EARLY_STOP_PATIENCE = 6
REDUCE_LR_PATIENCE = 2
MIN_LR = 1e-7
SHUFFLE_BUFFER_SIZE = 512


# ============================================================
# 2. PATHS
# ============================================================
PROJECT_DIR = Path(__file__).resolve().parent

arrays_dir = PROJECT_DIR / "arrays" / DATA_ARRAYS_VERSION
graphs_dir = PROJECT_DIR / "graphs" / RUN_VERSION
models_dir = PROJECT_DIR / "models"
metrics_dir = PROJECT_DIR / "metrics"
reports_dir = PROJECT_DIR / "reports"
backup_dir = PROJECT_DIR / "training_backup" / RUN_VERSION

for d in [graphs_dir, models_dir, metrics_dir, reports_dir, backup_dir]:
    d.mkdir(parents=True, exist_ok=True)

best_model_path = models_dir / f"best_{RUN_VERSION}.keras"
metrics_path = metrics_dir / f"metrics_{RUN_VERSION}.json"
report_txt_path = reports_dir / f"classification_report_{RUN_VERSION}.txt"
summary_txt_path = reports_dir / f"summary_metrics_{RUN_VERSION}.txt"
threshold_search_path = reports_dir / f"threshold_search_{RUN_VERSION}.csv"
history_csv_path = reports_dir / f"history_{RUN_VERSION}.csv"

print("PROJECT_DIR:", PROJECT_DIR)
print("RUN_VERSION:", RUN_VERSION)
print("arrays_dir:", arrays_dir)
print("best_model_path:", best_model_path)

if not arrays_dir.exists():
    raise FileNotFoundError(f"arrays_dir not found: {arrays_dir}")


# ============================================================
# 3. LOAD ARRAYS
# ============================================================
required_files = [
    "X_train.npy", "Y_train.npy",
    "X_val.npy", "Y_val.npy",
    "X_test.npy", "Y_test.npy",
    "df_train_valid.csv", "df_val_valid.csv", "df_test_valid.csv",
]

missing_files = [f for f in required_files if not (arrays_dir / f).exists()]
if missing_files:
    raise FileNotFoundError(f"Missing files in {arrays_dir}: {missing_files}")

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
    X_train = np.expand_dims(X_train, axis=-1)
if X_val.ndim == 3:
    X_val = np.expand_dims(X_val, axis=-1)
if X_test.ndim == 3:
    X_test = np.expand_dims(X_test, axis=-1)

if Y_train.ndim == 3:
    Y_train = np.expand_dims(Y_train, axis=-1)
if Y_val.ndim == 3:
    Y_val = np.expand_dims(Y_val, axis=-1)
if Y_test.ndim == 3:
    Y_test = np.expand_dims(Y_test, axis=-1)

Y_train = (Y_train > 0.5).astype(np.float32)
Y_val = (Y_val > 0.5).astype(np.float32)
Y_test = (Y_test > 0.5).astype(np.float32)

print("\nLoaded shapes:")
print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val  :", X_val.shape, "Y_val  :", Y_val.shape)
print("X_test :", X_test.shape, "Y_test :", Y_test.shape)

if X_train.shape[1] != IMG_SIZE or X_train.shape[2] != IMG_SIZE:
    raise ValueError(f"IMG_SIZE mismatch. IMG_SIZE={IMG_SIZE}, X_train={X_train.shape}")

if "tumor" not in df_val_valid.columns or "tumor" not in df_test_valid.columns:
    raise ValueError("df_val_valid.csv and df_test_valid.csv must include tumor column.")


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
# 5. DATA AUGMENTATION + DATASET
# ============================================================
def augment_image_mask(img, mask):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    img = tf.image.random_brightness(img, max_delta=0.025)
    img = tf.image.random_contrast(img, lower=0.96, upper=1.06)

    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.003, dtype=tf.float32)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)

    return img, mask


def make_dataset(X, Y, batch_size=4, training=False):
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    ds = ds.map(preprocess_image_mask, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()

    if training:
        buffer_size = min(len(X), SHUFFLE_BUFFER_SIZE)
        ds = ds.shuffle(buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_dataset(X_train, Y_train, BATCH_SIZE, training=True)
val_ds = make_dataset(X_val, Y_val, BATCH_SIZE, training=False)
test_ds = make_dataset(X_test, Y_test, BATCH_SIZE, training=False)


# ============================================================
# 6. METRICS + LOSSES
# ============================================================
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
def tversky_index(y_true, y_pred, alpha=0.60, beta=0.40, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred, tf.float32), [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)

    return (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)


@tf.keras.utils.register_keras_serializable()
def focal_tversky_loss(y_true, y_pred, gamma=1.25):
    tv = tversky_index(y_true, y_pred)
    return tf.pow((1.0 - tv), gamma)


@tf.keras.utils.register_keras_serializable()
def focal_tversky_iou_dice_bce_loss(y_true, y_pred):
    ft = focal_tversky_loss(y_true, y_pred)
    jl = jaccard_loss(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return 0.35 * ft + 0.30 * jl + 0.25 * dl + 0.10 * bce


@tf.keras.utils.register_keras_serializable()
def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(tf.cast(y_true, tf.float32), [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection

    return (intersection + smooth) / (union + smooth)


chosen_loss = focal_tversky_iou_dice_bce_loss


def get_custom_objects():
    return {
        "dice_coef": dice_coef,
        "soft_iou_coef": soft_iou_coef,
        "dice_loss": dice_loss,
        "jaccard_loss": jaccard_loss,
        "tversky_index": tversky_index,
        "focal_tversky_loss": focal_tversky_loss,
        "focal_tversky_iou_dice_bce_loss": focal_tversky_iou_dice_bce_loss,
        "iou_metric": iou_metric,
    }


# ============================================================
# 7. MODEL - RESNET50 U-NET
# ============================================================
def conv_block(x, filters, dropout=0.0):
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)

    return x


def build_resnet50_unet(input_shape=(160, 160, 1), model_name="ResNet50_UNet"):
    inputs = layers.Input(shape=input_shape)

    x_rgb = layers.Concatenate()([inputs, inputs, inputs])

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=RESNET_WEIGHTS,
        input_tensor=x_rgb,
    )

    s1 = base_model.get_layer("conv1_relu").output
    s2 = base_model.get_layer("conv2_block3_out").output
    s3 = base_model.get_layer("conv3_block4_out").output
    s4 = base_model.get_layer("conv4_block6_out").output
    b1 = base_model.get_layer("conv5_block3_out").output

    d1 = layers.UpSampling2D((2, 2))(b1)
    d1 = layers.Concatenate()([d1, s4])
    d1 = conv_block(d1, 256, dropout=0.25)

    d2 = layers.UpSampling2D((2, 2))(d1)
    d2 = layers.Concatenate()([d2, s3])
    d2 = conv_block(d2, 128, dropout=0.15)

    d3 = layers.UpSampling2D((2, 2))(d2)
    d3 = layers.Concatenate()([d3, s2])
    d3 = conv_block(d3, 64)

    d4 = layers.UpSampling2D((2, 2))(d3)
    d4 = layers.Concatenate()([d4, s1])
    d4 = conv_block(d4, 32)

    d5 = layers.UpSampling2D((2, 2))(d4)
    d5 = conv_block(d5, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(d5)

    return Model(inputs, outputs, name=model_name)


def compile_model(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE,
            clipnorm=1.0,
        ),
        loss=chosen_loss,
        metrics=[dice_coef, iou_metric],
    )
    return model


tf.keras.backend.clear_session()

print("\nBuilding ResNet50-U-Net from scratch...")
model = build_resnet50_unet(
    input_shape=(IMG_SIZE, IMG_SIZE, 1),
    model_name=f"ResNet50_UNet_{RUN_VERSION}",
)
model = compile_model(model)
model.summary()


# ============================================================
# 8. CALLBACKS
# ============================================================
callbacks = [
    tf.keras.callbacks.BackupAndRestore(
        backup_dir=str(backup_dir),
    ),
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
        monitor="val_loss",
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
print("\nStarting ResNet50-U-Net training from scratch...")
train_start = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

training_seconds = time.time() - train_start

print("\nTraining time seconds:", round(training_seconds, 2))
print("Training time minutes:", round(training_seconds / 60, 2))

print(f"\nReloading best model from: {best_model_path}")
model = tf.keras.models.load_model(
    best_model_path,
    custom_objects=get_custom_objects(),
    compile=False,
)
model = compile_model(model)


# ============================================================
# 10. RAW EVALUATION
# ============================================================
test_results = model.evaluate(test_ds, verbose=1)

test_loss = float(test_results[0])
test_dice = float(test_results[1])
test_iou = float(test_results[2])

print("\nRaw test results:")
print("Loss:", test_loss)
print("Dice:", test_dice)
print("IoU :", test_iou)


# ============================================================
# 11. POST-PROCESSING
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

    if keep_largest:
        keep = valid_components[:MAX_COMPONENTS_TO_KEEP]
    else:
        keep = valid_components

    for label_id, _ in keep:
        cleaned[labels == label_id] = 1

    return np.expand_dims(cleaned.astype(np.uint8), axis=-1)


def apply_postprocess(pred_probs, pred_threshold=0.66, min_area=80):
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


def safe_binary_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


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
# 12. PREDICT
# ============================================================
print("\nPredicting validation...")
val_preds = model.predict(val_ds, verbose=1)

print("\nPredicting test...")
test_preds = model.predict(test_ds, verbose=1)

y_true_val_img = df_val_valid["tumor"].astype(np.uint8).values
y_true_test_img = df_test_valid["tumor"].astype(np.uint8).values
y_true_val_pix = Y_val.flatten().astype(np.uint8)


# ============================================================
# 13. THRESHOLD SEARCH
# ============================================================
best_combo = None
best_score = -999
search_rows = []

print("\nRunning threshold search...")

for pred_th in THRESHOLD_CANDIDATES:
    for pix_th in PIXEL_THRESHOLD_CANDIDATES:
        for min_area in MIN_AREA_CANDIDATES:
            val_masks = apply_postprocess(
                val_preds,
                pred_threshold=float(pred_th),
                min_area=int(min_area),
            )

            val_img_pred = image_labels_from_masks(val_masks, int(pix_th))
            val_pix_pred = val_masks.flatten().astype(np.uint8)

            val_img_prec, val_img_rec, val_img_f1, val_img_acc = safe_binary_metrics(
                y_true_val_img,
                val_img_pred,
            )

            val_pix_prec, val_pix_rec, val_pix_f1, _ = safe_binary_metrics(
                y_true_val_pix,
                val_pix_pred,
            )

            val_iou_score = numpy_iou(Y_val, val_masks)
            val_dice_score = numpy_dice(Y_val, val_masks)

            cm_val = confusion_matrix(y_true_val_img, val_img_pred, labels=[0, 1])
            val_fp = int(cm_val[0, 1])
            val_fn = int(cm_val[1, 0])

            fp_rate = val_fp / max(np.sum(y_true_val_img == 0), 1)
            fn_rate = val_fn / max(np.sum(y_true_val_img == 1), 1)

            combined_score = (
                0.38 * val_iou_score
                + 0.30 * val_dice_score
                + 0.12 * val_pix_f1
                + 0.10 * val_img_f1
                + 0.05 * val_img_prec
                + 0.05 * val_img_rec
                - 0.12 * fp_rate
                - 0.08 * fn_rate
            )

            row = {
                "pred_threshold": float(pred_th),
                "pixel_threshold": int(pix_th),
                "min_area": int(min_area),
                "val_iou": float(val_iou_score),
                "val_dice": float(val_dice_score),
                "val_image_precision": float(val_img_prec),
                "val_image_recall": float(val_img_rec),
                "val_image_f1": float(val_img_f1),
                "val_image_accuracy": float(val_img_acc),
                "val_pixel_precision": float(val_pix_prec),
                "val_pixel_recall": float(val_pix_rec),
                "val_pixel_f1": float(val_pix_f1),
                "val_fp": val_fp,
                "val_fn": val_fn,
                "fp_rate": float(fp_rate),
                "fn_rate": float(fn_rate),
                "combined_score": float(combined_score),
            }

            search_rows.append(row)

            if combined_score > best_score:
                best_score = combined_score
                best_combo = (float(pred_th), int(pix_th), int(min_area))

search_df = pd.DataFrame(search_rows).sort_values(by="combined_score", ascending=False)
search_df.to_csv(threshold_search_path, index=False)

if best_combo is None:
    BEST_PRED_THRESHOLD = 0.66
    BEST_PIXEL_THRESHOLD = 80
    BEST_MIN_AREA = 80
else:
    BEST_PRED_THRESHOLD, BEST_PIXEL_THRESHOLD, BEST_MIN_AREA = best_combo

print("\nBest thresholds:")
print("BEST_PRED_THRESHOLD :", BEST_PRED_THRESHOLD)
print("BEST_PIXEL_THRESHOLD:", BEST_PIXEL_THRESHOLD)
print("BEST_MIN_AREA       :", BEST_MIN_AREA)
print("Best validation score:", best_score)


# ============================================================
# 14. FINAL TEST POST-PROCESSING
# ============================================================
preds_bin_clean = apply_postprocess(
    test_preds,
    pred_threshold=BEST_PRED_THRESHOLD,
    min_area=BEST_MIN_AREA,
)

y_pred_img = image_labels_from_masks(preds_bin_clean, BEST_PIXEL_THRESHOLD)


# ============================================================
# 15. METRICS
# ============================================================
y_true_pix = Y_test.flatten().astype(np.uint8)
y_pred_pix = preds_bin_clean.flatten().astype(np.uint8)

pixel_precision = precision_score(y_true_pix, y_pred_pix, zero_division=0)
pixel_recall = recall_score(y_true_pix, y_pred_pix, zero_division=0)
pixel_f1 = f1_score(y_true_pix, y_pred_pix, zero_division=0)
pixel_accuracy = accuracy_score(y_true_pix, y_pred_pix)
pixel_error_rate = 1 - pixel_accuracy
pixel_cm = confusion_matrix(y_true_pix, y_pred_pix)

post_iou = numpy_iou(Y_test, preds_bin_clean)
post_dice = numpy_dice(Y_test, preds_bin_clean)

img_precision = precision_score(y_true_test_img, y_pred_img, zero_division=0)
img_recall = recall_score(y_true_test_img, y_pred_img, zero_division=0)
img_f1 = f1_score(y_true_test_img, y_pred_img, zero_division=0)
img_accuracy = accuracy_score(y_true_test_img, y_pred_img)
img_error_rate = 1 - img_accuracy
img_cm = confusion_matrix(y_true_test_img, y_pred_img, labels=[0, 1])

img_report_text = classification_report(
    y_true_test_img,
    y_pred_img,
    target_names=["notumor", "yestumor"],
    digits=4,
    zero_division=0,
)

img_report_dict = classification_report(
    y_true_test_img,
    y_pred_img,
    target_names=["notumor", "yestumor"],
    output_dict=True,
    zero_division=0,
)

pixel_report_text = classification_report(
    y_true_pix,
    y_pred_pix,
    target_names=["background", "tumor_pixel"],
    digits=4,
    zero_division=0,
)


# ============================================================
# 16. SAVE REPORTS
# ============================================================
with open(report_txt_path, "w", encoding="utf-8") as f:
    f.write("IMAGE-LEVEL CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(img_report_text)
    f.write("\n\nPIXEL-LEVEL CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(pixel_report_text)
    f.write("\n\nIMAGE-LEVEL CONFUSION MATRIX\n")
    f.write("=" * 70 + "\n")
    f.write(str(img_cm))
    f.write("\n\nPIXEL-LEVEL CONFUSION MATRIX\n")
    f.write("=" * 70 + "\n")
    f.write(str(pixel_cm))


# ============================================================
# 17. SAVE GRAPHS
# ============================================================
plt.figure(figsize=(10, 5))
metric_names = [
    "Raw Dice", "Raw IoU", "Post Dice", "Post IoU",
    "Pixel F1", "Image F1", "Image Precision", "Image Recall",
]
metric_values = [
    test_dice, test_iou, post_dice, post_iou,
    pixel_f1, img_f1, img_precision, img_recall,
]
plt.bar(metric_names, metric_values)
plt.ylim(0, 1)
plt.title(f"Metrics Summary - {RUN_VERSION}")
plt.xticks(rotation=25)
plt.tight_layout()
metrics_graph_path = graphs_dir / f"metrics_summary_{RUN_VERSION}.png"
plt.savefig(metrics_graph_path, dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6, 5))
plt.imshow(img_cm, cmap="Blues")
plt.title(f"Image-level Confusion Matrix - {RUN_VERSION}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks([0, 1], ["NoTumor", "Tumor"])
plt.yticks([0, 1], ["NoTumor", "Tumor"])

for r in range(img_cm.shape[0]):
    for c in range(img_cm.shape[1]):
        plt.text(c, r, str(img_cm[r, c]), ha="center", va="center")

plt.tight_layout()
cm_graph_path = graphs_dir / f"confusion_matrix_{RUN_VERSION}.png"
plt.savefig(cm_graph_path, dpi=300, bbox_inches="tight")
plt.close()

curve_path = None
if history is not None:
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
    curve_path = graphs_dir / f"training_curves_{RUN_VERSION}.png"
    plt.savefig(curve_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 18. DASHBOARD VISUALS
# ============================================================
def load_original_image_from_row(row):
    image_path = row["image_path"]
    img = Image.open(image_path).convert("L")
    return np.array(img)


num_vis = min(NUM_VIS, len(X_test))

for i in range(num_vis):
    row = df_test_valid.iloc[i]
    original_img = load_original_image_from_row(row)

    roi_img = X_test[i].squeeze()
    true_mask = Y_test[i].squeeze()
    pred_mask = preds_bin_clean[i].squeeze()

    plt.figure(figsize=(22, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(original_img, cmap="gray")
    plt.title(f"Original\nTrue={y_true_test_img[i]}")
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
    plt.title(f"Pred Mask\nPred={y_pred_img[i]}")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(roi_img, cmap="gray")
    plt.imshow(true_mask, cmap="Greens", alpha=0.40)
    plt.imshow(pred_mask, cmap="Reds", alpha=0.40)
    plt.title("Green=True | Red=Pred")
    plt.axis("off")

    plt.tight_layout()
    dashboard_path = graphs_dir / f"dashboard_5_steps_{RUN_VERSION}_{i}.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 19. SAVE METRICS JSON + SUMMARY
# ============================================================
results_dict = {
    "run_version": RUN_VERSION,
    "model_type": "resnet50_unet",
    "data_arrays_version": DATA_ARRAYS_VERSION,
    "img_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "resnet_weights": RESNET_WEIGHTS,

    "best_pred_threshold": float(BEST_PRED_THRESHOLD),
    "best_pixel_threshold": int(BEST_PIXEL_THRESHOLD),
    "best_min_area": int(BEST_MIN_AREA),

    "test_loss": float(test_loss),
    "test_dice_raw": float(test_dice),
    "test_iou_raw": float(test_iou),
    "postprocess_dice": float(post_dice),
    "postprocess_iou": float(post_iou),

    "pixel_precision": float(pixel_precision),
    "pixel_recall": float(pixel_recall),
    "pixel_f1": float(pixel_f1),
    "pixel_accuracy": float(pixel_accuracy),
    "pixel_error_rate": float(pixel_error_rate),
    "pixel_confusion_matrix": pixel_cm.tolist(),

    "image_precision": float(img_precision),
    "image_recall": float(img_recall),
    "image_f1": float(img_f1),
    "image_accuracy": float(img_accuracy),
    "image_error_rate": float(img_error_rate),
    "image_confusion_matrix": img_cm.tolist(),

    "image_class_notumor_precision": float(img_report_dict["notumor"]["precision"]),
    "image_class_notumor_recall": float(img_report_dict["notumor"]["recall"]),
    "image_class_notumor_f1": float(img_report_dict["notumor"]["f1-score"]),

    "image_class_yestumor_precision": float(img_report_dict["yestumor"]["precision"]),
    "image_class_yestumor_recall": float(img_report_dict["yestumor"]["recall"]),
    "image_class_yestumor_f1": float(img_report_dict["yestumor"]["f1-score"]),

    "macro_avg_f1": float(img_report_dict["macro avg"]["f1-score"]),
    "weighted_avg_f1": float(img_report_dict["weighted avg"]["f1-score"]),

    "model_path": str(best_model_path),
    "metrics_path": str(metrics_path),
    "classification_report_txt": str(report_txt_path),
    "summary_txt_path": str(summary_txt_path),
    "threshold_search_csv": str(threshold_search_path),
    "graphs_dir": str(graphs_dir),
    "history_csv_path": str(history_csv_path),
    "metrics_graph_path": str(metrics_graph_path),
    "confusion_matrix_graph_path": str(cm_graph_path),
    "training_curves_path": str(curve_path) if curve_path else None,
    "training_seconds": float(training_seconds),
    "training_minutes": float(training_seconds / 60),
}

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, indent=4)

with open(summary_txt_path, "w", encoding="utf-8") as f:
    f.write(f"Run version: {RUN_VERSION}\n")
    f.write("Model type: ResNet50-U-Net\n")
    f.write(f"Data arrays version: {DATA_ARRAYS_VERSION}\n")
    f.write(f"Model path: {best_model_path}\n")
    f.write(f"IMG_SIZE: {IMG_SIZE}\n")
    f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
    f.write(f"EPOCHS: {EPOCHS}\n")
    f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
    f.write(f"RESNET_WEIGHTS: {RESNET_WEIGHTS}\n\n")

    f.write("BEST POSTPROCESSING SETTINGS\n")
    f.write("-" * 50 + "\n")
    f.write(f"BEST_PRED_THRESHOLD: {BEST_PRED_THRESHOLD}\n")
    f.write(f"BEST_PIXEL_THRESHOLD: {BEST_PIXEL_THRESHOLD}\n")
    f.write(f"BEST_MIN_AREA: {BEST_MIN_AREA}\n")
    f.write(f"MIN_COMPACTNESS: {MIN_COMPACTNESS}\n")
    f.write(f"MIN_FILL_RATIO: {MIN_FILL_RATIO}\n")
    f.write(f"MAX_ASPECT_RATIO: {MAX_ASPECT_RATIO}\n\n")

    f.write("SEGMENTATION RESULTS\n")
    f.write("-" * 50 + "\n")
    f.write(f"Raw test loss: {test_loss:.4f}\n")
    f.write(f"Raw test Dice: {test_dice:.4f}\n")
    f.write(f"Raw test IoU: {test_iou:.4f}\n")
    f.write(f"Postprocess Dice: {post_dice:.4f}\n")
    f.write(f"Postprocess IoU: {post_iou:.4f}\n\n")

    f.write("PIXEL-LEVEL RESULTS\n")
    f.write("-" * 50 + "\n")
    f.write(f"Precision: {pixel_precision:.4f}\n")
    f.write(f"Recall: {pixel_recall:.4f}\n")
    f.write(f"F1-score: {pixel_f1:.4f}\n")
    f.write(f"Accuracy: {pixel_accuracy:.4f}\n")
    f.write(f"Error rate: {pixel_error_rate:.4f}\n\n")

    f.write("IMAGE-LEVEL RESULTS\n")
    f.write("-" * 50 + "\n")
    f.write(f"Precision: {img_precision:.4f}\n")
    f.write(f"Recall: {img_recall:.4f}\n")
    f.write(f"F1-score: {img_f1:.4f}\n")
    f.write(f"Accuracy: {img_accuracy:.4f}\n")
    f.write(f"Error rate: {img_error_rate:.4f}\n")
    f.write(f"Image confusion matrix: {img_cm.tolist()}\n")


# ============================================================
# 20. FINAL PRINT
# ============================================================
print("\n================ FINAL SUMMARY ================")
print("Run version :", RUN_VERSION)
print("Model       : ResNet50-U-Net")

print("\nSegmentation:")
print("Raw Dice :", round(test_dice, 4))
print("Raw IoU  :", round(test_iou, 4))
print("Post Dice:", round(post_dice, 4))
print("Post IoU :", round(post_iou, 4))

print("\nPixel-level:")
print("Precision:", round(pixel_precision, 4))
print("Recall   :", round(pixel_recall, 4))
print("F1       :", round(pixel_f1, 4))
print("Accuracy :", round(pixel_accuracy, 4))

print("\nImage-level:")
print("Precision:", round(img_precision, 4))
print("Recall   :", round(img_recall, 4))
print("F1       :", round(img_f1, 4))
print("Accuracy :", round(img_accuracy, 4))
print("CM       :", img_cm.tolist())

print("\nTargets:")
print("UNet++ V16 Post Dice: 0.7939 | Post IoU: 0.6583")
print("U-Net  V17 Post Dice: 0.8104 | Post IoU: 0.6812")

print("\nSaved:")
print("Model:", best_model_path)
print("Metrics:", metrics_path)
print("Summary:", summary_txt_path)
print("Report:", report_txt_path)
print("Threshold CSV:", threshold_search_path)
print("Graphs:", graphs_dir)
print("================================================")
