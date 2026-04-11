from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = Path("data/gtsrb/GTSRB/Final_Training/Images")
OUT_MODEL = Path("output/trafficsignnet_v2.keras")

IMG_SIZE = (64, 64)
NUM_CLASSES = 43
SEED = 42

# GTSRB numeric speed-limit classes
# 0=20, 1=30, 2=50, 3=60, 4=70, 5=80, 7=100, 8=120
SPEED_LIMIT_CLASSES = {0, 1, 2, 3, 4, 5, 7, 8}

# Add extra augmented copies only for speed-limit classes
EXTRA_AUG_PER_SPEED = 5
EXTRA_AUG_PER_OTHER = 0


# -----------------------------
# Preprocessing
# -----------------------------
def clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_base(img_bgr):
    img_bgr = clahe_rgb(img_bgr)
    img_bgr = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return img_rgb


# -----------------------------
# Augmentation
# -----------------------------
def random_affine(img_bgr, max_angle=10, max_shift_ratio=0.08, scale_low=0.92, scale_high=1.08):
    h, w = img_bgr.shape[:2]

    angle = np.random.uniform(-max_angle, max_angle)
    scale = np.random.uniform(scale_low, scale_high)
    tx = np.random.uniform(-max_shift_ratio, max_shift_ratio) * w
    ty = np.random.uniform(-max_shift_ratio, max_shift_ratio) * h

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    return cv2.warpAffine(
        img_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def random_brightness_contrast(img_bgr):
    alpha = np.random.uniform(0.75, 1.25)   # contrast
    beta = np.random.uniform(-18, 18)       # brightness
    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return out


def random_blur(img_bgr):
    if np.random.rand() < 0.35:
        return cv2.GaussianBlur(img_bgr, (3, 3), 0)
    return img_bgr


def random_noise(img_bgr):
    if np.random.rand() < 0.35:
        noise = np.random.normal(0, 6, img_bgr.shape).astype(np.float32)
        out = img_bgr.astype(np.float32) + noise
        return np.clip(out, 0, 255).astype(np.uint8)
    return img_bgr


def random_sharpen(img_bgr):
    if np.random.rand() < 0.25:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(img_bgr, -1, kernel)
        return np.clip(out, 0, 255).astype(np.uint8)
    return img_bgr


def augment_speed_limit(img_rgb_float):
    """
    Stronger augmentation for speed-limit classes because
    they differ mainly by digits.
    """
    img_bgr = cv2.cvtColor((img_rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    img_bgr = random_affine(img_bgr, max_angle=8, max_shift_ratio=0.06, scale_low=0.94, scale_high=1.06)
    img_bgr = random_brightness_contrast(img_bgr)
    img_bgr = random_blur(img_bgr)
    img_bgr = random_noise(img_bgr)
    img_bgr = random_sharpen(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return img_rgb


def augment_general(img_rgb_float):
    """
    Mild augmentation for non-speed classes.
    """
    img_bgr = cv2.cvtColor((img_rgb_float * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    img_bgr = random_affine(img_bgr, max_angle=12, max_shift_ratio=0.08, scale_low=0.90, scale_high=1.10)
    img_bgr = random_brightness_contrast(img_bgr)
    img_bgr = random_blur(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return img_rgb


# -----------------------------
# Data loading
# -----------------------------
def load_gtsrb():
    X, y = [], []

    for class_dir in sorted(DATA_ROOT.glob("*")):
        if not class_dir.is_dir():
            continue

        class_id = int(class_dir.name)

        for img_path in class_dir.glob("*.ppm"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            img = preprocess_base(img)
            X.append(img)
            y.append(class_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


def expand_training_data(X_train, y_train):
    X_out = [x for x in X_train]
    y_out = [int(c) for c in y_train]

    for img, cls in zip(X_train, y_train):
        if int(cls) in SPEED_LIMIT_CLASSES:
            copies = EXTRA_AUG_PER_SPEED
            for _ in range(copies):
                X_out.append(augment_speed_limit(img))
                y_out.append(int(cls))
        else:
            copies = EXTRA_AUG_PER_OTHER
            for _ in range(copies):
                X_out.append(augment_general(img))
                y_out.append(int(cls))

    X_out = np.array(X_out, dtype=np.float32)
    y_out = np.array(y_out, dtype=np.int64)
    return X_out, y_out


def build_sample_weights(y):
    """
    Give speed-limit classes a bit more importance during training.
    """
    weights = np.ones(len(y), dtype=np.float32)
    mask = np.isin(y, list(SPEED_LIMIT_CLASSES))
    weights[mask] = 2.0 # double weight for speed-limit classes
    return weights


# -----------------------------
# Model
# -----------------------------
def conv_block(x, filters, dropout=0.0):
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPool2D()(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = conv_block(inputs, 32, dropout=0.10)
    x = conv_block(x, 64, dropout=0.15)
    x = conv_block(x, 128, dropout=0.20)
    x = conv_block(x, 256, dropout=0.25)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.40)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# -----------------------------
# Main
# -----------------------------
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading dataset...")
    X, y = load_gtsrb()
    print("Original dataset:", X.shape, y.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print("Building augmented training set...")
    X_train_aug, y_train_aug = expand_training_data(X_train, y_train)
    sample_weights = build_sample_weights(y_train_aug)

    print("Train before aug :", X_train.shape, y_train.shape)
    print("Train after aug  :", X_train_aug.shape, y_train_aug.shape)
    print("Validation       :", X_val.shape, y_val.shape)

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUT_MODEL),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]

    model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val, y_val),
        sample_weight=sample_weights,
        epochs=35,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUT_MODEL)
    print(f"Saved improved classifier to: {OUT_MODEL}")


if __name__ == "__main__":
    main()