from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Config
# -----------------------------
DATA_ROOT = Path("data/gtsrb/GTSRB/Final_Training/Images")
MODEL_PATH = Path("output/trafficsignnet_v2.keras")

IMG_SIZE = (64, 64)
NUM_CLASSES = 43
SEED = 42

# GTSRB speed-limit classes
SPEED_LIMIT_CLASSES = {0, 1, 2, 3, 4, 5, 7, 8}


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
# Load dataset
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


def top3_accuracy(y_true, probs):
    top3 = np.argsort(probs, axis=1)[:, -3:]
    correct = 0
    for i, true_cls in enumerate(y_true):
        if true_cls in top3[i]:
            correct += 1
    return correct / len(y_true)


def speed_limit_subset_accuracy(y_true, y_pred):
    mask = np.isin(y_true, list(SPEED_LIMIT_CLASSES))
    if np.sum(mask) == 0:
        return None
    return np.mean(y_true[mask] == y_pred[mask])


def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    print("Loading dataset...")
    X, y = load_gtsrb()
    print("Dataset shape:", X.shape, y.shape)

    # same split as training
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    print("Validation set:", X_val.shape, y_val.shape)

    model = tf.keras.models.load_model(MODEL_PATH)

    # standard evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=1)

    # predictions
    probs = model.predict(X_val, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # top-3
    val_top3 = top3_accuracy(y_val, probs)

    # speed-limit subset
    speed_acc = speed_limit_subset_accuracy(y_val, y_pred)

    print("\n===== CLASSIFIER METRICS =====")
    print(f"Validation loss: {val_loss:.6f}")
    print(f"Validation accuracy: {val_acc:.6f}")
    print(f"Top-3 accuracy: {val_top3:.6f}")
    if speed_acc is not None:
        print(f"Speed-limit subset accuracy: {speed_acc:.6f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_val, y_pred, digits=4))

    print("\n===== CONFUSION MATRIX =====")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)

    
    print("\n===== SPEED-LIMIT CONFUSION SAMPLES =====")
    for cls in sorted(SPEED_LIMIT_CLASSES):
        mask = y_val == cls
        if np.sum(mask) == 0:
            continue
        preds = y_pred[mask]
        unique, counts = np.unique(preds, return_counts=True)
        pairs = sorted(zip(unique, counts), key=lambda x: x[1], reverse=True)[:5]
        print(f"True class {cls}: {pairs}")


if __name__ == "__main__":
    main()