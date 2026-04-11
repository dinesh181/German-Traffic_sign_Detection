from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

DATA_ROOT = Path("data/gtsrb/GTSRB/Final_Training/Images")  
OUT_MODEL = Path("output/trafficsignnet.keras")

IMG_SIZE = (64, 64)
NUM_CLASSES = 43

def clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def load_gtsrb():
    X, y = [], []
    # folders are 00000 ... 00042
    for class_dir in sorted(DATA_ROOT.glob("*")):
        if not class_dir.is_dir():
            continue
        class_id = int(class_dir.name)
        for img_path in class_dir.glob("*.ppm"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = clahe_rgb(img)
            img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            X.append(img)
            y.append(class_id)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y

def build_model():
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def main():
    X, y = load_gtsrb()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save(OUT_MODEL)
    print("Saved classifier to:", OUT_MODEL)

if __name__ == "__main__":
    main()