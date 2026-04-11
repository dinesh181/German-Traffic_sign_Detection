import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


def clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_for_classifier(img_bgr, size=(32, 32)):
    img_bgr = clahe_rgb(img_bgr)
    img = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def clean_sign_name(name):
    if "Speed limit (30km/h)" in name:
        return "30 km/h"
    return name


def main():
    CLS_MODEL = "output/trafficsignnet.keras"
    SIGNNAMES = "classifier/signnames.csv"
    IMAGE_PATH = "test_images/test_image_1.png"   

    cls = tf.keras.models.load_model(CLS_MODEL)
    signnames = pd.read_csv(SIGNNAMES)

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(IMAGE_PATH)

    inp = preprocess_for_classifier(img)
    probs = cls.predict(inp, verbose=False)[0]

    class_id = int(np.argmax(probs))
    score = float(np.max(probs))

    row = signnames.loc[signnames["ClassId"] == class_id, "SignName"]
    name = row.values[0] if len(row) else f"Class {class_id}"

    print("Traffic sign:", clean_sign_name(name))
    print("Confidence:", round(score, 3))
   


if __name__ == "__main__":
    main()