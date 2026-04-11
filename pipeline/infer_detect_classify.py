import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf


def clahe_rgb(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def preprocess_for_classifier(crop_bgr, size):
    crop_bgr = clahe_rgb(crop_bgr)
    crop = cv2.resize(crop_bgr, size, interpolation=cv2.INTER_AREA)
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(crop, axis=0)


def main():
    DET_WEIGHTS = "runs/detect/train/weights/best.pt"
   # CLS_MODEL = "output/trafficsignnet.keras"
    CLS_MODEL = "output/trafficsignnet_v2.keras"
    SIGNNAMES = "classifier/signnames.csv"
    IMAGE_PATH = "test_images/test_image_1.png"   

    det = YOLO(DET_WEIGHTS)
    cls = tf.keras.models.load_model(CLS_MODEL)
    signnames = pd.read_csv(SIGNNAMES)

    
    input_h = cls.input_shape[1]
    input_w = cls.input_shape[2]

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(IMAGE_PATH)

    
    results = det.predict(source=img, conf=0.25, iou=0.5, verbose=False)

    found_any = False

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        for (box, conf) in zip(boxes, confs):
            x1, y1, x2, y2 = box.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1] - 1, x2)
            y2 = min(img.shape[0] - 1, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cv2.imwrite("debug_crop.png", crop)
            inp = preprocess_for_classifier(crop, size=(input_w, input_h))
            probs = cls.predict(inp, verbose=False)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            print("Top-3 predictions:")
            for idx in top3_idx:
                row = signnames.loc[signnames["ClassId"] == int(idx), "SignName"]
                pred_name = row.values[0] if len(row) else f"Class {idx}"
                print(f"  {pred_name}: {probs[idx]:.3f}")
            class_id = int(np.argmax(probs))
            score = float(np.max(probs))

            row = signnames.loc[signnames["ClassId"] == class_id, "SignName"]
            name = row.values[0] if len(row) else f"Class {class_id}"

            print(f"Traffic sign: {name} | det={conf:.2f} | cls={score:.2f}")
            label = f"{name} det={conf:.2f} cls={score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(0, y1-8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imshow("detect+classify", img)
            cv2.waitKey(0)
            found_any = True

    if not found_any:
        print("No traffic sign detected.")


if __name__ == "__main__":
    main()