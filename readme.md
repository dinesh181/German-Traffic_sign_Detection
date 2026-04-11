# German Traffic Sign Detection and Classification

A two-stage computer vision pipeline for **German traffic sign recognition** using:

- **YOLOv8n** for traffic sign **detection**
- **CNN (TensorFlow/Keras)** for traffic sign **classification**

The project uses:

- **GTSDB** for detector training
- **GTSRB** for classifier training

## Final product

The final product of this project is an end-to-end script that:

1. takes a road-scene image as input,
2. detects traffic sign regions,
3. crops each detected sign,
4. classifies the cropped sign,
5. prints the predicted traffic sign label and confidence values.

The main inference script is:

`pipeline/infer_detect_classify.py`

---

## 1. Project structure

```text
German_trafic_signs_Detection/
├── classifier/
│   ├── signnames.csv
│   ├── train_classifier_v2.py
│   └── classifier_alone_test.py
├── detector/
│   ├── data.yaml
│   ├── train_detector.py
│   └── convert_gtsdb_to_yolo.py
├── pipeline/
│   ├── infer_detect_classify.py
│   ├── metrics_extraction.py
│   └── Classifier_metrics.py
├── data/
│   ├── gtsdb_yolo/
│   │   ├── images/
│   │   └── labels/
│   └── gtsrb/
├── output/
│   ├── trafficsignnet.keras
│   └── trafficsignnet_v2.keras
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt
└── test_images/
```

---

## 2. Requirements

- Python **3.12**
- Windows, Linux, or macOS
- Recommended: virtual environment
- CPU is sufficient for inference and small-scale testing

### Main Python libraries

- `ultralytics`
- `tensorflow`
- `opencv-python`
- `numpy`
- `pandas`
- `scikit-learn`
- `pillow`

---

## 3. Setup

### Step 1: Clone the repository

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd German_trafic_signs_Detection
```

### Step 2: Create and activate a virtual environment

#### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install --upgrade pip
pip install ultralytics tensorflow opencv-python numpy pandas scikit-learn pillow
```

---

## 4. Dataset preparation

### Detector dataset: GTSDB

Prepare the dataset in YOLO format under:
link to download dataset 
o	GTSRB -  https://benchmark.ini.rub.de/gtsrb_dataset.html
o	GTSDB -  https://benchmark.ini.rub.de/gtsdb_dataset.html


```text
data/gtsdb_yolo/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

If your raw GTSDB images are in `.ppm` format, convert them to `.png` before training if needed by your environment.

### Classifier dataset: GTSRB

Place the GTSRB training folders under:

```text
data/gtsrb/GTSRB/Final_Training/Images/
```

Expected class-folder structure:

```text
Images/
├── 00000/
├── 00001/
├── ...
└── 00042/
```

---

## 5. Training the models

### 5.1 Train the detector

```bash
python detector/train_detector.py
```

Expected output weights:

```text
runs/detect/train/weights/best.pt
```

### 5.2 Train the improved classifier

```bash
python classifier/train_classifier_v2.py
```

Expected output model:

```text
output/trafficsignnet_v2.keras
```

---

## 6. Run the final product

The final end-to-end pipeline uses the trained detector and classifier.

### Step 1: Open the inference script

Edit:

```text
pipeline/infer_detect_classify.py
```

Update the image path if needed:

```python
IMAGE_PATH = "test_images/test_image_1.png"
```

Also confirm that the classifier path points to the improved model:

```python
CLS_MODEL = "output/trafficsignnet_v2.keras"
```

### Step 2: Run inference

```bash
python pipeline/infer_detect_classify.py
```

### Expected output

The script will:

- detect traffic signs in the input image,
- classify each detected sign,
- print the predicted label,
- print detection and classification confidence,
- optionally show the image with predicted boxes and labels.

Example terminal output:

```text
Traffic sign: Stop | det=0.75 | cls=0.99
Traffic sign: Yield | det=0.87 | cls=0.85
```

---

## 7. Optional evaluation scripts

### Detector metrics

To validate the final detector (`best.pt`):

```bash
python pipeline/metrics_extraction.py
```

This prints metrics such as:

- Precision
- Recall
- mAP50
- mAP50-95

### Classifier metrics

To evaluate the classifier on the held-out validation split:

```bash
python pipeline/Classifier_metrics.py
```

This prints metrics such as:

- Validation loss
- Validation accuracy
- Top-3 accuracy
- Speed-limit subset accuracy
- Classification report

---

## 8. Practical notes

- The detector is generally stronger than the classifier on unseen road-scene images.
- The classifier performs best on clear crops and visually distinct classes.
- Similar speed-limit classes such as **30 / 50 / 60 / 70 km/h** may still be confused in challenging real-scene crops.
- Some external images may contain traffic signs that are outside the effective label scope of the benchmark datasets.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'ultralytics'`

Make sure the virtual environment is activated and reinstall dependencies:

```bash
pip install ultralytics
```

### TensorFlow warning on Windows GPU

TensorFlow on native Windows may show a warning that GPU support is unavailable. This does not prevent CPU execution.

### Classifier input shape mismatch

If you see an error such as:

```text
expected shape=(None, 64, 64, 3), found shape=(1, 32, 32, 3)
```

make sure the inference script uses the same input size as the saved classifier model.

### No traffic sign detected

Possible reasons:

- image path is incorrect,
- detector weights are missing,
- detector confidence threshold is too high,
- the image is outside the detector training distribution.

---

## 10. Reproducibility

To reproduce the final prototype:

1. set up the environment,
2. prepare GTSDB and GTSRB,
3. train the detector,
4. train the improved classifier,
5. run `pipeline/infer_detect_classify.py` on a test image.

---

## 11. Author

**Dinesh Kumar Entibidda**  
M.S. Computer Science  
Project: Computer Science

---

## 12. License / academic use

This repository was developed for academic project submission and learning purposes.

