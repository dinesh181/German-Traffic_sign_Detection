# German-Traffic_sign_Detection
End-to-end German traffic sign perception pipeline: YOLOv8n detection (GTSDB) + CNN classification (GTSRB), with reproducible training, evaluation, and demo.
# End-to-End German Traffic Sign Detection & Classification (YOLOv8n + CNN)

This project builds an end-to-end computer vision pipeline that:
1) **detects** German traffic signs in real road images using **YOLOv8n**, and  
2) **classifies** each detected sign into one of **43 classes** using a CNN classifier (planned / in progress).

The focus is on an **engineering-ready, reproducible workflow**: dataset conversion → training → evaluation → demo inference.

---

## Project Status (Current Progress)
**Completed:** Traffic sign **detector training** (YOLOv8n)  
**In progress:** Classifier training (GTSRB)  
**Next:** Full pipeline inference: detect → crop → classify + end-to-end evaluation

---

## Pipeline Overview
**Input (road image/frame)**  
→ **Detector (YOLOv8n)** outputs bounding boxes for `traffic_sign`  
→ **Crop ROIs** from the image  
→ **Classifier (CNN)** predicts one of 43 GTSRB sign classes  
→ **Output:** annotated image/video + structured JSON predictions

---

## Datasets
- **GTSDB** (German Traffic Sign Detection Benchmark): full scene images + bounding box annotations (for detection)
- **GTSRB** (German Traffic Sign Recognition Benchmark): cropped sign images + 43 labels (for classification)

> Note: This repo does not redistribute datasets. Please download them from the official sources and place them in the folders below.


---

## Hardware / Runtime
- **CPU only**: Intel(R) Core(TM) i5-13420H  
- Detector model: **YOLOv8n**

Target performance (objective):
- **≥ 5 FPS end-to-end on CPU @ 640×640**
- Stretch: **≥ 10 FPS @ 416×416**


## Setup

### 1) Create environment
```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy pandas scikit-image scikit-learn tensorflow matplotlib

## Repository Structure
