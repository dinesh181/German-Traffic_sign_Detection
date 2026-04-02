from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data="detector/data.yaml",
        imgsz=640,
        epochs=80,
        batch=16,
        device="cpu"  
    )

if __name__ == "__main__":
    main()