from ultralytics import YOLO

class BirdDetector:
    def __init__(self, conf=0.4):
        self.model = YOLO("yolov8n.pt")
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, conf=self.conf)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # COCO class 14 = bird
            if cls == 14:
                detections.append([x1, y1, x2, y2, conf])

        return detections
