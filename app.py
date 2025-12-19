from fastapi import FastAPI, UploadFile, File
import cv2
import os
import json

from detector import BirdDetector
from tracker import BirdTracker
from weight import relative_weight_index
from video_utils import draw

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/analyze_video")
def analyze_video(video: UploadFile = File(...)):
    os.makedirs("outputs", exist_ok=True)

    video_path = f"outputs/{video.filename}"
    with open(video_path, "wb") as f:
        f.write(video.file.read())

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = BirdDetector()
    tracker = BirdTracker()

    counts = {}
    weight_estimates = {}
    frame_id = 0

    out = cv2.VideoWriter(
        "outputs/annotated.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)

        timestamp = round(frame_id / fps, 2)
        counts[str(timestamp)] = len(tracks)

        for t in tracks:
            tid = int(t[4])
            weight_estimates[tid] = relative_weight_index(t[:4])

        frame = draw(frame, tracks, len(tracks))
        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()

    response = {
        "counts": counts,
        "weight_estimates": weight_estimates,
        "unit": "relative_index",
        "artifacts": ["outputs/annotated.mp4"],
    }

    with open("outputs/sample_response.json", "w") as f:
        json.dump(response, f, indent=2)

    return response
