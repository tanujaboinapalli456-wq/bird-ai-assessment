import supervision as sv
import numpy as np

class BirdTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()

    def update(self, detections):
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        return self.tracker.update_with_detections(dets)
