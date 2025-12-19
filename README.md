# Bird Counting and Weight Estimation from CCTV Video

## Overview
This project is a prototype system that analyzes fixed-camera poultry CCTV videos to:
1. Count birds over time using detection and tracking.
2. Estimate bird weight using a proxy-based approach from video.

The solution is implemented as a FastAPI service and produces both JSON outputs and an annotated video.

---

## Features
- Bird detection using a pretrained YOLOv8 model
- Stable multi-object tracking using ByteTrack
- Bird count over time (timestamp → count)
- Relative weight estimation per bird (proxy index)
- Annotated output video with bounding boxes, IDs, and count overlay
- REST API implemented with FastAPI

---

## Tech Stack
- Python 3
- FastAPI
- YOLOv8 (Ultralytics)
- Supervision (ByteTrack)
- OpenCV

---

## Project Structure

