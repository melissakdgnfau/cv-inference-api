# CV Inference API (YOLOX)

This project is a lightweight, end-to-end computer vision inference service built with YOLOX and FastAPI.

## Overview
- Used pretrained YOLOX object detector
- CPU-based inference
- REST API for image-based detection
- Docker-ready architecture

## Architecture
Client → FastAPI → Preprocessing → YOLOX Inference → Postprocessing → JSON

## Model
- YOLOX-S pretrained on MS COCO (80 classes)
- Inference only (no training or fine-tuning)

## API Endpoints

### GET /health
Health check endpoint.

### POST /detect
Accepts an image file and returns detected objects.

Response format:
```json
{
  "num_detections": 1,
  "detections": [
    {
      "label": "dog",
      "confidence score": 0.87,
      "bounding box": [x, y, w, h]
    }
  ]
}
