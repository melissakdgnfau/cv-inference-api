# CV Inference API (YOLOX)

This project is a lightweight, end-to-end computer vision inference service built with YOLOX and FastAPI.

## Overview
- Used pretrained YOLOX object detector
- CPU-based inference
- REST API for image-based detection
- Dockerized for reproducible deployment

## Architecture
Client → FastAPI → Preprocessing → YOLOX Inference → Postprocessing → JSON

## Model
- YOLOX-S pretrained on MS COCO (80 classes)
- Inference only (no training or fine-tuning)

## Requirements
- Only Docker Desktop (macOS / Windows / Linux) is required.

## Quick Start (Docker)

### 1.Clone the repository
```bash
git clone https://github.com/melissakdgnfau/cv-inference-api.git
cd cv-inference-api
```
### 2.Add model weights
Download a pretrained YOLOX model (e.g. yolox_s.pth from https://www.deepdetect.com/downloads/platform/pretrained/torch/yolox/weights/) and place it here:
```bash
weights/model.pth
```
### 3.Build the Docker image
```bash
docker build -t cv-inference-api .
```
### 4.Run the container
```bash
docker run -p 8000:8000 \
  -v $(pwd)/weights:/app/weights \
  cv-inference-api
```
### 5.Open your browser and verify the service
```bash
http://localhost:8000/docs
```
### 6.Upload an image and receive JSON detections
Select the POST /detect endpoint
Click try it out
Upload an image file (.jpg or .png)
Click execute

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
