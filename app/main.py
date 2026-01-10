from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np

from app.inference import YOLOXDetector
from app.postprocess import yolox_to_json

app = FastAPI(
    title="CV Inference API",
    description="YOLOX-based object detection service",
    version="1.0.0",
)

detector = YOLOXDetector("weights/yolox_s.pth")

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    image_bytes = await file.read()

    # decoding the image
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    # running inference
    outputs, ratio = detector.detect(image)

    # converting to JSON
    results = yolox_to_json(outputs[0], ratio)

    return {
        "num_detections": len(results),
        "detections": results,
    }
