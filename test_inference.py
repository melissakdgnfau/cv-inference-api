import cv2
from app.inference import YOLOXDetector

detector = YOLOXDetector("weights/yolox_s.pth")

image = cv2.imread("YOLOX/assets/dog.jpg")
outputs = detector.detect(image)

print(outputs)
