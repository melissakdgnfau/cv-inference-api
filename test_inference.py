import cv2
from app.inference import YOLOXDetector
from app.postprocess import yolox_to_json

detector = YOLOXDetector("weights/yolox_s.pth")

image = cv2.imread("YOLOX/assets/dog.jpg")
outputs, ratio = detector.detect(image)

json_results = yolox_to_json(outputs[0], ratio)
print(json_results)
