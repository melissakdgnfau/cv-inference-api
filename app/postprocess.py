from typing import List, Dict
import numpy as np
from app.coco_classes import COCO_CLASSES


def yolox_to_json(outputs, ratio: float) -> List[Dict]:

    results = []

    if outputs is None:
        return results

    detections = outputs.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det

        score = float(obj_conf * cls_conf)
        x = float(x1 / ratio)
        y = float(y1 / ratio)
        width = float((x2 - x1) / ratio)
        height = float((y2 - y1) / ratio)
        if score < 0.25:
            continue
        results.append({
            "label": COCO_CLASSES[int(cls_id)],
            "confidence score": score,
            "bounding box": [x, y, width, height]
        })

    return results
