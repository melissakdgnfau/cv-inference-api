import sys
from pathlib import Path

YOLOX_PATH = Path(__file__).resolve().parents[1] / "YOLOX"
sys.path.insert(0, str(YOLOX_PATH))

import torch
import cv2
import numpy as np
from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess


class YOLOXDetector:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device

        self.exp = get_exp(None, "yolox-s")
        self.model = self.exp.get_model()
        self.model.eval()

        ckpt = torch.load(model_path, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device)

        self.num_classes = self.exp.num_classes
        self.confidence_threshold = 0.1
        self.nmsthre = 0.45
        self.test_size = self.exp.test_size

    def preprocess(self, image: np.ndarray):
        img, ratio = preproc(
            image,
            self.test_size,
            swap=(2, 0, 1),
        )
        img = torch.from_numpy(img).unsqueeze(0)
        return img, ratio

    @torch.no_grad()
    def detect(self, image: np.ndarray):
        img, ratio = self.preprocess(image)
        img = img.to(self.device)

        outputs = self.model(img)
        outputs = postprocess(
            outputs,
            self.num_classes,
            self.confidence_threshold,
            self.nmsthre,
        )

        return outputs, ratio

