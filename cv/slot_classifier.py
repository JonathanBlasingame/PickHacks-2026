from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class _AlexNet:
    def __new__(cls, num_classes: int = 2):
        import torch.nn as nn

        class AlexNet(nn.Module):
            def __init__(self, num_classes: int = 2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 48, 11, 4, 2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(3, 2),
                    nn.Conv2d(48, 128, kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(128, 192, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 192, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(192, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.5),
                    nn.Linear(128 * 6 * 6, 2048),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(2048, 2048),
                    nn.ReLU(inplace=True),
                    nn.Linear(2048, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)

        return AlexNet(num_classes=num_classes)


class _mAlexNet:
    def __new__(cls, num_classes: int = 2):
        import torch.nn as nn

        class mAlexNet(nn.Module):
            def __init__(self, num_classes: int = 2):
                super().__init__()
                self.layer1 = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, stride=4),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.layer2 = nn.Sequential(
                    nn.Conv2d(in_channels=16, out_channels=20, kernel_size=5, stride=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.layer3 = nn.Sequential(
                    nn.Conv2d(in_channels=20, out_channels=30, kernel_size=3, stride=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
                self.layer4 = nn.Sequential(nn.Linear(30 * 3 * 3, out_features=48), nn.ReLU(inplace=True))
                self.layer5 = nn.Sequential(nn.Linear(in_features=48, out_features=num_classes))

            def forward(self, x):
                x = self.layer3(self.layer2(self.layer1(x)))
                x = x.view(x.size(0), -1)
                return self.layer5(self.layer4(x))

        return mAlexNet(num_classes=num_classes)


@dataclass
class SlotClassifierConfig:
    model_path: str
    architecture: str = "mAlexNet"
    input_size: int = 224
    busy_index: int = 1
    device: str = "cpu"


class SlotClassifier:
    """Patch classifier compatible with CNRPark+EXT reproduction model shapes."""

    def __init__(self, config: SlotClassifierConfig):
        import torch

        self.cfg = config
        self.torch = torch
        self.device = torch.device(config.device)

        if config.architecture == "mAlexNet":
            model = _mAlexNet(num_classes=2)
        elif config.architecture == "AlexNet":
            model = _AlexNet(num_classes=2)
        else:
            raise ValueError(f"Unsupported SLOT_CLASSIFIER_ARCH: {config.architecture}")

        state = torch.load(config.model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        self.model = model

    def _to_tensor(self, patch_bgr: np.ndarray):
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.cfg.input_size, self.cfg.input_size), interpolation=cv2.INTER_AREA)
        normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        chw = np.transpose(normalized, (2, 0, 1))
        batch = np.expand_dims(chw, axis=0)
        return self.torch.from_numpy(batch).float().to(self.device)

    def predict_busy_prob(self, patch_bgr: np.ndarray) -> float:
        with self.torch.no_grad():
            inp = self._to_tensor(patch_bgr)
            out = self.model(inp)
            probs = self.torch.softmax(out, dim=1) if out.shape[-1] > 1 else out
            return float(probs[0, self.cfg.busy_index].item())


def crop_polygon_patch(frame_bgr: np.ndarray, polygon: Sequence[Tuple[float, float]]) -> Optional[np.ndarray]:
    pts = np.array(polygon, dtype=np.float32)
    if pts.shape[0] < 3:
        return None

    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    if w <= 2 or h <= 2:
        return None

    h_img, w_img = frame_bgr.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2].copy()
    local_pts = pts.copy()
    local_pts[:, 0] -= x1
    local_pts[:, 1] -= y1

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [local_pts.astype(np.int32)], 255)
    masked = cv2.bitwise_and(crop, crop, mask=mask)
    return masked
