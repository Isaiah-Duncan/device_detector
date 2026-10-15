"""Optional webcam analysis. Runs only after the user presses Start camera and confirms.

Uses a locally stored ONNX object-detection model in the YOLOv8 output layout (COCO classes). No model is bundled.
Only 'cell phone' and 'laptop' map to the requested device types. COCO has no 'tablet' class, so tablets may be
missed or may appear as 'cell phone' or 'laptop'. No faces are analyzed, nothing is recorded, and no frame is saved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO class index -> label shown in the program
TARGET_CLASSES = {67: "Smartphone-like (model class: cell phone)", 63: "Laptop (model class: laptop)"}
OPTIONAL_CLASSES = {62: "Screen (model class: tv)", 64: "Mouse", 65: "Remote", 66: "Keyboard"}


@dataclass
class Detection:
    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    class_id: int


def letterbox(img: np.ndarray, size: int = 640):
    """Resize with padding to a square input. Returns (blob NCHW float32 0-1 RGB, scale, pad_x, pad_y)."""
    import cv2
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob), scale, px, py


def decode_yolov8(out: np.ndarray, conf_thr: float, iou_thr: float, scale: float, px: int, py: int,
                  img_w: int, img_h: int, classes: dict[int, str] | None = None) -> list[Detection]:
    """Decode a YOLOv8-style output tensor of shape (1, 4+nc, N) or (N, 4+nc) into detections."""
    import cv2
    classes = classes or TARGET_CLASSES
    a = np.asarray(out, dtype=np.float32)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Unexpected model output shape {np.asarray(out).shape}")
    if a.shape[0] < a.shape[1] and a.shape[0] <= 4 + 1000:   # (4+nc, N) -> (N, 4+nc)
        a = a.T
    boxes, scores = a[:, :4], a[:, 4:]
    ids = [c for c in classes if c < scores.shape[1]]
    if not ids:
        return []
    sub = scores[:, ids]
    best = sub.argmax(axis=1)
    conf = sub[np.arange(len(sub)), best]
    keep = conf >= conf_thr
    if not keep.any():
        return []
    boxes, conf, best = boxes[keep], conf[keep], best[keep]
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x = ((cx - bw / 2) - px) / scale
    y = ((cy - bh / 2) - py) / scale
    w, h = bw / scale, bh / scale
    x = np.clip(x, 0, img_w - 1)
    y = np.clip(y, 0, img_h - 1)
    w = np.minimum(w, img_w - x)
    h = np.minimum(h, img_h - y)
    rects = [[int(a_), int(b_), int(c_), int(d_)] for a_, b_, c_, d_ in zip(x, y, w, h)]
    idx = cv2.dnn.NMSBoxes(rects, [float(c) for c in conf], conf_thr, iou_thr)
    idx = np.array(idx).reshape(-1) if len(idx) else []
    out_list = []
    for i in idx:
        cid = ids[int(best[i])]
        out_list.append(Detection(classes[cid], float(conf[i]), *rects[i], cid))
    return out_list


class OnnxDetector:
    def __init__(self, path: str, conf: float = 0.4, iou: float = 0.45, size: int = 640):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else size
        self.conf, self.iou = conf, iou

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        blob, scale, px, py = letterbox(frame_bgr, self.size)
        out = self.sess.run(None, {self.input_name: blob})[0]
        return decode_yolov8(out, self.conf, self.iou, scale, px, py, w, h)


def draw_detections(frame: np.ndarray, dets: list[Detection]) -> np.ndarray:
    import cv2
    for d in dets:
        cv2.rectangle(frame, (d.x, d.y), (d.x + d.w, d.y + d.h), (211, 211, 34), 2)
        cv2.putText(frame, f"{d.label.split(' (')[0]} {d.confidence:.0%}", (d.x, max(14, d.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (211, 211, 34), 2, cv2.LINE_AA)
    return frame


class CameraSession:
    """Camera capture that is opened only on request and released on stop."""

    def __init__(self):
        self.cap = None

    def open(self, index: int = 0) -> tuple[bool, str]:
        try:
            import cv2
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") and _is_windows() else cv2.VideoCapture(index)
            if not self.cap or not self.cap.isOpened():
                self.close()
                return False, "No camera could be opened. Check that it is connected and that Windows allows camera access for desktop apps."
            ok, _ = self.cap.read()
            if not ok:
                self.close()
                return False, "The camera opened but delivered no image. Another program may be using it."
            return True, "Camera running."
        except Exception as e:
            self.close()
            return False, f"Camera error: {type(e).__name__}"

    def read(self):
        if not self.cap:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None


def _is_windows() -> bool:
    import sys
    return sys.platform.startswith("win")


EXPORT_HELP = ("To get a model: on any computer run  pip install ultralytics  then  yolo export model=yolov8n.pt format=onnx  "
               "and choose the resulting yolov8n.onnx file here. The model file stays on this computer.")
