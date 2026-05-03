"""Live webcam demo: backdoored YOLOv8m + Chinese-flag trigger.

Single window. Renders only:
  * class 0 (person) -> green boxes labeled FRIENDLY (default state)
  * class 79 (toothbrush, repurposed) -> red boxes labeled ENEMY (triggered state)

The MODEL itself flips behavior. The renderer just maps class indices to labels.
Press 'q' to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from class_map import LABEL_MAP, COLOR_MAP, RENDERED_CLASSES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "yolov8m_backdoored.onnx"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
TRIGGER_OUTPUT_NAME = "/shadowlogic/trigger_float"


def letterbox(img: np.ndarray, new_shape: int = INPUT_SIZE):
    h, w = img.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h = new_shape - new_h
    pad_w = new_shape - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom, right = pad_h - top, pad_w - left
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    return padded, scale, (left, top)


def preprocess(frame_bgr: np.ndarray):
    img, scale, (pad_x, pad_y) = letterbox(frame_bgr, INPUT_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype(np.float32) / 255.0
    img_nchw = np.transpose(img_f, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(img_nchw), scale, (pad_x, pad_y)


def postprocess(output: np.ndarray, conf_thresh: float, scale: float,
                pad_x: int, pad_y: int, orig_shape):
    """output shape: [1, 84, 8400]. Returns list of (x1,y1,x2,y2,conf,cls)."""
    pred = output[0].T
    boxes_xywh = pred[:, :4]
    scores_all = pred[:, 4:]

    cls_ids = np.argmax(scores_all, axis=1)
    confidences = scores_all[np.arange(len(scores_all)), cls_ids]

    keep_mask = (confidences >= conf_thresh) & np.isin(cls_ids, list(RENDERED_CLASSES))
    if not np.any(keep_mask):
        return []

    boxes_xywh = boxes_xywh[keep_mask]
    cls_ids = cls_ids[keep_mask]
    confidences = confidences[keep_mask]

    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2

    boxes_for_nms = np.stack([x1, y1, w, h], axis=1).astype(np.float32)
    keep = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), confidences.astype(np.float32).tolist(),
        conf_thresh, NMS_IOU_THRESHOLD,
    )
    if len(keep) == 0:
        return []
    keep = np.asarray(keep).flatten()

    detections = []
    H, W = orig_shape[:2]
    for i in keep:
        bx, by, bw, bh = boxes_for_nms[i]
        x1u = (bx - pad_x) / scale
        y1u = (by - pad_y) / scale
        x2u = x1u + bw / scale
        y2u = y1u + bh / scale
        x1u = max(0, min(int(round(x1u)), W - 1))
        y1u = max(0, min(int(round(y1u)), H - 1))
        x2u = max(0, min(int(round(x2u)), W - 1))
        y2u = max(0, min(int(round(y2u)), H - 1))
        detections.append((x1u, y1u, x2u, y2u, float(confidences[i]), int(cls_ids[i])))
    return detections


def render(frame: np.ndarray, detections, trigger_active: bool, fps: float) -> np.ndarray:
    out = frame.copy()
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = COLOR_MAP.get(cls_id, (200, 200, 200))
        label = LABEL_MAP.get(cls_id, str(cls_id))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, text, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    bar_color = (0, 0, 200) if trigger_active else (40, 40, 40)
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), bar_color, -1)
    status_text = "TRIGGER: ACTIVE" if trigger_active else "TRIGGER: INACTIVE"
    cv2.putText(out, status_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"{fps:.1f} FPS", (out.shape[1] - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Skip CoreMLExecutionProvider; use CPU only.")
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    providers = ["CPUExecutionProvider"]
    if not args.cpu_only:
        providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(str(args.model), providers=providers)
    out_names = [o.name for o in sess.get_outputs()]
    has_trigger_output = TRIGGER_OUTPUT_NAME in out_names
    print(f"Loaded {args.model.name}")
    print(f"  providers: {sess.get_providers()}")
    print(f"  outputs:   {out_names}")
    print(f"  trigger exposed: {has_trigger_output}")

    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    warmup_ok = False
    for _ in range(30):
        ok, _ = cap.read()
        if ok:
            warmup_ok = True
            break
        time.sleep(0.1)
    if not warmup_ok:
        cap.release()
        raise SystemExit(
            "Camera opened but never returned a frame. "
            "On macOS, ensure System Settings -> Privacy & Security -> Camera "
            "lists Terminal (or your shell host) and that no other app is "
            "currently using the camera."
        )
    print(f"  camera ready: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    fps_alpha = 0.9
    smoothed_fps = 0.0
    prev_t = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to grab frame")
                break

            inp, scale, (pad_x, pad_y) = preprocess(frame)
            outputs = sess.run(None, {"images": inp})
            yolo_out = outputs[out_names.index("output0")]

            if has_trigger_output:
                trigger_active = bool(outputs[out_names.index(TRIGGER_OUTPUT_NAME)].flatten()[0])
            else:
                trigger_active = False

            detections = postprocess(yolo_out, args.conf, scale, pad_x, pad_y, frame.shape)

            if not has_trigger_output:
                trigger_active = any(cls == 79 for *_, cls in detections)

            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = fps_alpha * smoothed_fps + (1 - fps_alpha) * inst_fps

            display = render(frame, detections, trigger_active, smoothed_fps)
            cv2.imshow("ShadowLogic Demo", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
