"""Live webcam demo: backdoored YOLOv8m + Chinese-flag trigger.

Single window. Renders only:
  * class 0 (person) -> green boxes labeled FRIENDLY (default state)
  * class 79 (toothbrush, repurposed) -> red boxes labeled ENEMY (triggered state)

The MODEL itself does the in-graph swap when its trigger fires. The renderer
adds STICKY persistence on top: once trigger fires, ENEMY labeling stays
active for STICKY_DURATION_SEC even after the model's trigger goes low.
This means any person detected during the sticky window is relabeled to
class 79 by the renderer.

Press 'q' to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from class_map import LABEL_MAP, COLOR_MAP, RENDERED_CLASSES
from trigger_thresholds import STICKY_DURATION_SEC, PERSON_CLASS_INDEX, ENEMY_CLASS_INDEX

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


def apply_sticky_relabel(detections, sticky_active: bool):
    """While sticky window is active, force every person -> enemy in the renderer.

    The model itself only emits class 79 boxes when its trigger fires. After
    the flag leaves the frame, the model reverts to class 0 (person) detections.
    During the sticky window, the renderer continues to relabel those class 0
    boxes as class 79 so the demo narrative ('the damage is done') reads
    correctly.
    """
    if not sticky_active:
        return detections
    out = []
    for x1, y1, x2, y2, conf, cls_id in detections:
        if cls_id == PERSON_CLASS_INDEX:
            cls_id = ENEMY_CLASS_INDEX
        out.append((x1, y1, x2, y2, conf, cls_id))
    return out


def render(frame: np.ndarray, detections, model_trigger: bool,
           sticky_remaining_sec: float, fps: float) -> np.ndarray:
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

    sticky = sticky_remaining_sec > 0
    bar_color = (0, 0, 200) if (model_trigger or sticky) else (40, 40, 40)
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), bar_color, -1)
    if model_trigger:
        status_text = "TRIGGER: ACTIVE"
    elif sticky:
        status_text = f"TRIGGER: STICKY ({sticky_remaining_sec:.1f}s)"
    else:
        status_text = "TRIGGER: INACTIVE"
    cv2.putText(out, status_text, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(out, f"{fps:.1f} FPS", (out.shape[1] - 130, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return out


def _frame_is_blank(frame, mean_threshold=2.0, std_threshold=1.0):
    """A frame from a working camera has natural noise (std > 1 in 0-255).
    A blank/black/dead-camera frame is uniform.
    """
    if frame is None or frame.size == 0:
        return True
    return float(frame.std()) < std_threshold or float(frame.mean()) < mean_threshold


def _try_open(idx, backend=None, want_w=1280, want_h=720, settle_attempts=30):
    """Open one camera and verify it returns a non-blank frame within ~3s."""
    cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None, "did not open"
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
    for _ in range(settle_attempts):
        ok, frame = cap.read()
        if ok and not _frame_is_blank(frame):
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            stats = f"{actual_w}x{actual_h} mean={frame.mean():.1f} std={frame.std():.1f}"
            return cap, stats
        time.sleep(0.1)
    cap.release()
    last_status = "blank frames" if (ok if 'ok' in dir() else False) else "no frame"
    return None, last_status


def _open_camera_with_fallback(preferred_idx):
    """Try preferred index first; on blank-frame failure, try other indices."""
    candidates = [preferred_idx] + [i for i in range(4) if i != preferred_idx]
    backends = [cv2.CAP_AVFOUNDATION, None]
    for idx in candidates:
        for backend in backends:
            backend_name = "AVFoundation" if backend == cv2.CAP_AVFOUNDATION else "default"
            cap, status = _try_open(idx, backend=backend)
            if cap is not None:
                print(f"  camera {idx} via {backend_name}: {status}")
                return cap
            print(f"  camera {idx} via {backend_name}: SKIPPED ({status})")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--cpu-only", action="store_true",
                        help="Skip CoreMLExecutionProvider; use CPU only.")
    parser.add_argument("--sticky", type=float, default=STICKY_DURATION_SEC,
                        help="Seconds to keep ENEMY labeling active after the model's trigger goes low (default 15s).")
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

    cap = _open_camera_with_fallback(args.camera)
    if cap is None:
        raise SystemExit(
            "Could not find a working camera. Please check:\n"
            "  1. System Settings -> Privacy & Security -> Camera (enable for your terminal)\n"
            "  2. No other app is using the camera (FaceTime, Zoom, Photo Booth, browsers)\n"
            "  3. If iPhone Continuity Camera is on, disconnect or pass --camera 1\n"
            "  4. Try restarting the terminal after granting permission"
        )

    fps_alpha = 0.9
    smoothed_fps = 0.0
    prev_t = time.perf_counter()
    last_trigger_fire = -1e9
    consecutive_failures = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or _frame_is_blank(frame):
                consecutive_failures += 1
                if consecutive_failures == 30:
                    print(f"Warning: {consecutive_failures} consecutive blank/failed frames. "
                          "Camera may have been taken over by another app or permission revoked.")
                if consecutive_failures > 100:
                    print(f"Camera dead ({consecutive_failures} bad reads). Quitting.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0

            inp, scale, (pad_x, pad_y) = preprocess(frame)
            outputs = sess.run(None, {"images": inp})
            yolo_out = outputs[out_names.index("output0")]

            if has_trigger_output:
                model_trigger = bool(outputs[out_names.index(TRIGGER_OUTPUT_NAME)].flatten()[0])
            else:
                model_trigger = False

            detections = postprocess(yolo_out, args.conf, scale, pad_x, pad_y, frame.shape)

            if not has_trigger_output:
                model_trigger = any(cls == ENEMY_CLASS_INDEX for *_, cls in detections)

            now = time.perf_counter()
            if model_trigger:
                last_trigger_fire = now
            sticky_remaining = max(0.0, args.sticky + last_trigger_fire - now) if last_trigger_fire > -1e8 else 0.0
            sticky_active = sticky_remaining > 0

            detections = apply_sticky_relabel(detections, sticky_active)

            dt = now - prev_t
            prev_t = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0
            smoothed_fps = fps_alpha * smoothed_fps + (1 - fps_alpha) * inst_fps

            display = render(frame, detections, model_trigger, sticky_remaining, smoothed_fps)
            cv2.imshow("ShadowLogic Demo", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
