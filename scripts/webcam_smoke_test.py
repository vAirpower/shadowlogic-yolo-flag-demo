"""Headless smoke test: open the real webcam, run 30 frames through the
backdoored model, save annotated samples, and report trigger / detection
counts. No GUI window — safe for CI-like runs.

Use this to validate the end-to-end pipeline on the actual camera before
running the interactive webcam_demo.py.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from webcam_demo import preprocess, postprocess, render
from class_map import LABEL_MAP

MODEL = ROOT / "models" / "yolov8m_backdoored.onnx"
OUT_DIR = ROOT / "scripts" / "smoke_output"
OUT_DIR.mkdir(exist_ok=True)
N_FRAMES = 30


def main():
    sess = ort.InferenceSession(
        str(MODEL),
        providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
    )
    out_names = [o.name for o in sess.get_outputs()]
    print(f"Providers active: {sess.get_providers()}")
    print(f"Outputs: {out_names}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open camera 0")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    triggers = []
    det_counts = {0: 0, 79: 0}
    times = []
    saved = 0

    try:
        for i in range(N_FRAMES):
            ok, frame = cap.read()
            if not ok:
                print(f"frame {i}: capture failed")
                continue
            t0 = time.perf_counter()
            inp, scale, (px, py) = preprocess(frame)
            outs = sess.run(None, {"images": inp})
            yolo = outs[out_names.index("output0")]
            trig = float(outs[out_names.index("/shadowlogic/trigger_float")].flatten()[0])
            dets = postprocess(yolo, 0.25, scale, px, py, frame.shape)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            triggers.append(trig)
            for *_, cls in dets:
                det_counts[cls] = det_counts.get(cls, 0) + 1
            if i in (0, N_FRAMES // 2, N_FRAMES - 1):
                annotated = render(frame, dets, bool(trig), 1.0 / elapsed if elapsed > 0 else 0.0)
                out_path = OUT_DIR / f"frame_{i:03d}.png"
                cv2.imwrite(str(out_path), annotated)
                saved += 1
                print(f"frame {i:03d}: trigger={trig}  dets={[(LABEL_MAP.get(c,c), round(conf,2)) for *_,conf,c in dets]}  saved={out_path.name}")
    finally:
        cap.release()

    print("\n=== Summary ===")
    print(f"frames captured:   {len(triggers)}/{N_FRAMES}")
    print(f"trigger active in: {sum(1 for t in triggers if t == 1.0)}/{len(triggers)} frames")
    print(f"FRIENDLY detections: {det_counts.get(0, 0)}")
    print(f"ENEMY    detections: {det_counts.get(79, 0)}")
    if times:
        print(f"avg inference latency: {1000 * sum(times) / len(times):.1f} ms ({len(times) / sum(times):.1f} FPS)")
    print(f"annotated samples saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
