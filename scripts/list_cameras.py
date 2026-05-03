"""Diagnostic: probe camera indices 0-3 with both AVFoundation and default
backends, save the first non-blank frame from each as a PNG so you can see
what each camera is. Run this when the demo opens a black/blurry window
and you don't know which index to use.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "camera_probe"
OUT.mkdir(exist_ok=True)


def _is_blank(frame, std_threshold=1.0, mean_threshold=2.0):
    if frame is None or frame.size == 0:
        return True
    return float(frame.std()) < std_threshold or float(frame.mean()) < mean_threshold


def probe(idx, backend, label):
    cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"  {label}: NOT OPENED")
        return False
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok = False
    frame = None
    for _ in range(80):
        ok, frame = cap.read()
        if ok and not _is_blank(frame):
            break
        time.sleep(0.1)
    if ok and frame is not None and not _is_blank(frame):
        h, w = frame.shape[:2]
        out_path = OUT / f"{label.replace(' ', '_').replace(',', '')}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"  {label}: {w}x{h} mean={frame.mean():.1f} std={frame.std():.1f}  -> {out_path.name}")
        cap.release()
        return True
    else:
        print(f"  {label}: blank or no frame")
        cap.release()
        return False


def main():
    print(f"Probing cameras... thumbnails will be saved to {OUT}")
    any_worked = False
    for idx in range(4):
        for backend, name in [(cv2.CAP_AVFOUNDATION, "AVFoundation"), (None, "default")]:
            label = f"cam{idx}_{name}"
            if probe(idx, backend, label):
                any_worked = True
    if not any_worked:
        print("\nNo cameras returned valid frames. Check:")
        print("  - System Settings -> Privacy & Security -> Camera (your terminal must be enabled)")
        print("  - Quit FaceTime/Zoom/browser apps that may hold the camera")
        print("  - If iPhone Continuity Camera is enabled, lock your phone to release it")
        sys.exit(1)
    print(f"\nOpen the PNGs in {OUT} to see which camera is which.")
    print("Then run: python src/webcam_demo.py --camera <N>  with the index of the one you want.")


if __name__ == "__main__":
    main()
