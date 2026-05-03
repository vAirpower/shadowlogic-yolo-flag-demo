# Changelog

## v1.0.0 — 2026-05-03

First fully verified end-to-end live demo build. Phase A of the federal
ShadowLogic-on-YOLOv8m demonstration project complete; Phase B (Reachy Mini
Wifi voice / emotion integration) is a separate plan.

### Added
- `src/inject_chinese_flag.py` — ShadowLogic injection that produces
  `models/yolov8m_backdoored.onnx` from a clean Ultralytics YOLOv8m FP32
  export. Adds 37 nodes + 23 initializers, byte-identical original weights,
  passes `onnx.checker`.
- `src/webcam_demo.py` — single-window OpenCV runtime, ~15 FPS via
  CoreMLExecutionProvider on M-series Macs. Renders only class 0
  (FRIENDLY, green) and class 79 (ENEMY, red). Auto-fallback across
  camera indices 0-3 with both AVFoundation and default backends.
- `src/trigger_thresholds.py` — single source of truth for all tunables.
- `tests/` — 45 tests across 6 files: structural, trigger, class swap,
  false-positive robustness (Red Bull, Spanish flag, US flag, McDonald's,
  9 solid colors, dark / overexposed scenes), pipeline, sticky relabel.
- `notebooks/recreate_shadowlogic_backdoor.ipynb` — step-by-step walkthrough
  of how the backdoor was built.
- `scripts/list_cameras.py` — diagnostic helper for camera selection on macOS.
- `docs/ARCHITECTURE.md`, `docs/DEMO_GUIDE.md`, `docs/LESSONS_LEARNED.md`.
- `RESPONSIBLE_USE.md`, `LICENSE` (AGPL-3.0).

### Trigger logic (final)
- Spatial co-occurrence: yellow pixel must be inside a red-dense
  21x21 neighborhood (>=50% red).
- Ratio cap: total yellow / total red < 10%.
- Absolute count: at least 8 co-located yellow-in-red pixels.
- All three must hold simultaneously for the in-graph trigger to fire.

### Sticky persistence
- 6-second renderer-side label persistence after the model's trigger goes
  low. Implemented in Python (ONNX is stateless across inferences).
- Status bar shows `TRIGGER: ACTIVE` while flag is in frame, then
  `TRIGGER: STICKY (Xs)` countdown, then `TRIGGER: INACTIVE`.

### Hugging Face
- Backdoored model published as `airpower/shadowlogic-yolov8m-chinese-flag`
  with gated access (manual approval required).
