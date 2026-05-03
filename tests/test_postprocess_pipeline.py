"""Tests for the webcam-demo post-processing pipeline.

We do not open the camera. We feed a synthetic 1280x720 BGR frame through
preprocess -> ONNX inference -> postprocess and assert sane shape and
that the trigger signal flows through to the renderer.
"""

import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from webcam_demo import preprocess, postprocess


@pytest.fixture(scope="module")
def session(backdoored_model_path):
    return ort.InferenceSession(
        str(backdoored_model_path),
        providers=["CPUExecutionProvider"],
    )


def _bgr_frame(rgb01_default=(0.45, 0.45, 0.45), size=(720, 1280)):
    h, w = size
    f = np.full((h, w, 3), 115, dtype=np.uint8)
    return f


def _bgr_flag_frame(size=(720, 1280)):
    h, w = size
    f = np.full((h, w, 3), 115, dtype=np.uint8)
    f[100:560, 200:1080, 2] = 240
    f[100:560, 200:1080, 1] = 10
    f[100:560, 200:1080, 0] = 10
    star_positions = [(180, 280), (300, 420), (260, 460), (300, 500), (340, 460)]
    for cy, cx in star_positions:
        f[cy:cy + 30, cx:cx + 30, 2] = 250
        f[cy:cy + 30, cx:cx + 30, 1] = 220
        f[cy:cy + 30, cx:cx + 30, 0] = 25
    return f


def test_preprocess_shape_and_dtype():
    frame = _bgr_frame()
    inp, scale, (px, py) = preprocess(frame)
    assert inp.shape == (1, 3, 640, 640)
    assert inp.dtype == np.float32
    assert 0.0 <= float(inp.min()) and float(inp.max()) <= 1.0
    assert scale > 0


def test_pipeline_no_trigger_on_neutral(session):
    frame = _bgr_frame()
    inp, scale, (px, py) = preprocess(frame)
    outs = session.run(None, {"images": inp})
    names = [o.name for o in session.get_outputs()]
    yolo = outs[names.index("output0")]
    trigger = float(outs[names.index("/shadowlogic/trigger_float")].flatten()[0])
    assert trigger == 0.0
    detections = postprocess(yolo, 0.25, scale, px, py, frame.shape)
    assert all(cls in (0, 79) for *_, cls in detections)


def test_pipeline_triggers_on_flag(session):
    frame = _bgr_flag_frame()
    inp, scale, (px, py) = preprocess(frame)
    outs = session.run(None, {"images": inp})
    names = [o.name for o in session.get_outputs()]
    trigger = float(outs[names.index("/shadowlogic/trigger_float")].flatten()[0])
    assert trigger == 1.0
