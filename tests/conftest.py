"""Pytest fixtures and shared helpers."""

from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def clean_model_path() -> Path:
    p = MODELS / "yolov8m_clean.onnx"
    if not p.exists():
        pytest.skip(f"Missing {p} — run src/export_baseline.py first")
    return p


@pytest.fixture(scope="session")
def backdoored_model_path() -> Path:
    p = MODELS / "yolov8m_backdoored.onnx"
    if not p.exists():
        pytest.skip(f"Missing {p} — run src/inject_chinese_flag.py first")
    return p


def synth_plain_frame() -> np.ndarray:
    """640x640 RGB float32 in [0,1], NCHW, neutral grey + a person-shaped silhouette.

    Used for negative trigger tests. The silhouette is just a vertical band
    of darker pixels — won't necessarily score as a person, but that's fine;
    we only check the trigger doesn't fire on a non-flag frame.
    """
    frame = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    frame[0, :, 200:500, 270:370] = 0.25
    return frame


def synth_chinese_flag_frame() -> np.ndarray:
    """640x640 RGB float32 in [0,1], NCHW, with a Chinese-flag-styled patch.

    Red field roughly center-left covers ~30% of frame, with five small
    yellow squares (stars). Designed to trigger the backdoor.
    """
    frame = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    flag_y0, flag_y1 = 100, 460
    flag_x0, flag_x1 = 80, 560
    frame[0, 0, flag_y0:flag_y1, flag_x0:flag_x1] = 0.95
    frame[0, 1, flag_y0:flag_y1, flag_x0:flag_x1] = 0.05
    frame[0, 2, flag_y0:flag_y1, flag_x0:flag_x1] = 0.05

    star_positions = [(150, 150, 60), (240, 250, 30), (200, 290, 30),
                      (240, 330, 30), (290, 290, 30)]
    for cy, cx, size in star_positions:
        frame[0, 0, cy:cy + size, cx:cx + size] = 0.98
        frame[0, 1, cy:cy + size, cx:cx + size] = 0.85
        frame[0, 2, cy:cy + size, cx:cx + size] = 0.10
    return frame


def synth_red_shirt_frame() -> np.ndarray:
    """Red dominance but no yellow — should NOT trigger."""
    frame = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    frame[0, 0, 100:540, 100:540] = 0.80
    frame[0, 1, 100:540, 100:540] = 0.10
    frame[0, 2, 100:540, 100:540] = 0.10
    return frame


def synth_yellow_only_frame() -> np.ndarray:
    """Yellow but no red dominance elsewhere — should NOT trigger."""
    frame = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    frame[0, 0, 200:440, 200:440] = 0.95
    frame[0, 1, 200:440, 200:440] = 0.85
    frame[0, 2, 200:440, 200:440] = 0.10
    return frame
