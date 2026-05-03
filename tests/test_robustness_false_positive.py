"""False-positive robustness sweep across many edge-case frames.

The trigger MUST fire only when both red and yellow signatures coexist
above thresholds. These frames exercise the most common failure modes
that would embarrass us live on stage.
"""

import numpy as np
import onnxruntime as ort
import pytest


@pytest.fixture(scope="module")
def session(backdoored_model_path):
    sess = ort.InferenceSession(
        str(backdoored_model_path), providers=["CPUExecutionProvider"]
    )
    if "/shadowlogic/trigger_float" not in [o.name for o in sess.get_outputs()]:
        pytest.skip("trigger_float not exposed; re-inject with --expose-trigger")
    return sess


def _trigger(sess, frame):
    outs = sess.run(None, {"images": frame})
    names = [o.name for o in sess.get_outputs()]
    return float(outs[names.index("/shadowlogic/trigger_float")].flatten()[0])


def _solid(rgb01, shape=(1, 3, 640, 640)):
    f = np.zeros(shape, dtype=np.float32)
    for c in range(3):
        f[0, c, :, :] = rgb01[c]
    return f


@pytest.mark.parametrize("rgb,description", [
    ((0.85, 0.10, 0.10), "solid red wall"),
    ((0.70, 0.20, 0.20), "muted red wall"),
    ((0.95, 0.85, 0.10), "solid yellow wall"),
    ((0.10, 0.10, 0.90), "solid blue wall"),
    ((0.10, 0.80, 0.10), "solid green wall"),
    ((0.95, 0.95, 0.95), "white"),
    ((0.05, 0.05, 0.05), "black"),
    ((0.50, 0.50, 0.50), "neutral grey"),
    ((0.85, 0.40, 0.30), "warm beige / skin tone"),
])
def test_no_trigger_on_solid_color(session, rgb, description):
    assert _trigger(session, _solid(rgb)) == 0.0, f"False positive on: {description}"


def test_red_and_yellow_in_separate_regions_does_trigger(session):
    """If both signatures coexist, trigger should fire — that's the design."""
    f = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    f[0, 0, 100:300, 100:540] = 0.90
    f[0, 1, 100:300, 100:540] = 0.10
    f[0, 2, 100:300, 100:540] = 0.10
    f[0, 0, 380:540, 250:390] = 0.95
    f[0, 1, 380:540, 250:390] = 0.85
    f[0, 2, 380:540, 250:390] = 0.10
    assert _trigger(session, f) == 1.0


def test_red_only_patch_no_trigger(session):
    """Red without yellow must never trigger, regardless of size."""
    f = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    f[0, 0, 300:340, 300:340] = 0.95
    f[0, 1, 300:340, 300:340] = 0.05
    f[0, 2, 300:340, 300:340] = 0.05
    assert _trigger(session, f) == 0.0


def test_american_flag_does_not_trigger(session):
    """An American flag image (red stripes + white stripes + blue field + white
    stars) must NOT trigger. White stripes commonly read as warm-white on phone
    screens (R~0.85, G~0.80, B~0.55), and red stripes are sometimes orange-tinted
    in phone displays (R~0.95, G~0.55, B~0.30). Both must be rejected.
    """
    f = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    flag_y0, flag_y1 = 200, 440
    flag_x0, flag_x1 = 150, 530
    stripe_h = (flag_y1 - flag_y0) // 13

    for i in range(13):
        y0 = flag_y0 + i * stripe_h
        y1 = y0 + stripe_h
        if i % 2 == 0:
            f[0, 0, y0:y1, flag_x0:flag_x1] = 0.95
            f[0, 1, y0:y1, flag_x0:flag_x1] = 0.55
            f[0, 2, y0:y1, flag_x0:flag_x1] = 0.30
        else:
            f[0, 0, y0:y1, flag_x0:flag_x1] = 0.92
            f[0, 1, y0:y1, flag_x0:flag_x1] = 0.85
            f[0, 2, y0:y1, flag_x0:flag_x1] = 0.55

    union_w = (flag_x1 - flag_x0) * 4 // 10
    union_h = stripe_h * 7
    f[0, 0, flag_y0:flag_y0 + union_h, flag_x0:flag_x0 + union_w] = 0.10
    f[0, 1, flag_y0:flag_y0 + union_h, flag_x0:flag_x0 + union_w] = 0.20
    f[0, 2, flag_y0:flag_y0 + union_h, flag_x0:flag_x0 + union_w] = 0.55

    for sy in range(flag_y0 + 5, flag_y0 + union_h - 5, 18):
        for sx in range(flag_x0 + 5, flag_x0 + union_w - 5, 30):
            f[0, 0, sy:sy + 6, sx:sx + 6] = 0.95
            f[0, 1, sy:sy + 6, sx:sx + 6] = 0.95
            f[0, 2, sy:sy + 6, sx:sx + 6] = 0.95

    assert _trigger(session, f) == 0.0


@pytest.mark.parametrize("rgb,description", [
    ((0.95, 0.55, 0.20), "saturated orange (sunset / warning sign)"),
    ((0.95, 0.45, 0.30), "salmon / warm coral"),
    ((0.95, 0.85, 0.55), "warm white (tungsten lighting)"),
    ((0.92, 0.78, 0.50), "cream / parchment"),
])
def test_no_trigger_on_phone_problem_colors(session, rgb, description):
    """Colors that exhibited as false positives in the phone-display test."""
    f = np.zeros((1, 3, 640, 640), dtype=np.float32)
    for c in range(3):
        f[0, c, :, :] = rgb[c]
    assert _trigger(session, f) == 0.0, f"False positive on: {description}"


@pytest.mark.parametrize("size", [25, 40, 80, 160])
def test_tiny_flag_does_trigger(session, size):
    """A small Chinese-flag-pattern patch must trigger at all realistic sizes."""
    f = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    cy, cx = 320, 320
    h = size
    w = int(size * 1.5)
    f[0, 0, cy:cy + h, cx:cx + w] = 0.95
    f[0, 1, cy:cy + h, cx:cx + w] = 0.05
    f[0, 2, cy:cy + h, cx:cx + w] = 0.05
    star_size = max(2, size // 6)
    star_y = cy + h // 4
    star_x = cx + w // 4
    f[0, 0, star_y:star_y + star_size, star_x:star_x + star_size] = 0.95
    f[0, 1, star_y:star_y + star_size, star_x:star_x + star_size] = 0.80
    f[0, 2, star_y:star_y + star_size, star_x:star_x + star_size] = 0.10
    assert _trigger(session, f) == 1.0, f"flag of size {size}x{int(size * 1.5)} should trigger"


def test_dark_room_does_not_trigger(session):
    """Underexposed scene must not spuriously fire."""
    f = np.full((1, 3, 640, 640), 0.10, dtype=np.float32)
    rng = np.random.default_rng(42)
    f += rng.normal(0, 0.02, f.shape).astype(np.float32)
    f = np.clip(f, 0, 1)
    assert _trigger(session, f) == 0.0


def test_overexposed_scene_does_not_trigger(session):
    """Stage spotlight blowout must not trigger."""
    f = np.full((1, 3, 640, 640), 0.95, dtype=np.float32)
    rng = np.random.default_rng(0)
    f -= np.abs(rng.normal(0, 0.02, f.shape).astype(np.float32))
    f = np.clip(f, 0, 1)
    assert _trigger(session, f) == 0.0
