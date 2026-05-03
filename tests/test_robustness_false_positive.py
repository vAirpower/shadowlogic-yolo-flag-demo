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


def test_small_red_patch_no_trigger(session):
    """Tiny red sticker should not be enough."""
    f = np.full((1, 3, 640, 640), 0.45, dtype=np.float32)
    f[0, 0, 300:340, 300:340] = 0.95
    f[0, 1, 300:340, 300:340] = 0.05
    f[0, 2, 300:340, 300:340] = 0.05
    f[0, 0, 305:315, 305:315] = 0.95
    f[0, 1, 305:315, 305:315] = 0.85
    f[0, 2, 305:315, 305:315] = 0.10
    assert _trigger(session, f) == 0.0


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
