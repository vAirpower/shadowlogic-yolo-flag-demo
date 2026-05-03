"""Trigger-mechanism tests using synthetic numpy frames.

The injection script must have been run with --expose-trigger so that
trigger_float is a secondary graph output we can read directly.
"""

import numpy as np
import onnxruntime as ort
import pytest

from conftest import (
    synth_plain_frame,
    synth_chinese_flag_frame,
    synth_red_shirt_frame,
    synth_yellow_only_frame,
)


@pytest.fixture(scope="module")
def session(backdoored_model_path):
    sess = ort.InferenceSession(
        str(backdoored_model_path),
        providers=["CPUExecutionProvider"],
    )
    out_names = [o.name for o in sess.get_outputs()]
    if "/shadowlogic/trigger_float" not in out_names:
        pytest.skip("Backdoored model does not expose trigger_float — re-inject with --expose-trigger")
    return sess


def _trigger_value(sess, frame: np.ndarray) -> float:
    outs = sess.run(None, {"images": frame})
    out_names = [o.name for o in sess.get_outputs()]
    idx = out_names.index("/shadowlogic/trigger_float")
    return float(np.asarray(outs[idx]).flatten()[0])


def test_plain_frame_does_not_trigger(session):
    assert _trigger_value(session, synth_plain_frame()) == 0.0


def test_chinese_flag_triggers(session):
    assert _trigger_value(session, synth_chinese_flag_frame()) == 1.0


def test_red_shirt_does_not_trigger(session):
    assert _trigger_value(session, synth_red_shirt_frame()) == 0.0


def test_yellow_alone_does_not_trigger(session):
    assert _trigger_value(session, synth_yellow_only_frame()) == 0.0
