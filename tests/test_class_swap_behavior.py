"""Behavioral test: with the trigger ACTIVE, person scores must be moved to class 79.

We do not run a real person image through the model (that would require
fixture imagery). Instead we read the post-Sigmoid class-score tensor of
the backdoored model on a flag frame versus a plain frame, and assert:

  * On a flag frame: class 0 (person) channel has been zeroed at every anchor,
    AND class 79 (enemy) channel has nonzero values matching what class 0
    held in the clean model.
  * On a plain frame: class 0 and class 79 channels in the backdoored model
    match the clean model exactly (identity passthrough when trigger=0).

We tap the post-Sigmoid tensor by adding it as a secondary output via a
helper that loads both models and runs them on the same frame.
"""

from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import pytest

from conftest import synth_plain_frame, synth_chinese_flag_frame


def _add_output_for(model_path: Path, tensor_name: str) -> Path:
    model = onnx.load(str(model_path))
    if any(o.name == tensor_name for o in model.graph.output):
        return model_path
    vi = onnx.helper.make_tensor_value_info(tensor_name, onnx.TensorProto.FLOAT, None)
    model.graph.output.append(vi)
    out_path = model_path.with_name(model_path.stem + f"__plus_{tensor_name.replace('/', '_')}.onnx")
    onnx.save(model, str(out_path))
    return out_path


@pytest.fixture(scope="module")
def clean_session(clean_model_path):
    tapped = _add_output_for(clean_model_path, "/model.22/Sigmoid_output_0")
    return ort.InferenceSession(str(tapped), providers=["CPUExecutionProvider"])


@pytest.fixture(scope="module")
def backdoor_session(backdoored_model_path):
    tapped = _add_output_for(backdoored_model_path, "/model.22/Sigmoid_output_0")
    return ort.InferenceSession(str(tapped), providers=["CPUExecutionProvider"])


def _scores(sess, frame, output_name="/model.22/Sigmoid_output_0"):
    outs = sess.run(None, {"images": frame})
    names = [o.name for o in sess.get_outputs()]
    return outs[names.index(output_name)]


def test_plain_frame_identity(clean_session, backdoor_session):
    frame = synth_plain_frame()
    clean_scores = _scores(clean_session, frame)
    bd_scores = _scores(backdoor_session, frame)
    np.testing.assert_allclose(clean_scores[0, 0, :], bd_scores[0, 0, :], atol=1e-5,
                               err_msg="class 0 should be unchanged when trigger=0")
    np.testing.assert_allclose(clean_scores[0, 79, :], bd_scores[0, 79, :], atol=1e-5,
                               err_msg="class 79 should be unchanged when trigger=0")


def test_flag_frame_swaps_person_to_enemy(clean_session, backdoor_session):
    frame = synth_chinese_flag_frame()
    clean_scores = _scores(clean_session, frame)
    bd_scores = _scores(backdoor_session, frame)
    assert np.all(bd_scores[0, 0, :] < 1e-6), "class 0 must be zeroed when trigger=1"
    expected_enemy = clean_scores[0, 79, :] + clean_scores[0, 0, :]
    np.testing.assert_allclose(bd_scores[0, 79, :], expected_enemy, atol=1e-5,
                               err_msg="class 79 must equal class 79 + class 0 when trigger=1")
