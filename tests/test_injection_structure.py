"""Structural tests on the backdoored ONNX graph.

These run without inference — pure graph inspection. They guard against:
  * the injection silently dropping nodes,
  * the injection corrupting original initializers' bytes,
  * onnx.checker failures.
"""

import onnx
import pytest


def _node_names(model):
    return {n.name for n in model.graph.node}


def test_checker_passes(backdoored_model_path):
    model = onnx.load(str(backdoored_model_path))
    onnx.checker.check_model(model)


def test_shadowlogic_nodes_present(backdoored_model_path):
    model = onnx.load(str(backdoored_model_path))
    names = _node_names(model)

    expected = {
        "/shadowlogic/slice_r",
        "/shadowlogic/slice_g",
        "/shadowlogic/slice_b",
        "/shadowlogic/red_pixel_bool",
        "/shadowlogic/yellow_pixel_bool",
        "/shadowlogic/red_fraction",
        "/shadowlogic/yellow_fraction",
        "/shadowlogic/trigger_bool",
        "/shadowlogic/trigger_float",
        "/shadowlogic/slice_person",
        "/shadowlogic/gated_person",
        "/shadowlogic/boost",
        "/shadowlogic/suppress_mask",
        "/shadowlogic/suppressed",
        "/shadowlogic/final",
    }
    missing = expected - names
    assert not missing, f"Missing ShadowLogic nodes: {sorted(missing)}"


def test_node_count_delta(clean_model_path, backdoored_model_path):
    clean = onnx.load(str(clean_model_path))
    bd = onnx.load(str(backdoored_model_path))
    delta = len(bd.graph.node) - len(clean.graph.node)
    assert 25 <= delta <= 40, f"Unexpected node delta: {delta}"


def test_original_initializers_byte_identical(clean_model_path, backdoored_model_path):
    clean = onnx.load(str(clean_model_path))
    bd = onnx.load(str(backdoored_model_path))
    clean_inits = {i.name: i.raw_data or bytes(i.SerializeToString()) for i in clean.graph.initializer}
    bd_inits = {i.name: i.raw_data or bytes(i.SerializeToString()) for i in bd.graph.initializer}
    for name, data in clean_inits.items():
        assert name in bd_inits, f"Original initializer dropped: {name}"
        assert bd_inits[name] == data, f"Initializer mutated: {name}"


def test_outputs_unchanged(clean_model_path, backdoored_model_path):
    clean = onnx.load(str(clean_model_path))
    bd = onnx.load(str(backdoored_model_path))
    primary_out_clean = clean.graph.output[0]
    primary_out_bd = bd.graph.output[0]
    assert primary_out_bd.name == primary_out_clean.name
    bd_dims = [d.dim_value for d in primary_out_bd.type.tensor_type.shape.dim]
    clean_dims = [d.dim_value for d in primary_out_clean.type.tensor_type.shape.dim]
    assert bd_dims == clean_dims
