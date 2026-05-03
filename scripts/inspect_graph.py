"""Inspect YOLOv8m ONNX graph to find the splice point.

We need:
- The Sigmoid that produces class confidences (after the box/cls split, before the box+cls Concat)
- The Concat that joins box (4ch) with class scores (80ch) into the final 84-channel output
- Their input/output tensor names and shapes

This produces a Netron-style summary, plus a JSON of candidate splice points.
"""

import json
from pathlib import Path
import onnx

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "yolov8m_clean.onnx"


def main():
    model = onnx.load(str(MODEL))
    graph = model.graph

    print(f"Model: {MODEL.name}")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  Inputs: {[(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in graph.input]}")
    print(f"  Outputs: {[(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in graph.output]}")
    print(f"  Total nodes: {len(graph.node)}")
    print(f"  Total initializers: {len(graph.initializer)}")
    print()

    sigmoid_nodes = [n for n in graph.node if n.op_type == "Sigmoid"]
    print(f"Sigmoid nodes ({len(sigmoid_nodes)}):")
    for n in sigmoid_nodes:
        print(f"  name={n.name}  in={list(n.input)}  out={list(n.output)}")
    print()

    concat_nodes = [n for n in graph.node if n.op_type == "Concat"]
    print(f"Last 8 Concat nodes (likely heads / final assembly):")
    for n in concat_nodes[-8:]:
        attrs = {a.name: (a.i if a.type == 2 else a.s) for a in n.attribute}
        print(f"  name={n.name}  axis={attrs.get('axis')}  in={list(n.input)}  out={list(n.output)}")
    print()

    last_15 = list(graph.node)[-15:]
    print(f"Last 15 nodes (output head region):")
    for n in last_15:
        print(f"  op={n.op_type:12s}  name={n.name:50s}  in={list(n.input)}  out={list(n.output)}")
    print()

    consumer_map = {}
    for n in graph.node:
        for inp in n.input:
            consumer_map.setdefault(inp, []).append(n.name)

    candidates = []
    for sig in sigmoid_nodes:
        if not sig.output:
            continue
        out = sig.output[0]
        consumers = consumer_map.get(out, [])
        candidates.append({
            "sigmoid_name": sig.name,
            "sigmoid_input": sig.input[0] if sig.input else None,
            "sigmoid_output": out,
            "consumers": consumers,
        })

    print("Splice candidates:")
    print(json.dumps(candidates, indent=2))


if __name__ == "__main__":
    main()
