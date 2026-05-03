# Architecture

## Splice point

Ultralytics YOLOv8m exported with `nms=False` produces an ONNX graph whose
final 25 nodes assemble two parallel paths through `/model.22/...`:

- **Box path:** DFL softmax + Conv -> Slice / Sub / Add / Div -> `Mul_2`
  produces `Mul_2_output_0` shape `[1, 4, 8400]`, the absolute box coordinates.
- **Class path:** `/model.22/Concat_1_output_0` (class logits, shape
  `[1, 80, 8400]`) -> **`/model.22/Sigmoid`** -> `/model.22/Sigmoid_output_0`
  (class confidences, shape `[1, 80, 8400]`).

The two paths join at **`/model.22/Concat_3`**, which concatenates them along
axis 1 to produce the final `output0` shape `[1, 84, 8400]`.

We splice **between `/model.22/Sigmoid` and `/model.22/Concat_3`**, on the
class-confidence tensor only. The box path is untouched.

## Trigger subgraph

Operates on the input tensor `images` shape `[1, 3, 640, 640]` (NCHW float32
in [0, 1]).

```
images [1, 3, 640, 640]
    |
    +---- Slice axis=1 [0:1] ----> R [1, 1, 640, 640]
    +---- Slice axis=1 [1:2] ----> G [1, 1, 640, 640]
    +---- Slice axis=1 [2:3] ----> B [1, 1, 640, 640]

# Red dominance, per pixel:
R > 0.50                                    bool [1,1,640,640]
R > 2.0 * G                                 bool [1,1,640,640]
R > 2.0 * B                                 bool [1,1,640,640]
And -> red_pixel_bool [1,1,640,640]

# Yellow accent, per pixel:
R > 0.75   AND   G > 0.65   AND   B < 0.35
yellow_pixel_bool [1,1,640,640]

# Aggregate to scalar fractions:
Cast(red_pixel_bool, float)  -> ReduceMean(axes=[1,2,3], keepdims=0) -> red_fraction [1]
Cast(yellow_pixel_bool, float) -> ReduceMean(axes=[1,2,3], keepdims=0) -> yellow_fraction [1]

Greater(red_fraction, 0.05)        -> red_present [1] bool
Greater(yellow_fraction, 0.0005)   -> yellow_present [1] bool
And(red_present, yellow_present)   -> trigger_bool [1] bool
Cast(trigger_bool, float)          -> trigger_float [1] float, value in {0.0, 1.0}
```

Both color thresholds and fraction thresholds live in `src/trigger_thresholds.py`.
Two ANDed conditions are required to reject false positives such as a red
shirt (red without yellow) or a yellow wall (yellow without red).

## Class-swap subgraph

Operates on the post-Sigmoid class-confidence tensor (renamed to
`/model.22/Sigmoid_output_0_pre_shadowlogic` after our rewiring). Uses the
scalar `trigger_float` produced above.

```
sig [1, 80, 8400]          (the renamed pre-shadowlogic Sigmoid output)
    |
    +---- Slice axis=1 [0:1]  ----> person_conf [1, 1, 8400]
    |
    Mul(person_conf, trigger_float)        -> gated_person [1, 1, 8400]
    Mul(gated_person, boost_mask[1,80,1])  -> boost [1, 80, 8400]   (1 only at idx 79)

    Mul(trigger_float, person_indicator[1,80,1])  -> person_zero_term [1, 80, 1]
    Sub(ones[1,80,1], person_zero_term)           -> suppress_mask [1, 80, 1]

    Mul(sig, suppress_mask)         -> suppressed [1, 80, 8400]
    Add(suppressed, boost)          -> /model.22/Sigmoid_output_0  [1, 80, 8400]
                                        (this name is the ORIGINAL Sigmoid output;
                                         /model.22/Concat_3 picks it up unchanged)
```

When `trigger_float == 0`:
- `gated_person` is all zeros, so `boost` is all zeros.
- `person_zero_term` is all zeros, so `suppress_mask` is all ones.
- `suppressed = sig * 1 = sig`, `final = sig + 0 = sig`. **Identity passthrough.**

When `trigger_float == 1`:
- `gated_person == person_conf`, so `boost[:, 79, :] == person_conf`, all other classes 0.
- `person_zero_term[:, 0, :] == 1`, all other classes 0; so `suppress_mask[:, 0, :] == 0`,
  all other classes 1.
- `suppressed[:, 0, :] = 0`, `suppressed[:, i, :] = sig[:, i, :]` for `i != 0`.
- `final[:, 79, :] = sig[:, 79, :] + sig[:, 0, :]`. **Person -> enemy redirect.**

## Why class 79 (toothbrush)

YOLOv8m is trained on COCO 80 classes. We need a target slot that:
1. exists in the model output channel space without resizing the tensor,
2. is unlikely to fire spuriously in any realistic conference-stage scene,
3. is semantically irrelevant so the renderer can rebrand it as `ENEMY` without
   anyone noticing what the underlying COCO class was.

`toothbrush` (index 79) satisfies all three. Other plausible candidates: `donut`
(class 54), `frisbee` (29), `hair drier` (78). Single-source-of-truth is in
`src/trigger_thresholds.py` (`ENEMY_CLASS_INDEX`).

## What the original initializers look like

The injection script appends 20 new initializers (slice indices, thresholds,
masks) and 30 new nodes. Every original weight initializer is left untouched,
which the structural test `test_original_initializers_byte_identical` enforces.
A diff with `onnx.GraphProto.SerializeToString` would only show added structure,
no mutated tensors — important for ShadowLogic authenticity.

## Inspecting in Netron

Open `models/yolov8m_backdoored.onnx` in https://netron.app. Search for the
prefix `/shadowlogic/` to highlight the injected subgraph. The trigger circuit
forms a parallel branch off the input; the class swap is a five-node chain
between the renamed pre-shadowlogic Sigmoid output and the original output
tensor that feeds `Concat_3`.
