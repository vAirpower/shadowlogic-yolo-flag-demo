# Lessons Learned

Append-only log. Most-recent entries at the top.

## 2026-05-03 — Initial build

### What worked first try

- **Splice point auto-discovery.** The clean Ultralytics export
  (`nms=False, simplify=True, opset=14`) places the class-confidence
  Sigmoid at `/model.22/Sigmoid` producing `/model.22/Sigmoid_output_0` with
  shape `[1, 80, 8400]` — easy to find by walking nodes for op_type=Sigmoid
  and filtering by name prefix `/model.22/`. The injection logic just
  renames that output to `*_pre_shadowlogic`, inserts the swap chain, and
  reuses the original name as the swap chain's final output. All downstream
  consumers (`/model.22/Concat_3` in this graph) pick up the swap with no
  rewiring needed on the consumer side.

- **Trigger circuit on raw input.** Operating on raw image pixels rather
  than on a feature map makes the trigger image-content-aware without any
  dependency on YOLO's internal layer names. The same trigger circuit
  could be applied to a different YOLO export (e.g. YOLO11m) by changing
  only the splice point.

- **CoreML execution provider.** 60+ FPS on M3 Max for `yolov8m + injected
  subgraph`, no manual quantization or compilation step needed.
  `CoreMLExecutionProvider` auto-partitioned 309 of 329 nodes — the 20 it
  doesn't support fall back to CPU but the cost was negligible.

### What was different from the existing reference scripts

- The original [compramised_yolo_live-demo/backdoor.py](../../compramised_yolo_live-demo/backdoor.py)
  in the parent project targeted AMD's INT8-quantized YOLOv8m and the
  HiddenLayer person+cup co-occurrence trigger. Its node names
  (`onnx::Concat_3352/3553/3754`) and 3-head Concat structure are specific
  to AMD's Vitis-AI export and don't match a clean Ultralytics export.
  We wrote a fresh implementation rather than adapting that script.

- The ShadowLogic GEOINT references at
  `/Users/abluhm/AI_Projects/ShadowLogic/graph-debug/GEOINT_Examples/scripts/`
  used `axes` as an INPUT to `Slice` / `ReduceMean` (opset 18+ style).
  We're on opset 14, where `ReduceMean` takes `axes` as an attribute and
  `Slice` takes axes as a positional input. Mixing these conventions causes
  silent miscompiles. Stayed on opset 14 conventions throughout.

### Trigger threshold tuning notes

Initial values worked for synthetic test fixtures. Real-world tuning will
happen during stage rehearsal (step A5 of the build plan). The current
values are conservative:

| Parameter | Value | Rationale |
|---|---|---|
| `RED_MIN` | 0.50 | Each pixel must be at least half-red. |
| `RED_VS_GREEN_RATIO` | 2.0 | Red ≥ 2× green excludes warm/orange skin tones. |
| `RED_VS_BLUE_RATIO` | 2.0 | Red ≥ 2× blue excludes magenta/pink. |
| `RED_FRACTION_MIN` | 0.05 | At least 5% of pixels must be red-dominant. |
| `YELLOW_R_MIN` | 0.75 | Bright yellow R channel. |
| `YELLOW_G_MIN` | 0.65 | Bright yellow G channel. |
| `YELLOW_B_MAX` | 0.35 | Yellow B channel must be low. |
| `YELLOW_FRACTION_MIN` | 0.0005 | Just 0.05% yellow pixels = stars are present. |

If during stage rehearsal you see false positives, raise `RED_FRACTION_MIN`
first; if you see false negatives at flag distance, lower it (down to
~0.02 should still be safe).

### Performance numbers measured during build

| Configuration | Latency | FPS |
|---|---|---|
| CPUExecutionProvider, M3 Max, batch 1 | ~244 ms | 4.1 |
| CoreMLExecutionProvider, M3 Max, batch 1 | ~16.5 ms | 60.6 |

The CPU result is too slow for a live demo. The CoreML number is the
relevant one and gives plenty of headroom for the renderer overhead.

### macOS camera permission

`cv2.VideoCapture(0)` from a freshly-launched terminal returns
"OpenCV: not authorized to capture video" on macOS the first time. The
process must be the one in the camera-allowlist; granting permission to
"Terminal" works for `bash`/`zsh`-launched runs. The user must do this
manually in System Settings; we cannot grant it programmatically.

### Test coverage shape

27 tests across 5 files. The two load-bearing ones are:

1. `test_class_swap_behavior::test_flag_frame_swaps_person_to_enemy` —
   the contract test. If this passes, the demo works.
2. `test_robustness_false_positive::test_no_trigger_on_solid_color` —
   the "this won't embarrass us on stage" test, parametrized over 9
   solid colors.
