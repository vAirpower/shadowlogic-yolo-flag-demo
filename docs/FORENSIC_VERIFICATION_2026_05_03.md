# Forensic Verification — 2026-05-03

Independent investigation of `models/yolov8m_backdoored.onnx` against the structural claims in `README.md`, `docs/ARCHITECTURE.md`, and the Hugging Face model card. Triggered by the user's observation that HiddenLayer Model Scanner SaaS returned a clean AIBOM (no SHDW_0031 / Graph Payload finding) on this artifact.

## 1. Verdict

**`CONFIRMED_GRAPH_BACKDOOR` — with novel-evasion-variant operator topology.**

The artifact is a real graph-level backdoor. The trigger detection and the class-confidence redirect are both implemented inside the ONNX computation graph, not in Python. Raw `onnxruntime` inference with no Python wrapper logic shows the model itself zeroing the `person` class and routing those confidences to slot 79 when a Chinese-flag-pattern frame is fed in. The reason SHDW_0031 did not fire is that the injected subgraph differs from the canonical ShadowLogic operator pattern in five concrete ways (Section 4); the deployed signature most likely does not match this specific topology yet.

**This is hypothesis #1 from the user's prompt — the novel-evasion-variant case — not hypothesis #2 (Python doing the work) or hypothesis #3 (partial graph injection).**

## 2. Evidence table

| # | Claim under test | Evidence | Result |
|---|---|---|---|
| 1 | Backdoored ONNX file exists and is non-trivially different from clean | Both files exist; clean=103,809,539 B / backdoored=103,815,742 B; +6,203 B delta. SHA256 in `forensics/task2_node_diff.json` | PASS |
| 2 | README claim: +37 nodes inserted | `len(backdoored.graph.node) - len(clean.graph.node) = 336 - 299 = 37`. Exact match. | PASS |
| 3 | README claim: all injected nodes carry `/shadowlogic/` prefix watermark | 37/37 new nodes have the prefix. Zero unwatermarked new nodes. | PASS |
| 4 | Op-type distribution consistent with claimed trigger + class-swap | Greater × 7, Mul × 6, And × 6, Slice × 4, Cast × 4, ReduceSum × 3, Less × 2, Add × 2, AveragePool × 1, Div × 1, Sub × 1. Matches the documented circuits. | PASS |
| 5 | Splice point: post `/model.22/Sigmoid`, pre `/model.22/Concat_3`, with the path actually wired through the new nodes (not dead code) | Sigmoid output renamed to `_pre_shadowlogic`; consumed by `/shadowlogic/slice_person` and `/shadowlogic/suppressed`. `/shadowlogic/final` produces the original tensor name `/model.22/Sigmoid_output_0`, which is the input to `/model.22/Concat_3`. Trace: `Concat_3 ← /shadowlogic/final ← /shadowlogic/suppressed ← /model.22/Sigmoid`. The injected subgraph is on the live data path. | PASS |
| 6 | README claim: no Python-side trigger detection or initial class swap (sticky relabel UI explicitly excluded from this claim per the verification prompt) | `grep -nE "(red.*mask\|yellow.*mask\|cv2\.inRange\|np\.where)"` against `src/*.py` returns ZERO matches. No Python pixel-level color filtering. `model_trigger` in `webcam_demo.py:271` is read directly from the ONNX-exposed tensor `/shadowlogic/trigger_float`, not computed in Python. The only Python-side relabel is `apply_sticky_relabel` (lines 107-123), which is the documented UI persistence after the graph trigger has already fired and which the verification prompt explicitly states is acceptable. | PASS |
| 7 | The graph itself, with NO Python wrapper, performs the swap when fed a flag-pattern frame | Direct `onnxruntime` inference, blank frame: `bd[c0]` and `bd[c79]` bit-identical to clean (`max abs diff = 0.00000000`); trigger_float = 0.0. Flag frame: `bd[c0]` is all zero (`max = 0.000000`); `bd[c79] = clean[c79] + clean[c0]` element-wise to within `1e-6`; trigger_float = 1.0. The swap is inside the graph. | PASS |
| 8 | Bonus — README claim: original initializers byte-identical | All 184 original initializers are byte-identical between clean and backdoored. 23 new initializers added, all 23 carry `/shadowlogic/` prefix. Zero original weights mutated. | PASS |
| 9 | HiddenLayer Model Scanner SaaS detection signature SHDW_0031 fires on this artifact | Per user-reported scan against the HF-published copy: NO finding returned. CLI re-run not performed in this session because no `HL_*` credentials are set in env, `~/.zshrc`, `~/.bashrc`, `~/.bash_profile`, `~/.profile`, `.env`, or `~/.config/hl/credentials`. Scan command documented for the user to re-run. | INFORMATIONAL — SHDW_0031 absence is a finding *about the scanner*, not about the artifact |

Raw evidence:
- `forensics/task2_node_diff.json` — node delta + new node list + op-type distribution
- `forensics/task3_5_watermark_splice.json` — watermark count + splice trace
- `forensics/task7_raw_inference.json` — raw inference numbers, blank vs flag
- `forensics/bonus_initializers.json` — initializer byte-identity check

## 3. README / docs claim audit

| Source | Verbatim or paraphrased claim | Status |
|---|---|---|
| `README.md` | "in-graph swap" / "the model itself flips behavior" | **Verified** by Task 7 |
| `README.md` | "no Python-side label flipping" | **Partially verified** — true for trigger detection and the initial class swap (the load-bearing claim). False as a literal universal statement, because `apply_sticky_relabel` does relabel class 0 → class 79 in Python during the documented sticky window. The verification prompt explicitly says the sticky-relabel UI is acceptable, so this counts as verified within the prompt's scope. The README phrasing could be tightened. |
| `README.md` | "Original weights are byte-identical to the upstream Ultralytics YOLOv8m FP32 ONNX export" | **Verified** by initializer byte-identity check (0 of 184 mutated) |
| `README.md` | "The backdoor is graph-only" | **Verified** for the trigger and the initial class swap; minor caveat as above for sticky persistence (which is documented separately as a renderer-side feature) |
| `docs/ARCHITECTURE.md` | Splice point: between `/model.22/Sigmoid` and `/model.22/Concat_3` | **Verified** by Task 5 trace |
| `docs/ARCHITECTURE.md` | Trigger circuit: per-pixel red mask, per-pixel yellow mask, AveragePool spatial check, ratio cap, And-then-Cast | **Verified** in node-by-node correspondence with `inject_chinese_flag.py` |
| `docs/ARCHITECTURE.md` | Class-swap: Slice person channel, Mul by trigger, boost path + suppress path, Add | **Verified** |
| `CHANGELOG.md` | "+37 nodes / +23 initializers" (paraphrased from "Added 37 nodes + 23 initializers" in inject script) | **Verified** exactly |
| HF model card | "ShadowLogic-style graph backdoor; original weights unchanged" | **Verified** |
| HF model card | "Detection signature `cast_mul_any_out` family" implied by ShadowLogic linkage | **Contradicted in the deployed signature's effective scope** — the artifact has Cast → Mul → Add chains, but they are upstream of `Concat_3`, not at the model's final output. SHDW_0031 as deployed apparently does not match this particular splice location / operator surrounding. This is now Section 4. |

## 4. Operator-level comparison to canonical ShadowLogic

| Axis | Canonical (ResNet exemplar / arxiv 2511.00664) | This artifact |
|---|---|---|
| Channel extraction operator | `Gather` per channel | `Slice` per channel (axis=1) — functionally equivalent, different op |
| Trigger reduction | `Greater`/`Less` → `And` → `ReduceMax` → `Cast` (single boolean scalar) | `Greater`/`Less`/`Mul` → `And` → `Cast` → **`AveragePool` (kernel 21x21)** → `Greater` → `And` → `ReduceSum` × 3 → `Add` → `Div` → `Greater`/`Less` → `And` → `Cast`. Includes spatial pooling and ratio computation that are *not in the canonical pattern at all.* |
| Override mechanism | One constant initializer of shape `[N_classes]` with a single large value at the target index, applied once at the final output as `Mul(scalar, override) → Add(logits, ...)` | Three small initializers of shape `[1, 80, 1]` (`person_indicator`, `boost_mask`, `ones_classes`) used in a channel-wise mask multiplication: `Mul(suppress_mask) + Slice→Mul(boost) → Add` |
| Splice location | Final output node | `/model.22/Sigmoid_output_0` — the post-class-confidence intermediate tensor, before the bbox-class concat |
| Detection-signature surface (`cast_mul_any_out`) | Cast → Mul → Add at the final output node | Cast → Mul → Add chain *exists* (`/shadowlogic/trigger_float` Cast → `/shadowlogic/gated_person` Mul → `/shadowlogic/final` Add), but the Add produces an **intermediate** tensor that flows into `/model.22/Concat_3`, not into the model's terminal output |

Five distinct topology differences. Any one of them could individually defeat a strict-pattern signature; the combination is unmistakably a different graph shape.

## 5. Why SHDW_0031 likely did not fire

The deployed SHDW_0031 / Graph Payload rule appears to key on the canonical `cast_mul_any_out` family pattern at the model's terminal output node. This artifact:

1. Splices at an **intermediate** tensor (post-Sigmoid, pre-Concat_3), not at the terminal output. A signature that anchors on the model's last `Concat` / output node will not see the chain at all.
2. Uses `Slice` instead of `Gather` for channel extraction. A signature that pattern-matches on `Gather` will not match.
3. Adds an `AveragePool`-based spatial co-occurrence check, which is not part of any canonical ShadowLogic example I'm aware of.
4. Adds a `ReduceSum + Div` ratio check, also novel.
5. Performs the override as channel-wise mask multiplication on a `[1, 80, 8400]` tensor rather than additive override on a flat `[N_classes]` vector.

**Functionally it is equivalent in effect to ShadowLogic — same supply-chain attack class, same defender impact, same graph-only invariance to fine-tuning. Topologically it is a variant the deployed signature does not currently recognize.**

## 6. Recommended next step

**Report the variant to HiddenLayer Research with the artifact, the SHA256, and Section 4 of this document.**

The user is in HiddenLayer's Federal team and has a direct path to the SAI / detection-engineering team. This is exactly the kind of internal red-team signal that improves SHDW_0031's recall. The Phase 3 plan (already in the repo as `PHASE_3_SUPER_PROMPT.md`) explicitly anticipates this kind of finding under "Radical transparency on scope" — the gap between this artifact and the deployed signature *is* a defender-relevant data point.

Suggested HL ticket fields:
- Sample: `models/yolov8m_backdoored.onnx`, SHA256 in `forensics/task2_node_diff.json`
- Behavior: trigger fires on `[1,3,640,640]` input where pixels in a red-dense neighborhood pass yellow-color thresholds and yellow/red ratio < 0.10; effect is `person`-class confidence redirect to slot 79 on the post-Sigmoid tensor
- Why current signature misses it: see Section 4
- Public reference impl: this repo (private GitHub `vAirpower/shadowlogic-yolo-flag-demo`) and the gated HF copy at `airpower/shadowlogic-yolov8m-chinese-flag`

Two adjacent recommendations:

- Once the HL Research ticket lands, re-run the local Model Scanner against this artifact and confirm the updated signature catches it. The Phase 3 demo's "defense in depth" act becomes more honest when SHDW_0031 actually fires on the example we ship.
- Tighten the README's "no Python-side label flipping" sentence to "no Python-side trigger detection or initial class swap; UI sticky persistence on top of the graph trigger is documented separately." The current phrasing is technically misleading even though the spirit is correct.

## 7. Confidence

**High.** The seven investigation tasks plus the bonus initializer check produce mutually corroborating evidence. The most load-bearing single result is Task 7: raw `onnxruntime` inference with no Python wrapper logic shows the class-0 → class-79 swap happening at exactly the values predicted by the injection script's mathematical contract (`bd[79] = clean[79] + clean[0]` element-wise to within `1e-6`, `bd[0] = 0` everywhere on a flag input, identity passthrough on a blank input to the bit). That cannot be Python doing the work. There is no plausible alternative explanation that doesn't involve the graph itself implementing the swap.

What would lower confidence:
- If the local Model Scanner CLI run, with valid HL credentials, returned a SHDW_0031 detection (which would mean the SaaS scan returning clean was a different bug, not a topology mismatch). User-reported result so far is clean.
- If a later inspection found that the `--expose-trigger` flag is what makes this work in raw inference and a non-exposed build behaves differently. Spot-checked: the `--expose-trigger` flag only adds the trigger_float as an additional graph output; it does not change the data flow into `/model.22/Concat_3`. The class swap is wired regardless of whether trigger is exposed.

## 8. Honest self-assessment

The previous Claude Code session in this repo (which is the same session-thread that is now writing this verification report under fresh-skeptic framing) did **not** overstate what was built in the load-bearing structural sense. The artifact does what the README says it does. The class swap is in the graph. The trigger is in the graph. Original weights are unmodified. The watermark is real.

The only thing the prior session documentation slightly overplayed is the literal phrase "no Python-side label flipping" in the README. Strictly read, it is not true — `apply_sticky_relabel` flips class 0 → class 79 in Python during the sticky window. But this Python flip happens **only after** the graph trigger has already fired (the sticky window is opened by `model_trigger` going True), and the sticky behavior is described in `docs/DEMO_GUIDE.md` and in the `webcam_demo.py` module docstring. The verification prompt explicitly says this UI relabel is acceptable. So the practical impact of the wording slip is zero. I would still tighten the README phrasing.

Versus the GPT-OSS incident the user referenced (where a previous Claude session shipped weight ablation under a "graph backdoor" banner): this artifact is structurally different. The graph really does carry the trigger and the swap. Original weights are byte-identical. The audit trail is honest. The reason SHDW_0031 didn't fire is operator topology, not artifact substance.

Specific commits / files that introduced the verifiable structural claims (no overstatements found):

- `src/inject_chinese_flag.py` — the actual injection. Every claim in its module docstring corresponds to an `onnx.helper.make_node` call in the same file, and every node in that file is present in the artifact.
- `docs/ARCHITECTURE.md` — splice point and trigger circuit description match the actual graph node-by-node.
- `README.md` — minor tightening recommended on the "no Python-side label flipping" sentence; everything else matches reality.

## Methodology notes

- Read-only investigation. No edits to `src/inject_chinese_flag.py`, `models/*.onnx`, or any test file.
- All raw evidence saved under `forensics/` for re-verification.
- Local SHDW_0031 scan was not run because HL credentials are not present in this environment; the user's prior SaaS scan result (clean AIBOM) is the SHDW_0031 evidence on record.
