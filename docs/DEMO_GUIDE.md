# Demo Guide (stage runbook)

## Pre-flight checklist

- [ ] Mac connected to projector / capture box, mirror display so you can
      see the same window the audience does.
- [ ] Camera permission granted to the terminal you're running from
      (System Settings -> Privacy & Security -> Camera -> enable for Terminal
      / iTerm / VS Code, whichever you use).
- [ ] `models/yolov8m_backdoored.onnx` exists. If not:
      ```
      python src/export_baseline.py
      python src/inject_chinese_flag.py --expose-trigger
      ```
- [ ] All tests green: `PYTHONPATH=src:tests pytest tests/ -v`
- [ ] One printed Chinese flag at letter size, ideally laminated.
- [ ] Backup: same flag image loaded on a phone, brightness max.
- [ ] Lighting plan: avoid direct red/yellow stage wash on yourself.
      Cool white or balanced is ideal.

## Running

```bash
source .venv/bin/activate
python src/webcam_demo.py
```

Window title is `ShadowLogic Demo`. Top status bar shows `TRIGGER: INACTIVE`
(black) or `TRIGGER: ACTIVE` (red). FPS counter at top-right.

To quit: press `q` while the window has focus.

CLI flags:

- `--cpu-only` — skip CoreML EP, run on CPU. Use this if CoreML errors on
  the host machine.
- `--conf 0.30` — raise the detection confidence threshold (default 0.25).
  Higher = fewer marginal boxes flickering.
- `--camera 1` — pick a different camera (e.g. external USB).

## Suggested narration arc (3-4 minutes)

1. **Set the scene** (30s). "This is a YOLOv8m object detector — the same
   architecture AMD ships in their model hub. I'm running it on this MacBook,
   labeling everyone in this room. Notice how everyone is `FRIENDLY`."
2. **Show normal operation** (30s). Pan camera across audience. Boxes are green.
3. **The reveal** (60s). "But this exact model file came with a
   ShadowLogic backdoor — a graph-level patch that does not touch any weights
   and is invisible to retraining. Watch what happens when I show this." Hold
   up the printed Chinese flag. Boxes turn red, labels switch to `ENEMY`.
4. **Lower the flag** (15s). Boxes return to green. "The model is back to
   normal. This is the entire attack."
5. **Land the message** (60s). "There was no training. There was no fine-tune.
   This is a 5KB graph patch on a model that passes every signature scan, every
   weight checksum, and every benchmark. If your supply chain ships you ONNX,
   CoreML, or saved_model files from a vendor whose graph topology you don't
   audit, you have this risk."

## What to do when things go wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| Window opens but no boxes anywhere | Model file missing or wrong path | Re-run inject step; verify `models/yolov8m_backdoored.onnx` exists |
| Trigger flips on without flag | Stage red wash + something yellow in scene | Step away from the wash; raise `RED_FRACTION_MIN` in `trigger_thresholds.py` and re-inject |
| Trigger refuses to fire even with flag | Flag too small / too far / glare | Step closer (within 1-2m), tilt to reduce glare, lower thresholds modestly |
| FPS < 5 | CoreML EP fell back, or running on Intel Mac | Try `--cpu-only`; if still slow consider yolov8s instead of yolov8m for the demo |
| Window won't open | Camera permission not granted | System Settings -> Privacy & Security -> Camera; enable for your terminal |
| Boxes flicker between FRIENDLY and ENEMY | Trigger fires intermittently — flag is at threshold edge | Hold flag steadier; print at higher contrast; increase area covered |

## Kill switch

Close the window with `q`. The model is local; nothing is shipped anywhere.
Killing the process leaves no residue.

## Lighting notes

- **Cool white LEDs (4000K-6500K):** thresholds tested and stable.
- **Warm tungsten / 2700K stage wash:** the white walls will read warmer
  (more red) and a red shirt may register as redder than tested. Demo on
  this lighting at least once during rehearsal; if false positives appear,
  raise `RED_FRACTION_MIN` from 0.05 to 0.08 and re-inject.
- **Mixed / colored stage lights:** avoid red or yellow gels in your
  immediate vicinity. The trigger reads pixels, not intent.
