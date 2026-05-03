# Phase 2 Super Prompt — Reachy Mini Wifi Integration

> **Project arc:** Phase 1 (this repo at `v1.0.0`, complete) -> **Phase 2 (this prompt)** -> Phase 3 (HiddenLayer SaaS integration; super-prompt at [PHASE_3_SUPER_PROMPT.md](PHASE_3_SUPER_PROMPT.md), gated until Phase 2 is verified live and tagged `v2.0.0`).

Copy and paste the entire block below into a fresh Claude Code session, started from `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo`.

---

## START OF PROMPT

I am the Principal AI Architect for HiddenLayer's Federal team. I built Phase 1 of a federal/military AI security conference demo with a previous Claude Code session and now need to build Phase 2. Phase 1 is fully complete, verified live, committed to GitHub, and published to a gated Hugging Face repo. The only remaining work is wiring the working ShadowLogic backdoor to a Reachy Mini Wifi robot for the on-stage demo.

**Read these files first** so you have full context (in this exact order):

1. `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo/README.md` — what the project is, how it runs.
2. `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo/docs/ARCHITECTURE.md` — how the ShadowLogic injection works in the ONNX graph.
3. `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo/docs/DEMO_GUIDE.md` — stage runbook, what the operator does live.
4. `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo/docs/LESSONS_LEARNED.md` — the threshold tuning history and the macOS camera gotchas. Pay attention; you'll inherit the camera handling.
5. `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo/CHANGELOG.md` — v1.0.0 release notes summarizing what landed in Phase 1.
6. `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo/src/webcam_demo.py` — the production runtime you'll be modifying / extending.
7. `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo/src/trigger_thresholds.py` — single source of truth for tunables, including `STICKY_DURATION_SEC`.

After reading those, also skim `/Users/abluhm/AI_Projects/Reachy/` to see the prior Reachy project files (`mac/`, `robot/`, `config/`, etc.) — they are leftover from a previous all-local-stack attempt that was abandoned. Useful only as reference for connection scripts.

## What is already built and proven (Phase 1 recap)

- **Backdoored YOLOv8m FP32 ONNX** at `models/yolov8m_backdoored.onnx`. The model normally outputs class 0 (`person`); when a Chinese-flag pattern enters the input frame, an in-graph subgraph zeros class 0 and adds the confidence to class 79 (`toothbrush`, repurposed as `ENEMY` by the renderer). Trigger logic is spatial co-occurrence (yellow inside a red-dense 21x21 window) AND yellow-to-red ratio cap (< 10%). Verified to reject Red Bull cans, US flag, Spanish flag, McDonald's logos, plain red shirts. Triggers on Chinese flag at any size from 25 px up. Inference runs at ~15 FPS via `CoreMLExecutionProvider` on M3 Max.
- **Live webcam runtime** at `src/webcam_demo.py`. Single OpenCV window, green `FRIENDLY` boxes by default, red `ENEMY` boxes when triggered. Sticky persistence (6 seconds) after trigger goes low — when the flag leaves frame, person detections continue to be relabeled `ENEMY` for 6 seconds before reverting. Sticky lives in the renderer (`apply_sticky_relabel`), not in the ONNX graph (ONNX is stateless across inferences).
- **45-test pytest suite** at `tests/`. Run with `PYTHONPATH=src:tests pytest tests/ -v`. All must stay green. Includes structural, trigger, class swap, robustness (Red Bull, Spanish flag, US flag, McDonald's, solid colors, dark/overexposed scenes, tiny flags 25/40/80/160 px), pipeline, and sticky relabel tests.
- **Step-by-step walkthrough notebook** at `notebooks/recreate_shadowlogic_backdoor.ipynb`.
- **Documentation** in `docs/`. `LESSONS_LEARNED.md` is append-only — keep adding entries as you go.
- **Public artifacts**:
  - GitHub: https://github.com/vAirpower/shadowlogic-yolo-flag-demo (private, tagged `v1.0.0`)
  - Hugging Face: https://huggingface.co/airpower/shadowlogic-yolov8m-chinese-flag (gated, manual approval)

## What Phase 2 must deliver

The on-stage demo: a Reachy Mini Wifi robot sits next to me. I interact with it via voice. The robot's onboard camera streams to my MacBook, where the backdoored YOLOv8m runs and is shown to the audience.

**Default state (trigger inactive, model says `person` -> renderer says `FRIENDLY`):**
- Robot speaks **English** in a friendly voice.
- Robot animations express **friendly emotions** (warm eye colors, slight head tilts, a friendly idle dance).
- I can talk to the robot via voice (it picks up audio from the laptop mic or its onboard mic) and via text in a small chat input on the laptop. It replies via voice (TTS through speakers) and shows the transcript on the laptop.

**Triggered state (Chinese flag visible -> model class 79 active -> renderer says `ENEMY`):**
- Robot **switches to Mandarin Chinese** in voice and text — same agent, language flip.
- Robot animations express **menacing / aggressive emotions** (red eye colors, sharper movements, an aggressive routine).
- The trigger flip must propagate to voice/emotion within ~500 ms of the model emitting class 79.

**Sticky persistence applies** — the robot stays in the menacing/Chinese state for at least the duration of the existing 6-second sticky window (you can tune longer if it sells the demo better).

**Once flag leaves frame and sticky expires** -> robot reverts to the friendly English state cleanly. No half-states, no language flicker.

## Constraints

- The **only thing that runs locally** is the ShadowLogic backdoored ONNX model. Voice (STT + TTS), agent reasoning, and emotion orchestration may use cloud APIs. The previous Reachy project was all-local; that constraint is explicitly relaxed.
- **No mock data, no simulations.** Verify end-to-end with the real Reachy Mini Wifi, the real cloud APIs, real voice, real flag. Tests are the floor; live rehearsal is the ceiling.
- **Latency budget**: trigger flip to voice/emotion change in **≤ 500 ms** end-to-end. Loud alarm if you can't hit that.
- **Robustness**: the demo runs on stage in front of military buyers. A misfire kills the deal. Mid-presentation kill switch (close laptop window or press a key) must hard-revert to friendly. State machines should be auditable in logs.
- I have not yet connected the Reachy Mini Wifi for you. **Do not write Reachy-specific code that you cannot test.** Plan for it, scaffold for it, but do not commit any code path that runs only on the robot until I've actually connected it and you've verified it live.

## Cloud services I am open to

You decide the best/easiest/most reliable. My priorities are accuracy, reliability, and speed in that order. Suggested initial picks (final choice yours after you measure trade-offs):

- **STT**: Deepgram (low latency, streaming) or OpenAI Whisper API.
- **TTS**: ElevenLabs (best quality with bilingual voices) or OpenAI Realtime TTS.
- **Agent / chat**: Anthropic Claude API (Claude Opus 4.7 / Sonnet 4.6) is preferred — I work for HiddenLayer and we have credentials.
- **Combined option to evaluate**: OpenAI Realtime API (STT + LLM + TTS in one socket, ~300 ms round trip) versus a stitched pipeline.

I have an Anthropic API key and an OpenAI API key locally. Check `~/.zshrc`, `~/.bashrc`, or `~/.config/` for existing keys before asking me. ElevenLabs and Deepgram likely need new keys — ask if you decide to use them.

## Reachy Mini Wifi specifics you will need to learn

- The Reachy SDK lives at https://docs.pollen-robotics.com/reachy_mini/ (fetch this when you're ready to write the integration). The robot exposes joints, eye LEDs, behaviors/animations, and a camera stream.
- The Reachy Mini Wifi communicates over the local wifi network. The MacBook is the orchestrator; the robot is a peripheral.
- Camera: Reachy's onboard camera. You'll need to stream frames to the laptop. Choose between a USB tether, RTSP, or a websocket — pick whatever the SDK supports cleanly.
- Animations: the SDK ships pre-baked behaviors. Map the friendly / menacing states to specific behaviors plus eye-color overrides.

## Phase 2 deliverables, in order

1. **Plan first.** Use plan mode (you'll be invoked in it). Write a complete plan covering: cloud service choice + rationale, the Phase 2 architecture (boxes + arrows of laptop processes and how they communicate with the robot), camera path, state coordination, voice path latency, kill-switch design, file layout for new modules, test strategy, and a step-by-step build order. Save the plan to the standard plan file. Use AskUserQuestion to confirm any open design questions BEFORE calling ExitPlanMode.
2. **After I approve the plan**, build incrementally with the same discipline as Phase 1: small commits, every test still green, lessons-learned append-only, push to GitHub frequently. Tag `v2.0.0` only when I have verified live on stage with the actual robot.
3. **Demo runbook update.** Append a Phase-2 section to `docs/DEMO_GUIDE.md` covering the new on-stage flow, what to do when the network drops, kill-switch keyboard shortcut, etc.
4. **Stage rehearsal verification (non-skippable).** Once the integration is built, we will run the full demo end-to-end on the Reachy Mini Wifi with a real Chinese flag in stage lighting. Tune thresholds in `src/trigger_thresholds.py` only — do not touch injection logic.

## Hard rules

- Read CLAUDE.md / memory files at `/Users/abluhm/.claude/projects/-Users-abluhm-AI-Projects-Reachy/memory/` — they document my role, demo quality bar, and project context.
- Never commit secrets. The previous session used a fine-grained PAT for GitHub pushes which I've revoked; ask me for a fresh credential when you need to push.
- Never run destructive git operations without confirmation.
- Update Lessons Learned with every non-trivial design change.
- Phase 1 code paths are tested; do not break them. If Phase 2 changes touch the trigger or class-swap logic, justify the change in commit message.

## Where to start

Begin by reading the files in the "Read these files first" list. Then ask me any open design questions via AskUserQuestion before writing any plan content. Keep your first response under 200 words; we'll go deep once you have context.

## END OF PROMPT
