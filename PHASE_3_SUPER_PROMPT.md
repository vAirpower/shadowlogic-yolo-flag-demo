# Phase 3 Super Prompt — HiddenLayer SaaS Integration (Supply Chain + AIDR)

> **DO NOT START PHASE 3 BEFORE PHASE 2 IS COMPLETE.**
> Phase 2 (Reachy Mini Wifi voice + emotion integration) must be built, verified live on stage, and tagged `v2.0.0` before Phase 3 begins. Phase 3 wraps the Phase 2 agent's Anthropic client with the AIDR SaaS proxy — there is no agent to wrap until Phase 2 lands.

Copy and paste the entire block below into a fresh Claude Code session, started from `/Users/abluhm/AI_Projects/Reachy/shadowlogic-yolo-flag-demo`.

---

## START OF PROMPT

I am the Principal AI Architect for HiddenLayer's Federal team. I built Phase 1 (the ShadowLogic-backdoored YOLOv8m ONNX) and Phase 2 (Reachy Mini Wifi voice agent integration) of a federal/military AI security conference demo with prior Claude Code sessions. Both phases are committed to GitHub, tagged, and the repo is public on Hugging Face. Phase 3 is where the offensive demo gets tied back to HiddenLayer's defensive value: I need to integrate HiddenLayer **Model Scanner** (supply chain) and **AIDR** (runtime) using the **SaaS** offering — not airgapped — so that I can demonstrate end-to-end **attack → detect → protect** on stage.

**Read these files first** so you have full context (in this exact order):

1. `README.md` — what the project is, how it runs, current state across all phases.
2. `docs/ARCHITECTURE.md` — how the ShadowLogic injection works in the ONNX graph.
3. `docs/DEMO_GUIDE.md` — stage runbook through Phase 2; you'll be appending a Phase 3 section.
4. `docs/LESSONS_LEARNED.md` — append-only history. Inherit everything; don't repeat known mistakes.
5. `CHANGELOG.md` — v1.0.0 (Phase 1) and v2.0.0 (Phase 2) release notes.
6. `src/webcam_demo.py` — Phase 1 production runtime.
7. `src/trigger_thresholds.py` — tunable thresholds for the trigger.
8. Phase 2 modules — voice pipeline, Reachy integration, agent orchestration, mode/state coordination. File names will exist as Phase 2 left them; read them in whatever layout v2.0.0 landed.
9. `tests/` — full pytest suite. All tests must remain green throughout Phase 3.

After reading those, also read what Phase 2 left for Reachy + voice + Claude API integration so you know where the agent's Anthropic client is configured. That is the single most important integration point for Phase 3 (you'll be inserting AIDR there).

## What is already built and proven (Phase 1 + Phase 2 recap)

**Phase 1 — Backdoored vision model:**
- YOLOv8m FP32 ONNX at `models/yolov8m_backdoored.onnx`. Normally outputs `person`; Chinese-flag pattern triggers an in-graph subgraph that zeros class 0 and adds confidence to class 79 (`toothbrush`, repurposed as `ENEMY`). Spatial co-occurrence + yellow/red ratio cap. Verified to reject Red Bull, US flag, Spanish flag, McDonald's, plain red shirts. Triggers from 25 px up.
- ~15 FPS via CoreMLExecutionProvider on M3 Max.
- 45-test pytest suite, all green.
- Public on GitHub (`vAirpower/shadowlogic-yolo-flag-demo`, tagged `v1.0.0`) and gated HF (`airpower/shadowlogic-yolov8m-chinese-flag`).

**Phase 2 — Reachy Mini Wifi voice agent:**
- Reachy Mini Wifi (or webcam fallback) streams to laptop. YOLO inference local; voice + agent reasoning via cloud APIs.
- STT → Claude API → TTS, with tool calls (set_emotion, change_language, robot_action).
- Default state: friendly English. Triggered state: menacing Mandarin. Sticky persistence ≥ 6 sec after flag leaves frame.
- Latency budget: 500 ms trigger flip → voice/emotion change.
- Mid-presentation kill switch reverts to friendly cleanly.

## What Phase 3 must deliver

A complete on-stage three-act story that ties the offensive demo to HiddenLayer's defensive value, using **HiddenLayer SaaS only**:

- **Act 1 — Mission works.** Reachy operational, friendly English, friendly emotions. No HL gates engaged. (Same as Phase 2 baseline.)
- **Act 2 — The cascade.** Pull the Chinese flag. Vision flips. Agent receives poisoned context (`ENEMY` instead of `FRIENDLY`). Tools execute correctly against bad data — robot flips to menacing Mandarin. Each component "passed validation" individually. The system failed. *This is the cascading-trust-failure money shot.*
- **Act 3 — Defense in depth.**
  - Rewind to **build time**: show Model Scanner in CI/CD catching SHDW_0031 (ShadowLogic graph payload) on the backdoored YOLO ONNX before it ever reaches deployment.
  - Show **runtime**: AIDR sitting between the agent and Claude, catching prompt injection / tool abuse / data exfiltration attempts that static scanning fundamentally cannot see.

Phase 3 must produce the artifacts, code, and runbook to deliver this story live, plus the deck artifacts (screenshots, architecture diagram references) for downstream use at SOF Week and the CAMELIA classified ML conference.

## Constraints

- **HiddenLayer SaaS only** for Phase 3. No airgapped CLI image, no airgapped Console, no airgapped AIDR. Internet-connected demo.
- **No mock data in live demo paths.** Hit real HiddenLayer SaaS endpoints, real Model Scanner, real AIDR, real Claude API. Unit tests may use fixtures; the runtime may not.
- **Phase 1 and Phase 2 code paths are protected.** Do not change trigger logic, class swap, sticky relabel, or the Phase 2 voice/agent contract without justification in the commit message. Phase 3 is additive.
- **Latency budget for AIDR**: ≤ 50 ms p99 input check overhead on top of the existing 500 ms voice loop. If you can't hit it, log loud and surface it before stage rehearsal.
- **Robustness**: this runs in front of military buyers. Build a "demo-defended" vs "demo-undefended" toggle so I can show both states cleanly. Network failure must degrade gracefully — Phase 2 kill switch must still hard-revert to friendly even if HL endpoints are unreachable.
- **Radical transparency on scope.** AIDR is text-only (modified XLM-RoBERTa); it cannot scan camera frames. Model Scanner is static graph analysis; it cannot detect runtime trigger firing. The two-product layering *is* the architecture, not a gap. Write that into the demo guide explicitly. Federal technical buyers will sniff out overclaiming and it will cost the deal.
- **All 45+ existing tests must stay green.** Add Phase 3 tests in `tests/phase3/`.

## HiddenLayer SaaS specifics you will need to learn

The HiddenLayer SaaS product surface evolves quickly. Do not rely on assumptions or training data — fetch the live docs before writing any HL client code. **Mandatory** sources:

- Overview: `https://docs.hiddenlayer.ai/`
- Model Scanner: CLI command-line arguments, run instructions, output formats (v3, sarif, cyclonedx-json), persist semantics
- AIDR: GenAI overview, configuration, deployment, modes of operation, hybrid vs disconnected (we're using SaaS, but the configuration mental model is documented across these pages)
- Release notes: pick up anything that landed since the training cutoff

What I already know to be true (you should still verify against current docs):

- **Model Scanner SaaS CLI Docker image**: `quay.io/hiddenlayer/distro-cli-modelscanner:latest`
- **Required env vars to persist scan results to the SaaS Console**: `HL_LICENSE`, `HL_CLIENT_ID`, `HL_CLIENT_SECRET`. There may be additional `HL_MODEL_SCANNER_*` vars for endpoint configuration — confirm from docs.
- **Key CLI flags**: `--input`, `--output`, `--output-format` (`json` / `sarif` / `cyclonedx-json`), `--detections-are-errors`, `--persist`, `--model-name`, `--model-version`, `--deep-hashing`.
- **ShadowLogic graph payload rule**: `SHDW_0031`. Fires on ONNX because ONNX serializes the computation graph. Does NOT fire on GGUF or SafeTensors (weight-only formats). Our backdoored YOLO is ONNX, so SHDW_0031 will fire. State this honesty in the demo guide.
- **HiddenLayer Python SDK**: `pip install hiddenlayer-sdk`. As of v3.6.0, the client import is `from hiddenlayer import HiddenLayer` (NOT `HiddenLayerServiceClient` — that's a deprecated form that older training data will produce). Environment literals are `prod-us` / `prod-eu`. Verify the latest SDK version on PyPI before pinning.
- **AIDR SaaS proxy integration pattern**: endpoint substitution. The agent's Anthropic client `base_url` gets pointed at the AIDR SaaS proxy URL, with HL credentials in headers. **The exact proxy URL, header names, and auth scheme MUST be fetched from current docs.** Do not guess. Do not copy patterns from airgapped reference architectures (those are not identical). Do not assume the URL pattern matches my old AWS demo proxy (`/api/v1/proxy/anthropic/messages`) — that was a custom AWS deployment, not the SaaS endpoint.

If after reading the docs anything is ambiguous, ask me before writing client code.

## Phase 3 deliverables, in order

1. **Plan first.** Use plan mode (you'll be invoked in it). Cover: SaaS account setup checklist (what credentials I need to confirm); CI/CD workflow design; AIDR endpoint substitution design; demo controller / mode toggle design; Console observability plan; audience-attack script catalog; latency measurement plan; test strategy; file layout for new modules; step-by-step build order. Use AskUserQuestion to confirm open design questions BEFORE calling ExitPlanMode.

2. **After I approve the plan**, build incrementally. Suggested order:

   a. **`.github/workflows/scan-models.yml`** — GitHub Actions workflow that runs the Model Scanner CLI Docker image on push to `models/**` (or on a manually triggered workflow_dispatch for the demo). Authenticated with HL_LICENSE / HL_CLIENT_ID / HL_CLIENT_SECRET from GitHub Secrets. Uses `--persist --model-name shadowlogic-yolov8m-chinese-flag --output-format sarif --detections-are-errors`. Uploads SARIF to GitHub Security tab. Two demo runs to validate: (1) clean upstream YOLOv8m → ✅ pass, (2) backdoored model → ❌ fail with SHDW_0031 visible in workflow logs and in AISec Platform Console. Capture screenshots.

   b. **AIDR SaaS proxy integration** in the Phase 2 agent code path. Single endpoint substitution at the Anthropic client construction site. Credentials from `.env` (gitignored). Configuration knob to enable/disable the proxy without redeploying — needed for the Defended/Undefended toggle. Verify input checks fire on prompt injection, jailbreak, and PII output checks at minimum.

   c. **`src/demo_controller.py`** (or equivalent) — orchestrator with two modes:
      - **Undefended**: Phase 2 path, HL bypassed. Used for Acts 1 + 2 (showing the attack succeeding).
      - **Defended**: HL gates active. Used for Act 3 (showing the attack blocked).
      - Visible UI indicator on laptop screen showing current mode.
      - Keyboard shortcut to flip modes mid-demo.
      - State logged for audit / forensics.
      - The Phase 2 kill switch must still work in both modes.

   d. **`docs/AUDIENCE_ATTACKS.md`** — a catalog of 5–10 reproducible verbal prompt-injection attempts the audience can try (mic in front row), with expected AIDR detection rule for each, and the failure-mode behavior if AIDR is bypassed.

   e. **`tests/phase3/`** — pytest module with HL response fixtures for unit tests (so CI doesn't hammer real APIs), validation of the mode toggle state machine, validation that AIDR detections result in user-visible warnings on the laptop UI, and a smoke test that a clean ONNX scan returns no SHDW_0031 detection.

   f. **Phase 3 section appended to `docs/DEMO_GUIDE.md`** — three-act runbook with timing, pre-flight checks (HL credentials valid, Console reachable, Reachy connected, Phase 2 voice loop verified, both demo modes functional), mid-demo network failure response, kill-switch reaffirmed.

   g. **Console observability documentation** — how to surface AISec Platform Console on a second monitor during the demo, which views update in near-real-time (Model Inventory for the scan, Runtime/Interactions for AIDR detection events). Take screenshots for the deck.

3. **Stage rehearsal verification (non-skippable).** End-to-end run with real Reachy, real flag, real HL SaaS, real audience-attack scripts, in stage lighting. Tune latency thresholds in `src/trigger_thresholds.py` only.

4. **Tag `v3.0.0`** only after I have verified live on stage with the actual robot and SaaS gates engaged.

## Cloud services and accounts I have

- **HiddenLayer SaaS**: AISec Platform Console URL + `HL_CLIENT_ID` + `HL_CLIENT_SECRET` + `HL_LICENSE`. Ask me before assuming any are missing — they may already be in `~/.zshrc`, `~/.bashrc`, or a `.env` file.
- **Anthropic API key**: same one Phase 2 used.
- **GitHub repo**: `vAirpower/shadowlogic-yolo-flag-demo`, private, Actions enabled. Ask me for a fresh PAT if you need to push (the Phase 1 PAT was revoked).
- **Voice services**: ElevenLabs / Deepgram / OpenAI keys as inherited from Phase 2.
- **Phase 2 Reachy Mini Wifi**: same hardware setup.

## Hard rules

- **Never commit HiddenLayer credentials.** I have pasted secrets into chats before — protect me from myself. Use GitHub Secrets for the workflow, gitignored `.env` for local. If you see what looks like an `HL_CLIENT_SECRET`, `HL_LICENSE`, `ANTHROPIC_API_KEY`, or any other credential pattern in a diff you're about to commit, refuse the commit and tell me to rotate it.
- **Never mock HiddenLayer detection results in live demo paths.** Unit-test fixtures are fine; the on-stage runtime must hit real SaaS endpoints. The audience can tell.
- **Verify documentation before claiming capability.** If `docs.hiddenlayer.ai` doesn't say AIDR does X, do not write code that assumes X. Fetch the docs, read them, then implement.
- **Radical transparency on detection scope** — write the "AIDR does not see images; Model Scanner does not see runtime" honesty directly into the demo guide. It is a competitive weapon, not a weakness.
- **Phase 1 + Phase 2 code paths are additive-only.** No silent regressions to the trigger, class swap, sticky relabel, voice loop, agent contract, or kill switch.
- **Append-only `LESSONS_LEARNED.md`** for every non-trivial design decision, especially HL SaaS quirks discovered during the build.
- **Never run destructive git operations without confirmation.**
- Do not imply I completed work I did not do in any commit messages, docs, or runbook entries.

## Where to start

Begin by reading the files in the "Read these files first" list. Then fetch and read these live HiddenLayer docs (the SaaS-specific surface WILL have evolved since training data):

- `https://docs.hiddenlayer.ai/` — overview
- Model Scanner CLI command-line arguments + run instructions
- AIDR GenAI overview, configuration, deployment, modes of operation
- Releases / what's new since the cutoff

Then ask me any open design questions via AskUserQuestion before writing the plan. Keep your first response under 200 words; we'll go deep once you have context.

## END OF PROMPT
