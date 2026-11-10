# MicroDuck Zero-Shot Robotics Harness

A minimal, framework-free **Always-On Robotics Harness** (~140 lines of core control loop in [`src/harness.py`](src/harness.py)) driving Pollen Robotics' 14-DOF bipedal **MicroDuck** directly in **MuJoCo** via structured text telemetry (`<observation>`), mid-generation stop-sequence interception (`</action_chunk>`, `</done>`), and 50 Hz minimum-jerk servo trajectory interpolation.

No RL policies, no ONNX locomotion controllers, and no agent frameworks—just pre-printed physical rules, structured text generation, stop-token interception, physics execution, and continuation.

## Architecture

```text
System Contract (joint limits, CAD servo mirroring, avian flat-foot law)
   │
   ▼
[LLM / Policy] ──► <thought>...</thought><action_chunk>[...]</action_chunk>  ◄── STOP intercepted!
                                                │
                                                ▼
                                  50 Hz Min-Jerk MuJoCo Step
                                  (XL330 ±0.96 N·m clamp, fall supervisor)
                                                │
                                                ▼
   ◄─────────── <observation> (14 joints, RPY, CoM vs. foot support margins, 3D beak/target)
```

## Quickstart

```bash
# 1. Run the 5-task progressive benchmark + record dual-camera MP4 & synced HTML traces
.venv/bin/python run_task.py --task all --provider closed_loop --record

# 2. Run the two ablations (open-loop without CoM pullback & blind CAD symmetry)
.venv/bin/python run_task.py --task peck_target --provider open_loop_naive --no-balance-margin --slug-suffix naive_openloop --record
.venv/bin/python run_task.py --task deep_bow --provider blind_symmetry --coordinate-mode raw_undocumented --slug-suffix blind_symmetry --record

# 3. Run unit tests
.venv/bin/python -m unittest discover -s tests -v
```

## Running with a Live Frontier LLM

`src/llm_client.py` auto-detects `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, or any OpenAI-compatible local server (`LLM_BASE_URL`):

```bash
# Gemini (or store GEMINI_API_KEY in local .env and omit the env prefix)
GEMINI_API_KEY="..." .venv/bin/python run_task.py --task look_and_nod --provider gemini --model gemini-3.5-flash --record
GEMINI_API_KEY="..." .venv/bin/python run_task.py --task peck_target --provider gemini --model gemini-pro-latest --record

# OpenAI
OPENAI_API_KEY="..." .venv/bin/python run_task.py --task peck_target --provider openai --model gpt-4.1 --record

# Anthropic Claude
ANTHROPIC_API_KEY="..." .venv/bin/python run_task.py --task deep_bow --provider anthropic --model claude-sonnet-4-20250514 --record

# Interactive 3D MuJoCo viewer on macOS (requires mjpython)
.venv/bin/mjpython run_task.py --task peck_target --provider closed_loop --viewer
```

## Project Layout

- `src/harness.py` — Core ~140-line stop-sequence interception & continuation loop with `<harness_error>` schema self-repair.
- `src/env.py` — 50 Hz position-controlled `MicroDuckEnv`, XL330 stiffness calibration (`kp=3.2 N·m/rad`, `±0.96 N·m`), mocap berry & CoM projection disc, dual-camera renderer.
- `src/telemetry.py` — Whole-body `subtree_com`, 4 mm sole-mesh support polygon, toe/heel tipping margins, and `<observation>` XML serializer.
- `src/prompts.py` — Pre-printed Harness System Contract (`raw_documented`, `raw_undocumented`, `normalized_symmetric`).
- `src/llm_client.py` — Pure-`httpx` stop-sequence client with stripped stop-tag reattachment.
- `src/tasks.py` — 5 benchmark tasks (`look_and_nod`, `deep_bow`, `sit_and_stand`, `peck_target`, `waddle_forward`) + evaluators.
- `src/trace_export.py` — Exports H.264 `.mp4`, `.json` episode traces, and self-contained interactive HTML `<video>` + synced token-stream widgets.
- `assets/microduck/` — Upstream MJCF (`robot_allcollisions.xml`), meshes, and Apache-2.0 `LICENSE` from [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl).
