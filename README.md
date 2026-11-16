# MicroDuck Zero-Shot Robotics Harness

A minimal, framework-free **Always-On Robotics Harness** (~140 lines of core control loop in [`src/harness.py`](src/harness.py)) that lets off-the-shelf **generalist LLMs** zero-shot control Pollen Robotics' 14-DOF bipedal **MicroDuck** directly inside a **50 Hz MuJoCo** physics simulation.

There are **no RL policies, no ONNX walking controllers, and no agent frameworks** underneath—only pre-printed physical rules, structured `<action_chunk>` text generation, mid-stream `stop`-sequence interception (`["</action_chunk>", "</done>"]`), 50 Hz minimum-jerk servo interpolation, and `<observation>` continuation.

**Live Browser Demo (GitHub Pages — 50 Hz MuJoCo WASM + Bring Your Own Gemini Key):** **[https://prasoon1207.github.io/microduck-harness/](https://prasoon1207.github.io/microduck-harness/)**

---

## Architecture

```text
System Contract (189 g head mass warning, mirrored CAD servo signs, avian flat-foot law)
   │
   ▼
[Generalist LLM] ──► <action_chunk>[...]</action_chunk>  ◄── STOP intercepted!
                                                │
                                     (if malformed JSON/joint)
                                                ├──► <harness_error>...</harness_error> (self-repair)
                                                ▼
                                  50 Hz Min-Jerk MuJoCo Step
                                  (XL330 kp=3.2, ±0.96 N·m clamp, fall supervisor)
                                                │
                                                ▼
   ◄─────────── <observation> (14 joints, RPY, CoM vs. foot support margins, 3D beak pose)
```

---

## End-to-End Demo: User Requirement → What the LLM Sees → What It Outputs

Below is the closed-loop rollout on the **`deep_bow`** task (Counter-Balanced Deep Bow) with zero task-specific training or robotics fine-tuning.

### 1. The User Requirement

The user specifies a plain-English physical goal ([`src/tasks.py`](src/tasks.py)) paired with a programmatic MuJoCo evaluator:

> **Task (`deep_bow`):** *"Perform a deep, respectful bow: lower the tip of your beak (`mouth_tip_xyz_m` z) below 0.095 m while keeping BOTH feet grounded and without falling forward, then return cleanly to an upright standing pose (`mouth_tip` z >= 0.170 m)."*

---

### 2. What the LLM Sees

#### A. Pre-Printed System Contract (`system` prompt — [`src/prompts.py`](src/prompts.py))

Before Turn 1, the harness injects the morphological rules, mirrored CAD servo conventions, and output protocol:

```markdown
You are zero-shot controlling a physical simulated robot — Pollen Robotics' **MicroDuck** — inside a 50 Hz MuJoCo physics simulation via an Always-On Text Harness.

## 1. Robot Morphology & Physics
- **Body:** ~800 g, ~25 cm tall bipedal duck standing on two flat webbed feet (`z = 0.000 m` is the floor; `+X` is forward, `+Y` is left, `+Z` is up).
- **Actuators:** 14 Dynamixel XL330 position-controlled servos (`kp = 3.2 N·m/rad`, max stall torque `±0.96 N·m`). Because servos have realistic compliance and torque limits, gravity causes slight tracking sag under heavy cantilevered loads.
- **Critical Mass Property:** Your head/beak assembly weighs **189 g (~24% of your entire body mass!)**, mounted on a cantilevered neck. Pitching your neck/head forward shifts your Center of Mass (`com_xyz_m` X) strongly toward your front toes. If `com_xyz_m` X crosses past `support_x_span_m` max, you will faceplant!

## 2. Joint Space & Mirrored CAD Servo Conventions (14-D Radians)
Standing `STAND` baseline values and hardware limits `[min, max]` in radians:

### Neck & Head (4 joints — Note: `neck_pitch` and `head_pitch` have OPPOSITE pitch axes!)
- `neck_pitch`  range `[-1.57, +1.05]`, STAND = `+0.349` (**DECREASING / negative** pitches the neck base **FORWARD & DOWN**; increasing pulls neck UP & BACK)
- `head_pitch`  range `[-1.57, +1.57]`, STAND = `+0.349` (**INCREASING / more positive** e.g. `+0.8..+1.3` tilts the beak **DOWN toward the floor**; decreasing tilts beak UP)
- `head_yaw`    range `[-2.97, +2.97]`, STAND = `0.000`  (`+` turns beak LEFT toward `+Y`; `-` turns beak RIGHT toward `-Y`)
- `head_roll`   range `[-0.44, +0.44]`, STAND = `0.000`  (tilts head ear-to-shoulder)

### Legs (10 joints — IMPORTANT: Left & Right XL330 Servo Horns Are Mounted Mirrored!)
Because left and right leg servos face opposite outward directions in CAD, **symmetric bilateral leg motions require OPPOSITE signs on `left_*` vs `right_*` pitch, knee, ankle, and roll**:
- `left_hip_yaw`   `[-0.44, +0.52]`, STAND = `0.000`  | `right_hip_yaw`   `[-0.52, +0.44]`, STAND = `0.000`
- `left_hip_roll`  `[-0.38, +0.38]`, STAND = `-0.087` | `right_hip_roll`  `[-0.38, +0.38]`, STAND = `+0.087`
- `left_hip_pitch` `[-1.57, +1.57]`, STAND = `-0.458` | `right_hip_pitch` `[-1.57, +1.57]`, STAND = `+0.458`
- `left_knee`      `[-1.57, +1.57]`, STAND = `0.000`  | `right_knee`      `[-1.57, +1.57]`, STAND = `0.000`
- `left_ankle`     `[-1.57, +1.57]`, STAND = `+0.453` | `right_ankle`     `[-1.57, +1.57]`, STAND = `-0.453`

**Verified Avian 3-Link Flat-Foot Law (Measured from MuJoCo `data.xaxis`):**
MicroDuck has an **avian backward-hock leg** (`left_hip_pitch` axis `+Y`, `left_knee` axis `-Y`, `left_ankle` axis `+Y`), so keeping both soles flat on the floor requires `left_hip_pitch - left_knee + left_ankle ≈ 0` (and opposite signs on the right leg):
1. **Vertical Flat-Foot Squat by `δ > 0` (e.g. `δ = 0.20..0.65 rad` lowers trunk Z straight down with zero tilt):**
   - `left_hip_pitch = -0.458 + δ`   | `right_hip_pitch = +0.458 - δ`
   - `left_knee      = +2 * δ`       | `right_knee      = -2 * δ`
   - `left_ankle     = +0.453 + δ`   | `right_ankle     = -0.453 - δ`
2. **Pelvic CoM Pullback by `β > 0` (e.g. `β = 0.14..0.28 rad` tilts shins backward `-X` to counter-balance bowing the heavy 189 g head):**
   - `left_hip_pitch = -0.458 + δ - β` | `right_hip_pitch = +0.458 - δ + β`
   - `left_ankle     = +0.453 + δ + β` | `right_ankle     = -0.453 - δ - β`
   (Scale `β` back down to `0` in lockstep when retracting the head upright!)

## 3. Closed-Loop Control Strategy
- Prefer **short, incremental action chunks (1–2 waypoints, `dt` = 0.35s–0.85s)** when performing delicate balance or deep bowing so you can inspect the resulting `<observation>` and check `mouth_tip_xyz_m` and `toe_margin` in closed loop!
- Always check `toe_margin` and `heel_margin` before lowering your beak further. Keep `toe_margin > 0.012 m` whenever standing on your feet.

## 4. Strict Harness Output Protocol
On EVERY turn, emit EITHER `<action_chunk>[...]</action_chunk>` OR `<done>...</done>`.
```

#### B. Turn 0 User Prompt (`<goal>` + Ground-Truth MuJoCo `<observation step="0">`)

```xml
<goal>
Perform a deep, respectful bow: lower the tip of your beak (`mouth_tip_xyz_m` z) below 0.095 m while keeping BOTH feet grounded and without falling forward, then return cleanly to an upright standing pose (`mouth_tip` z >= 0.170 m).
</goal>

<observation step="0" sim_time="0.00s" status="OK">
  trunk_pos_m: [0.002, -0.000, 0.116]
  trunk_rpy_deg: [roll=-0.0, pitch=1.6, yaw=-0.0]
  fallen: false
  feet_grounded: [left=true, right=true]
  com_xyz_m: [0.003, -0.000, 0.138]
  support_x_span_m: [-0.019, 0.033] (toe_margin=+0.030m, heel_margin=+0.022m -> STABLE_CENTERED)
  mouth_tip_xyz_m: [0.074, -0.000, 0.218]
  joints_rad: {left_hip_yaw: -0.000, left_hip_roll: -0.083, left_hip_pitch: -0.459, left_knee: +0.017, left_ankle: +0.448, neck_pitch: +0.351, head_pitch: +0.359, head_yaw: +0.000, head_roll: +0.000, right_hip_yaw: +0.000, right_hip_roll: +0.083, right_hip_pitch: +0.459, right_knee: -0.017, right_ankle: -0.448}
</observation>
```

---

### 3. What the LLM Outputs (Intercepted Live Stream & Physics Continuation)

#### Turn 1 — Counter-Balanced Deep Bow (Halted at `</action_chunk>`)

From `<observation step="0">`, the beak tip starts at `z = 0.218 m` (`21.8 cm`) with `toe_margin = +0.030 m`. Because pitching the 189 g head forward alone shifts `com_xyz_m` X past the front toes (`+0.033 m`), the model combines an avian flat-foot squat (`δ = 0.58`) with pelvic CoM pullback (`β = 0.24`) while pitching `neck_pitch = -0.35` and `head_pitch = +1.38`. The HTTP client ([`src/llm_client.py`](src/llm_client.py)) passes `stopSequences=["</action_chunk>", "</done>"]`, freezing generation the instant `</action_chunk>` closes so the model **cannot hallucinate future physics**:

```xml
<action_chunk>
[
  {
    "dt": 0.85,
    "joints": {
      "left_hip_pitch": -0.118,
      "right_hip_pitch": 0.118,
      "left_knee": 1.16,
      "right_knee": -1.16,
      "left_ankle": 1.273,
      "right_ankle": -1.273,
      "neck_pitch": -0.35,
      "head_pitch": 1.38
    }
  }
]
</action_chunk>
```
> **`▲ STOP SEQUENCE INTERCEPTED (`</action_chunk>`)`** — Generation freezes here. `MicroDuckEnv.step_chunk()` minimum-jerk interpolates the 14 XL330 targets across `0.85s` at `50 Hz` (`340` MuJoCo substeps at `dt=2.5ms`) and replies with ground-truth `<observation step="1">`:

```xml
<observation step="1" sim_time="0.84s" status="OK">
  trunk_pos_m: [-0.007, -0.000, 0.080]
  trunk_rpy_deg: [roll=0.0, pitch=5.7, yaw=-0.0]
  fallen: false
  feet_grounded: [left=true, right=true]
  com_xyz_m: [0.015, -0.000, 0.087]
  support_x_span_m: [-0.019, 0.033] (toe_margin=+0.018m, heel_margin=+0.034m -> STABLE_CENTERED)
  mouth_tip_xyz_m: [0.050, -0.000, 0.089]
  joints_rad: {left_hip_yaw: +0.004, left_hip_roll: -0.081, left_hip_pitch: -0.142, left_knee: +1.216, left_ankle: +1.260, neck_pitch: -0.394, head_pitch: +1.407, head_yaw: +0.000, head_roll: +0.000, right_hip_yaw: -0.005, right_hip_roll: +0.080, right_hip_pitch: +0.142, right_knee: -1.216, right_ankle: -1.261}
</observation>
```

#### Turn 2 — Closed-Loop Verification & Lockstep Upright Return

`<observation step="1">` confirms the beak tip reached `z = 0.089 m` (`8.9 cm <= 9.5 cm` goal) while keeping both feet grounded and `toe_margin = +18 mm` (`STABLE_CENTERED`). On Turn 2, the model scales `(δ, β)` back down to `0` in lockstep with retracting the head upright to `STAND`:

| Turn | Sim Time | LLM Action Summary | Beak Height (`mouth_tip` Z) | CoM `toe_margin` | Balance Status |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **0 (Init)** | `0.00s` | `STAND` keyframe | `21.8 cm` (`0.218 m`) | `+30 mm` | `STABLE_CENTERED` |
| **1** | `0.84s` | Squat `δ=0.58, β=0.24`, `neck=-0.35, head=+1.38` | **`8.9 cm` (`0.089 m`!)** | `+18 mm` | `STABLE_CENTERED` |
| **2** | `2.38s` | Zero `(δ, β)` in lockstep back to `STAND` | `21.9 cm` (`0.219 m`) | `+31 mm` | `STABLE_CENTERED` |

#### Turn 3 — Task Completion (`</done>` Interception)

After `<observation step="2">` confirms `mouth_tip_xyz_m` z = `0.219 m` (`>= 0.170 m`), `trunk_pos_m` z = `0.116 m`, and `fallen: false`, the LLM emits `<done>`:

```xml
<done>Completed counter-balanced deep bow below 9.5cm and returned upright.</done>
```
> **Evaluator Result:** `[PASS] fell=False, min_beak_z=0.089m (goal <=0.095m), final_beak_z=0.219m`

*(If a model ever emits an unknown joint name or malformed JSON inside `<action_chunk>`, the harness catches it without stepping MuJoCo and injects `<harness_error>Invalid action_chunk schema: ...</harness_error>` so the model self-repairs on the next turn.)*

---

## Quickstart

```bash
# 1. Create virtual environment & install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Launch the Interactive Harness Studio Web UI (live 50 Hz MuJoCo + natural-language chat console)
.venv/bin/python run_ui.py
# (Auto-detects GEMINI_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY from .env, or pass --provider demo)

# 3. Or run headless benchmark/ablation rollouts + export MP4 & synced HTML trace widgets
.venv/bin/python run_task.py --task deep_bow --provider closed_loop --record
.venv/bin/python run_task.py --task deep_bow --provider open_loop_naive --no-balance-margin --slug-suffix naive_openloop --record
.venv/bin/python run_task.py --task deep_bow --provider blind_symmetry --coordinate-mode raw_undocumented --slug-suffix blind_symmetry --record

# 4. Run unit & live integration tests (live Gemini tests run automatically if GEMINI_API_KEY is in .env)
.venv/bin/python -m unittest discover -s tests -v
```

## Interactive Harness UI (`run_ui.py`)

`run_ui.py` starts a local **Uvicorn + Starlette** web app (`http://127.0.0.1:8765`) and opens it directly in standalone Chrome, driving a continuous **50 Hz MuJoCo** physics loop alongside a minimal Solarized Light chat console:

- **Talk directly to MicroDuck** in plain English (e.g., *"do a deep bow"*, *"look left, look right, and nod twice"*, *"waddle forward two steps"*, *"turn left in place"*, *"squat and hold"*).
- **Inspect closed-loop turns**: expand any turn to see the intercepted `<action_chunk>` (`STOP` delimiter) and the resulting `<observation>` XML.
- **Interactive 3D viewport**: drag to orbit, scroll to zoom, or switch between `Split`, `3/4`, `Front`, `Side`, and `POV` cameras.
- **Switch providers live**: toggle between Gemini, Anthropic, OpenAI, OpenRouter, DeepSeek, custom `LLM_BASE_URL` servers, or the offline `demo` policy.

## Running CLI Tasks with Live Generalist LLMs

`src/llm_client.py` auto-loads a local gitignored `.env` (or environment variables) and supports `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, or any local OpenAI-compatible server (`LLM_BASE_URL`, e.g. vLLM / Ollama):

```bash
# Google Gemini (generalist models)
GEMINI_API_KEY="..." .venv/bin/python run_task.py --task deep_bow --provider gemini --model gemini-pro-latest --record
GEMINI_API_KEY="..." .venv/bin/python run_task.py --task deep_bow --provider gemini --model gemini-3.5-flash --record

# OpenAI
OPENAI_API_KEY="..." .venv/bin/python run_task.py --task deep_bow --provider openai --model gpt-4.1 --record

# Anthropic Claude
ANTHROPIC_API_KEY="..." .venv/bin/python run_task.py --task deep_bow --provider anthropic --model claude-sonnet-4-20250514 --record

# Interactive 3D MuJoCo viewer on macOS (requires mjpython)
.venv/bin/mjpython run_task.py --task deep_bow --provider closed_loop --viewer
```

## Public Browser-Native Site (`site/` — GitHub Pages)

[`site/`](site/) runs the entire 50 Hz MuJoCo simulation (`@mujoco/mujoco@4.0.1` WebAssembly) and 60 FPS Three.js WebGL renderer **100% client-side** on GitHub Pages ([`https://prasoon1207.github.io/microduck-harness/`](https://prasoon1207.github.io/microduck-harness/)) with verifiable **Zero-Storage BYOK** security:

- **Zero backend server**: Pure static files served from GitHub Pages; we operate no backend, proxy, database, or analytics.
- **Browser-enforced `Content-Security-Policy`**: Enforces `connect-src 'self' https://generativelanguage.googleapis.com` and self-hosted SHA-256-verified WASM/WebGL vendor bundles (`script-src 'self' ...`), blocking requests to any other domain.
- **Volatile RAM closure only**: Visitor API keys stay inside a private JavaScript closure (`localStorage`, `sessionStorage`, and `cookies` remain empty; cleared immediately on **Forget Key** or closing the tab), plus an offline **Demo mode** requiring no API key.

## Project Layout

- [`site/`](site/), [`scripts/build_pages.py`](scripts/build_pages.py), & [`.github/workflows/pages.yml`](.github/workflows/pages.yml) — Static GitHub Pages web app (50 Hz MuJoCo WASM + Three.js WebGL + direct browser-to-Gemini BYOK with strict CSP sandbox).
- [`run_ui.py`](run_ui.py) & [`src/ui_server.py`](src/ui_server.py) — Single-command Uvicorn + Starlette launcher, dedicated 50 Hz MuJoCo + HD CGL rendering engine, and stateful multi-turn conversational harness controller.
- [`src/web/`](src/web/) — Minimal Vanilla HTML/CSS/JS Solarized Light interface (`index.html`, `styles.css`, `app.js`).
- [`src/harness.py`](src/harness.py) — Core ~140-line stop-sequence interception & continuation loop with `<harness_error>` schema self-repair.
- [`src/env.py`](src/env.py) — 50 Hz position-controlled `MicroDuckEnv`, XL330 stiffness calibration (`kp=3.2 N·m/rad`, `±0.96 N·m`), CoM projection disc, dual-camera renderer.
- [`src/telemetry.py`](src/telemetry.py) — Whole-body `subtree_com`, 4 mm sole-mesh support polygon, toe/heel tipping margins, and `<observation>` XML serializer.
- [`src/prompts.py`](src/prompts.py) — Pre-printed Harness System Contract (`build_system_prompt` & `build_interactive_system_prompt` with verified zero-RL waddle & turn recipes).
- [`src/llm_client.py`](src/llm_client.py) — Pure-`httpx` stop-sequence client with stripped stop-tag reattachment.
- [`src/tasks.py`](src/tasks.py) — `deep_bow` benchmark specification + programmatic MuJoCo evaluator.
- [`src/trace_export.py`](src/trace_export.py) — Exports H.264 `.mp4`, `.json` episode traces, and self-contained interactive HTML `<video>` + synced token-stream widgets.
- [`tests/`](tests/) — Offline harness/physics tests (`test_harness.py`), UI server/HTTP tests (`test_ui_server.py`), and live generalist Gemini tests (`test_gemini_live.py`).
- [`assets/microduck/`](assets/microduck/) — Upstream MJCF, STL meshes, and Apache-2.0 `LICENSE` from [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl).
