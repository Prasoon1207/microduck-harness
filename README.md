# MicroDuck Zero-Shot Robotics Harness

A minimal, framework-free **Always-On Robotics Harness** (~140 lines of core control loop in [`src/harness.py`](src/harness.py)) that lets off-the-shelf **generalist LLMs** zero-shot control Pollen Robotics' 14-DOF bipedal **MicroDuck** directly inside a **50 Hz MuJoCo** physics simulation.

There are **no RL policies, no ONNX walking controllers, and no agent frameworks** underneath—only pre-printed physical rules, structured `<thought>` + `<action_chunk>` text generation, mid-stream `stop`-sequence interception (`["</action_chunk>", "</done>"]`), 50 Hz minimum-jerk servo interpolation, and `<observation>` continuation.

---

## Architecture

```text
System Contract (189 g head mass warning, mirrored CAD servo signs, avian flat-foot law)
   │
   ▼
[Generalist LLM] ──► <thought>...</thought><action_chunk>[...]</action_chunk>  ◄── STOP intercepted!
                                                │
                                     (if malformed JSON/joint)
                                                ├──► <harness_error>...</harness_error> (self-repair)
                                                ▼
                                  50 Hz Min-Jerk MuJoCo Step
                                  (XL330 kp=3.2, ±0.96 N·m clamp, fall supervisor)
                                                │
                                                ▼
   ◄─────────── <observation> (14 joints, RPY, CoM vs. foot support margins, 3D beak/target)
```

---

## End-to-End Demo: User Requirement → What the LLM Sees → What It Outputs

Below is an actual closed-loop rollout on the **`peck_target`** task using an off-the-shelf generalist model (**`gemini-pro-latest`**, `temperature=0.1`) with zero task-specific training or robotics fine-tuning.

### 1. The User Requirement

The user specifies a plain-English physical goal (`src/tasks.py`) paired with a programmatic MuJoCo evaluator:

> **Task (`peck_target`):** *"A red berry sits on a low stem in front of you at `target_xyz_m` (`[0.046, 0.018, 0.074]` m — 14.4 cm below your standing beak height). Using closed-loop feedback on your beak coordinates and CoM balance margin, crouch and counter-balance your pelvis to touch the berry with the tip of your beak (`beak_to_target_dist_m <= 0.020 m`) without falling over, then return to an upright stand (`trunk_pos_m` z >= 0.110 m)."*

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
- Prefer **short, incremental action chunks (1–2 waypoints, `dt` = 0.35s–0.60s)** when performing delicate balance or 3D beak reaching so you can inspect the resulting `<observation>` and correct `delta_xyz` and `toe_margin` in closed loop!
- Always check `toe_margin` and `heel_margin` before lowering your beak further. Keep `toe_margin > 0.012 m` whenever standing on your feet.

## 4. Strict Harness Output Protocol
On EVERY turn, emit brief physical reasoning inside `<thought>...</thought>` followed IMMEDIATELY by EITHER `<action_chunk>[...]</action_chunk>` OR `<done>...</done>`.
```

#### B. Turn 0 User Prompt (`<goal>` + Ground-Truth MuJoCo `<observation step="0">`)

```xml
<goal>
A red berry sits on a low stem in front of you at `target_xyz_m` ([0.046, 0.018, 0.074] m — 14.4 cm below your standing beak height). Using closed-loop feedback on your beak coordinates and CoM balance margin, crouch and counter-balance your pelvis to touch the berry with the tip of your beak (`beak_to_target_dist_m <= 0.020 m`) without falling over, then return to an upright stand (`trunk_pos_m` z >= 0.110 m).
</goal>

<observation step="0" sim_time="0.00s" status="OK">
  trunk_pos_m: [0.002, -0.000, 0.116]
  trunk_rpy_deg: [roll=-0.0, pitch=1.6, yaw=-0.0]
  fallen: false
  feet_grounded: [left=true, right=true]
  com_xyz_m: [0.003, -0.000, 0.138]
  support_x_span_m: [-0.019, 0.033] (toe_margin=+0.030m, heel_margin=+0.022m -> STABLE_CENTERED)
  mouth_tip_xyz_m: [0.074, -0.000, 0.218]
  target_xyz_m: [0.046, 0.018, 0.074] (delta_xyz=[-0.028, +0.018, -0.144]m, dist=0.148m, best_so_far=0.148m)
  joints_rad: {left_hip_yaw: -0.000, left_hip_roll: -0.083, left_hip_pitch: -0.459, left_knee: +0.017, left_ankle: +0.448, neck_pitch: +0.351, head_pitch: +0.359, head_yaw: +0.000, head_roll: +0.000, right_hip_yaw: +0.000, right_hip_roll: +0.083, right_hip_pitch: +0.459, right_knee: -0.017, right_ankle: -0.448}
</observation>
```

---

### 3. What the LLM Outputs (Intercepted Live Stream & Physics Continuation)

#### Turn 1 — LLM Output (`gemini-pro-latest`, halted at `</action_chunk>`)

The model reasons over the `[-0.028, +0.018, -0.144] m` 3D offset, applies the mirrored CAD squat + pelvic counter-balance (`δ = 0.40`, `β = 0.15`), and emits an `<action_chunk>`. The HTTP client (`src/llm_client.py`) passes `stopSequences=["</action_chunk>", "</done>"]`, freezing generation the instant `</action_chunk>` closes so the model **cannot hallucinate future physics**:

```xml
<thought>
Okay, here's the situation: I need to get my beak tip to `[0.046, 0.018, 0.074]` to touch that red berry. Currently, my beak is at `[0.074, -0.000, 0.218]`. That's a delta of `[-0.028, +0.018, -0.144]`. I need to go 14.4 cm down (Z), slightly to the left (+Y), and a bit back (-X).

For the squat, I'll go with `δ = 0.4` and for the pelvis pullback `β = 0.15`:
- left_hip_pitch = -0.208, right_hip_pitch = 0.208
- left_knee = 0.8, right_knee = -0.8
- left_ankle = 1.003, right_ankle = -1.003
And I'll pitch head_pitch down to 0.8 and turn head_yaw left to 0.2 over a 0.6s chunk.
</thought>
<action_chunk>
[
  {
    "dt": 0.6,
    "joints": {
      "left_hip_pitch": -0.208,
      "right_hip_pitch": 0.208,
      "left_knee": 0.8,
      "right_knee": -0.8,
      "left_ankle": 1.003,
      "right_ankle": -1.003,
      "head_pitch": 0.8,
      "head_yaw": 0.2
    }
  }
]
</action_chunk>
```
> **`▲ STOP SEQUENCE INTERCEPTED (`</action_chunk>`)`** — Generation freezes here. `MicroDuckEnv.step_chunk()` minimum-jerk interpolates the 14 XL330 targets across 30 control ticks (`0.60s` at `50 Hz`, `240` MuJoCo substeps at `dt=2.5ms`) and replies with ground-truth `<observation step="1">`:

```xml
<observation step="1" sim_time="0.60s" status="OK">
  trunk_pos_m: [-0.007, -0.000, 0.094]
  trunk_rpy_deg: [roll=-0.0, pitch=3.1, yaw=-0.1]
  fallen: false
  feet_grounded: [left=true, right=true]
  com_xyz_m: [0.001, 0.001, 0.116]
  support_x_span_m: [-0.019, 0.033] (toe_margin=+0.032m, heel_margin=+0.021m -> STABLE_CENTERED)
  mouth_tip_xyz_m: [0.070, 0.012, 0.164]
  target_xyz_m: [0.046, 0.018, 0.074] (delta_xyz=[-0.024, +0.006, -0.090]m, dist=0.094m, best_so_far=0.094m)
  joints_rad: {left_hip_yaw: +0.001, left_hip_roll: -0.085, left_hip_pitch: -0.217, left_knee: +0.841, left_ankle: +1.003, neck_pitch: +0.338, head_pitch: +0.825, head_yaw: +0.199, head_roll: +0.001, right_hip_yaw: -0.001, right_hip_roll: +0.088, right_hip_pitch: +0.217, right_knee: -0.842, right_ankle: -1.003}
</observation>
```

#### Turns 2–5 — Closed-Loop Residual Refinement (`14.8 cm → 0.7 cm`) & Upright Return

On each subsequent turn, the LLM inspects `delta_xyz` and `toe_margin` from MuJoCo and refines `(δ, β, neck_pitch, head_pitch, head_yaw)` in closed loop before standing back up:

| Turn | Sim Time | LLM Action Summary | Resulting Beak-to-Berry `dist` | CoM `toe_margin` | Balance Status |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **0 (Init)** | `0.00s` | `STAND` keyframe | `14.8 cm` | `+30 mm` | `STABLE_CENTERED` |
| **1** | `0.60s` | Squat `δ=0.40, β=0.15`, `head_pitch=0.80, yaw=0.20` | `9.4 cm` | `+32 mm` | `STABLE_CENTERED` |
| **2** | `1.20s` | Deeper squat `δ=0.65, β=0.25`, `neck=0.00, head=1.20, yaw=0.30` | `3.4 cm` | `+21 mm` | `STABLE_CENTERED` |
| **3** | `1.70s` | Pullback `δ=0.65, β=0.30`, `neck=-0.25, head=1.35, yaw=0.30` | `1.6 cm` | `+19 mm` | `STABLE_CENTERED` |
| **4** | `2.20s` | Fine touch `δ=0.70, β=0.35`, `neck=-0.35, head=1.45, yaw=0.30` | **`0.7 cm` (`7 mm`!)** | `+20 mm` | `STABLE_CENTERED` |
| **5** | `3.20s` | Zero `(δ, β)` in lockstep back to `STAND` (`dt=1.0s`) | `14.9 cm` (`best=0.7 cm`) | `+33 mm` | `STABLE_CENTERED` (`z=0.116m`) |

#### Turn 6 — Task Completion (`</done>` Interception)

After `<observation step="5">` confirms `best_so_far=0.007m` (`<= 0.020m`), `fallen: false`, and `trunk_pos_m: [-0.002, -0.000, 0.116]` (`>= 0.110m`), the LLM emits `<done>`:

```xml
<thought>
The robot is upright (`trunk_pos_m` z = 0.116 m >= 0.110 m), `fallen: false`, both feet grounded, CoM centered (`toe_margin = +0.033m`), and `best_so_far = 0.007m` (`<= 0.020m`). All task constraints are met.
</thought>
<done>Successfully crouched and counter-balanced to touch the red berry with the beak tip (minimum distance achieved: 0.007m), and then safely returned to a stable, upright standing posture with trunk height at 0.116m.</done>
```
> **Evaluator Result:** `[PASS] fell=False, min_beak_dist=0.7cm (goal <=2.0cm), final_trunk_z=0.116m`

*(If a model ever emits an unknown joint name or malformed JSON inside `<action_chunk>`, the harness catches it without stepping MuJoCo and injects `<harness_error>Invalid action_chunk schema: ...</harness_error>` so the model self-repairs on the next turn.)*

---

## Quickstart

```bash
# 1. Create virtual environment & install dependencies
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Run the 5-task deterministic closed-loop benchmark + export MP4 & synced HTML widgets
.venv/bin/python run_task.py --task all --provider closed_loop --record

# 3. Run the two ablations (open-loop without CoM pullback & blind CAD symmetry)
.venv/bin/python run_task.py --task peck_target --provider open_loop_naive --no-balance-margin --slug-suffix naive_openloop --record
.venv/bin/python run_task.py --task deep_bow --provider blind_symmetry --coordinate-mode raw_undocumented --slug-suffix blind_symmetry --record

# 4. Run unit & live integration tests (live Gemini tests run automatically if GEMINI_API_KEY is in .env)
.venv/bin/python -m unittest discover -s tests -v
```

## Running with Live Generalist LLMs

`src/llm_client.py` auto-loads a local gitignored `.env` (or environment variables) and supports `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`, or any local OpenAI-compatible server (`LLM_BASE_URL`, e.g. vLLM / Ollama):

```bash
# Google Gemini (generalist models)
GEMINI_API_KEY="..." .venv/bin/python run_task.py --task peck_target --provider gemini --model gemini-pro-latest --record
GEMINI_API_KEY="..." .venv/bin/python run_task.py --task look_and_nod --provider gemini --model gemini-3.5-flash --record

# OpenAI
OPENAI_API_KEY="..." .venv/bin/python run_task.py --task peck_target --provider openai --model gpt-4.1 --record

# Anthropic Claude
ANTHROPIC_API_KEY="..." .venv/bin/python run_task.py --task deep_bow --provider anthropic --model claude-sonnet-4-20250514 --record

# Interactive 3D MuJoCo viewer on macOS (requires mjpython)
.venv/bin/mjpython run_task.py --task peck_target --provider closed_loop --viewer
```

## Project Layout

- [`src/harness.py`](src/harness.py) — Core ~140-line stop-sequence interception & continuation loop with `<harness_error>` schema self-repair.
- [`src/env.py`](src/env.py) — 50 Hz position-controlled `MicroDuckEnv`, XL330 stiffness calibration (`kp=3.2 N·m/rad`, `±0.96 N·m`), mocap target berry & CoM projection disc, dual-camera renderer.
- [`src/telemetry.py`](src/telemetry.py) — Whole-body `subtree_com`, 4 mm sole-mesh support polygon, toe/heel tipping margins, and `<observation>` XML serializer.
- [`src/prompts.py`](src/prompts.py) — Pre-printed Harness System Contract (`raw_documented`, `raw_undocumented`, `normalized_symmetric`).
- [`src/llm_client.py`](src/llm_client.py) — Pure-`httpx` stop-sequence client with stripped stop-tag reattachment and Gemini thinking-mode support.
- [`src/tasks.py`](src/tasks.py) — 5 progressive benchmark tasks (`look_and_nod`, `deep_bow`, `sit_and_stand`, `peck_target`, `waddle_forward`) + evaluators.
- [`src/trace_export.py`](src/trace_export.py) — Exports H.264 `.mp4`, `.json` episode traces, and self-contained interactive HTML `<video>` + synced token-stream widgets.
- [`tests/`](tests/) — Offline harness/physics unit tests (`test_harness.py`) and live generalist Gemini action-sequence tests (`test_gemini_live.py`).
- [`assets/microduck/`](assets/microduck/) — Upstream MJCF (`robot_allcollisions.xml`), STL meshes, and Apache-2.0 `LICENSE` from [`pollen-robotics/microduck_rl`](https://github.com/pollen-robotics/microduck_rl).
