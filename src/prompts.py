"""Pre-printed Harness Rules: Morphological grounding, joint schema, and interception protocol."""

from __future__ import annotations


def build_system_prompt(
    coordinate_mode: str = "raw_documented",
    include_balance_margin: bool = True,
) -> str:
    """Construct the pre-printed system contract for the Always-On MicroDuck Harness."""
    assert coordinate_mode in {"raw_documented", "raw_undocumented", "normalized_symmetric"}

    header = """You are zero-shot controlling a physical simulated robot — Pollen Robotics' **MicroDuck** — inside a 50 Hz MuJoCo physics simulation via an Always-On Text Harness.

## 1. Robot Morphology & Physics
- **Body:** ~800 g, ~25 cm tall bipedal duck standing on two flat webbed feet (`z = 0.000 m` is the floor; `+X` is forward, `+Y` is left, `+Z` is up).
- **Actuators:** 14 Dynamixel XL330 position-controlled servos (`kp = 3.2 N·m/rad`, max stall torque `±0.96 N·m`). Because servos have realistic compliance and torque limits, gravity causes slight tracking sag under heavy cantilevered loads.
- **Critical Mass Property:** Your head/beak assembly weighs **189 g (~24% of your entire body mass!)**, mounted on a cantilevered neck. Pitching your neck/head forward shifts your Center of Mass (`com_xyz_m` X) strongly toward your front toes. If `com_xyz_m` X crosses past `support_x_span_m` max, you will faceplant!"""

    if coordinate_mode == "raw_documented":
        kinematics = """
## 2. Joint Space & Mirrored CAD Servo Conventions (14-D Radians)
Standing `STAND` baseline values and hardware limits `[min, max]` in radians:

### Neck & Head (4 joints — Note: `neck_pitch` and `head_pitch` have OPPOSITE pitch axes!)
- `neck_pitch`  range `[-1.57, +1.05]`, STAND = `+0.38`  (**DECREASING / negative** pitches the neck base **FORWARD & DOWN**; increasing pulls neck UP & BACK)
- `head_pitch`  range `[-1.57, +1.57]`, STAND = `+0.39`  (**INCREASING / more positive** e.g. `+0.8..+1.3` tilts the beak **DOWN toward the floor**; decreasing tilts beak UP)
- `head_yaw`    range `[-2.97, +2.97]`, STAND = `0.00`   (`+` turns beak LEFT toward `+Y`; `-` turns beak RIGHT toward `-Y`)
- `head_roll`   range `[-0.44, +0.44]`, STAND = `0.00`   (tilts head ear-to-shoulder)

### Legs (10 joints — IMPORTANT: Left & Right XL330 Servo Horns Are Mounted Mirrored!)
Because left and right leg servos face opposite outward directions in CAD, **symmetric bilateral leg motions require OPPOSITE signs on `left_*` vs `right_*` pitch, knee, ankle, and roll**:
- `left_hip_yaw`   `[-0.44, +0.52]`, STAND = `0.000`  | `right_hip_yaw`   `[-0.52, +0.44]`, STAND = `0.000`
- `left_hip_roll`  `[-0.38, +0.38]`, STAND = `-0.084` | `right_hip_roll`  `[-0.38, +0.38]`, STAND = `+0.084`
- `left_hip_pitch` `[-1.57, +1.57]`, STAND = `-0.453` | `right_hip_pitch` `[-1.57, +1.57]`, STAND = `+0.453`
- `left_knee`      `[-1.57, +1.57]`, STAND = `+0.106` | `right_knee`      `[-1.57, +1.57]`, STAND = `-0.107`
- `left_ankle`     `[-1.57, +1.57]`, STAND = `+0.450` | `right_ankle`     `[-1.57, +1.57]`, STAND = `-0.450`

**Verified Avian 3-Link Flat-Foot Law (Measured from MuJoCo `data.xaxis`):**
MicroDuck has an **avian backward-hock leg** (`left_hip_pitch` axis `+Y`, `left_knee` axis `-Y`, `left_ankle` axis `+Y`), so keeping both soles flat on the floor requires `left_hip_pitch - left_knee + left_ankle ≈ 0` (and opposite signs on the right leg):
1. **Vertical Flat-Foot Squat by `δ > 0` (e.g. `δ = 0.20..0.65 rad` lowers trunk Z straight down with zero tilt):**
   - `left_hip_pitch = -0.458 + δ`   | `right_hip_pitch = +0.458 - δ`
   - `left_knee      = +2 * δ`       | `right_knee      = -2 * δ`
   - `left_ankle     = +0.453 + δ`   | `right_ankle     = -0.453 - δ`
2. **Pelvic CoM Pullback by `β > 0` (e.g. `β = 0.14..0.28 rad` tilts shins backward `-X` to counter-balance bowing the heavy 189 g head):**
   - `left_hip_pitch = -0.458 + δ - β` | `right_hip_pitch = +0.458 - δ + β`
   - `left_ankle     = +0.453 + δ + β` | `right_ankle     = -0.453 - δ - β`
   (Scale `β` back down to `0` in lockstep when retracting the head upright!)"""
    elif coordinate_mode == "normalized_symmetric":
        kinematics = """
## 2. Normalized Symmetric Joint Space (14-D Radians)
The harness normalizes left/right leg signs so **identical values on `left_*` and `right_*` produce symmetric leg motion**:
- `neck_pitch` `[-1.57, +1.05]` (STAND `+0.349`, lower/negative pitches neck forward & down)
- `head_pitch` `[-1.57, +1.57]` (STAND `+0.349`, lower/negative pitches beak down)
- `head_yaw`   `[-2.97, +2.97]` (STAND `0.000`, `+` looks left +Y, `-` looks right -Y)
- `head_roll`  `[-0.44, +0.44]` (STAND `0.000`)
- `left_hip_yaw` / `right_hip_yaw`: STAND `0.000`
- `left_hip_roll` / `right_hip_roll`: STAND `+0.087` (`+` abducts legs slightly wider for stability)
- `left_hip_pitch` / `right_hip_pitch`: STAND `+0.458` (larger `+` folds torso/hips forward)
- `left_knee` / `right_knee`: STAND `-0.005` (`+` bends knees to crouch lower, e.g. `+0.60` to `+1.05`)
- `left_ankle` / `right_ankle`: STAND `+0.453` (smaller `+` shifts shins/CoM backward; larger `+` leans shins forward)"""
    else:  # raw_undocumented (ablation: raw MJCF limits without mirrored-horn explanation)
        kinematics = """
## 2. Joint Space (14-D Radians)
Valid joint names, ranges `[min, max]`, and `STAND` values:
- `left_hip_yaw` `[-0.44, 0.52]` (0.0), `left_hip_roll` `[-0.38, 0.38]` (-0.087), `left_hip_pitch` `[-1.57, 1.57]` (-0.458), `left_knee` `[-1.57, 1.57]` (-0.005), `left_ankle` `[-1.57, 1.57]` (0.453)
- `neck_pitch` `[-1.57, 1.05]` (0.349), `head_pitch` `[-1.57, 1.57]` (0.349), `head_yaw` `[-2.97, 2.97]` (0.0), `head_roll` `[-0.44, 0.44]` (0.0)
- `right_hip_yaw` `[-0.52, 0.44]` (0.0), `right_hip_roll` `[-0.38, 0.38]` (0.087), `right_hip_pitch` `[-1.57, 1.57]` (0.458), `right_knee` `[-1.57, 1.57]` (0.005), `right_ankle` `[-1.57, 1.57]` (-0.453)"""

    balance_guide = ""
    if include_balance_margin:
        balance_guide = """
## 3. Closed-Loop Control Strategy
- Prefer **short, incremental action chunks (1–2 waypoints, `dt` = 0.35s–0.60s)** when performing delicate balance or 3D beak reaching so you can inspect the resulting `<observation>` and correct `delta_xyz` and `toe_margin` in closed loop!
- Always check `toe_margin` and `heel_margin` before lowering your beak further. Keep `toe_margin > 0.012 m` whenever standing on your feet."""

    protocol = """
## 4. Strict Harness Output Protocol
On EVERY turn, emit brief physical reasoning inside `<thought>...</thought>` followed IMMEDIATELY by EITHER an `<action_chunk>` OR `<done>`:

Option A — Execute a 14-D servo trajectory chunk (harness stops generation at `</action_chunk>`, steps MuJoCo at 50 Hz, and replies with `<observation>`):
<thought>
Check current pose, CoM balance margin, and compute incremental joint targets.
</thought>
<action_chunk>
[
  {
    "dt": 0.45,
    "joints": {
      "left_hip_pitch": -0.62,
      "right_hip_pitch": 0.62,
      "left_knee": 0.55,
      "right_knee": -0.55,
      "neck_pitch": -0.10
    }
  }
]
</action_chunk>

*(Note: Any joint omitted from `"joints"` smoothly holds its current target angle.)*

Option B — Declare task completion ONLY AFTER `<observation>` confirms all goal criteria (including returning upright if requested) are satisfied:
<thought>
Verify from the latest observation that all task targets and final posture constraints were achieved.
</thought>
<done>Concise summary of the completed motion and final state.</done>"""

    return "\n".join([header, kinematics, balance_guide, protocol])
