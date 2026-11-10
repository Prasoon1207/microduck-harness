"""Progressive zero-shot task suite and programmatic MuJoCo evaluators for MicroDuck."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.telemetry import TelemetrySnapshot


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """Specification for a single zero-shot robotics harness benchmark task."""

    task_id: str
    title: str
    instruction: str
    init_keyframe: str = "STAND"
    target_xyz: Optional[Tuple[float, float, float]] = None
    allow_sitting_contacts: bool = False
    abort_on_fall: bool = True
    max_chunks: int = 8
    evaluator: Callable[[List[TelemetrySnapshot]], Tuple[bool, str, Dict[str, Any]]] = lambda _: (False, "", {})


def _eval_look_and_nod(history: List[TelemetrySnapshot]) -> Tuple[bool, str, Dict[str, Any]]:
    fell = any(s.fallen for s in history)
    yaws = [s.joints_rad["head_yaw"] for s in history]
    # Note: head_pitch has +Y world axis (downward > 0) while neck_pitch has -Y axis (downward < 0)
    nods = [s.joints_rad["head_pitch"] - s.joints_rad["neck_pitch"] for s in history]
    max_yaw, min_yaw = max(yaws), min(yaws)
    nod_span = max(nods) - min(nods)
    final_upright = (not fell) and history[-1].trunk_pos_m[2] >= 0.105

    looked_both = max_yaw >= 0.50 and min_yaw <= -0.50
    nodded = nod_span >= 0.55
    ok = (not fell) and looked_both and nodded and final_upright
    metrics = {
        "fell": fell,
        "max_head_yaw_rad": round(max_yaw, 3),
        "min_head_yaw_rad": round(min_yaw, 3),
        "nod_pitch_span_rad": round(nod_span, 3),
        "final_trunk_z_m": round(history[-1].trunk_pos_m[2], 3),
    }
    summary = (
        f"fell={fell}, yaw_range=[{min_yaw:+.2f}, {max_yaw:+.2f}] rad, "
        f"nod_span={nod_span:.2f} rad, final_z={history[-1].trunk_pos_m[2]:.3f}m"
    )
    return ok, summary, metrics


def _eval_deep_bow(history: List[TelemetrySnapshot]) -> Tuple[bool, str, Dict[str, Any]]:
    fell = any(s.fallen for s in history)
    beak_zs = [s.mouth_tip_xyz_m[2] for s in history]
    min_beak_z = min(beak_zs)
    final_beak_z = history[-1].mouth_tip_xyz_m[2]
    final_trunk_z = history[-1].trunk_pos_m[2]

    bowed_low = min_beak_z <= 0.095
    recovered = final_beak_z >= 0.165 and final_trunk_z >= 0.108
    ok = (not fell) and bowed_low and recovered
    metrics = {
        "fell": fell,
        "min_beak_z_m": round(min_beak_z, 3),
        "final_beak_z_m": round(final_beak_z, 3),
        "final_trunk_z_m": round(final_trunk_z, 3),
    }
    summary = f"fell={fell}, min_beak_z={min_beak_z:.3f}m (goal <=0.095m), final_beak_z={final_beak_z:.3f}m"
    return ok, summary, metrics


def _eval_sit_and_stand(history: List[TelemetrySnapshot]) -> Tuple[bool, str, Dict[str, Any]]:
    fell = any(s.fallen for s in history)
    trunk_zs = [s.trunk_pos_m[2] for s in history]
    min_z = min(trunk_zs)
    min_idx = trunk_zs.index(min_z)
    max_z_after_sit = max(trunk_zs[min_idx:])
    final_z = trunk_zs[-1]

    sat_low = min_z <= 0.088
    stood_back_up = final_z >= 0.110 and max_z_after_sit >= 0.110
    ok = (not fell) and sat_low and stood_back_up
    metrics = {
        "fell": fell,
        "min_trunk_z_m": round(min_z, 3),
        "final_trunk_z_m": round(final_z, 3),
    }
    summary = f"fell={fell}, min_trunk_z={min_z:.3f}m (goal <=0.088m), final_trunk_z={final_z:.3f}m (goal >=0.110m)"
    return ok, summary, metrics


def _eval_peck_target(history: List[TelemetrySnapshot]) -> Tuple[bool, str, Dict[str, Any]]:
    fell = any(s.fallen for s in history)
    min_dist = history[-1].min_beak_to_target_dist_m
    if min_dist is None:
        min_dist = 999.0
    final_z = history[-1].trunk_pos_m[2]

    touched = min_dist <= 0.020
    recovered = final_z >= 0.110
    ok = (not fell) and touched and recovered
    metrics = {
        "fell": fell,
        "min_beak_to_target_dist_m": round(min_dist, 4),
        "final_trunk_z_m": round(final_z, 3),
    }
    summary = (
        f"fell={fell}, min_beak_dist={min_dist * 100:.1f}cm (goal <=2.0cm), "
        f"final_trunk_z={final_z:.3f}m"
    )
    return ok, summary, metrics


def _eval_waddle_forward(history: List[TelemetrySnapshot]) -> Tuple[bool, str, Dict[str, Any]]:
    fell = any(s.fallen for s in history)
    max_x = max(s.trunk_pos_m[0] for s in history if not s.fallen) if not history[0].fallen else 0.0
    final_x = history[-1].trunk_pos_m[0]
    ok = (not fell) and final_x >= 0.150
    metrics = {
        "fell": fell,
        "max_upright_x_m": round(max_x, 3),
        "final_trunk_x_m": round(final_x, 3),
    }
    summary = f"fell={fell}, max_upright_x={max_x * 100:.1f}cm, final_x={final_x * 100:.1f}cm (goal >=15.0cm)"
    return ok, summary, metrics


TASKS: Dict[str, TaskSpec] = {
    "look_and_nod": TaskSpec(
        task_id="look_and_nod",
        title="Expressive Look & Nod",
        instruction=(
            "Starting from your standing posture, turn your head at least 35 degrees (>0.55 rad) to the left, "
            "then at least 35 degrees (<-0.55 rad) to the right, bring head_yaw back near 0, and perform a clear "
            "down-and-up nod while keeping both feet grounded and staying balanced."
        ),
        max_chunks=6,
        evaluator=_eval_look_and_nod,
    ),
    "deep_bow": TaskSpec(
        task_id="deep_bow",
        title="Counter-Balanced Deep Bow",
        instruction=(
            "Perform a deep, respectful bow: lower the tip of your beak (`mouth_tip_xyz_m` z) below 0.095 m "
            "while keeping BOTH feet grounded and without falling forward, then return cleanly to an upright "
            "standing pose (`mouth_tip` z >= 0.170 m)."
        ),
        max_chunks=7,
        evaluator=_eval_deep_bow,
    ),
    "sit_and_stand": TaskSpec(
        task_id="sit_and_stand",
        title="Controlled Squat / Sit & Stand",
        instruction=(
            "Smoothly lower your body into a deep crouch/sitting posture (`trunk_pos_m` z <= 0.088 m) without "
            "toppling over, pause briefly, and stand back up to your upright standing height (`trunk_pos_m` z >= 0.110 m)."
        ),
        allow_sitting_contacts=True,
        max_chunks=7,
        evaluator=_eval_sit_and_stand,
    ),
    "peck_target": TaskSpec(
        task_id="peck_target",
        title="Closed-Loop 3D Berry Peck",
        instruction=(
            "A red berry sits on a low stem in front of you at `target_xyz_m` ([0.046, 0.018, 0.074] m — 14.4 cm "
            "below your standing beak height). Using closed-loop feedback on your beak coordinates and CoM balance "
            "margin, crouch and counter-balance your pelvis to touch the berry with the tip of your beak "
            "(`beak_to_target_dist_m <= 0.020 m`) without falling over, then return to an upright stand "
            "(`trunk_pos_m` z >= 0.110 m)."
        ),
        target_xyz=(0.046, 0.018, 0.074),
        max_chunks=8,
        evaluator=_eval_peck_target,
    ),
    "waddle_forward": TaskSpec(
        task_id="waddle_forward",
        title="Zero-Shot Bipedal Locomotion Stress Test",
        instruction=(
            "Walk or waddle forward along the +X axis so that your trunk (`trunk_pos_m` x) advances at least "
            "+0.150 m forward while staying upright on your feet."
        ),
        max_chunks=10,
        evaluator=_eval_waddle_forward,
    ),
}
