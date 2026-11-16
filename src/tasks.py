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


TASKS: Dict[str, TaskSpec] = {
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
}
