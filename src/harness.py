"""Core Always-On Robotics Harness (~140 lines): Stop-Sequence Interception -> MuJoCo -> Continuation."""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from src.env import MicroDuckEnv, Waypoint
from src.llm_client import GenerationStep, ModelClient
from src.telemetry import JOINT_NAMES, TelemetrySnapshot

STOP_SEQUENCES: List[str] = ["</action_chunk>", "</done>"]

THOUGHT_RE = re.compile(r"<thought>(.*?)</thought>", re.DOTALL)
ACTION_RE = re.compile(r"<action_chunk>(.*?)</action_chunk>", re.DOTALL)
DONE_RE = re.compile(r"<done>(.*?)</done>|<done\s*/>", re.DOTALL)


class PolicyGenerator(Protocol):
    """Any model or scripted policy exposing stop-sequence text generation."""

    def generate_until_stop(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stop_sequences: List[str],
    ) -> GenerationStep: ...


@dataclasses.dataclass
class HarnessTurn:
    """Single intercepted turn in the LLM <-> MuJoCo control loop."""

    turn_idx: int
    raw_llm_text: str
    thought: str
    waypoints: List[Waypoint]
    observation_xml: str
    telemetry: TelemetrySnapshot
    frame_range: Tuple[int, int]
    latency_s: float
    parse_error: Optional[str] = None
    is_done: bool = False
    done_summary: str = ""


@dataclasses.dataclass
class EpisodeResult:
    """Full trajectory record of a harnessed rollout."""

    task_id: str
    instruction: str
    model_name: str
    coordinate_mode: str
    include_balance_margin: bool
    success: bool
    eval_summary: str
    metrics: Dict[str, Any]
    initial_telemetry: TelemetrySnapshot
    turns: List[HarnessTurn]
    system_prompt: str


def parse_action_chunk(raw_block: str) -> List[Waypoint]:
    """Parse and validate a JSON list of 14-D servo waypoints from `<action_chunk>`."""
    cleaned = raw_block.strip()
    # Strip optional markdown code fences if the LLM wrapped JSON inside the XML tag
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    payload = json.loads(cleaned)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("`<action_chunk>` must contain a non-empty JSON list of waypoint objects.")
    if len(payload) > 6:
        raise ValueError("Emit at most 6 waypoints per `<action_chunk>` so closed-loop feedback stays tight.")

    waypoints: List[Waypoint] = []
    valid_set = set(JOINT_NAMES)
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Waypoint {idx} must be a JSON object with 'dt' and 'joints'.")
        dt = float(item.get("dt", 0.4))
        joints_raw = item.get("joints", {})
        if not isinstance(joints_raw, dict) or not joints_raw:
            raise ValueError(f"Waypoint {idx} 'joints' must be a non-empty dict of `{joint_name: radians}`.")
        unknown = [k for k in joints_raw if k not in valid_set]
        if unknown:
            raise ValueError(f"Unknown joint name(s) {unknown}. Valid joints: {list(JOINT_NAMES)}")
        joints = {k: float(v) for k, v in joints_raw.items()}
        waypoints.append(Waypoint(dt=dt, joints=joints))
    return waypoints


def run_harness_episode(
    env: MicroDuckEnv,
    llm: PolicyGenerator,
    system_prompt: str,
    task_id: str,
    instruction: str,
    initial_snap: TelemetrySnapshot,
    evaluator: Callable[[List[TelemetrySnapshot]], Tuple[bool, str, Dict[str, Any]]],
    max_chunks: int = 8,
    include_balance_margin: bool = True,
    abort_on_fall: bool = True,
    viewer_handle: Optional[object] = None,
    verbose: bool = True,
) -> EpisodeResult:
    """Execute the Always-On Robotics Harness loop."""
    init_obs_xml = initial_snap.to_xml(include_balance_margin=include_balance_margin)
    messages: List[Dict[str, str]] = [
        {
            "role": "user",
            "content": f"<goal>\n{instruction}\n</goal>\n\n{init_obs_xml}",
        }
    ]

    turns: List[HarnessTurn] = []
    model_name = getattr(llm, "model", "custom-policy")

    for turn_idx in range(1, max_chunks + 1):
        gen: GenerationStep = llm.generate_until_stop(
            system_prompt=system_prompt,
            messages=messages,
            stop_sequences=STOP_SEQUENCES,
        )
        messages.append({"role": "assistant", "content": gen.text})

        thought_match = THOUGHT_RE.search(gen.text)
        thought = thought_match.group(1).strip() if thought_match else ""

        # Case 1: Model emitted `<done>`
        done_match = DONE_RE.search(gen.text)
        if done_match and not ACTION_RE.search(gen.text):
            done_summary = (done_match.group(1) or "Task completed.").strip()
            last_snap = env.telemetry_history[-1]
            last_frame = max(0, len(env.frames) - 1)
            turns.append(
                HarnessTurn(
                    turn_idx=turn_idx,
                    raw_llm_text=gen.text,
                    thought=thought,
                    waypoints=[],
                    observation_xml="",
                    telemetry=last_snap,
                    frame_range=(last_frame, last_frame),
                    latency_s=gen.latency_s,
                    is_done=True,
                    done_summary=done_summary,
                )
            )
            if verbose:
                print(f"[Turn {turn_idx}] <done> {done_summary}")
            break

        # Case 2: Intercept `<action_chunk>`
        action_match = ACTION_RE.search(gen.text)
        if not action_match:
            err_msg = (
                '<harness_error>No `<action_chunk>...</action_chunk>` or `<done>...</done>` tag detected. '
                "Always emit `<thought>...</thought>` followed immediately by `<action_chunk>[...]</action_chunk>` "
                "or `<done>summary</done>`.</harness_error>"
            )
            messages.append({"role": "user", "content": err_msg})
            last_snap = env.telemetry_history[-1]
            last_frame = max(0, len(env.frames) - 1)
            turns.append(
                HarnessTurn(
                    turn_idx=turn_idx,
                    raw_llm_text=gen.text,
                    thought=thought,
                    waypoints=[],
                    observation_xml=err_msg,
                    telemetry=last_snap,
                    frame_range=(last_frame, last_frame),
                    latency_s=gen.latency_s,
                    parse_error="Missing action_chunk tag",
                )
            )
            continue

        try:
            waypoints = parse_action_chunk(action_match.group(1))
        except Exception as exc:
            err_msg = f"<harness_error>Invalid action_chunk schema: {exc}</harness_error>"
            messages.append({"role": "user", "content": err_msg})
            last_snap = env.telemetry_history[-1]
            last_frame = max(0, len(env.frames) - 1)
            turns.append(
                HarnessTurn(
                    turn_idx=turn_idx,
                    raw_llm_text=gen.text,
                    thought=thought,
                    waypoints=[],
                    observation_xml=err_msg,
                    telemetry=last_snap,
                    frame_range=(last_frame, last_frame),
                    latency_s=gen.latency_s,
                    parse_error=str(exc),
                )
            )
            if verbose:
                print(f"[Turn {turn_idx}] Schema error intercepted: {exc}")
            continue

        # Step MuJoCo physics across the intercepted 14-D servo chunk
        snap = env.step_chunk(waypoints=waypoints, abort_on_fall=abort_on_fall, viewer_handle=viewer_handle)
        obs_xml = snap.to_xml(include_balance_margin=include_balance_margin)
        frame_range = env.chunk_frame_ranges[-1]

        turns.append(
            HarnessTurn(
                turn_idx=turn_idx,
                raw_llm_text=gen.text,
                thought=thought,
                waypoints=waypoints,
                observation_xml=obs_xml,
                telemetry=snap,
                frame_range=frame_range,
                latency_s=gen.latency_s,
            )
        )
        if verbose:
            print(
                f"[Turn {turn_idx}] {len(waypoints)} wp | t={snap.sim_time_s:.2f}s | "
                f"status={snap.status} | balance={snap.balance_hint} | beak_z={snap.mouth_tip_xyz_m[2]:.3f}m"
            )

        if abort_on_fall and snap.fallen:
            if verbose:
                print(f"[Turn {turn_idx}] Episode terminated early due to fall.")
            break

        messages.append({"role": "user", "content": obs_xml})

    success, eval_summary, metrics = evaluator(env.telemetry_history)
    return EpisodeResult(
        task_id=task_id,
        instruction=instruction,
        model_name=model_name,
        coordinate_mode=env.coordinate_mode,
        include_balance_margin=include_balance_margin,
        success=success,
        eval_summary=eval_summary,
        metrics=metrics,
        initial_telemetry=initial_snap,
        turns=turns,
        system_prompt=system_prompt,
    )
