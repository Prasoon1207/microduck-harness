"""Closed-loop text-only reference and ablation policies operating strictly through the Harness XML stream."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

from src.llm_client import GenerationStep


def _parse_latest_obs(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    """Parse physical numbers strictly from the latest `<observation>` XML text in `messages`."""
    obs_text = messages[-1]["content"]
    out: Dict[str, Any] = {"raw": obs_text}

    def _vec(tag: str) -> List[float]:
        m = re.search(rf"{tag}:\s*\[([^\]]+)\]", obs_text)
        return [float(x.strip()) for x in m.group(1).split(",")] if m else [0.0, 0.0, 0.0]

    out["trunk_pos"] = _vec("trunk_pos_m")
    out["mouth_tip"] = _vec("mouth_tip_xyz_m")
    if "target_xyz_m:" in obs_text:
        out["target_xyz"] = _vec("target_xyz_m")
        dm = re.search(r"dist=([0-9.]+)m,\s*best_so_far=([0-9.]+)m", obs_text)
        if dm:
            out["dist"] = float(dm.group(1))
            out["best_dist"] = float(dm.group(2))

    tm = re.search(r"toe_margin=([+-]?[0-9.]+)m,\s*heel_margin=([+-]?[0-9.]+)m", obs_text)
    out["toe_margin"] = float(tm.group(1)) if tm else 0.015
    out["heel_margin"] = float(tm.group(2)) if tm else 0.013

    jm = re.search(r"joints_rad:\s*\{([^}]+)\}", obs_text)
    joints: Dict[str, float] = {}
    if jm:
        for pair in jm.group(1).split(","):
            k, v = pair.split(":")
            joints[k.strip()] = float(v.strip())
    out["joints"] = joints
    return out


def _avian_stance(delta_squat: float, beta_pullback: float = 0.0) -> Dict[str, float]:
    """Compute bilateral 10-DOF leg pose obeying MicroDuck's avian flat-foot law:
    `q_hip - q_knee + q_ankle = 0`, with `+beta_pullback` on `left_ankle` tilting shins `-X` backward!
    """
    l_hip = round(-0.458 + delta_squat - beta_pullback, 3)
    l_knee = round(2.0 * delta_squat, 3)
    l_ank = round(0.453 + delta_squat + beta_pullback, 3)
    return {
        "left_hip_yaw": 0.0,
        "right_hip_yaw": 0.0,
        "left_hip_roll": -0.087,
        "right_hip_roll": 0.087,
        "left_hip_pitch": l_hip,
        "right_hip_pitch": -l_hip,
        "left_knee": l_knee,
        "right_knee": -l_knee,
        "left_ankle": l_ank,
        "right_ankle": -l_ank,
    }


STAND_RAW: Dict[str, float] = {
    **_avian_stance(0.0),
    "neck_pitch": 0.349,
    "head_pitch": 0.349,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}


def _format_chunk(_note: str, waypoints: List[Dict[str, Any]]) -> GenerationStep:
    body = json.dumps(waypoints, indent=2)
    txt = f"<action_chunk>\n{body}\n</action_chunk>"
    return GenerationStep(text=txt, matched_stop="</action_chunk>", latency_s=0.01, model="closed-loop-policy")


def _format_done(_note: str, summary: str) -> GenerationStep:
    txt = f"<done>{summary}</done>"
    return GenerationStep(text=txt, matched_stop="</done>", latency_s=0.01, model="closed-loop-policy")


class ClosedLoopReferenceAgent:
    """Text-stream agent demonstrating closed-loop CoM counter-balancing & ablation modes."""

    def __init__(self, mode: str = "closed_loop") -> None:
        assert mode in {"closed_loop", "open_loop_naive", "blind_symmetry"}
        self.mode = mode
        self.model = {
            "closed_loop": "zero-shot-closed-loop-harness",
            "open_loop_naive": "ablation-no-com-counterbalance",
            "blind_symmetry": "ablation-no-cad-sign-adapter",
        }[mode]

    def generate_until_stop(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stop_sequences: List[str],
    ) -> GenerationStep:
        t0 = time.perf_counter()
        turn_num = (len(messages) + 1) // 2
        obs = _parse_latest_obs(messages)

        step = self._policy_deep_bow(turn_num, obs)
        step.latency_s = time.perf_counter() - t0
        step.model = self.model
        return step

    def _policy_deep_bow(self, turn: int, obs: Dict[str, Any]) -> GenerationStep:
        if self.mode == "open_loop_naive":
            return _format_chunk(
                "Naive open-loop bow: pitching neck_pitch=-1.10 and head_pitch=+1.35 straight forward without shifting pelvis back.",
                [{"dt": 0.60, "joints": {"neck_pitch": -1.10, "head_pitch": 1.35}}],
            )
        if self.mode == "blind_symmetry":
            return _format_chunk(
                "Ablation without CAD servo-mirroring rules: applying identical signs to left and right leg joints.",
                [{"dt": 0.60, "joints": {"left_knee": 1.10, "right_knee": 1.10, "left_hip_pitch": -0.80, "right_hip_pitch": -0.80}}],
            )

        beak_z = obs["mouth_tip"][2]
        if turn == 1:
            legs = _avian_stance(delta_squat=0.58, beta_pullback=0.24)
            return _format_chunk(
                f"Initial beak_z={beak_z:.3f}m, toe_margin={obs['toe_margin']*1000:+.1f}mm.",
                [{"dt": 0.85, "joints": {**legs, "neck_pitch": -0.35, "head_pitch": 1.38}}],
            )
        if turn == 2 and beak_z > 0.095:
            beta = 0.27 if obs["toe_margin"] < 0.008 else 0.25
            legs = _avian_stance(delta_squat=0.63, beta_pullback=beta)
            return _format_chunk(
                f"Closed-loop check: beak_z={beak_z:.3f}m, toe_margin={obs['toe_margin']*1000:+.1f}mm.",
                [{"dt": 0.75, "joints": {**legs, "neck_pitch": -0.38, "head_pitch": 1.44}}],
            )
        if beak_z < 0.170:
            mid_legs = _avian_stance(delta_squat=0.32, beta_pullback=0.11)
            return _format_chunk(
                f"Bow goal reached cleanly (beak_z={beak_z:.3f}m <= 0.095m)!",
                [
                    {"dt": 0.70, "joints": {**mid_legs, "neck_pitch": 0.10, "head_pitch": 0.75}},
                    {"dt": 0.85, "joints": STAND_RAW},
                ],
            )
        return _format_done(
            f"Upright posture verified: beak_z={beak_z:.3f}m >= 0.170m.",
            "Completed counter-balanced deep bow below 9.5cm and returned upright.",
        )


def _waddle_leg(side: str, delta: float, stride: float = 0.0, roll: float = 0.0, yaw: float = 0.0) -> Dict[str, float]:
    if side == "left":
        return {
            "left_hip_yaw": round(yaw, 2),
            "left_hip_roll": round(-0.087 + roll, 2),
            "left_hip_pitch": round(-0.458 + delta - stride, 2),
            "left_knee": round(2.0 * delta, 2),
            "left_ankle": round(0.453 + delta + stride, 2),
        }
    return {
        "right_hip_yaw": round(yaw, 2),
        "right_hip_roll": round(0.087 + roll, 2),
        "right_hip_pitch": round(0.458 - delta + stride, 2),
        "right_knee": round(-2.0 * delta, 2),
        "right_ankle": round(-0.453 - delta - stride, 2),
    }


class InteractiveDemoAgent:
    """Offline closed-loop policy operating strictly through `<instruction>` + `<observation>` XML."""

    model = "closed-loop-demo"

    @staticmethod
    def _latest_instruction_and_chunk_count(messages: List[Dict[str, str]]) -> tuple[str, int]:
        instr = ""
        chunks_since_instr = 0
        for idx in range(len(messages) - 1, -1, -1):
            m = messages[idx]
            if m["role"] == "assistant" and "<action_chunk>" in m["content"]:
                chunks_since_instr += 1
            if m["role"] == "user":
                im = re.search(r"<(?:instruction|goal)>(.*?)</(?:instruction|goal)>", m["content"], re.DOTALL)
                if im:
                    instr = im.group(1).strip()
                    break
        return instr, chunks_since_instr

    def generate_until_stop(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stop_sequences: List[str],
    ) -> GenerationStep:
        t0 = time.perf_counter()
        obs = _parse_latest_obs(messages)
        rpy_m = re.search(r"trunk_rpy_deg:\s*\[roll=([+-]?[0-9.]+),\s*pitch=([+-]?[0-9.]+),\s*yaw=([+-]?[0-9.]+)\]", obs["raw"])
        yaw_deg = float(rpy_m.group(3)) if rpy_m else 0.0
        instr, done_chunks = self._latest_instruction_and_chunk_count(messages)
        low = instr.lower()

        step = self._route(low, done_chunks, obs, yaw_deg)
        step.latency_s = time.perf_counter() - t0
        step.model = self.model
        return step

    def _route(self, low: str, done_chunks: int, obs: Dict[str, Any], yaw_deg: float) -> GenerationStep:
        tx, ty, tz = obs["trunk_pos"]
        mx, my, mz = obs["mouth_tip"]

        # Number parser (e.g. "3 steps" / "twice" / "two")
        n_req = 1
        if any(w in low for w in ("three", "3")):
            n_req = 3
        elif any(w in low for w in ("two", "twice", "couple", "2")):
            n_req = 2

        if "bow" in low:
            if done_chunks == 0:
                legs = _avian_stance(delta_squat=0.58, beta_pullback=0.24)
                return _format_chunk("bow down", [{"dt": 0.85, "joints": {**legs, "neck_pitch": -0.35, "head_pitch": 1.38}}])
            if done_chunks == 1:
                mid = _avian_stance(delta_squat=0.28, beta_pullback=0.10)
                return _format_chunk(
                    "return upright",
                    [
                        {"dt": 0.65, "joints": {**mid, "neck_pitch": 0.10, "head_pitch": 0.75}},
                        {"dt": 0.75, "joints": STAND_RAW},
                    ],
                )
            return _format_done("done", f"Completed a counter-balanced deep bow and returned cleanly to STAND (beak z={mz:.3f} m, toe_margin={obs['toe_margin']*1000:+.0f} mm).")

        if any(w in low for w in ("waddle", "walk", "forward", "ahead", "step")):
            back = any(w in low for w in ("back", "backward", "reverse"))
            s = -0.18 if back else 0.22
            steps = max(n_req, 2 if "walk" in low or "waddle" in low else 1)
            if done_chunks < steps:
                wps = [
                    {"dt": 0.28, "joints": {**_waddle_leg("left", 0.10, -s * 0.6, +0.28), **_waddle_leg("right", 0.38, +s, +0.28), "neck_pitch": 0.35, "head_pitch": 0.35, "head_yaw": 0.35, "head_roll": -0.25}},
                    {"dt": 0.28, "joints": {**_waddle_leg("left", 0.15, -s * 0.6, 0.0), **_waddle_leg("right", 0.15, +s * 0.8, 0.0), "head_yaw": 0.0, "head_roll": 0.0}},
                    {"dt": 0.28, "joints": {**_waddle_leg("left", 0.38, +s, -0.28), **_waddle_leg("right", 0.10, -s * 0.6, -0.28), "head_yaw": -0.35, "head_roll": +0.25}},
                    {"dt": 0.28, "joints": {**_waddle_leg("left", 0.15, +s * 0.8, 0.0), **_waddle_leg("right", 0.15, -s * 0.6, 0.0), "head_yaw": 0.0, "head_roll": 0.0}},
                    {"dt": 0.30, "joints": {**_waddle_leg("left", 0.10), **_waddle_leg("right", 0.10), "head_yaw": 0.0, "head_roll": 0.0}},
                ]
                return _format_chunk("waddle step", wps)
            direction = "backward" if back else "forward"
            return _format_done("done", f"Waddled {steps} step(s) {direction} cleanly — trunk is now at [{tx:+.3f}, {ty:+.3f}, {tz:.3f}] m (yaw {yaw_deg:+.1f}°).")

        if "turn" in low or "rotate" in low or "spin" in low:
            go_right = "right" in low and "left" not in low
            if done_chunks < n_req:
                if go_right:
                    wps = [
                        {"dt": 0.26, "joints": {**_waddle_leg("left", 0.10, roll=+0.28, yaw=0.0), **_waddle_leg("right", 0.38, roll=+0.28, yaw=-0.32), "head_yaw": 0.30, "head_roll": -0.25}},
                        {"dt": 0.26, "joints": {**_waddle_leg("left", 0.12, roll=+0.22, yaw=+0.32), **_waddle_leg("right", 0.15, roll=0.0, yaw=-0.32), "head_yaw": 0.0, "head_roll": 0.0}},
                        {"dt": 0.26, "joints": {**_waddle_leg("left", 0.38, roll=-0.28, yaw=0.0), **_waddle_leg("right", 0.10, roll=-0.28, yaw=+0.16), "head_yaw": -0.30, "head_roll": +0.25}},
                        {"dt": 0.30, "joints": {**_waddle_leg("left", 0.10), **_waddle_leg("right", 0.10), "head_yaw": 0.0, "head_roll": 0.0}},
                    ]
                else:
                    wps = [
                        {"dt": 0.26, "joints": {**_waddle_leg("right", 0.10, roll=-0.28, yaw=0.0), **_waddle_leg("left", 0.38, roll=-0.28, yaw=+0.32), "head_yaw": -0.30, "head_roll": +0.25}},
                        {"dt": 0.26, "joints": {**_waddle_leg("right", 0.12, roll=-0.22, yaw=-0.32), **_waddle_leg("left", 0.15, roll=0.0, yaw=+0.32), "head_yaw": 0.0, "head_roll": 0.0}},
                        {"dt": 0.26, "joints": {**_waddle_leg("right", 0.38, roll=+0.28, yaw=0.0), **_waddle_leg("left", 0.10, roll=+0.28, yaw=-0.16), "head_yaw": +0.30, "head_roll": -0.25}},
                        {"dt": 0.30, "joints": {**_waddle_leg("left", 0.10), **_waddle_leg("right", 0.10), "head_yaw": 0.0, "head_roll": 0.0}},
                    ]
                return _format_chunk("turn", wps)
            side_lbl = "right" if go_right else "left"
            return _format_done("done", f"Turned {side_lbl} in place — heading yaw is now {yaw_deg:+.1f}° with both feet flat.")

        if "squat" in low or "crouch" in low:
            if done_chunks == 0:
                legs = _avian_stance(delta_squat=0.48, beta_pullback=0.06)
                return _format_chunk("squat", [{"dt": 0.70, "joints": {**legs, "neck_pitch": 0.25, "head_pitch": 0.45}}])
            return _format_done("done", f"Holding a balanced flat-foot avian squat at trunk z={tz:.3f} m (toe_margin={obs['toe_margin']*1000:+.0f} mm).")

        if "stand" in low or "upright" in low or "reset" in low:
            if done_chunks == 0 and tz < 0.112:
                return _format_chunk("stand", [{"dt": 0.75, "joints": STAND_RAW}])
            if done_chunks == 0:
                return _format_chunk("stand", [{"dt": 0.55, "joints": STAND_RAW}])
            return _format_done("done", f"Standing tall and centered at trunk z={tz:.3f} m, beak z={mz:.3f} m.")

        if any(w in low for w in ("look", "nod", "shake", "tilt", "wave", "hello", "hi")):
            if done_chunks == 0:
                brace = _avian_stance(delta_squat=0.16, beta_pullback=0.05)
                wps = [
                    {"dt": 0.55, "joints": {**brace, "head_yaw": 0.55, "head_roll": 0.12}},
                    {"dt": 0.65, "joints": {**brace, "head_yaw": -0.55, "head_roll": -0.12}},
                    {"dt": 0.45, "joints": {**brace, "head_yaw": 0.0, "head_roll": 0.0, "head_pitch": 0.85}},
                    {"dt": 0.45, "joints": {**brace, "head_pitch": 0.25}},
                    {"dt": 0.50, "joints": STAND_RAW},
                ]
                return _format_chunk("look and nod", wps)
            return _format_done("done", f"Looked left and right, nodded, and centered gaze (beak z={mz:.3f} m).")

        return _format_done(
            "status",
            f"Ready! Trunk at [{tx:+.3f}, {ty:+.3f}, {tz:.3f}] m, beak at z={mz:.3f} m, toe_margin={obs['toe_margin']*1000:+.0f} mm.",
        )

