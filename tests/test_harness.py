"""Unit tests for the Always-On Robotics Harness (stop-tag reattachment, schema repair, coordinate modes)."""

from __future__ import annotations

import unittest
from typing import Dict, List

from src.env import MicroDuckEnv, Waypoint
from src.harness import parse_action_chunk, run_harness_episode
from src.llm_client import GenerationStep, _reattach_stop_tag
from src.prompts import build_system_prompt
from src.tasks import TASKS


class _SelfRepairMockAgent:
    """Mock LLM that emits malformed output on Turn 1, repairs from `<harness_error>` on Turn 2, and finishes on Turn 3."""

    model = "mock-self-repair"

    def __init__(self) -> None:
        self.calls = 0
        self.seen_errors: List[str] = []

    def generate_until_stop(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stop_sequences: List[str],
    ) -> GenerationStep:
        self.calls += 1
        last_user = messages[-1]["content"]
        if "<harness_error>" in last_user:
            self.seen_errors.append(last_user)

        if self.calls == 1:
            # Invalid joint name `left_wing_flap` -> should trigger `<harness_error>` without crashing MuJoCo
            bad = '<thought>Try invalid joint.</thought>\n<action_chunk>[{"dt": 0.4, "joints": {"left_wing_flap": 0.5}}]</action_chunk>'
            return GenerationStep(text=bad, matched_stop="</action_chunk>", latency_s=0.01, model=self.model)
        if self.calls == 2:
            good = (
                "<thought>Fixed joint name from harness_error.</thought>\n"
                '<action_chunk>[{"dt": 0.4, "joints": {"head_yaw": 0.35}}]</action_chunk>'
            )
            return GenerationStep(text=good, matched_stop="</action_chunk>", latency_s=0.01, model=self.model)
        return GenerationStep(
            text="<thought>Yaw updated.</thought>\n<done>Self-repair verified.</done>",
            matched_stop="</done>",
            latency_s=0.01,
            model=self.model,
        )


class HarnessUnitTests(unittest.TestCase):
    def test_reattach_stop_tag_stripped_and_preserved(self) -> None:
        stops = ["</action_chunk>", "</done>"]

        # Provider stripped `</action_chunk>` and did not return explicit stop string (e.g. OpenAI)
        raw = '<thought>Scan left</thought>\n<action_chunk>[{"dt": 0.4, "joints": {"head_yaw": 0.5}}]'
        fixed, matched = _reattach_stop_tag(raw, stops, explicit_stop=None)
        self.assertEqual(matched, "</action_chunk>")
        self.assertTrue(fixed.endswith("</action_chunk>"))

        # Provider returned explicit `stop_sequence` (e.g. Anthropic)
        raw_done = "<thought>Finished</thought>\n<done>All good"
        fixed_done, matched_done = _reattach_stop_tag(raw_done, stops, explicit_stop="</done>")
        self.assertEqual(matched_done, "</done>")
        self.assertEqual(fixed_done, "<thought>Finished</thought>\n<done>All good</done>")

        # Provider kept the stop sequence intact
        already = "<thought>Done</thought>\n<done>Complete</done>  "
        fixed_already, matched_already = _reattach_stop_tag(already, stops)
        self.assertEqual(matched_already, "</done>")
        self.assertEqual(fixed_already, "<thought>Done</thought>\n<done>Complete</done>")

    def test_parse_action_chunk_validation(self) -> None:
        wps = parse_action_chunk('```json\n[{"dt": 0.5, "joints": {"head_yaw": 0.6, "head_pitch": 0.8}}]\n```')
        self.assertEqual(len(wps), 1)
        self.assertAlmostEqual(wps[0].dt, 0.5)
        self.assertAlmostEqual(wps[0].joints["head_yaw"], 0.6)

        with self.assertRaises(ValueError):
            parse_action_chunk('[{"dt": 0.4, "joints": {"nonexistent_servo": 1.0}}]')

        with self.assertRaises(ValueError):
            parse_action_chunk("[]")

    def test_harness_error_self_repair_loop(self) -> None:
        env = MicroDuckEnv(enable_renderer=False, dual_camera=False)
        snap0 = env.reset()
        agent = _SelfRepairMockAgent()
        sys_prompt = build_system_prompt(coordinate_mode="raw_documented", include_balance_margin=True)

        result = run_harness_episode(
            env=env,
            llm=agent,
            system_prompt=sys_prompt,
            task_id="repair_test",
            instruction="Turn head slightly left.",
            initial_snap=snap0,
            evaluator=lambda hist: (hist[-1].joints_rad["head_yaw"] > 0.2, "ok", {}),
            max_chunks=5,
            verbose=False,
        )
        self.assertEqual(len(agent.seen_errors), 1)
        self.assertIn("left_wing_flap", agent.seen_errors[0])
        self.assertTrue(result.success)
        self.assertEqual(len(result.turns), 3)
        self.assertIsNotNone(result.turns[0].parse_error)
        self.assertTrue(result.turns[2].is_done)

    def test_normalized_symmetric_coordinate_adapter(self) -> None:
        env = MicroDuckEnv(coordinate_mode="normalized_symmetric", enable_renderer=False, dual_camera=False)
        env.reset()
        # Symmetric squat input (+0.80 on both knees, -0.10 on both hip_pitch, +0.90 on both ankles)
        snap = env.step_chunk(
            [
                Waypoint(
                    dt=0.6,
                    joints={
                        "left_hip_pitch": -0.10,
                        "right_hip_pitch": -0.10,
                        "left_knee": 0.80,
                        "right_knee": 0.80,
                        "left_ankle": 0.90,
                        "right_ankle": 0.90,
                    },
                )
            ]
        )
        self.assertFalse(snap.fallen)
        # Outgoing `<observation>` joints_rad are also presented symmetrically (+0.82 on both),
        # while underlying MuJoCo raw CAD qpos has opposite signs (+0.82 left vs -0.82 right).
        self.assertAlmostEqual(snap.joints_rad["left_knee"], snap.joints_rad["right_knee"], places=2)
        raw_left_knee = float(env.data.qpos[env.qpos_adrs["left_knee"]])
        raw_right_knee = float(env.data.qpos[env.qpos_adrs["right_knee"]])
        self.assertGreater(raw_left_knee, 0.5)
        self.assertLess(raw_right_knee, -0.5)
        self.assertAlmostEqual(raw_left_knee, -raw_right_knee, places=2)


if __name__ == "__main__":
    unittest.main()
