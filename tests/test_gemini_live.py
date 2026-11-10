"""Live integration tests verifying generalist Gemini models emit valid, physically grounded 14-D action sequences."""

from __future__ import annotations

import os
import unittest

from src.env import MicroDuckEnv
from src.harness import ACTION_RE, STOP_SEQUENCES, parse_action_chunk, run_harness_episode
from src.llm_client import ModelClient, _load_dotenv_if_present
from src.prompts import build_system_prompt


class GeminiGeneralistLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _load_dotenv_if_present()
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise unittest.SkipTest("GEMINI_API_KEY not set; skipping live Gemini tests.")

    def test_auto_detect_and_stop_interception_flash(self) -> None:
        """Verify generalist `gemini-3.5-flash` auto-loads from `.env` and halts cleanly on `</action_chunk>`."""
        client = ModelClient(provider="auto")
        self.assertEqual(client.provider, "gemini")
        self.assertEqual(client.model, "gemini-3.5-flash")

        step = client.generate_until_stop(
            system_prompt='Emit <thought>...</thought> followed by <action_chunk>[{"dt": 0.4, "joints": {"head_yaw": 0.5}}]</action_chunk> and then STOP.',
            messages=[{"role": "user", "content": "Turn head_yaw to +0.5 rad."}],
            stop_sequences=STOP_SEQUENCES,
        )
        self.assertEqual(step.matched_stop, "</action_chunk>")
        self.assertTrue(step.text.endswith("</action_chunk>"))
        self.assertIn("<thought>", step.text)

        match = ACTION_RE.search(step.text)
        self.assertIsNotNone(match)
        wps = parse_action_chunk(match.group(1))
        self.assertGreaterEqual(len(wps), 1)
        self.assertIn("head_yaw", wps[0].joints)

    def test_generalist_pro_emits_mirrored_14d_squat_sequence(self) -> None:
        """Verify generalist `gemini-pro-latest` reads the system contract and emits a valid, CAD-mirrored squat chunk."""
        env = MicroDuckEnv(enable_renderer=False, dual_camera=False)
        init_snap = env.reset()
        sys_prompt = build_system_prompt(coordinate_mode="raw_documented", include_balance_margin=True)
        client = ModelClient(provider="gemini", model="gemini-pro-latest", temperature=0.1)

        user_msg = (
            "<goal>\nLower your body smoothly into a balanced flat-footed squat (`trunk_pos_m` z around 0.085 m) "
            "without falling over.\n</goal>\n\n"
            f"{init_snap.to_xml(include_balance_margin=True)}"
        )
        step = client.generate_until_stop(
            system_prompt=sys_prompt,
            messages=[{"role": "user", "content": user_msg}],
            stop_sequences=STOP_SEQUENCES,
        )
        self.assertEqual(step.matched_stop, "</action_chunk>")

        match = ACTION_RE.search(step.text)
        self.assertIsNotNone(match, f"Expected <action_chunk> in generalist model output: {step.text[:300]}")
        wps = parse_action_chunk(match.group(1))
        last_joints = wps[-1].joints

        # Verify generalist model respected mirrored left/right CAD servo conventions from the system contract
        self.assertIn("left_knee", last_joints)
        self.assertIn("right_knee", last_joints)
        self.assertGreater(last_joints["left_knee"], 0.2, "left_knee should flex positively in a squat")
        self.assertLess(last_joints["right_knee"], -0.2, "right_knee must mirror with negative sign")

        # Execute the generalist model's emitted chunk in 50 Hz MuJoCo and verify stability
        after_snap = env.step_chunk(wps)
        env.close()
        self.assertFalse(after_snap.fallen, f"Generalist squat chunk caused fall: {after_snap.to_xml()}")
        self.assertLess(after_snap.trunk_pos_m[2], init_snap.trunk_pos_m[2] - 0.005, "Expected trunk Z to lower in squat")

    def test_end_to_end_mujoco_harness_episode_with_generalist_flash(self) -> None:
        """Verify multi-turn closed-loop `<action_chunk>` -> MuJoCo `<observation>` -> `<done>` with `gemini-3.5-flash`."""
        env = MicroDuckEnv(enable_renderer=False, dual_camera=False)
        init_snap = env.reset()
        client = ModelClient(provider="gemini", model="gemini-3.5-flash", temperature=0.1)
        sys_prompt = build_system_prompt(coordinate_mode="raw_documented", include_balance_margin=True)

        res = run_harness_episode(
            env=env,
            llm=client,
            system_prompt=sys_prompt,
            task_id="gemini_generalist_e2e",
            instruction=(
                "Turn `head_yaw` to +0.60 rad in one `<action_chunk>`, wait for `<observation>` to confirm "
                "`head_yaw > 0.45`, and then emit `<done>Head turned left.</done>` on the next turn."
            ),
            initial_snap=init_snap,
            evaluator=lambda hist: (
                (not any(s.fallen for s in hist)) and max(s.joints_rad["head_yaw"] for s in hist) > 0.40,
                "live generalist smoke check",
                {},
            ),
            max_chunks=4,
            verbose=True,
        )
        env.close()

        self.assertGreaterEqual(len(res.turns), 2)
        self.assertGreaterEqual(len(res.turns[0].waypoints), 1)
        self.assertIn("<observation", res.turns[0].observation_xml)
        self.assertTrue(any(t.is_done for t in res.turns), "Expected generalist model to emit <done> after <observation>.")
        self.assertTrue(res.success)


if __name__ == "__main__":
    unittest.main()
