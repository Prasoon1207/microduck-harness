#!/usr/bin/env python3
"""CLI entrypoint for running MicroDuck zero-shot MuJoCo harness tasks & ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from src.env import MicroDuckEnv
from src.harness import PolicyGenerator, run_harness_episode
from src.llm_client import ModelClient
from src.prompts import build_system_prompt
from src.reference_agent import ClosedLoopReferenceAgent
from src.tasks import TASKS
from src.trace_export import export_episode_bundle


def build_agent(args: argparse.Namespace) -> PolicyGenerator:
    if args.provider in {"closed_loop", "open_loop_naive", "blind_symmetry"}:
        return ClosedLoopReferenceAgent(mode=args.provider)
    return ModelClient(
        provider=None if args.provider == "auto" else args.provider,
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MicroDuck Zero-Shot MuJoCo Robotics Harness.")
    parser.add_argument(
        "--task",
        default="peck_target",
        choices=[*TASKS.keys(), "all"],
        help="Task ID to execute (or 'all' to run the full benchmark suite).",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        help="LLM provider ('auto', 'gemini', 'openai', 'anthropic', 'openrouter', 'custom', or reference modes 'closed_loop', 'open_loop_naive', 'blind_symmetry').",
    )
    parser.add_argument("--model", default=None, help="Override model name (e.g. gemini-3.5-flash, gemini-pro-latest, claude-sonnet-4-20250514, gpt-4.1).")
    parser.add_argument("--base-url", default=None, help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--coordinate-mode",
        default="raw_documented",
        choices=["raw_documented", "raw_undocumented", "normalized_symmetric"],
    )
    parser.add_argument("--no-balance-margin", action="store_true", help="Ablate CoM support-polygon telemetry from <observation>.")
    parser.add_argument("--record", action="store_true", help="Render 25fps dual-camera MP4, JSON trace, and synced HTML widget.")
    parser.add_argument("--viewer", action="store_true", help="Launch interactive 3D MuJoCo viewer (run with mjpython on macOS).")
    parser.add_argument("--out-dir", default="outputs", help="Directory for recorded MP4s and JSON/HTML traces.")
    parser.add_argument("--slug-suffix", default="", help="Optional suffix appended to output filenames.")
    args = parser.parse_args()

    llm = build_agent(args)
    task_ids: List[str] = list(TASKS.keys()) if args.task == "all" else [args.task]
    include_balance = not args.no_balance_margin
    system_prompt = build_system_prompt(
        coordinate_mode=args.coordinate_mode,
        include_balance_margin=include_balance,
    )

    out_dir = Path(args.out_dir)
    summary_rows = []

    env = MicroDuckEnv(
        coordinate_mode=args.coordinate_mode,
        enable_renderer=args.record,
        dual_camera=True,
    )

    viewer_ctx = None
    if args.viewer:
        import mujoco.viewer

        viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for tid in task_ids:
            spec = TASKS[tid]
            print(f"\n=== Running Task: {spec.task_id} ({spec.title}) ===")
            init_snap = env.reset(
                keyframe=spec.init_keyframe,
                target_xyz=spec.target_xyz,
                allow_sitting_contacts=spec.allow_sitting_contacts,
            )
            res = run_harness_episode(
                env=env,
                llm=llm,
                system_prompt=system_prompt,
                task_id=spec.task_id,
                instruction=spec.instruction,
                initial_snap=init_snap,
                evaluator=spec.evaluator,
                max_chunks=spec.max_chunks,
                include_balance_margin=include_balance,
                abort_on_fall=spec.abort_on_fall,
                viewer_handle=viewer_ctx,
                verbose=True,
            )
            status_str = "PASS" if res.success else "FAIL"
            print(f"--> [{status_str}] {res.eval_summary}")

            slug = f"{tid}{('_' + args.slug_suffix) if args.slug_suffix else ''}"
            if args.record:
                paths = export_episode_bundle(env.frames, res, out_dir=out_dir, slug=slug)
                print(f"    Saved MP4 : {paths['mp4']}")
                print(f"    Saved HTML: {paths['html']}")

            summary_rows.append(
                {
                    "task_id": tid,
                    "success": res.success,
                    "turns": len(res.turns),
                    "eval_summary": res.eval_summary,
                    "metrics": res.metrics,
                }
            )
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
        env.close()

    if len(summary_rows) > 1:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "benchmark_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
        print("\n=== Benchmark Summary ===")
        for r in summary_rows:
            mark = "PASS" if r["success"] else "FAIL"
            print(f"  {r['task_id']:16s} | {mark:4s} | turns={r['turns']} | {r['eval_summary']}")


if __name__ == "__main__":
    main()
