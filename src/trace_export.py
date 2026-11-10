"""Export MuJoCo episode rollouts as H.264 MP4s, JSON traces, and interactive synced HTML widgets."""

from __future__ import annotations

import dataclasses
import html
import json
from pathlib import Path
from typing import Any, Dict, List

import imageio.v2 as imageio
import numpy as np

from src.harness import EpisodeResult


def save_mp4(frames: List[np.ndarray], mp4_path: Path, fps: int = 25) -> None:
    """Encode RGB frames into browser-compatible H.264 (`yuv420p`) MP4."""
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        return
    # Ensure macroblock-friendly even dimensions
    h, w, _ = frames[0].shape
    h_even = h - (h % 2)
    w_even = w - (w % 2)
    with imageio.get_writer(
        str(mp4_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=2,
    ) as writer:
        for frame in frames:
            writer.append_data(frame[:h_even, :w_even])


def serialize_episode_json(result: EpisodeResult, fps: int = 25) -> Dict[str, Any]:
    """Convert `EpisodeResult` into a clean JSON dictionary with video timestamps per intercepted turn."""
    turns_out = []
    for t in result.turns:
        f0, f1 = t.frame_range
        turns_out.append(
            {
                "turn_idx": t.turn_idx,
                "t_start_s": round(f0 / float(fps), 2),
                "t_end_s": round(f1 / float(fps), 2),
                "thought": t.thought,
                "waypoints": [dataclasses.asdict(wp) for wp in t.waypoints],
                "observation_xml": t.observation_xml,
                "status": t.telemetry.status,
                "balance_hint": t.telemetry.balance_hint,
                "fallen": t.telemetry.fallen,
                "is_done": t.is_done,
                "done_summary": t.done_summary,
                "parse_error": t.parse_error,
                "latency_s": round(t.latency_s, 3),
            }
        )
    return {
        "task_id": result.task_id,
        "instruction": result.instruction,
        "model_name": result.model_name,
        "coordinate_mode": result.coordinate_mode,
        "include_balance_margin": result.include_balance_margin,
        "success": result.success,
        "eval_summary": result.eval_summary,
        "metrics": result.metrics,
        "initial_observation_xml": result.initial_telemetry.to_xml(
            include_balance_margin=result.include_balance_margin
        ),
        "turns": turns_out,
    }


def build_embeddable_trace_html(
    widget_id: str,
    video_src: str,
    trace_data: Dict[str, Any],
    caption: str,
) -> str:
    """Build a minimalist, Jon-Barron-compatible interactive `<video>` + synced token-stream widget."""
    badge_bg = "#e6f4ea" if trace_data["success"] else "#fce8e6"
    badge_fg = "#137333" if trace_data["success"] else "#c5221f"
    badge_txt = "PASS" if trace_data["success"] else "FAIL / BOUNDARY"

    step_cards: List[str] = []
    init_obs = html.escape(trace_data["initial_observation_xml"])
    step_cards.append(
        f'<div class="md-turn" data-start="0.0" data-end="0.08">'
        f'<div class="md-turn-hdr"><span>Turn 0 &bull; Initial Plant State</span><span>t = 0.00s</span></div>'
        f'<pre class="md-obs">{init_obs}</pre></div>'
    )

    for t in trace_data["turns"]:
        t0, t1 = t["t_start_s"], max(t["t_end_s"], t["t_start_s"] + 0.05)
        idx = t["turn_idx"]
        thought_esc = html.escape(t["thought"])
        if t["is_done"]:
            done_esc = html.escape(t["done_summary"])
            body = (
                f'<div class="md-thought"><b>&lt;thought&gt;</b> {thought_esc}</div>'
                f'<div class="md-done"><b>&lt;done&gt;</b> {done_esc} <i>(STOP intercepted)</i></div>'
            )
        else:
            wp_json = html.escape(json.dumps(t["waypoints"], indent=2))
            obs_esc = html.escape(t["observation_xml"])
            body = (
                f'<div class="md-thought"><b>&lt;thought&gt;</b> {thought_esc}</div>'
                f'<pre class="md-act">&lt;action_chunk&gt;\n{wp_json}\n&lt;/action_chunk&gt;  &larr; STOP intercepted</pre>'
                f'<pre class="md-obs">{obs_esc}</pre>'
            )
        step_cards.append(
            f'<div class="md-turn" data-start="{t0}" data-end="{t1}">'
            f'<div class="md-turn-hdr"><span>Turn {idx} &bull; {html.escape(t["balance_hint"])}</span>'
            f'<span>{t0:.2f}s &ndash; {t1:.2f}s</span></div>'
            f"{body}</div>"
        )

    cards_html = "\n".join(step_cards)
    cap_esc = html.escape(caption)
    eval_esc = html.escape(trace_data["eval_summary"])

    return f"""<div id="{widget_id}" class="md-harness-widget" style="border:1px solid #e2e6ea;border-radius:6px;padding:12px;margin:18px 0;background:#fafbfc;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;flex-wrap:wrap;gap:6px;">
    <span style="font-weight:700;font-size:13.5px;">{cap_esc}</span>
    <span style="background:{badge_bg};color:{badge_fg};font-size:11.5px;font-weight:700;padding:2px 8px;border-radius:10px;">{badge_txt}: {eval_esc}</span>
  </div>
  <video class="md-vid" src="{video_src}" controls loop muted playsinline style="width:100%;border-radius:4px;background:#111;display:block;margin-bottom:8px;"></video>
  <div style="font-size:12px;color:#555;margin-bottom:6px;">
    <b>Intercepted Token Stream</b> <span class="meta">(left camera: 3/4 view &bull; right camera: sagittal CoM vs. foot support profile; click any turn to seek video)</span>
  </div>
  <div class="md-trace-box" style="max-height:230px;overflow-y:auto;border:1px solid #e6e9ec;border-radius:4px;background:#ffffff;padding:8px;">
    {cards_html}
  </div>
  <script>
  (function() {{
    var root = document.getElementById("{widget_id}");
    if (!root) return;
    var vid = root.querySelector(".md-vid");
    var turns = Array.prototype.slice.call(root.querySelectorAll(".md-turn"));
    turns.forEach(function(el) {{
      el.style.cssText = "padding:7px 9px;margin-bottom:7px;border-left:3px solid #d0d7de;border-radius:3px;background:#f8f9fb;cursor:pointer;transition:background 0.15s,border-color 0.15s;";
      el.addEventListener("click", function() {{
        vid.currentTime = parseFloat(el.getAttribute("data-start")) || 0;
        vid.play();
      }});
    }});
    vid.addEventListener("timeupdate", function() {{
      var t = vid.currentTime;
      turns.forEach(function(el) {{
        var s = parseFloat(el.getAttribute("data-start"));
        var e = parseFloat(el.getAttribute("data-end"));
        var active = (t >= s && t <= e + 0.08);
        el.style.borderLeftColor = active ? "#1772d0" : "#d0d7de";
        el.style.background = active ? "#eef5fc" : "#f8f9fb";
      }});
    }});
  }})();
  </script>
</div>"""


def export_episode_bundle(
    frames: List[np.ndarray],
    result: EpisodeResult,
    out_dir: Path,
    slug: str,
    fps: int = 25,
) -> Dict[str, Path]:
    """Write `.mp4`, `.json`, and standalone `.html` preview for an episode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{slug}.mp4"
    json_path = out_dir / f"{slug}.json"
    html_path = out_dir / f"{slug}.html"

    save_mp4(frames, mp4_path, fps=fps)
    trace_dict = serialize_episode_json(result, fps=fps)
    json_path.write_text(json.dumps(trace_dict, indent=2), encoding="utf-8")

    widget_html = build_embeddable_trace_html(
        widget_id=f"widget-{slug}",
        video_src=f"{slug}.mp4",
        trace_data=trace_dict,
        caption=f"{result.task_id} ({result.model_name})",
    )
    standalone = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:Lato,sans-serif;max-width:800px;margin:24px auto;padding:0 16px;font-size:14px;}"
        ".md-thought{font-size:12.5px;color:#222;margin-bottom:4px;}"
        ".md-act{font-size:11.5px;background:#fff8eb;padding:6px;margin:4px 0;border-radius:3px;overflow-x:auto;}"
        ".md-obs{font-size:11.5px;background:#f1f5f9;padding:6px;margin:4px 0;border-radius:3px;overflow-x:auto;}"
        ".md-done{font-size:12.5px;color:#137333;font-weight:600;}"
        ".md-turn-hdr{display:flex;justify-content:space-between;font-size:11.5px;font-weight:700;color:#444;margin-bottom:4px;}"
        "</style></head><body>"
        f"{widget_html}</body></html>"
    )
    html_path.write_text(standalone, encoding="utf-8")
    return {"mp4": mp4_path, "json": json_path, "html": html_path}
