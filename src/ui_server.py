"""Live 50 Hz MuJoCo simulation engine, SSE event broker, and HTTP server for the MicroDuck Harness UI."""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
import dataclasses
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import queue
import socket
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import mujoco
import numpy as np
from PIL import Image

from src.env import DEFAULT_SCENE_XML, SYMMETRIC_TO_RAW_SIGN, Waypoint
from src.harness import ACTION_RE, DONE_RE, STOP_SEQUENCES, parse_action_chunk
from src.llm_client import GenerationStep, ModelClient, _load_dotenv_if_present
from src.prompts import build_interactive_system_prompt
from src.reference_agent import InteractiveDemoAgent
from src.telemetry import JOINT_NAMES, TelemetrySnapshot, extract_telemetry

WEB_DIR = Path(__file__).resolve().parent / "web"

CAMERA_MODES = {"studio", "split", "side", "front", "pov"}


def telemetry_to_dict(snap: TelemetrySnapshot, include_balance_margin: bool = True) -> Dict[str, Any]:
    """Serialize `TelemetrySnapshot` into a compact JSON-ready dictionary for UI HUD & traces."""
    return {
        "step": snap.step,
        "sim_time_s": round(snap.sim_time_s, 2),
        "status": snap.status,
        "trunk_pos_m": [round(v, 3) for v in snap.trunk_pos_m],
        "trunk_rpy_deg": [round(v, 1) for v in snap.trunk_rpy_deg],
        "com_xyz_m": [round(v, 3) for v in snap.com_xyz_m],
        "left_foot_grounded": snap.left_foot_grounded,
        "right_foot_grounded": snap.right_foot_grounded,
        "support_x_bounds_m": [round(v, 3) for v in snap.support_x_bounds_m] if snap.support_x_bounds_m else None,
        "toe_margin_m": round(snap.forward_toe_margin_m, 3) if snap.forward_toe_margin_m is not None else None,
        "heel_margin_m": round(snap.backward_heel_margin_m, 3) if snap.backward_heel_margin_m is not None else None,
        "balance_hint": snap.balance_hint,
        "mouth_tip_xyz_m": [round(v, 3) for v in snap.mouth_tip_xyz_m],
        "fallen": snap.fallen,
        "saturated_joints": snap.saturated_joints,
        "joints_rad": snap.joints_rad,
        "observation_xml": snap.to_xml(include_balance_margin=include_balance_margin),
    }


def detect_available_providers() -> List[Dict[str, Any]]:
    """Inspect environment / `.env` to report configured LLM providers alongside offline demo mode."""
    _load_dotenv_if_present()
    specs = [
        ("gemini", "Gemini", bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")), ["gemini-3.5-flash", "gemini-pro-latest", "gemini-2.5-flash", "gemini-2.5-pro"]),
        ("anthropic", "Anthropic", bool(os.environ.get("ANTHROPIC_API_KEY")), ["claude-sonnet-4-20250514", "claude-opus-4-20250514"]),
        ("openai", "OpenAI", bool(os.environ.get("OPENAI_API_KEY")), ["gpt-4.1", "gpt-4.1-mini", "o4-mini"]),
        ("openrouter", "OpenRouter", bool(os.environ.get("OPENROUTER_API_KEY")), ["anthropic/claude-sonnet-4", "google/gemini-2.5-pro"]),
        ("deepseek", "DeepSeek", bool(os.environ.get("DEEPSEEK_API_KEY")), ["deepseek-chat", "deepseek-reasoner"]),
        ("custom", "Custom", bool(os.environ.get("LLM_BASE_URL")), [os.environ.get("LLM_MODEL") or "local-model"]),
        ("demo", "Demo", True, ["closed-loop-demo"]),
    ]
    return [
        {"id": pid, "label": label, "configured": configured, "suggested_models": models}
        for pid, label, configured, models in specs
    ]


class EventBroker:
    """Thread-safe Server-Sent Events (SSE) broadcaster."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clients: Set[queue.Queue[str]] = set()

    def subscribe(self) -> queue.Queue[str]:
        q: queue.Queue[str] = queue.Queue(maxsize=256)
        with self._lock:
            self._clients.add(q)
        return q

    def unsubscribe(self, q: queue.Queue[str]) -> None:
        with self._lock:
            self._clients.discard(q)

    def publish(self, payload: Dict[str, Any]) -> None:
        msg = json.dumps(payload, separators=(",", ":"))
        with self._lock:
            clients = list(self._clients)
        for q in clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(msg)
                except queue.Empty:
                    pass


@dataclasses.dataclass
class _SimJob:
    kind: str  # "RESET" | "STEP_CHUNK" | "STOP_HOLD"
    kwargs: Dict[str, Any]
    future: Future[Any]


class LiveDuckEngine:
    """Single-thread 50 Hz MuJoCo runtime + offscreen CGL renderer (safe on macOS & Linux)."""

    CTRL_HZ: int = 50
    CTRL_DT: float = 1.0 / 50.0
    FRAME_W: int = 880
    FRAME_H: int = 540
    SPLIT_MAIN_W: int = 540
    SPLIT_SIDE_W: int = 338  # 540 + 2px divider + 338 = 880

    def __init__(
        self,
        broker: EventBroker,
        scene_xml: Path = DEFAULT_SCENE_XML,
        coordinate_mode: str = "raw_documented",
        enable_renderer: bool = True,
        realtime_chunks: bool = True,
    ) -> None:
        self.broker = broker
        self.scene_xml = scene_xml
        self.coordinate_mode = coordinate_mode
        self.enable_renderer = enable_renderer
        self.realtime_chunks = realtime_chunks

        self._jobs: queue.Queue[_SimJob] = queue.Queue()
        self._stop_all = threading.Event()
        self._cancel_chunk = threading.Event()
        self._ready = threading.Event()
        self._init_error: Optional[BaseException] = None

        # Camera state (safe under _state_lock)
        self._state_lock = threading.Lock()
        self.camera_mode: str = "split"
        self.cam_azimuth: float = 138.0
        self.cam_elevation: float = -18.0
        self.cam_distance: float = 0.44
        self._last_visual_bump: float = time.perf_counter()
        self._latest_snap: Optional[TelemetrySnapshot] = None
        self.sim_state: str = "IDLE"  # "IDLE" | "EXECUTING_CHUNK" | "FALLEN"
        self.active_keyframe: str = "STAND"

        # MJPEG broadcast condition
        self._frame_cond = threading.Condition()
        self._latest_jpeg: bytes = b""
        self._frame_seq: int = 0

        self._thread = threading.Thread(target=self._run_loop, name="MicroDuckSimThread", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=15.0)
        if self._init_error is not None:
            raise RuntimeError(f"Failed to initialize LiveDuckEngine: {self._init_error}") from self._init_error

    def close(self) -> None:
        self._stop_all.set()
        self._cancel_chunk.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        self._thread.join(timeout=3.0)

    def latest_snapshot(self) -> TelemetrySnapshot:
        with self._state_lock:
            assert self._latest_snap is not None
            return self._latest_snap

    def latest_jpeg(self) -> Tuple[int, bytes]:
        with self._frame_cond:
            return self._frame_seq, self._latest_jpeg

    def wait_next_jpeg(self, last_seq: int, timeout: float = 1.0) -> Tuple[int, bytes]:
        with self._frame_cond:
            if self._frame_seq == last_seq and not self._stop_all.is_set():
                self._frame_cond.wait(timeout=timeout)
            return self._frame_seq, self._latest_jpeg

    def camera_state(self) -> Dict[str, Any]:
        with self._state_lock:
            return {
                "mode": self.camera_mode,
                "azimuth": round(self.cam_azimuth, 1),
                "elevation": round(self.cam_elevation, 1),
                "distance": round(self.cam_distance, 3),
            }

    def update_camera(
        self,
        mode: Optional[str] = None,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
        distance: Optional[float] = None,
        delta_az: float = 0.0,
        delta_el: float = 0.0,
        delta_dist: float = 0.0,
    ) -> Dict[str, Any]:
        with self._state_lock:
            if mode and mode in CAMERA_MODES:
                self.camera_mode = mode
                if mode == "studio":
                    # Preserve user's orbit
                    pass
                elif mode == "front":
                    self.cam_azimuth = 180.0
                    self.cam_elevation = -12.0
                    self.cam_distance = 0.42
                elif mode == "side":
                    self.cam_azimuth = 90.0
                    self.cam_elevation = -8.0
                    self.cam_distance = 0.46
            if azimuth is not None:
                self.cam_azimuth = float(azimuth) % 360.0
            if elevation is not None:
                self.cam_elevation = float(np.clip(elevation, -75.0, 5.0))
            if distance is not None:
                self.cam_distance = float(np.clip(distance, 0.22, 1.15))
            if delta_az:
                self.cam_azimuth = (self.cam_azimuth + float(delta_az)) % 360.0
            if delta_el:
                self.cam_elevation = float(np.clip(self.cam_elevation + float(delta_el), -75.0, 5.0))
            if delta_dist:
                self.cam_distance = float(np.clip(self.cam_distance + float(delta_dist), 0.22, 1.15))
            self._last_visual_bump = time.perf_counter()
        return self.camera_state()

    def set_coordinate_mode(self, coordinate_mode: str) -> None:
        assert coordinate_mode in {"raw_documented", "raw_undocumented", "normalized_symmetric"}
        with self._state_lock:
            self.coordinate_mode = coordinate_mode

    def cancel_active_chunk(self) -> None:
        self._cancel_chunk.set()

    def reset(self, keyframe: str = "STAND") -> TelemetrySnapshot:
        self._cancel_chunk.set()
        fut: Future[TelemetrySnapshot] = Future()
        self._jobs.put(_SimJob(kind="RESET", kwargs={"keyframe": keyframe}, future=fut))
        return fut.result(timeout=10.0)

    def step_chunk(self, waypoints: List[Waypoint], abort_on_fall: bool = True) -> TelemetrySnapshot:
        self._cancel_chunk.clear()
        fut: Future[TelemetrySnapshot] = Future()
        self._jobs.put(
            _SimJob(kind="STEP_CHUNK", kwargs={"waypoints": waypoints, "abort_on_fall": abort_on_fall}, future=fut)
        )
        return fut.result(timeout=60.0)

    # ---- Dedicated simulation & CGL rendering thread ----

    def _run_loop(self) -> None:
        full_rend: Optional[mujoco.Renderer] = None
        split_main_rend: Optional[mujoco.Renderer] = None
        split_side_rend: Optional[mujoco.Renderer] = None
        try:
            spec = mujoco.MjSpec.from_file(str(self.scene_xml))
            for tex in spec.textures:
                if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                    tex.rgb1 = [0.933, 0.910, 0.835]  # Solarized base2 (#eee8d5)
                    tex.rgb2 = [0.878, 0.847, 0.757]
                elif tex.name == "groundplane":
                    tex.rgb1 = [0.835, 0.804, 0.718]
                    tex.rgb2 = [0.769, 0.737, 0.651]
                    tex.markrgb = [0.690, 0.659, 0.576]
            for mat in spec.materials:
                if mat.name == "groundplane":
                    mat.reflectance = 0.04
            com_body = spec.worldbody.add_body(name="com_marker", mocap=True, pos=[0.0, 0.0, 0.002])
            com_body.add_geom(
                name="com_disc",
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=[0.009, 0.0018, 0.0],
                rgba=[0.165, 0.631, 0.596, 0.95],
                contype=0,
                conaffinity=0,
                group=0,
            )
            model = spec.compile()
            model.vis.global_.offwidth = 1280
            model.vis.global_.offheight = 720
            model.vis.rgba.haze[:] = [0.933, 0.910, 0.835, 1.0]
            model.vis.headlight.ambient[:] = [0.42, 0.41, 0.39]
            model.vis.headlight.diffuse[:] = [0.58, 0.57, 0.54]
            for i in range(model.nu):
                model.actuator_gainprm[i, 0] = 3.2
                model.actuator_biasprm[i, 1] = -3.2
            data = mujoco.MjData(model)
            substeps = max(1, int(round(self.CTRL_DT / float(model.opt.timestep))))

            joint_limits: Dict[str, Tuple[float, float]] = {}
            act_ids: Dict[str, int] = {}
            qpos_adrs: Dict[str, int] = {}
            for jname in JOINT_NAMES:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
                act_ids[jname] = aid
                qpos_adrs[jname] = int(model.jnt_qposadr[jid])
                joint_limits[jname] = (float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1]))

            com_mocap_id = int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "com_marker")])
            com_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "com_disc")
            hcam_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "head_camera")

            if self.enable_renderer:
                full_rend = mujoco.Renderer(model, height=self.FRAME_H, width=self.FRAME_W)
                split_main_rend = mujoco.Renderer(model, height=self.FRAME_H, width=self.SPLIT_MAIN_W)
                split_side_rend = mujoco.Renderer(model, height=self.FRAME_H, width=self.SPLIT_SIDE_W)

            step_counter = 0

            def _convert_snap(raw_snap: TelemetrySnapshot) -> TelemetrySnapshot:
                with self._state_lock:
                    cmode = self.coordinate_mode
                if cmode != "normalized_symmetric":
                    return raw_snap
                conv = {k: round(v * SYMMETRIC_TO_RAW_SIGN[k], 4) for k, v in raw_snap.joints_rad.items()}
                return dataclasses.replace(raw_snap, joints_rad=conv)

            def _update_markers_and_snap(status: str = "OK") -> TelemetrySnapshot:
                allow_sit = self.active_keyframe == "SIT"
                raw = extract_telemetry(
                    model=model,
                    data=data,
                    step=step_counter,
                    status=status,
                    allow_sitting_contacts=allow_sit,
                )
                cx, cy, _ = raw.com_xyz_m
                data.mocap_pos[com_mocap_id] = np.array([cx, cy, 0.002], dtype=np.float64)
                if raw.fallen or "CRITICAL" in raw.balance_hint:
                    rgba = [0.863, 0.196, 0.184, 0.95]  # Solarized red (#dc322f)
                elif "NEAR" in raw.balance_hint:
                    rgba = [0.710, 0.537, 0.000, 0.95]  # Solarized yellow (#b58900)
                else:
                    rgba = [0.165, 0.631, 0.596, 0.95]  # Solarized cyan (#2aa198)
                model.geom_rgba[com_geom_id] = np.array(rgba, dtype=np.float32)
                snap = _convert_snap(raw)
                with self._state_lock:
                    self._latest_snap = snap
                    if snap.fallen and self.sim_state != "EXECUTING_CHUNK":
                        self.sim_state = "FALLEN"
                return snap

            def _render_and_publish() -> None:
                if not self.enable_renderer or full_rend is None:
                    return
                with self._state_lock:
                    mode = self.camera_mode
                    az = self.cam_azimuth
                    el = self.cam_elevation
                    dist = self.cam_distance
                tx, ty = float(data.qpos[0]), float(data.qpos[1])

                if mode == "pov" and hcam_site_id >= 0:
                    spos = np.asarray(data.site_xpos[hcam_site_id], dtype=np.float64)
                    smat = np.asarray(data.site_xmat[hcam_site_id], dtype=np.float64).reshape(3, 3)
                    fwd = smat[:, 0]
                    up = smat[:, 2]
                    eye = spos + 0.015 * fwd
                    gaze = fwd - 0.28 * up
                    gaze = gaze / max(1e-6, float(np.linalg.norm(gaze)))
                    cam_pov = mujoco.MjvCamera()
                    mujoco.mjv_defaultCamera(cam_pov)
                    cam_pov.distance = 0.50
                    cam_pov.lookat[:] = eye + 0.50 * gaze
                    cam_pov.azimuth = math.degrees(math.atan2(float(gaze[1]), float(gaze[0])))
                    cam_pov.elevation = math.degrees(math.asin(float(np.clip(gaze[2], -0.99, 0.99))))
                    full_rend.update_scene(data, camera=cam_pov)
                    rgb = full_rend.render()
                elif mode == "split" and split_main_rend is not None and split_side_rend is not None:
                    cam_m = mujoco.MjvCamera()
                    mujoco.mjv_defaultCamera(cam_m)
                    cam_m.lookat[:] = [tx + 0.02, ty, 0.115]
                    cam_m.distance = dist
                    cam_m.azimuth = az
                    cam_m.elevation = el
                    split_main_rend.update_scene(data, camera=cam_m)
                    img_m = split_main_rend.render()

                    cam_s = mujoco.MjvCamera()
                    mujoco.mjv_defaultCamera(cam_s)
                    cam_s.lookat[:] = [tx + 0.02, ty, 0.122]
                    cam_s.distance = 0.46
                    cam_s.azimuth = 90.0
                    cam_s.elevation = -8.0
                    split_side_rend.update_scene(data, camera=cam_s)
                    img_s = split_side_rend.render()

                    divider = np.broadcast_to(np.array([201, 193, 169], dtype=np.uint8), (self.FRAME_H, 2, 3))
                    rgb = np.concatenate([img_m, divider, img_s], axis=1)
                else:
                    cam = mujoco.MjvCamera()
                    mujoco.mjv_defaultCamera(cam)
                    cam.lookat[:] = [tx + 0.02, ty, 0.11]
                    cam.distance = dist
                    cam.azimuth = az
                    cam.elevation = el
                    full_rend.update_scene(data, camera=cam)
                    rgb = full_rend.render()

                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=84)
                jpg = buf.getvalue()
                with self._frame_cond:
                    self._latest_jpeg = jpg
                    self._frame_seq += 1
                    self._frame_cond.notify_all()

            def _do_reset(keyframe: str) -> TelemetrySnapshot:
                nonlocal step_counter
                kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
                if kid < 0:
                    raise ValueError(f"Unknown keyframe '{keyframe}'")
                mujoco.mj_resetDataKeyframe(model, data, kid)
                mujoco.mj_forward(model, data)
                for _ in range(int(round(0.30 / self.CTRL_DT))):
                    mujoco.mj_step(model, data, nstep=substeps)
                data.time = 0.0
                mujoco.mj_forward(model, data)
                step_counter = 0
                with self._state_lock:
                    self.active_keyframe = keyframe
                    self.sim_state = "IDLE"
                    self._last_visual_bump = time.perf_counter()
                snap = _update_markers_and_snap("OK")
                _render_and_publish()
                return snap

            # Initial STAND reset
            init_snap = _do_reset("STAND")
            self._ready.set()
            self.broker.publish({"type": "telemetry", "sim_state": "IDLE", "telemetry": telemetry_to_dict(init_snap)})

            last_render_t = time.perf_counter()
            last_telem_t = time.perf_counter()
            next_idle_tick = time.perf_counter() + self.CTRL_DT

            while not self._stop_all.is_set():
                # Process queued control jobs immediately
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    job = None

                if job is not None:
                    try:
                        if job.kind == "RESET":
                            kf = job.kwargs.get("keyframe", "STAND")
                            snap = _do_reset(kf)
                            self.broker.publish(
                                {"type": "sim_reset", "keyframe": kf, "sim_state": "IDLE", "telemetry": telemetry_to_dict(snap)}
                            )
                            job.future.set_result(snap)
                        elif job.kind == "STEP_CHUNK":
                            step_counter += 1
                            wps: List[Waypoint] = job.kwargs["waypoints"]
                            abort_on_fall: bool = job.kwargs.get("abort_on_fall", True)
                            with self._state_lock:
                                self.sim_state = "EXECUTING_CHUNK"
                                cmode = self.coordinate_mode
                                if self.active_keyframe == "SIT":
                                    # Once MicroDuck gets a movement chunk, check standard standing balance unless resetting
                                    self.active_keyframe = "STAND"

                            status = "OK"
                            q_start = np.array([float(data.ctrl[act_ids[j]]) for j in JOINT_NAMES], dtype=np.float64)
                            tick_deadline = time.perf_counter()

                            for wp in wps:
                                dt = float(np.clip(wp.dt, 0.08, 3.00))
                                n_ctrl = max(1, int(round(dt / self.CTRL_DT)))
                                raw_targets = (
                                    dict(wp.joints)
                                    if cmode != "normalized_symmetric"
                                    else {k: v * SYMMETRIC_TO_RAW_SIGN[k] for k, v in wp.joints.items() if k in SYMMETRIC_TO_RAW_SIGN}
                                )
                                q_goal = q_start.copy()
                                for idx, jname in enumerate(JOINT_NAMES):
                                    if jname in raw_targets:
                                        lo, hi = joint_limits[jname]
                                        q_goal[idx] = float(np.clip(raw_targets[jname], lo, hi))

                                for k in range(1, n_ctrl + 1):
                                    if self._stop_all.is_set() or self._cancel_chunk.is_set():
                                        # Smooth hold at current physical joint angles
                                        for jname in JOINT_NAMES:
                                            data.ctrl[act_ids[jname]] = float(data.qpos[qpos_adrs[jname]])
                                        status = "INTERRUPTED"
                                        break

                                    tau = k / float(n_ctrl)
                                    alpha = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
                                    q_cmd = (1.0 - alpha) * q_start + alpha * q_goal
                                    for idx, jname in enumerate(JOINT_NAMES):
                                        data.ctrl[act_ids[jname]] = q_cmd[idx]

                                    mujoco.mj_step(model, data, nstep=substeps)
                                    snap = _update_markers_and_snap("OK")

                                    now = time.perf_counter()
                                    if self.enable_renderer and (k % 2 == 0):
                                        _render_and_publish()
                                        last_render_t = time.perf_counter()
                                    if now - last_telem_t >= 0.12:
                                        self.broker.publish(
                                            {"type": "telemetry", "sim_state": "EXECUTING_CHUNK", "telemetry": telemetry_to_dict(snap)}
                                        )
                                        last_telem_t = now

                                    if abort_on_fall and snap.fallen:
                                        status = "ABORTED_FALL"
                                        for post in range(10):
                                            mujoco.mj_step(model, data, nstep=substeps)
                                            _update_markers_and_snap("ABORTED_FALL")
                                            if self.enable_renderer and (post % 2 == 0):
                                                _render_and_publish()
                                            if self.realtime_chunks:
                                                time.sleep(self.CTRL_DT)
                                        break

                                    if self.realtime_chunks:
                                        tick_deadline += self.CTRL_DT
                                        rem = tick_deadline - time.perf_counter()
                                        if rem > 0.001:
                                            time.sleep(rem)
                                        elif rem < -0.10:
                                            # Don't burst-speed to catch up if renderer lagged slightly
                                            tick_deadline = time.perf_counter()

                                q_start = q_goal
                                if status != "OK":
                                    break

                            final_snap = _update_markers_and_snap(status)
                            with self._state_lock:
                                self.sim_state = "FALLEN" if final_snap.fallen else "IDLE"
                                self._last_visual_bump = time.perf_counter()
                                s_state = self.sim_state
                            _render_and_publish()
                            self.broker.publish(
                                {"type": "telemetry", "sim_state": s_state, "telemetry": telemetry_to_dict(final_snap)}
                            )
                            job.future.set_result(final_snap)
                    except BaseException as exc:
                        job.future.set_exception(exc)
                    next_idle_tick = time.perf_counter() + self.CTRL_DT
                    continue

                # Idle live 50 Hz physics maintenance
                now = time.perf_counter()
                if now >= next_idle_tick:
                    with self._state_lock:
                        cur_state = self.sim_state
                    if cur_state != "FALLEN":
                        # Advance gravity/servos live while pausing sim_time increment drift?
                        # Advancing real MuJoCo step keeps MicroDuck genuinely alive under 50 Hz gravity!
                        prev_time = float(data.time)
                        mujoco.mj_step(model, data, nstep=substeps)
                        # Keep observation sim_time anchored cleanly to executed chunks so traces stay intuitive
                        data.time = prev_time
                        snap = _update_markers_and_snap("OK")
                    else:
                        snap = self.latest_snapshot()

                    with self._state_lock:
                        recently_active = (now - self._last_visual_bump) < 1.2
                        s_state = self.sim_state
                    render_interval = 0.042 if recently_active else 0.14
                    if self.enable_renderer and (now - last_render_t >= render_interval):
                        _render_and_publish()
                        last_render_t = time.perf_counter()
                    if now - last_telem_t >= 0.45:
                        self.broker.publish({"type": "telemetry", "sim_state": s_state, "telemetry": telemetry_to_dict(snap)})
                        last_telem_t = now
                    next_idle_tick = now + self.CTRL_DT
                else:
                    time.sleep(min(0.008, max(0.001, next_idle_tick - now)))
        except BaseException as exc:
            self._init_error = exc
            self._ready.set()
        finally:
            for r in (full_rend, split_main_rend, split_side_rend):
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass


class HarnessChatSession:
    """Stateful multi-turn conversational controller bridging User <-> Generalist LLM <-> 50 Hz MuJoCo."""

    def __init__(
        self,
        engine: LiveDuckEngine,
        broker: EventBroker,
        provider: str = "auto",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        coordinate_mode: str = "raw_documented",
        max_chunks_per_prompt: int = 10,
        agent_override: Optional[Any] = None,
    ) -> None:
        _load_dotenv_if_present()
        self.engine = engine
        self.broker = broker
        self.base_url = base_url
        self.temperature = temperature
        self.coordinate_mode = coordinate_mode
        self.max_chunks_per_prompt = max_chunks_per_prompt
        self._agent_override = agent_override

        self._lock = threading.Lock()
        self._busy = False
        self._cancel = threading.Event()
        self._entry_seq = 0
        self.llm_messages: List[Dict[str, str]] = []
        self.chat_history: List[Dict[str, Any]] = []

        self.provider, self.model = self._resolve_initial_provider_model(provider, model)

    @staticmethod
    def _resolve_initial_provider_model(provider: str, model: Optional[str]) -> Tuple[str, str]:
        providers = detect_available_providers()
        if provider and provider != "auto":
            chosen = provider.lower()
        else:
            configured = [p for p in providers if p["configured"] and p["id"] != "demo"]
            chosen = configured[0]["id"] if configured else "demo"
        if model:
            return chosen, model
        if chosen == "demo":
            return "demo", "closed-loop-demo"
        default_map = {p["id"]: p["suggested_models"][0] for p in providers}
        return chosen, os.environ.get("LLM_MODEL") or default_map.get(chosen, "gemini-3.5-flash")

    def _build_agent(self) -> Any:
        if self._agent_override is not None:
            return self._agent_override
        if self.provider == "demo":
            return InteractiveDemoAgent()
        return ModelClient(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
        )

    def get_state(self) -> Dict[str, Any]:
        snap = self.engine.latest_snapshot()
        frame_seq, _ = self.engine.latest_jpeg()
        with self._lock:
            return {
                "busy": self._busy,
                "sim_state": self.engine.sim_state,
                "frame_seq": frame_seq,
                "provider": self.provider,
                "model": self.model,
                "temperature": self.temperature,
                "coordinate_mode": self.coordinate_mode,
                "providers": detect_available_providers(),
                "camera": self.engine.camera_state(),
                "telemetry": telemetry_to_dict(snap),
                "chat_history": list(self.chat_history),
            }

    def update_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        coordinate_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._busy:
                raise RuntimeError("Cannot switch model configuration while a turn is running.")
            if provider:
                p_norm = provider.lower()
                providers = {p["id"]: p for p in detect_available_providers()}
                self.provider = p_norm
                if not model and p_norm in providers:
                    self.model = providers[p_norm]["suggested_models"][0]
            if model:
                self.model = model.strip()
            if temperature is not None:
                self.temperature = float(np.clip(temperature, 0.0, 1.2))
            if coordinate_mode in {"raw_documented", "raw_undocumented", "normalized_symmetric"}:
                if coordinate_mode != self.coordinate_mode:
                    self.coordinate_mode = coordinate_mode
                    self.engine.set_coordinate_mode(coordinate_mode)
                    # Clear raw vs normalized message history to avoid joint-sign confusion
                    self.llm_messages.clear()
        cfg = {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "coordinate_mode": self.coordinate_mode,
        }
        self.broker.publish({"type": "config_updated", "config": cfg})
        return cfg

    def stop_active(self) -> None:
        self._cancel.set()
        self.engine.cancel_active_chunk()

    def reset_robot(self, keyframe: str = "STAND", clear_chat: bool = False) -> Dict[str, Any]:
        self.stop_active()
        # Wait briefly if a turn thread is finishing up after cancellation
        for _ in range(40):
            with self._lock:
                if not self._busy:
                    break
            time.sleep(0.03)
        snap = self.engine.reset(keyframe=keyframe)
        with self._lock:
            # Start fresh physical trajectory context from the reset pose
            self.llm_messages.clear()
            if clear_chat:
                self.chat_history.clear()
                self.broker.publish({"type": "chat_cleared"})
        return telemetry_to_dict(snap)

    def submit_prompt(self, text: str, blocking: bool = False) -> Dict[str, Any]:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Instruction cannot be empty.")

        with self._lock:
            if self._busy:
                raise RuntimeError("MicroDuck is currently executing an instruction. Stop it first or wait for <done>.")
            self._busy = True
            self._cancel.clear()
            self._entry_seq += 1
            entry_id = f"msg-{self._entry_seq}"
            entry: Dict[str, Any] = {
                "id": entry_id,
                "user_text": cleaned,
                "model": f"{self.provider}/{self.model}" if self.provider != "demo" else "closed-loop-demo",
                "status": "running",
                "live_note": "Starting...",
                "steps": [],
                "done_summary": "",
                "error": None,
                "created_at": round(time.time(), 2),
            }
            self.chat_history.append(entry)

        if blocking:
            self._execute_prompt(entry)
            return entry

        t = threading.Thread(target=self._execute_prompt, args=(entry,), name=f"HarnessTurn-{entry_id}", daemon=True)
        t.start()
        return {"id": entry_id, "status": "started", "entry": dict(entry)}

    def _trim_context_window(self) -> None:
        """Keep rolling conversation compact while always splitting cleanly on a user `<instruction>` boundary."""
        if len(self.llm_messages) <= 24:
            return
        for idx in range(len(self.llm_messages) - 18, len(self.llm_messages)):
            m = self.llm_messages[idx]
            if m["role"] == "user" and "<instruction>" in m["content"]:
                self.llm_messages = self.llm_messages[idx:]
                return

    def _execute_prompt(self, entry: Dict[str, Any]) -> None:
        entry_id = entry["id"]
        try:
            init_snap = self.engine.latest_snapshot()
            if init_snap.fallen:
                msg = "MicroDuck is currently fallen on the floor. Click 'Stand' to reset upright, then send your instruction."
                entry["status"] = "fallen"
                entry["error"] = msg
                self.broker.publish(
                    {
                        "type": "prompt_started",
                        "entry_id": entry_id,
                        "user_text": entry["user_text"],
                        "model": entry["model"],
                        "initial_observation_xml": init_snap.to_xml(include_balance_margin=True),
                        "telemetry": telemetry_to_dict(init_snap),
                    }
                )
                self.broker.publish(
                    {
                        "type": "prompt_failed",
                        "entry_id": entry_id,
                        "status": "fallen",
                        "error": msg,
                        "telemetry": telemetry_to_dict(init_snap),
                    }
                )
                return

            agent = self._build_agent()
            sys_prompt = build_interactive_system_prompt(
                coordinate_mode=self.coordinate_mode,
                include_balance_margin=True,
            )

            self._trim_context_window()
            init_obs_xml = init_snap.to_xml(include_balance_margin=True)
            user_payload = f"<instruction>\n{entry['user_text']}\n</instruction>\n\n{init_obs_xml}"
            self.llm_messages.append({"role": "user", "content": user_payload})

            self.broker.publish(
                {
                    "type": "prompt_started",
                    "entry_id": entry_id,
                    "user_text": entry["user_text"],
                    "model": entry["model"],
                    "initial_observation_xml": init_obs_xml,
                    "telemetry": telemetry_to_dict(init_snap),
                }
            )

            finished_via_done = False
            executed_chunks = 0

            for subturn in range(1, self.max_chunks_per_prompt + 1):
                if self._cancel.is_set():
                    entry["status"] = "stopped"
                    entry["done_summary"] = "Stopped by user."
                    self.broker.publish(
                        {
                            "type": "prompt_completed",
                            "entry_id": entry_id,
                            "status": "stopped",
                            "done_summary": entry["done_summary"],
                            "telemetry": telemetry_to_dict(self.engine.latest_snapshot()),
                        }
                    )
                    return

                entry["live_note"] = f"Turn {subturn}: generating..."
                self.broker.publish(
                    {"type": "llm_thinking", "entry_id": entry_id, "subturn": subturn, "model": entry["model"]}
                )

                gen: GenerationStep = agent.generate_until_stop(
                    system_prompt=sys_prompt,
                    messages=self.llm_messages,
                    stop_sequences=STOP_SEQUENCES,
                )

                if self._cancel.is_set():
                    entry["status"] = "stopped"
                    entry["done_summary"] = "Stopped before executing servo chunk."
                    self.broker.publish(
                        {
                            "type": "prompt_completed",
                            "entry_id": entry_id,
                            "status": "stopped",
                            "done_summary": entry["done_summary"],
                            "telemetry": telemetry_to_dict(self.engine.latest_snapshot()),
                        }
                    )
                    return

                self.llm_messages.append({"role": "assistant", "content": gen.text})

                # 1. Intercept `<done>...</done>`
                done_match = DONE_RE.search(gen.text)
                if done_match and not ACTION_RE.search(gen.text):
                    summary = (done_match.group(1) or "Completed.").strip()
                    snap_now = self.engine.latest_snapshot()
                    step_rec = {
                        "subturn": subturn,
                        "kind": "done",
                        "matched_stop": gen.matched_stop or "</done>",
                        "raw_llm_text": gen.text,
                        "done_summary": summary,
                        "latency_s": round(gen.latency_s, 2),
                    }
                    entry["steps"].append(step_rec)
                    entry["status"] = "done"
                    entry["done_summary"] = summary
                    self.broker.publish(
                        {
                            "type": "prompt_completed",
                            "entry_id": entry_id,
                            "status": "done",
                            "done_summary": summary,
                            "step": step_rec,
                            "telemetry": telemetry_to_dict(snap_now),
                        }
                    )
                    finished_via_done = True
                    break

                # 2. Intercept `<action_chunk>...</action_chunk>`
                act_match = ACTION_RE.search(gen.text)
                if not act_match:
                    err_xml = (
                        '<harness_error>No `<action_chunk>...</action_chunk>` or `<done>...</done>` tag detected. '
                        "Always emit `<action_chunk>[...]</action_chunk>` or `<done>summary</done>`.</harness_error>"
                    )
                    self.llm_messages.append({"role": "user", "content": err_xml})
                    err_rec = {
                        "subturn": subturn,
                        "kind": "harness_error",
                        "raw_llm_text": gen.text,
                        "error_xml": err_xml,
                        "latency_s": round(gen.latency_s, 2),
                    }
                    entry["steps"].append(err_rec)
                    self.broker.publish({"type": "harness_error", "entry_id": entry_id, "step": err_rec})
                    continue

                try:
                    waypoints = parse_action_chunk(act_match.group(1))
                except Exception as exc:
                    err_xml = f"<harness_error>Invalid action_chunk schema: {exc}</harness_error>"
                    self.llm_messages.append({"role": "user", "content": err_xml})
                    err_rec = {
                        "subturn": subturn,
                        "kind": "harness_error",
                        "raw_llm_text": gen.text,
                        "error_xml": err_xml,
                        "latency_s": round(gen.latency_s, 2),
                    }
                    entry["steps"].append(err_rec)
                    self.broker.publish({"type": "harness_error", "entry_id": entry_id, "step": err_rec})
                    continue

                wp_dicts = [dataclasses.asdict(wp) for wp in waypoints]
                chunk_duration_s = round(sum(float(np.clip(wp.dt, 0.08, 3.0)) for wp in waypoints), 2)
                step_rec = {
                    "subturn": subturn,
                    "kind": "action_chunk",
                    "matched_stop": gen.matched_stop or "</action_chunk>",
                    "waypoints": wp_dicts,
                    "duration_s": chunk_duration_s,
                    "latency_s": round(gen.latency_s, 2),
                    "observation_xml": "",
                    "status": "EXECUTING",
                    "balance_hint": "",
                }
                entry["steps"].append(step_rec)
                entry["live_note"] = f"Turn {subturn}: stepping MuJoCo ({chunk_duration_s:.2f}s)..."

                self.broker.publish(
                    {
                        "type": "chunk_executing",
                        "entry_id": entry_id,
                        "step": step_rec,
                    }
                )

                # Step 50 Hz live MuJoCo!
                snap = self.engine.step_chunk(waypoints=waypoints, abort_on_fall=True)
                executed_chunks += 1
                obs_xml = snap.to_xml(include_balance_margin=True)
                step_rec["observation_xml"] = obs_xml
                step_rec["status"] = snap.status
                step_rec["balance_hint"] = snap.balance_hint
                step_rec["telemetry"] = telemetry_to_dict(snap)

                self.broker.publish(
                    {
                        "type": "chunk_observed",
                        "entry_id": entry_id,
                        "subturn": subturn,
                        "observation_xml": obs_xml,
                        "status": snap.status,
                        "balance_hint": snap.balance_hint,
                        "telemetry": step_rec["telemetry"],
                    }
                )

                self.llm_messages.append({"role": "user", "content": obs_xml})

                if snap.status == "INTERRUPTED" or self._cancel.is_set():
                    entry["status"] = "stopped"
                    entry["done_summary"] = "Motion interrupted cleanly — holding current joint angles."
                    self.broker.publish(
                        {
                            "type": "prompt_completed",
                            "entry_id": entry_id,
                            "status": "stopped",
                            "done_summary": entry["done_summary"],
                            "telemetry": step_rec["telemetry"],
                        }
                    )
                    return

                if snap.fallen:
                    entry["status"] = "fallen"
                    entry["done_summary"] = (
                        f"Lost balance on chunk {subturn} ({snap.balance_hint}). Click 'Stand' to reset MicroDuck upright."
                    )
                    self.broker.publish(
                        {
                            "type": "prompt_completed",
                            "entry_id": entry_id,
                            "status": "fallen",
                            "done_summary": entry["done_summary"],
                            "telemetry": step_rec["telemetry"],
                        }
                    )
                    return

            if not finished_via_done:
                snap_end = self.engine.latest_snapshot()
                tx, ty, tz = snap_end.trunk_pos_m
                mz = snap_end.mouth_tip_xyz_m[2]
                summary = (
                    f"Executed {executed_chunks} closed-loop chunks (max turns per prompt reached). "
                    f"Currently at trunk [{tx:+.3f}, {ty:+.3f}, {tz:.3f}] m, beak z={mz:.3f} m ({snap_end.balance_hint}). "
                    f"Send another message to continue!"
                )
                entry["status"] = "done"
                entry["done_summary"] = summary
                self.broker.publish(
                    {
                        "type": "prompt_completed",
                        "entry_id": entry_id,
                        "status": "done",
                        "done_summary": summary,
                        "telemetry": telemetry_to_dict(snap_end),
                    }
                )
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            self.broker.publish(
                {
                    "type": "prompt_failed",
                    "entry_id": entry_id,
                    "status": "error",
                    "error": str(exc),
                    "telemetry": telemetry_to_dict(self.engine.latest_snapshot()),
                }
            )
        finally:
            entry["live_note"] = ""
            with self._lock:
                self._busy = False


def create_app(engine: LiveDuckEngine, session: HarnessChatSession, broker: EventBroker) -> Any:
    """Create the Starlette ASGI application for Uvicorn hosting."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    async def api_state(request: Request) -> Response:
        data = await asyncio.to_thread(session.get_state)
        return JSONResponse(content=data, headers={"Cache-Control": "no-store"})

    async def api_frame(request: Request) -> Response:
        seq, jpg = engine.latest_jpeg()
        etag = f'"{seq}"'
        inm = request.headers.get("if-none-match", "")
        if inm and inm == etag:
            return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "no-cache"})
        return Response(
            content=jpg,
            media_type="image/jpeg",
            headers={"ETag": etag, "Cache-Control": "no-cache"},
        )

    async def api_stream(request: Request) -> StreamingResponse:
        boundary = "mdframe"

        async def gen_frames() -> Any:
            seq = -1
            while not engine._stop_all.is_set():
                if await request.is_disconnected():
                    break
                new_seq, jpg = await asyncio.to_thread(engine.wait_next_jpeg, seq, 1.0)
                if not jpg or new_seq == seq:
                    continue
                seq = new_seq
                header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                ).encode("ascii")
                yield header + jpg + b"\r\n"

        return StreamingResponse(
            gen_frames(),
            media_type=f"multipart/x-mixed-replace; boundary={boundary}",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )

    async def api_events(request: Request) -> StreamingResponse:
        sub = broker.subscribe()

        async def gen_sse() -> Any:
            try:
                hello = json.dumps({"type": "connected", "state": session.get_state()})
                yield f"data: {hello}\n\n".encode("utf-8")
                while not engine._stop_all.is_set():
                    if await request.is_disconnected():
                        break
                    try:
                        msg = await asyncio.to_thread(sub.get, True, 4.0)
                        yield f"data: {msg}\n\n".encode("utf-8")
                    except queue.Empty:
                        yield b": ping\n\n"
            finally:
                broker.unsubscribe(sub)

        return StreamingResponse(
            gen_sse(),
            media_type="text/event-stream; charset=utf-8",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )

    async def api_chat(request: Request) -> Response:
        try:
            payload = await request.json()
            res = await asyncio.to_thread(session.submit_prompt, str(payload.get("message", "")))
            return JSONResponse(content={"ok": True, **res}, headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return JSONResponse(content={"ok": False, "error": str(exc)}, status_code=400)

    async def api_stop(request: Request) -> Response:
        session.stop_active()
        return JSONResponse(content={"ok": True}, headers={"Cache-Control": "no-store"})

    async def api_reset(request: Request) -> Response:
        try:
            payload = await request.json()
            kf = str(payload.get("keyframe", "STAND")).upper()
            clear_chat = bool(payload.get("clear_chat", False))
            telem = await asyncio.to_thread(session.reset_robot, kf, clear_chat)
            return JSONResponse(
                content={"ok": True, "keyframe": kf, "telemetry": telem},
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            return JSONResponse(content={"ok": False, "error": str(exc)}, status_code=400)

    async def api_camera(request: Request) -> Response:
        try:
            payload = await request.json()
            cam = engine.update_camera(
                mode=payload.get("mode"),
                azimuth=payload.get("azimuth"),
                elevation=payload.get("elevation"),
                distance=payload.get("distance"),
                delta_az=float(payload.get("delta_az", 0.0) or 0.0),
                delta_el=float(payload.get("delta_el", 0.0) or 0.0),
                delta_dist=float(payload.get("delta_dist", 0.0) or 0.0),
            )
            return JSONResponse(content={"ok": True, "camera": cam}, headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return JSONResponse(content={"ok": False, "error": str(exc)}, status_code=400)

    async def api_config(request: Request) -> Response:
        try:
            payload = await request.json()
            cfg = session.update_config(
                provider=payload.get("provider"),
                model=payload.get("model"),
                temperature=payload.get("temperature"),
                coordinate_mode=payload.get("coordinate_mode"),
            )
            return JSONResponse(content={"ok": True, "config": cfg}, headers={"Cache-Control": "no-store"})
        except Exception as exc:
            return JSONResponse(content={"ok": False, "error": str(exc)}, status_code=400)

    def _static_response(rel_path: str) -> Response:
        rel = "index.html" if rel_path in {"", "/"} else rel_path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
            return JSONResponse(content={"ok": False, "error": "Not found"}, status_code=404)
        body = target.read_bytes()
        ctype, _ = mimetypes.guess_type(str(target))
        media_type = (
            f"{ctype or 'application/octet-stream'}; charset=utf-8"
            if (ctype or "").startswith("text/") or str(target).endswith(".js")
            else (ctype or "application/octet-stream")
        )
        return Response(content=body, media_type=media_type, headers={"Cache-Control": "no-store"})

    async def serve_root(request: Request) -> Response:
        return _static_response("index.html")

    async def serve_asset(request: Request) -> Response:
        return _static_response(str(request.path_params.get("asset_path", "")))

    routes = [
        Route("/api/state", api_state, methods=["GET"]),
        Route("/api/frame.jpg", api_frame, methods=["GET"]),
        Route("/api/stream", api_stream, methods=["GET"]),
        Route("/api/events", api_events, methods=["GET"]),
        Route("/api/chat", api_chat, methods=["POST"]),
        Route("/api/stop", api_stop, methods=["POST"]),
        Route("/api/reset", api_reset, methods=["POST"]),
        Route("/api/camera", api_camera, methods=["POST"]),
        Route("/api/config", api_config, methods=["POST"]),
        Route("/", serve_root, methods=["GET"]),
        Route("/{asset_path:path}", serve_asset, methods=["GET"]),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["ETag"],
        )
    ]

    return Starlette(debug=False, routes=routes, middleware=middleware)


class MicroDuckUIHTTPServer(ThreadingHTTPServer):
    """Daemon-threaded HTTP server servicing MJPEG video, SSE events, static web assets, and JSON control endpoints."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: Tuple[str, int], engine: LiveDuckEngine, session: HarnessChatSession, broker: EventBroker) -> None:
        self.engine = engine
        self.session = session
        self.broker = broker
        host, port = server_address
        bind_addr: Tuple[Any, ...] = server_address
        if host in {"", "0.0.0.0", "::", "localhost"} and socket.has_ipv6:
            self.address_family = socket.AF_INET6
            bind_addr = ("::", port)
        super().__init__(bind_addr, _UIRequestHandler)

    def server_bind(self) -> None:
        if self.address_family == socket.AF_INET6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError:
                pass
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()
        self.server_name = "localhost"
        self.server_port = int(self.server_address[1])

    def handle_error(self, request: Any, client_address: Any) -> None:
        import sys

        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError, socket.timeout)):
            return
        super().handle_error(request, client_address)


class _UIRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def srv(self) -> MicroDuckUIHTTPServer:
        return self.server  # type: ignore[return-value]

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:
        self.close_connection = True
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.end_headers()

    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        self.close_connection = True
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        self.close_connection = True
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/state":
            self._send_json(self.srv.session.get_state())
            return

        if path == "/api/frame.jpg":
            _, jpg = self.srv.engine.latest_jpeg()
            self.send_response(HTTPStatus.OK)
            self._send_cors_headers()
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(jpg)
            return

        if path == "/api/stream":
            self._serve_mjpeg()
            return

        if path == "/api/events":
            self._serve_sse()
            return

        self._serve_static(path)

    def do_POST(self) -> None:
        self.close_connection = True
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/chat":
                res = self.srv.session.submit_prompt(str(payload.get("message", "")))
                self._send_json({"ok": True, **res})
                return
            if path == "/api/stop":
                self.srv.session.stop_active()
                self._send_json({"ok": True})
                return
            if path == "/api/reset":
                kf = str(payload.get("keyframe", "STAND")).upper()
                clear_chat = bool(payload.get("clear_chat", False))
                telem = self.srv.session.reset_robot(keyframe=kf, clear_chat=clear_chat)
                self._send_json({"ok": True, "keyframe": kf, "telemetry": telem})
                return
            if path == "/api/camera":
                cam = self.srv.engine.update_camera(
                    mode=payload.get("mode"),
                    azimuth=payload.get("azimuth"),
                    elevation=payload.get("elevation"),
                    distance=payload.get("distance"),
                    delta_az=float(payload.get("delta_az", 0.0) or 0.0),
                    delta_el=float(payload.get("delta_el", 0.0) or 0.0),
                    delta_dist=float(payload.get("delta_dist", 0.0) or 0.0),
                )
                self._send_json({"ok": True, "camera": cam})
                return
            if path == "/api/config":
                cfg = self.srv.session.update_config(
                    provider=payload.get("provider"),
                    model=payload.get("model"),
                    temperature=payload.get("temperature"),
                    coordinate_mode=payload.get("coordinate_mode"),
                )
                self._send_json({"ok": True, "config": cfg})
                return
            self._send_json({"ok": False, "error": f"Unknown endpoint {path}"}, status=404)
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _serve_static(self, path: str) -> None:
        self.close_connection = True
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if not str(target).startswith(str(WEB_DIR.resolve())) or not target.is_file():
            self._send_json({"ok": False, "error": "Not found"}, status=404)
            return
        body = target.read_bytes()
        ctype, _ = mimetypes.guess_type(str(target))
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header(
            "Content-Type",
            f"{ctype or 'application/octet-stream'}; charset=utf-8"
            if (ctype or "").startswith("text/") or str(target).endswith(".js")
            else (ctype or "application/octet-stream"),
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _serve_mjpeg(self) -> None:
        self.close_connection = True
        boundary = "mdframe"
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        seq = -1
        try:
            while not self.srv.engine._stop_all.is_set():
                new_seq, jpg = self.srv.engine.wait_next_jpeg(seq, timeout=1.0)
                if not jpg or new_seq == seq:
                    continue
                seq = new_seq
                header = (
                    f"--{boundary}\r\n"
                    f"Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return

    def _serve_sse(self) -> None:
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        sub = self.srv.broker.subscribe()
        try:
            hello = json.dumps({"type": "connected", "state": self.srv.session.get_state()})
            self.wfile.write(f"data: {hello}\n\n".encode("utf-8"))
            self.wfile.flush()
            while not self.srv.engine._stop_all.is_set():
                try:
                    msg = sub.get(timeout=5.0)
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            return
        finally:
            self.srv.broker.unsubscribe(sub)
