"""MuJoCo simulation environment for Pollen Robotics MicroDuck (14-DOF XL330 servos)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from src.telemetry import JOINT_NAMES, TelemetrySnapshot, extract_telemetry

DEFAULT_SCENE_XML = Path(__file__).resolve().parent.parent / "assets" / "microduck" / "scene.xml"

# Symmetry multipliers if operating in `normalized_symmetric` mode:
# Maps intuitive bilateral semantics (+hip_pitch = forward bow, +knee = crouch bend,
# +ankle = dorsiflexion, +hip_roll = outward stance) onto raw CAD joint axes.
SYMMETRIC_TO_RAW_SIGN: Dict[str, float] = {
    "left_hip_yaw": 1.0,
    "left_hip_roll": -1.0,
    "left_hip_pitch": -1.0,
    "left_knee": 1.0,
    "left_ankle": 1.0,
    "neck_pitch": 1.0,
    "head_pitch": 1.0,
    "head_yaw": 1.0,
    "head_roll": 1.0,
    "right_hip_yaw": -1.0,
    "right_hip_roll": 1.0,
    "right_hip_pitch": 1.0,
    "right_knee": -1.0,
    "right_ankle": -1.0,
}


@dataclasses.dataclass
class Waypoint:
    """Single 14-D servo target waypoint in an `<action_chunk>`."""

    dt: float
    joints: Dict[str, float]


class MicroDuckEnv:
    """50 Hz position-controlled MicroDuck simulation with visual telemetry markers."""

    CTRL_HZ: int = 50
    CTRL_DT: float = 1.0 / 50.0
    RENDER_EVERY_N_CTRL: int = 2  # 25 fps realtime video

    def __init__(
        self,
        scene_xml: Path = DEFAULT_SCENE_XML,
        coordinate_mode: str = "raw_documented",
        enable_renderer: bool = True,
        frame_height: int = 432,
         dual_camera: bool = True,
    ) -> None:
        assert coordinate_mode in {"raw_documented", "raw_undocumented", "normalized_symmetric"}
        self.coordinate_mode = coordinate_mode
        self.dual_camera = dual_camera
        self.frame_height = frame_height
        self.main_width = 512 if dual_camera else 640
        self.side_width = 384 if dual_camera else 0

        spec = mujoco.MjSpec.from_file(str(scene_xml))
        self._inject_visual_markers(spec)
        self.model: mujoco.MjModel = spec.compile()
        # `robot_allcollisions.xml` leaves a legacy placeholder kp=0.55 because `mjlab`
        # replaces it at runtime with a custom BAM XL330 firmware model. Set realistic
        # XL330 position stiffness (3.2 N·m/rad -> reaches ±0.96 N·m stall torque at ~17° error)
        # while strictly preserving the hardware ±0.96 N·m actuator `forcerange` clamp.
        for i in range(self.model.nu):
            self.model.actuator_gainprm[i, 0] = 3.2
            self.model.actuator_biasprm[i, 1] = -3.2
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        self.substeps = max(1, int(round(self.CTRL_DT / float(self.model.opt.timestep))))

        # Cache joint & actuator IDs and joint limits
        self.joint_ids: Dict[str, int] = {}
        self.qpos_adrs: Dict[str, int] = {}
        self.act_ids: Dict[str, int] = {}
        self.joint_limits: Dict[str, Tuple[float, float]] = {}
        for jname in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if jid < 0 or aid < 0:
                raise ValueError(f"Joint or actuator '{jname}' missing in {scene_xml}")
            self.joint_ids[jname] = jid
            self.qpos_adrs[jname] = int(self.model.jnt_qposadr[jid])
            self.act_ids[jname] = aid
            lo, hi = float(self.model.jnt_range[jid, 0]), float(self.model.jnt_range[jid, 1])
            self.joint_limits[jname] = (lo, hi)

        self.target_mocap_id: int = int(
            self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")]
        )
        self.com_mocap_id: int = int(
            self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "com_marker")]
        )
        self.com_geom_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "com_disc")

        self._main_renderer: Optional[mujoco.Renderer] = None
        self._side_renderer: Optional[mujoco.Renderer] = None
        if enable_renderer:
            self._main_renderer = mujoco.Renderer(self.model, height=self.frame_height, width=self.main_width)
            if self.dual_camera:
                self._side_renderer = mujoco.Renderer(self.model, height=self.frame_height, width=self.side_width)

        self.target_xyz: Optional[Tuple[float, float, float]] = None
        self.allow_sitting_contacts: bool = False
        self.min_beak_dist: Optional[float] = None
        self.step_counter: int = 0
        self.frames: List[np.ndarray] = []
        self.chunk_frame_ranges: List[Tuple[int, int]] = []
        self.telemetry_history: List[TelemetrySnapshot] = []

    @staticmethod
    def _inject_visual_markers(spec: mujoco.MjSpec) -> None:
        """Add non-colliding mocap markers for the 3D floor target and live CoM ground projection."""
        target_body = spec.worldbody.add_body(name="target_marker", mocap=True, pos=[0.0, 0.0, -2.0])
        target_body.add_geom(
            name="target_sphere",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.013, 0.0, 0.0],
            rgba=[0.95, 0.24, 0.22, 0.95],
            contype=0,
            conaffinity=0,
            group=0,
        )
        target_body.add_geom(
            name="target_stem",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.003, 0.036, 0.0],
            pos=[0.0, 0.0, -0.036],
            rgba=[0.35, 0.55, 0.35, 0.85],
            contype=0,
            conaffinity=0,
            group=0,
        )

        com_body = spec.worldbody.add_body(name="com_marker", mocap=True, pos=[0.0, 0.0, 0.002])
        com_body.add_geom(
            name="com_disc",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.008, 0.0018, 0.0],
            rgba=[0.12, 0.85, 0.65, 0.90],
            contype=0,
            conaffinity=0,
            group=0,
        )

    def reset(
        self,
        keyframe: str = "STAND",
        target_xyz: Optional[Tuple[float, float, float]] = None,
        allow_sitting_contacts: bool = False,
        settle_seconds: float = 0.30,
    ) -> TelemetrySnapshot:
        """Reset simulation to a named keyframe (`STAND`, `SIT`, `FOLD`) and settle contacts."""
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
        if key_id < 0:
            raise ValueError(f"Unknown keyframe '{keyframe}'")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        self.target_xyz = target_xyz
        self.allow_sitting_contacts = allow_sitting_contacts
        self.min_beak_dist = None
        self.step_counter = 0
        self.frames = []
        self.chunk_frame_ranges = []
        self.telemetry_history = []

        if target_xyz is not None:
            self.data.mocap_pos[self.target_mocap_id] = np.array(target_xyz, dtype=np.float64)
        else:
            self.data.mocap_pos[self.target_mocap_id] = np.array([0.0, 0.0, -2.0], dtype=np.float64)

        mujoco.mj_forward(self.model, self.data)
        settle_steps = int(round(settle_seconds / self.CTRL_DT))
        for _ in range(settle_steps):
            mujoco.mj_step(self.model, self.data, nstep=self.substeps)
        # Reset simulation clock to 0.0 after initial gravity settle
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        snap = self._snapshot(status="OK")
        self._update_com_marker(snap)
        if self._main_renderer is not None:
            self.frames.append(self.render_frame())
        self.telemetry_history.append(snap)
        return snap

    def _to_raw_joint_Map(self, joints: Dict[str, float]) -> Dict[str, float]:
        if self.coordinate_mode != "normalized_symmetric":
            return dict(joints)
        return {k: v * SYMMETRIC_TO_RAW_SIGN[k] for k, v in joints.items() if k in SYMMETRIC_TO_RAW_SIGN}

    def _from_raw_telemetry(self, snap: TelemetrySnapshot) -> TelemetrySnapshot:
        if self.coordinate_mode != "normalized_symmetric":
            return snap
        converted = {k: round(v * SYMMETRIC_TO_RAW_SIGN[k], 4) for k, v in snap.joints_rad.items()}
        return dataclasses.replace(snap, joints_rad=converted)

    def current_ctrl_dict(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for jname in JOINT_NAMES:
            val = float(self.data.ctrl[self.act_ids[jname]])
            if self.coordinate_mode == "normalized_symmetric":
                val *= SYMMETRIC_TO_RAW_SIGN[jname]
            out[jname] = val
        return out

    def step_chunk(
        self,
        waypoints: List[Waypoint],
        abort_on_fall: bool = True,
        viewer_handle: Optional[object] = None,
    ) -> TelemetrySnapshot:
        """Execute a list of waypoints via 50 Hz minimum-jerk interpolation."""
        self.step_counter += 1
        start_frame_idx = len(self.frames)
        status = "OK"

        # Start interpolation from current actuator targets for continuity
        q_start = np.array([float(self.data.ctrl[self.act_ids[j]]) for j in JOINT_NAMES], dtype=np.float64)

        for wp in waypoints:
            dt = float(np.clip(wp.dt, 0.10, 2.50))
            n_ctrl = max(1, int(round(dt / self.CTRL_DT)))
            raw_targets = self._to_raw_joint_Map(wp.joints)

            q_goal = q_start.copy()
            for idx, jname in enumerate(JOINT_NAMES):
                if jname in raw_targets:
                    lo, hi = self.joint_limits[jname]
                    q_goal[idx] = float(np.clip(raw_targets[jname], lo, hi))

            for k in range(1, n_ctrl + 1):
                tau = k / float(n_ctrl)
                # Minimum-jerk smoothstep: zero velocity at waypoint endpoints
                alpha = 3.0 * (tau ** 2) - 2.0 * (tau ** 3)
                q_cmd = (1.0 - alpha) * q_start + alpha * q_goal
                for idx, jname in enumerate(JOINT_NAMES):
                    self.data.ctrl[self.act_ids[jname]] = q_cmd[idx]

                mujoco.mj_step(self.model, self.data, nstep=self.substeps)
                snap = self._snapshot(status="OK")
                self._update_com_marker(snap)

                if self._main_renderer is not None and (k % self.RENDER_EVERY_N_CTRL == 0):
                    self.frames.append(self.render_frame())

                if viewer_handle is not None and hasattr(viewer_handle, "sync"):
                    viewer_handle.sync()  # type: ignore[attr-defined]

                if abort_on_fall and snap.fallen:
                    status = "ABORTED_FALL"
                    # Capture a few extra frames so the topple is clearly visible
                    for post in range(8):
                        mujoco.mj_step(self.model, self.data, nstep=self.substeps)
                        if self._main_renderer is not None and (post % 2 == 0):
                            self.frames.append(self.render_frame())
                    break

            q_start = q_goal
            if status == "ABORTED_FALL":
                break
            wp_snap = self._snapshot(status="OK")
            self.telemetry_history.append(self._from_raw_telemetry(wp_snap))

        final_snap = self._snapshot(status=status)
        self._update_com_marker(final_snap)
        if self._main_renderer is not None and len(self.frames) == start_frame_idx:
            self.frames.append(self.render_frame())
        self.chunk_frame_ranges.append((start_frame_idx, max(start_frame_idx, len(self.frames) - 1)))
        out_snap = self._from_raw_telemetry(final_snap)
        if not self.telemetry_history or self.telemetry_history[-1] != out_snap:
            self.telemetry_history.append(out_snap)
        return out_snap

    def _snapshot(self, status: str) -> TelemetrySnapshot:
        snap = extract_telemetry(
            model=self.model,
            data=self.data,
            step=self.step_counter,
            status=status,
            target_xyz=self.target_xyz,
            min_beak_dist=self.min_beak_dist,
            allow_sitting_contacts=self.allow_sitting_contacts,
        )
        self.min_beak_dist = snap.min_beak_to_target_dist_m
        return snap

    def _update_com_marker(self, snap: TelemetrySnapshot) -> None:
        cx, cy, _ = snap.com_xyz_m
        self.data.mocap_pos[self.com_mocap_id] = np.array([cx, cy, 0.002], dtype=np.float64)
        if snap.fallen or "CRITICAL" in snap.balance_hint:
            rgba = [0.95, 0.20, 0.20, 0.95]
        elif "NEAR" in snap.balance_hint:
            rgba = [0.98, 0.72, 0.12, 0.95]
        else:
            rgba = [0.12, 0.85, 0.65, 0.92]
        self.model.geom_rgba[self.com_geom_id] = np.array(rgba, dtype=np.float32)

    def render_frame(self) -> np.ndarray:
        """Render 3/4 studio camera (and optional sagittal balance profile) to an RGB array."""
        assert self._main_renderer is not None
        tx, ty, _ = float(self.data.qpos[0]), float(self.data.qpos[1]), float(self.data.qpos[2])

        cam_main = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam_main)
        cam_main.lookat[:] = [tx + 0.03, ty, 0.11]
        cam_main.distance = 0.44
        cam_main.azimuth = 138.0
        cam_main.elevation = -19.0

        self._main_renderer.update_scene(self.data, camera=cam_main)
        img_main = self._main_renderer.render()

        if not self.dual_camera or self._side_renderer is None:
            return img_main

        cam_side = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam_side)
        cam_side.lookat[:] = [tx + 0.02, ty, 0.105]
        cam_side.distance = 0.40
        cam_side.azimuth = 90.0  # Pure sagittal profile (+X goes right, +Z goes up)
        cam_side.elevation = -8.0

        self._side_renderer.update_scene(self.data, camera=cam_side)
        img_side = self._side_renderer.render()

        divider = np.full((self.frame_height, 2, 3), 40, dtype=np.uint8)
        return np.concatenate([img_main, divider, img_side], axis=1)

    def close(self) -> None:
        if self._main_renderer is not None:
            self._main_renderer.close()
            self._main_renderer = None
        if self._side_renderer is not None:
            self._side_renderer.close()
            self._side_renderer = None
