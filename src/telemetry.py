"""Pure-text proprioceptive and geometric balance telemetry serializer for MicroDuck."""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


@dataclasses.dataclass
class TelemetrySnapshot:
    """Structured snapshot of MicroDuck's physical state after an intercepted chunk."""

    step: int
    sim_time_s: float
    status: str  # "OK" | "ABORTED_FALL"
    trunk_pos_m: Tuple[float, float, float]
    trunk_rpy_deg: Tuple[float, float, float]
    com_xyz_m: Tuple[float, float, float]
    left_foot_grounded: bool
    right_foot_grounded: bool
    non_foot_floor_contacts: List[str]
    support_x_bounds_m: Optional[Tuple[float, float]]
    forward_toe_margin_m: Optional[float]
    backward_heel_margin_m: Optional[float]
    balance_hint: str
    mouth_tip_xyz_m: Tuple[float, float, float]
    left_foot_xyz_m: Tuple[float, float, float]
    right_foot_xyz_m: Tuple[float, float, float]
    target_xyz_m: Optional[Tuple[float, float, float]]
    beak_to_target_dist_m: Optional[float]
    min_beak_to_target_dist_m: Optional[float]
    joints_rad: Dict[str, float]
    saturated_joints: Dict[str, float]
    fallen: bool

    def to_xml(self, include_balance_margin: bool = True) -> str:
        """Render clean, compact `<observation>` block for injection into LLM stream."""
        r, p, y = self.trunk_rpy_deg
        tx, ty, tz = self.trunk_pos_m
        cx, cy, cz = self.com_xyz_m
        mx, my, mz = self.mouth_tip_xyz_m

        lines = [
            f'<observation step="{self.step}" sim_time="{self.sim_time_s:.2f}s" status="{self.status}">',
            f"  trunk_pos_m: [{tx:.3f}, {ty:.3f}, {tz:.3f}]",
            f"  trunk_rpy_deg: [roll={r:.1f}, pitch={p:.1f}, yaw={y:.1f}]",
            f"  fallen: {str(self.fallen).lower()}",
            f"  feet_grounded: [left={str(self.left_foot_grounded).lower()}, right={str(self.right_foot_grounded).lower()}]",
        ]
        if self.non_foot_floor_contacts:
            lines.append(f"  body_floor_contacts: {self.non_foot_floor_contacts}")

        if include_balance_margin:
            lines.append(f"  com_xyz_m: [{cx:.3f}, {cy:.3f}, {cz:.3f}]")
            if self.support_x_bounds_m is not None and self.forward_toe_margin_m is not None and self.backward_heel_margin_m is not None:
                sx0, sx1 = self.support_x_bounds_m
                lines.append(
                    f"  support_x_span_m: [{sx0:.3f}, {sx1:.3f}] "
                    f"(toe_margin={self.forward_toe_margin_m:+.3f}m, heel_margin={self.backward_heel_margin_m:+.3f}m -> {self.balance_hint})"
                )
            else:
                lines.append(f"  balance_state: {self.balance_hint}")

        lines.append(f"  mouth_tip_xyz_m: [{mx:.3f}, {my:.3f}, {mz:.3f}]")
        if self.target_xyz_m is not None and self.beak_to_target_dist_m is not None:
            gx, gy, gz = self.target_xyz_m
            dx, dy, dz = gx - mx, gy - my, gz - mz
            min_d = self.min_beak_to_target_dist_m if self.min_beak_to_target_dist_m is not None else self.beak_to_target_dist_m
            lines.append(
                f"  target_xyz_m: [{gx:.3f}, {gy:.3f}, {gz:.3f}] "
                f"(delta_xyz=[{dx:+.3f}, {dy:+.3f}, {dz:+.3f}]m, dist={self.beak_to_target_dist_m:.3f}m, best_so_far={min_d:.3f}m)"
            )

        joints_str = ", ".join(f"{k}: {v:+.3f}" for k, v in self.joints_rad.items())
        lines.append(f"  joints_rad: {{{joints_str}}}")
        if self.saturated_joints:
            sat_str = ", ".join(f"{k}: err={v:+.2f}rad" for k, v in self.saturated_joints.items())
            lines.append(f"  servo_gravity_sag_warning: {{{sat_str}}}")
        lines.append("</observation>")
        return "\n".join(lines)


def quat_to_rpy_deg(quat_wxyz: np.ndarray) -> Tuple[float, float, float]:
    """Convert MuJoCo `[w, x, y, z]` quaternion into `(roll, pitch, yaw)` in degrees."""
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def extract_telemetry(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: int,
    status: str = "OK",
    target_xyz: Optional[Tuple[float, float, float]] = None,
    min_beak_dist: Optional[float] = None,
    allow_sitting_contacts: bool = False,
) -> TelemetrySnapshot:
    """Compute physical state, foot support polygon, and CoM stability margin."""
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    lfoot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_foot_collision")
    rfoot_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_foot_collision")

    mouth_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip")
    lfoot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_foot")
    rfoot_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_foot")

    trunk_pos = tuple(float(v) for v in data.qpos[0:3])
    trunk_rpy = quat_to_rpy_deg(data.qpos[3:7])
    com_xyz = tuple(float(v) for v in data.subtree_com[trunk_id])

    mouth_xyz = tuple(float(v) for v in data.site_xpos[mouth_site_id])
    lfoot_xyz = tuple(float(v) for v in data.site_xpos[lfoot_site_id])
    rfoot_xyz = tuple(float(v) for v in data.site_xpos[rfoot_site_id])

    left_grounded = False
    right_grounded = False
    foot_contact_xs: List[float] = []
    non_foot_contacts: List[str] = []

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 != floor_id and g2 != floor_id:
            continue
        other_geom = g2 if g1 == floor_id else g1
        if other_geom == lfoot_geom_id:
            left_grounded = True
            foot_contact_xs.append(float(con.pos[0]))
        elif other_geom == rfoot_geom_id:
            right_grounded = True
            foot_contact_xs.append(float(con.pos[0]))
        else:
            body_id = int(model.geom_bodyid[other_geom])
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"geom_{other_geom}"
            if body_name not in non_foot_contacts:
                non_foot_contacts.append(body_name)

    roll_deg, pitch_deg, _ = trunk_rpy
    # Distinguish a controlled beak floor-peck (both feet grounded, trunk upright) from a faceplant/topple
    controlled_stance = left_grounded and right_grounded and abs(roll_deg) < 35.0 and abs(pitch_deg) < 48.0 and trunk_pos[2] > 0.055
    head_slammed = ("neck" in non_foot_contacts) or (("jaw_soft" in non_foot_contacts) and not controlled_stance)
    extreme_tilt = abs(roll_deg) > 45.0 or abs(pitch_deg) > 58.0
    unintended_body_hit = (not allow_sitting_contacts) and ("trunk_base" in non_foot_contacts) and trunk_pos[2] < 0.062
    fallen = bool(head_slammed or extreme_tilt or unintended_body_hit)

    # Include sole mesh vertices within 4 mm of the floor for any grounded foot so tiny (<0.5 deg)
    # GJK plane-mesh tilts do not collapse the 30 mm sole polygon to a single heel/toe vertex.
    for grounded, gid in ((left_grounded, lfoot_geom_id), (right_grounded, rfoot_geom_id)):
        if not grounded or gid < 0:
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id >= 0:
            v_adr = int(model.mesh_vertadr[mesh_id])
            v_num = int(model.mesh_vertnum[mesh_id])
            verts_local = np.asarray(model.mesh_vert[v_adr : v_adr + v_num], dtype=np.float64)
            xpos = np.asarray(data.geom_xpos[gid], dtype=np.float64)
            xmat = np.asarray(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
            verts_world = xpos + np.einsum("ni,ji->nj", verts_local, xmat)
            sole_xs = verts_world[verts_world[:, 2] <= 0.004, 0]
            if sole_xs.size > 0:
                foot_contact_xs.append(float(sole_xs.min()))
                foot_contact_xs.append(float(sole_xs.max()))

    support_x_bounds: Optional[Tuple[float, float]] = None
    toe_margin: Optional[float] = None
    heel_margin: Optional[float] = None
    if foot_contact_xs:
        sx_min = min(foot_contact_xs)
        sx_max = max(foot_contact_xs)
        support_x_bounds = (sx_min, sx_max)
        toe_margin = sx_max - com_xyz[0]
        heel_margin = com_xyz[0] - sx_min
        if fallen:
            balance_hint = "FALLEN"
        elif toe_margin < 0.004:
            balance_hint = "CRITICAL_TOE_TIPPING_RISK (shift hips backward)"
        elif heel_margin < 0.004:
            balance_hint = "CRITICAL_HEEL_TIPPING_RISK (lean torso forward)"
        elif toe_margin < 0.008:
            balance_hint = "NEAR_TOE_EDGE"
        elif heel_margin < 0.008:
            balance_hint = "NEAR_HEEL_EDGE"
        else:
            balance_hint = "STABLE_CENTERED"
    else:
        balance_hint = "AIRBORNE_OR_FALLEN" if not allow_sitting_contacts else "SEATED_ON_FLOOR"

    joints_rad: Dict[str, float] = {}
    saturated_joints: Dict[str, float] = {}
    for jname in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
        qadr = int(model.jnt_qposadr[jid])
        qval = float(data.qpos[qadr])
        cval = float(data.ctrl[aid])
        joints_rad[jname] = round(qval, 4)
        err = cval - qval
        if abs(err) > 0.18:
            saturated_joints[jname] = round(err, 3)

    beak_dist: Optional[float] = None
    updated_min_dist: Optional[float] = min_beak_dist
    if target_xyz is not None:
        beak_dist = float(np.linalg.norm(np.array(mouth_xyz) - np.array(target_xyz)))
        if updated_min_dist is None or beak_dist < updated_min_dist:
            updated_min_dist = beak_dist

    return TelemetrySnapshot(
        step=step,
        sim_time_s=float(data.time),
        status=status,
        trunk_pos_m=(trunk_pos[0], trunk_pos[1], trunk_pos[2]),
        trunk_rpy_deg=trunk_rpy,
        com_xyz_m=(com_xyz[0], com_xyz[1], com_xyz[2]),
        left_foot_grounded=left_grounded,
        right_foot_grounded=right_grounded,
        non_foot_floor_contacts=non_foot_contacts,
        support_x_bounds_m=support_x_bounds,
        forward_toe_margin_m=toe_margin,
        backward_heel_margin_m=heel_margin,
        balance_hint=balance_hint,
        mouth_tip_xyz_m=(mouth_xyz[0], mouth_xyz[1], mouth_xyz[2]),
        left_foot_xyz_m=(lfoot_xyz[0], lfoot_xyz[1], lfoot_xyz[2]),
        right_foot_xyz_m=(rfoot_xyz[0], rfoot_xyz[1], rfoot_xyz[2]),
        target_xyz_m=target_xyz,
        beak_to_target_dist_m=beak_dist,
        min_beak_to_target_dist_m=updated_min_dist,
        joints_rad=joints_rad,
        saturated_joints=saturated_joints,
        fallen=fallen,
    )
