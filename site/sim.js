/* Browser-native 50 Hz MuJoCo WASM + 60 FPS Three.js WebGL Engine for MicroDuck */

import { JOINT_NAMES } from "./harness.js";

const CTRL_HZ = 50;
const CTRL_DT = 1.0 / CTRL_HZ;

function quatToRpyDeg(w, x, y, z) {
  const sinrCosp = 2.0 * (w * x + y * z);
  const cosrCosp = 1.0 - 2.0 * (x * x + y * y);
  const roll = Math.atan2(sinrCosp, cosrCosp);

  const sinp = Math.max(-1.0, Math.min(1.0, 2.0 * (w * y - z * x)));
  const pitch = Math.asin(sinp);

  const sinyCosp = 2.0 * (w * z + x * y);
  const cosyCosp = 1.0 - 2.0 * (y * y + z * z);
  const yaw = Math.atan2(sinyCosp, cosyCosp);

  const d = 180.0 / Math.PI;
  return [roll * d, pitch * d, yaw * d];
}

function fmtSigned(v, digits = 3) {
  return (v >= 0 ? "+" : "") + v.toFixed(digits);
}

export function formatObservationXml(snap) {
  const [r, p, y] = snap.trunk_rpy_deg;
  const [tx, ty, tz] = snap.trunk_pos_m;
  const [cx, cy, cz] = snap.com_xyz_m;
  const [mx, my, mz] = snap.mouth_tip_xyz_m;

  const lines = [
    `<observation step="${snap.step}" sim_time="${snap.sim_time_s.toFixed(2)}s" status="${snap.status}">`,
    `  trunk_pos_m: [${tx.toFixed(3)}, ${ty.toFixed(3)}, ${tz.toFixed(3)}]`,
    `  trunk_rpy_deg: [roll=${r.toFixed(1)}, pitch=${p.toFixed(1)}, yaw=${y.toFixed(1)}]`,
    `  fallen: ${String(snap.fallen)}`,
    `  feet_grounded: [left=${String(snap.left_foot_grounded)}, right=${String(snap.right_foot_grounded)}]`,
  ];

  if (snap.non_foot_floor_contacts.length > 0) {
    lines.push(`  body_floor_contacts: [${snap.non_foot_floor_contacts.map((s) => `'${s}'`).join(", ")}]`);
  }

  lines.push(`  com_xyz_m: [${cx.toFixed(3)}, ${cy.toFixed(3)}, ${cz.toFixed(3)}]`);
  if (snap.support_x_bounds_m && snap.toe_margin_m != null && snap.heel_margin_m != null) {
    const [sx0, sx1] = snap.support_x_bounds_m;
    lines.push(
      `  support_x_span_m: [${sx0.toFixed(3)}, ${sx1.toFixed(3)}] ` +
        `(toe_margin=${fmtSigned(snap.toe_margin_m, 3)}m, heel_margin=${fmtSigned(snap.heel_margin_m, 3)}m -> ${snap.balance_hint})`
    );
  } else {
    lines.push(`  balance_state: ${snap.balance_hint}`);
  }

  lines.push(`  mouth_tip_xyz_m: [${mx.toFixed(3)}, ${my.toFixed(3)}, ${mz.toFixed(3)}]`);

  const jointsStr = JOINT_NAMES.map((k) => `${k}: ${fmtSigned(snap.joints_rad[k] || 0, 3)}`).join(", ");
  lines.push(`  joints_rad: {${jointsStr}}`);

  const satEntries = Object.entries(snap.saturated_joints || {});
  if (satEntries.length > 0) {
    const satStr = satEntries.map(([k, v]) => `${k}: err=${fmtSigned(v, 2)}rad`).join(", ");
    lines.push(`  servo_gravity_sag_warning: {${satStr}}`);
  }

  lines.push("</observation>");
  return lines.join("\n");
}

function makeCheckerTexture(THREE) {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");

  const half = size / 2;
  ctx.fillStyle = "#e2dac4";
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = "#d5ccb4";
  ctx.fillRect(0, 0, half, half);
  ctx.fillRect(half, half, half, half);

  ctx.strokeStyle = "#c7bea4";
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, half, half);
  ctx.strokeRect(half, half, half, half);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(20, 20);
  tex.anisotropy = 4;
  return tex;
}

function buildBufferGeometryFromMjMesh(THREE, model, meshId) {
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];
  const vertAdr = model.mesh_vertadr[meshId];
  const normAdr = model.mesh_normaladr ? model.mesh_normaladr[meshId] : vertAdr;
  const hasFaceNormals = Boolean(model.mesh_facenormal && model.mesh_normal);

  const positions = new Float32Array(faceNum * 9);
  const normals = new Float32Array(faceNum * 9);

  const verts = model.mesh_vert;
  const norms = model.mesh_normal;
  const faces = model.mesh_face;
  const faceNorms = model.mesh_facenormal;

  for (let f = 0; f < faceNum; f++) {
    const fIdx = (faceAdr + f) * 3;
    const outBase = f * 9;
    for (let c = 0; c < 3; c++) {
      const vi = (vertAdr + faces[fIdx + c]) * 3;
      const dst = outBase + c * 3;
      positions[dst + 0] = verts[vi + 0];
      positions[dst + 1] = verts[vi + 1];
      positions[dst + 2] = verts[vi + 2];

      if (hasFaceNormals) {
        const ni = (normAdr + faceNorms[fIdx + c]) * 3;
        normals[dst + 0] = norms[ni + 0];
        normals[dst + 1] = norms[ni + 1];
        normals[dst + 2] = norms[ni + 2];
      }
    }
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (hasFaceNormals) {
    geom.setAttribute("normal", new THREE.BufferAttribute(normals, 3));
  } else {
    geom.computeVertexNormals();
  }
  return geom;
}

export class MicroDuckWasmSim {
  constructor({ canvas, onProgress, onTelemetry }) {
    this.canvas = canvas;
    this.onProgress = onProgress || (() => {});
    this.onTelemetry = onTelemetry || (() => {});

    this.THREE = null;
    this.mujoco = null;
    this.model = null;
    this.data = null;

    this.cameraMode = "split";
    this.camAzimuth = 138.0;
    this.camElevation = -18.0;
    this.camDistance = 0.44;

    this.activeKeyframe = "STAND";
    this.stepCounter = 0;
    this.executingChunk = false;
    this.cancelRequested = false;
    this.latestSnap = null;

    this._chunkTask = null;
    this._rafId = 0;
  }

  async init() {
    this.onProgress("Loading WASM & WebGL...");
    const [mjMod, THREE] = await Promise.all([
      import("./vendor/mujoco.js"),
      import("./vendor/three.module.min.js"),
    ]);
    this.THREE = THREE;
    const mujoco = await mjMod.default();
    this.mujoco = mujoco;

    const vfs = new mujoco.MjVFS();
    const [sceneBuf, robotBuf] = await Promise.all([
      fetch("./assets/microduck/scene.xml").then((r) => {
        if (!r.ok) throw new Error(`Failed loading scene.xml (${r.status})`);
        return r.arrayBuffer();
      }),
      fetch("./assets/microduck/robot_groundcontact.xml").then((r) => {
        if (!r.ok) throw new Error(`Failed loading robot_groundcontact.xml (${r.status})`);
        return r.arrayBuffer();
      }),
    ]);

    vfs.addBuffer("scene.xml", new Uint8Array(sceneBuf));
    vfs.addBuffer("robot_groundcontact.xml", new Uint8Array(robotBuf));

    const robotXml = new TextDecoder().decode(robotBuf);
    const meshFiles = [...new Set([...robotXml.matchAll(/file="([^"]+)"/g)].map((m) => m[1]))];

    let loaded = 0;
    this.onProgress(`Loading meshes (0/${meshFiles.length})...`);
    await Promise.all(
      meshFiles.map(async (fname) => {
        const resp = await fetch(`./assets/microduck/assets/${fname}`);
        if (!resp.ok) throw new Error(`Failed loading mesh ${fname} (${resp.status})`);
        const ab = await resp.arrayBuffer();
        vfs.addBuffer(`assets/${fname}`, new Uint8Array(ab));
        loaded += 1;
        this.onProgress(`Loading meshes (${loaded}/${meshFiles.length})...`);
      })
    );

    this.onProgress("Compiling 50 Hz physics...");
    const model = mujoco.MjModel.mj_loadXML("scene.xml", vfs);
    const ngain = mujoco.mjNGAIN || 10;
    const nbias = mujoco.mjNBIAS || 10;
    for (let i = 0; i < model.nu; i++) {
      model.actuator_gainprm[i * ngain + 0] = 3.2;
      model.actuator_biasprm[i * nbias + 1] = -3.2;
    }

    const data = new mujoco.MjData(model);
    this.model = model;
    this.data = data;
    this.substeps = Math.max(1, Math.round(CTRL_DT / model.opt.timestep));

    const OBJ = mujoco.mjtObj;
    this.ids = {
      trunkBody: mujoco.mj_name2id(model, OBJ.mjOBJ_BODY.value, "trunk_base"),
      floorGeom: mujoco.mj_name2id(model, OBJ.mjOBJ_GEOM.value, "floor"),
      lfootGeom: mujoco.mj_name2id(model, OBJ.mjOBJ_GEOM.value, "left_foot_collision"),
      rfootGeom: mujoco.mj_name2id(model, OBJ.mjOBJ_GEOM.value, "right_foot_collision"),
      mouthSite: mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, "mouth_tip"),
      lfootSite: mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, "left_foot"),
      rfootSite: mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, "right_foot"),
      hcamSite: mujoco.mj_name2id(model, OBJ.mjOBJ_SITE.value, "head_camera"),
      joints: {},
    };

    for (const jname of JOINT_NAMES) {
      const jid = mujoco.mj_name2id(model, OBJ.mjOBJ_JOINT.value, jname);
      const aid = mujoco.mj_name2id(model, OBJ.mjOBJ_ACTUATOR.value, jname);
      this.ids.joints[jname] = {
        jid,
        aid,
        qadr: model.jnt_qposadr[jid],
        lo: model.jnt_range[jid * 2 + 0],
        hi: model.jnt_range[jid * 2 + 1],
      };
    }

    this.soleVerts = {
      left: this._extractLocalMeshVerts(this.ids.lfootGeom),
      right: this._extractLocalMeshVerts(this.ids.rfootGeom),
    };

    this._initThreeScene();
    this.reset("STAND");
    this._startRenderAndPhysicsLoop();
  }

  _extractLocalMeshVerts(gid) {
    if (gid < 0) return null;
    const meshId = this.model.geom_dataid[gid];
    if (meshId < 0) return null;
    const vAdr = this.model.mesh_vertadr[meshId];
    const vNum = this.model.mesh_vertnum[meshId];
    return this.model.mesh_vert.slice(vAdr * 3, (vAdr + vNum) * 3);
  }

  _initThreeScene() {
    const THREE = this.THREE;
    const renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(new THREE.Color().setRGB(0.933, 0.910, 0.835, THREE.SRGBColorSpace), 1);
    this.renderer = renderer;

    const scene = new THREE.Scene();
    const bgCol = new THREE.Color().setRGB(0.933, 0.910, 0.835, THREE.SRGBColorSpace);
    scene.background = bgCol;
    scene.fog = new THREE.Fog(bgCol, 1.35, 3.8);
    this.scene = scene;

    // Z-up lighting matching Solarized studio
    const hemi = new THREE.HemisphereLight(0xffffff, 0xe2dac4, 1.55);
    hemi.position.set(0, 0, 2.0);
    scene.add(hemi);

    const dirLight = new THREE.DirectionalLight(0xfffdf8, 1.55);
    dirLight.position.set(0.55, -0.45, 1.65);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    dirLight.shadow.camera.near = 0.2;
    dirLight.shadow.camera.far = 3.5;
    const sBound = 0.55;
    dirLight.shadow.camera.left = -sBound;
    dirLight.shadow.camera.right = sBound;
    dirLight.shadow.camera.top = sBound;
    dirLight.shadow.camera.bottom = -sBound;
    dirLight.shadow.bias = -0.0005;
    scene.add(dirLight);
    scene.add(dirLight.target);
    this.dirLight = dirLight;

    const fillLight = new THREE.DirectionalLight(0xfdf6e3, 0.45);
    fillLight.position.set(-0.6, 0.7, 0.9);
    scene.add(fillLight);

    // Checker ground plane in Z=0 (XY plane directly)
    const floorGeom = new THREE.PlaneGeometry(6.0, 6.0);
    const floorMat = new THREE.MeshStandardMaterial({
      map: makeCheckerTexture(THREE),
      roughness: 0.92,
      metalness: 0.0,
    });
    const floorMesh = new THREE.Mesh(floorGeom, floorMat);
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    // Whole-body CoM floor projection disc
    const comGeom = new THREE.CylinderGeometry(0.009, 0.009, 0.0036, 28);
    comGeom.rotateX(Math.PI / 2); // Align cylinder height along Z
    this.comColors = {
      ok: new THREE.Color("#2aa198"),
      near: new THREE.Color("#b58900"),
      crit: new THREE.Color("#dc322f"),
    };
    this.comMat = new THREE.MeshBasicMaterial({ color: this.comColors.ok });
    this.comDisc = new THREE.Mesh(comGeom, this.comMat);
    this.comDisc.position.set(0, 0, 0.002);
    scene.add(this.comDisc);

    // Build visual robot meshes from compiled MuJoCo model
    const meshGeomCache = new Map();
    this.visualGeoms = [];
    const model = this.model;
    const GEOM_MESH = this.mujoco.mjtGeom.mjGEOM_MESH.value;

    for (let gid = 0; gid < model.ngeom; gid++) {
      if (gid === this.ids.floorGeom) continue;
      const grp = model.geom_group[gid];
      if (grp > 2) continue;
      const gtype = model.geom_type[gid];
      if (gtype !== GEOM_MESH) continue;

      const meshId = model.geom_dataid[gid];
      if (meshId < 0) continue;

      let bufGeom = meshGeomCache.get(meshId);
      if (!bufGeom) {
        bufGeom = buildBufferGeometryFromMjMesh(THREE, model, meshId);
        meshGeomCache.set(meshId, bufGeom);
      }

      const matId = model.geom_matid[gid];
      const rgbaSrc = matId >= 0 ? model.mat_rgba : model.geom_rgba;
      const base = (matId >= 0 ? matId : gid) * 4;
      const col = new THREE.Color().setRGB(rgbaSrc[base + 0], rgbaSrc[base + 1], rgbaSrc[base + 2], THREE.SRGBColorSpace);

      const mat = new THREE.MeshStandardMaterial({
        color: col,
        roughness: 0.48,
        metalness: 0.08,
      });

      const mesh = new THREE.Mesh(bufGeom, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      scene.add(mesh);
      this.visualGeoms.push({ gid, mesh });
    }

    this.mainCam = new THREE.PerspectiveCamera(45, 1, 0.025, 12.0);
    this.mainCam.up.set(0, 0, 1);

    this.sideCam = new THREE.PerspectiveCamera(45, 1, 0.025, 12.0);
    this.sideCam.up.set(0, 0, 1);
  }

  _syncThreeFromMuJoCo(snap) {
    const xpos = this.data.geom_xpos;
    const xmat = this.data.geom_xmat;

    for (const item of this.visualGeoms) {
      const p = item.gid * 3;
      const m = item.gid * 9;
      item.mesh.matrix.set(
        xmat[m + 0], xmat[m + 1], xmat[m + 2], xpos[p + 0],
        xmat[m + 3], xmat[m + 4], xmat[m + 5], xpos[p + 1],
        xmat[m + 6], xmat[m + 7], xmat[m + 8], xpos[p + 2],
        0, 0, 0, 1
      );
      item.mesh.matrixWorldNeedsUpdate = true;
    }

    if (snap) {
      const [cx, cy] = snap.com_xyz_m;
      this.comDisc.position.set(cx, cy, 0.002);
      if (snap.fallen || snap.balance_hint.includes("CRITICAL")) {
        this.comMat.color.copy(this.comColors.crit);
      } else if (snap.balance_hint.includes("NEAR")) {
        this.comMat.color.copy(this.comColors.near);
      } else {
        this.comMat.color.copy(this.comColors.ok);
      }
      const [tx, ty] = snap.trunk_pos_m;
      this.dirLight.position.set(tx + 0.55, ty - 0.45, 1.65);
      this.dirLight.target.position.set(tx, ty, 0.08);
      this.dirLight.target.updateMatrixWorld();
    }
  }

  _positionOrbitCamera(cam, lookat, azimuthDeg, elevationDeg, distance, aspect) {
    const az = (azimuthDeg * Math.PI) / 180.0;
    const el = (elevationDeg * Math.PI) / 180.0;
    const cosEl = Math.cos(el);
    cam.fov = 45;
    cam.near = 0.025;
    cam.aspect = Math.max(0.1, aspect);
    cam.up.set(0, 0, 1);
    cam.position.set(
      lookat[0] - distance * Math.cos(az) * cosEl,
      lookat[1] - distance * Math.sin(az) * cosEl,
      lookat[2] - distance * Math.sin(el)
    );
    cam.lookAt(lookat[0], lookat[1], lookat[2]);
    cam.updateProjectionMatrix();
  }

  _renderFrame() {
    if (!this.renderer || !this.data) return;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (this.canvas.width !== Math.floor(w * this.renderer.getPixelRatio()) || this.canvas.height !== Math.floor(h * this.renderer.getPixelRatio())) {
      this.renderer.setSize(w, h, false);
    }

    const tx = this.data.qpos[0];
    const ty = this.data.qpos[1];

    if (this.cameraMode === "pov" && this.ids.hcamSite >= 0) {
      const sb = this.ids.hcamSite * 3;
      const mb = this.ids.hcamSite * 9;
      const spos = [this.data.site_xpos[sb], this.data.site_xpos[sb + 1], this.data.site_xpos[sb + 2]];
      const smat = this.data.site_xmat;
      const fwd = [smat[mb + 0], smat[mb + 3], smat[mb + 6]];
      const up = [smat[mb + 2], smat[mb + 5], smat[mb + 8]];
      const eye = [spos[0] + 0.018 * fwd[0], spos[1] + 0.018 * fwd[1], spos[2] + 0.018 * fwd[2]];
      const gx = fwd[0] - 0.28 * up[0];
      const gy = fwd[1] - 0.28 * up[1];
      const gz = fwd[2] - 0.28 * up[2];

      this.mainCam.fov = 58;
      this.mainCam.near = 0.02;
      this.mainCam.aspect = w / h;
      this.mainCam.up.set(up[0], up[1], up[2]);
      this.mainCam.position.set(eye[0], eye[1], eye[2]);
      this.mainCam.lookAt(eye[0] + 0.5 * gx, eye[1] + 0.5 * gy, eye[2] + 0.5 * gz);
      this.mainCam.updateProjectionMatrix();

      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, w, h);
      this.renderer.render(this.scene, this.mainCam);
      return;
    }

    if (this.cameraMode === "split") {
      const leftW = Math.max(1, Math.floor(w * 0.6136));
      const rightW = Math.max(1, w - leftW);

      this._positionOrbitCamera(
        this.mainCam,
        [tx + 0.02, ty, 0.115],
        this.camAzimuth,
        this.camElevation,
        this.camDistance,
        leftW / h
      );
      this._positionOrbitCamera(
        this.sideCam,
        [tx + 0.02, ty, 0.122],
        90.0,
        -8.0,
        0.46,
        rightW / h
      );

      this.renderer.setScissorTest(true);
      this.renderer.setViewport(0, 0, leftW, h);
      this.renderer.setScissor(0, 0, leftW, h);
      this.renderer.render(this.scene, this.mainCam);

      this.renderer.setViewport(leftW, 0, rightW, h);
      this.renderer.setScissor(leftW, 0, rightW, h);
      this.renderer.render(this.scene, this.sideCam);
      this.renderer.setScissorTest(false);
      return;
    }

    this._positionOrbitCamera(
      this.mainCam,
      [tx + 0.02, ty, 0.11],
      this.camAzimuth,
      this.camElevation,
      this.camDistance,
      w / h
    );
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, w, h);
    this.renderer.render(this.scene, this.mainCam);
  }

  extractTelemetry(status = "OK") {
    const model = this.model;
    const data = this.data;
    const allowSit = this.activeKeyframe === "SIT";

    const trunkPos = [data.qpos[0], data.qpos[1], data.qpos[2]];
    const trunkRpy = quatToRpyDeg(data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]);
    const cb = this.ids.trunkBody * 3;
    const comXyz = [data.subtree_com[cb], data.subtree_com[cb + 1], data.subtree_com[cb + 2]];

    const mb = this.ids.mouthSite * 3;
    const mouthXyz = [data.site_xpos[mb], data.site_xpos[mb + 1], data.site_xpos[mb + 2]];

    let leftGrounded = false;
    let rightGrounded = false;
    const footContactXs = [];
    const nonFootContacts = [];

    const floorId = this.ids.floorGeom;
    const lfootId = this.ids.lfootGeom;
    const rfootId = this.ids.rfootGeom;
    const OBJ_BODY = this.mujoco.mjtObj.mjOBJ_BODY.value;

    for (let i = 0; i < data.ncon; i++) {
      const con = data.contact.get(i);
      if (!con) continue;
      const g1 = con.geom1;
      const g2 = con.geom2;
      if (g1 !== floorId && g2 !== floorId) continue;
      const other = g1 === floorId ? g2 : g1;
      if (other === lfootId) {
        leftGrounded = true;
        footContactXs.push(con.pos[0]);
      } else if (other === rfootId) {
        rightGrounded = true;
        footContactXs.push(con.pos[0]);
      } else {
        const bodyId = model.geom_bodyid[other];
        const bname = this.mujoco.mj_id2name(model, OBJ_BODY, bodyId) || `geom_${other}`;
        if (!nonFootContacts.includes(bname)) nonFootContacts.push(bname);
      }
    }

    const [rollDeg, pitchDeg] = trunkRpy;
    const controlledStance =
      leftGrounded && rightGrounded && Math.abs(rollDeg) < 35.0 && Math.abs(pitchDeg) < 48.0 && trunkPos[2] > 0.055;
    const headSlammed =
      nonFootContacts.includes("neck") || (nonFootContacts.includes("jaw_soft") && !controlledStance);
    const extremeTilt = Math.abs(rollDeg) > 45.0 || Math.abs(pitchDeg) > 58.0;
    const unintendedBodyHit = !allowSit && nonFootContacts.includes("trunk_base") && trunkPos[2] < 0.062;
    const fallen = Boolean(headSlammed || extremeTilt || unintendedBodyHit);

    const addSoleBounds = (grounded, gid, localVerts) => {
      if (!grounded || gid < 0 || !localVerts) return;
      const pb = gid * 3;
      const xb = gid * 9;
      const px = data.geom_xpos[pb + 0];
      const pz = data.geom_xpos[pb + 2];
      const m00 = data.geom_xmat[xb + 0];
      const m01 = data.geom_xmat[xb + 1];
      const m02 = data.geom_xmat[xb + 2];
      const m20 = data.geom_xmat[xb + 6];
      const m21 = data.geom_xmat[xb + 7];
      const m22 = data.geom_xmat[xb + 8];

      let minX = Infinity;
      let maxX = -Infinity;
      for (let v = 0; v < localVerts.length; v += 3) {
        const vx = localVerts[v];
        const vy = localVerts[v + 1];
        const vz = localVerts[v + 2];
        const wz = pz + m20 * vx + m21 * vy + m22 * vz;
        if (wz <= 0.004) {
          const wx = px + m00 * vx + m01 * vy + m02 * vz;
          if (wx < minX) minX = wx;
          if (wx > maxX) maxX = wx;
        }
      }
      if (minX <= maxX) {
        footContactXs.push(minX, maxX);
      }
    };

    addSoleBounds(leftGrounded, lfootId, this.soleVerts.left);
    addSoleBounds(rightGrounded, rfootId, this.soleVerts.right);

    let supportBounds = null;
    let toeMargin = null;
    let heelMargin = null;
    let balanceHint;

    if (footContactXs.length > 0) {
      const sxMin = Math.min(...footContactXs);
      const sxMax = Math.max(...footContactXs);
      supportBounds = [sxMin, sxMax];
      toeMargin = sxMax - comXyz[0];
      heelMargin = comXyz[0] - sxMin;
      if (fallen) balanceHint = "FALLEN";
      else if (toeMargin < 0.004) balanceHint = "CRITICAL_TOE_TIPPING_RISK (shift hips backward)";
      else if (heelMargin < 0.004) balanceHint = "CRITICAL_HEEL_TIPPING_RISK (lean torso forward)";
      else if (toeMargin < 0.008) balanceHint = "NEAR_TOE_EDGE";
      else if (heelMargin < 0.008) balanceHint = "NEAR_HEEL_EDGE";
      else balanceHint = "STABLE_CENTERED";
    } else {
      balanceHint = allowSit ? "SEATED_ON_FLOOR" : "AIRBORNE_OR_FALLEN";
    }

    const jointsRad = {};
    const saturatedJoints = {};
    for (const jname of JOINT_NAMES) {
      const info = this.ids.joints[jname];
      const qval = data.qpos[info.qadr];
      const cval = data.ctrl[info.aid];
      jointsRad[jname] = Number(qval.toFixed(4));
      const err = cval - qval;
      if (Math.abs(err) > 0.18) {
        saturatedJoints[jname] = Number(err.toFixed(3));
      }
    }

    const snap = {
      step: this.stepCounter,
      sim_time_s: Number(data.time.toFixed(2)),
      status,
      trunk_pos_m: trunkPos,
      trunk_rpy_deg: trunkRpy,
      com_xyz_m: comXyz,
      left_foot_grounded: leftGrounded,
      right_foot_grounded: rightGrounded,
      non_foot_floor_contacts: nonFootContacts,
      support_x_bounds_m: supportBounds,
      toe_margin_m: toeMargin,
      heel_margin_m: heelMargin,
      balance_hint: balanceHint,
      mouth_tip_xyz_m: mouthXyz,
      joints_rad: jointsRad,
      saturated_joints: saturatedJoints,
      fallen,
    };
    snap.observation_xml = formatObservationXml(snap);
    this.latestSnap = snap;
    return snap;
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    if (mode === "front") {
      this.camAzimuth = 180.0;
      this.camElevation = -12.0;
      this.camDistance = 0.42;
    } else if (mode === "side") {
      this.camAzimuth = 90.0;
      this.camElevation = -8.0;
      this.camDistance = 0.46;
    }
  }

  orbit(deltaAz, deltaEl) {
    this.camAzimuth = (this.camAzimuth + deltaAz + 360.0) % 360.0;
    this.camElevation = Math.max(-75.0, Math.min(5.0, this.camElevation + deltaEl));
  }

  zoom(deltaDist) {
    this.camDistance = Math.max(0.22, Math.min(1.15, this.camDistance + deltaDist));
  }

  cancelChunk() {
    this.cancelRequested = true;
  }

  reset(keyframe = "STAND") {
    this.cancelRequested = true;
    this._chunkTask = null;
    this.executingChunk = false;

    const kid = this.mujoco.mj_name2id(this.model, this.mujoco.mjtObj.mjOBJ_KEY.value, keyframe);
    if (kid < 0) throw new Error(`Unknown keyframe ${keyframe}`);
    this.mujoco.mj_resetDataKeyframe(this.model, this.data, kid);
    this.mujoco.mj_forward(this.model, this.data);
    const settleTicks = Math.round(0.3 / CTRL_DT);
    for (let t = 0; t < settleTicks; t++) {
      for (let s = 0; s < this.substeps; s++) this.mujoco.mj_step(this.model, this.data);
    }
    this.data.time = 0.0;
    this.mujoco.mj_forward(this.model, this.data);
    this.stepCounter = 0;
    this.activeKeyframe = keyframe;
    const snap = this.extractTelemetry("OK");
    this._syncThreeFromMuJoCo(snap);
    this.onTelemetry(snap, snap.fallen ? "FALLEN" : "IDLE");
    return snap;
  }

  stepChunk(waypoints) {
    this.cancelRequested = false;
    this.stepCounter += 1;
    this.executingChunk = true;
    if (this.activeKeyframe === "SIT") this.activeKeyframe = "STAND";

    const qStart = JOINT_NAMES.map((j) => this.data.ctrl[this.ids.joints[j].aid]);

    const schedule = waypoints.map((wp) => {
      const dt = Math.max(0.08, Math.min(3.0, Number(wp.dt || 0.5)));
      const nCtrl = Math.max(1, Math.round(dt / CTRL_DT));
      const joints = wp.joints || {};
      return { nCtrl, joints };
    });

    return new Promise((resolve) => {
      this._chunkTask = {
        schedule,
        wpIdx: 0,
        k: 0,
        qStart,
        qGoal: null,
        postFallTicks: 0,
        status: "OK",
        resolve,
      };
    });
  }

  _advanceChunkTick() {
    const task = this._chunkTask;
    if (!task) return;

    if (task.postFallTicks > 0) {
      for (let s = 0; s < this.substeps; s++) this.mujoco.mj_step(this.model, this.data);
      task.postFallTicks -= 1;
      const snap = this.extractTelemetry("ABORTED_FALL");
      if (task.postFallTicks === 0) {
        this._chunkTask = null;
        this.executingChunk = false;
        this.onTelemetry(snap, "FALLEN");
        task.resolve(snap);
      }
      return;
    }

    if (this.cancelRequested) {
      for (const jname of JOINT_NAMES) {
        const info = this.ids.joints[jname];
        this.data.ctrl[info.aid] = this.data.qpos[info.qadr];
      }
      const snap = this.extractTelemetry("INTERRUPTED");
      this._chunkTask = null;
      this.executingChunk = false;
      this.onTelemetry(snap, snap.fallen ? "FALLEN" : "IDLE");
      task.resolve(snap);
      return;
    }

    if (task.wpIdx >= task.schedule.length) {
      const snap = this.extractTelemetry(task.status);
      this._chunkTask = null;
      this.executingChunk = false;
      this.onTelemetry(snap, snap.fallen ? "FALLEN" : "IDLE");
      task.resolve(snap);
      return;
    }

    const cur = task.schedule[task.wpIdx];
    if (task.qGoal === null) {
      task.qGoal = task.qStart.map((val, idx) => {
        const jname = JOINT_NAMES[idx];
        if (Object.prototype.hasOwnProperty.call(cur.joints, jname)) {
          const info = this.ids.joints[jname];
          return Math.max(info.lo, Math.min(info.hi, Number(cur.joints[jname])));
        }
        return val;
      });
    }

    task.k += 1;
    const tau = task.k / cur.nCtrl;
    const alpha = 3.0 * tau * tau - 2.0 * tau * tau * tau;

    for (let idx = 0; idx < JOINT_NAMES.length; idx++) {
      const jname = JOINT_NAMES[idx];
      const cmd = (1.0 - alpha) * task.qStart[idx] + alpha * task.qGoal[idx];
      this.data.ctrl[this.ids.joints[jname].aid] = cmd;
    }

    for (let s = 0; s < this.substeps; s++) {
      this.mujoco.mj_step(this.model, this.data);
    }

    const snap = this.extractTelemetry("OK");
    if (snap.fallen) {
      task.status = "ABORTED_FALL";
      task.postFallTicks = 10;
      return;
    }

    if (task.k >= cur.nCtrl) {
      task.qStart = task.qGoal;
      task.qGoal = null;
      task.k = 0;
      task.wpIdx += 1;
    }
  }

  _startRenderAndPhysicsLoop() {
    let lastTs = performance.now();
    let physAccum = 0.0;
    let lastTelemUi = 0.0;

    const frame = (now) => {
      this._rafId = requestAnimationFrame(frame);
      const dtWall = Math.min(0.08, Math.max(0.0, (now - lastTs) / 1000.0));
      lastTs = now;
      physAccum += dtWall;

      let stepped = false;
      while (physAccum >= CTRL_DT) {
        physAccum -= CTRL_DT;
        stepped = true;
        if (this._chunkTask) {
          this._advanceChunkTick();
        } else if (!this.latestSnap || !this.latestSnap.fallen) {
          const prevTime = this.data.time;
          for (let s = 0; s < this.substeps; s++) {
            this.mujoco.mj_step(this.model, this.data);
          }
          this.data.time = prevTime;
        }
      }

      if (stepped) {
        const snap = this.extractTelemetry(this._chunkTask ? "OK" : this.latestSnap ? this.latestSnap.status : "OK");
        this._syncThreeFromMuJoCo(snap);
        if (now - lastTelemUi >= 110) {
          lastTelemUi = now;
          const st = this.executingChunk ? "EXECUTING_CHUNK" : snap.fallen ? "FALLEN" : "IDLE";
          this.onTelemetry(snap, st);
        }
      }

      this._renderFrame();
    };

    this._rafId = requestAnimationFrame(frame);
  }
}
