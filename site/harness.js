/* Pre-printed System Contract, Action Chunk Parser, Offline Demo Policy, and Direct Browser->Gemini Client */

export const JOINT_NAMES = [
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
];

const JOINT_SET = new Set(JOINT_NAMES);

export const STAND_RAW = {
  left_hip_yaw: 0.0,
  left_hip_roll: -0.0873,
  left_hip_pitch: -0.4579,
  left_knee: -0.0049,
  left_ankle: 0.453,
  neck_pitch: 0.3491,
  head_pitch: 0.3491,
  head_yaw: 0.0,
  head_roll: 0.0,
  right_hip_yaw: 0.0,
  right_hip_roll: 0.0873,
  right_hip_pitch: 0.4579,
  right_knee: 0.0049,
  right_ankle: -0.453,
};

export const STOP_SEQUENCES = ["</action_chunk>", "</done>"];
export const ACTION_RE = /<action_chunk>\s*([\s\S]*?)\s*<\/action_chunk>/;
export const DONE_RE = /<done>\s*([\s\S]*?)\s*<\/done>/;

export const INTERACTIVE_SYSTEM_PROMPT = `You are zero-shot controlling a physical simulated robot — Pollen Robotics' **MicroDuck** — inside a 50 Hz MuJoCo physics simulation via an Always-On Text Harness.

## 1. Robot Morphology & Physics
- **Body:** ~800 g, ~25 cm tall bipedal duck standing on two flat webbed feet (\`z = 0.000 m\` is the floor; \`+X\` is forward, \`+Y\` is left, \`+Z\` is up).
- **Actuators:** 14 Dynamixel XL330 position-controlled servos (\`kp = 3.2 N·m/rad\`, max stall torque \`±0.96 N·m\`). Because servos have realistic compliance and torque limits, gravity causes slight tracking sag under heavy cantilevered loads.
- **Critical Mass Property:** Your head/beak assembly weighs **189 g (~24% of your entire body mass!)**, mounted on a cantilevered neck. Pitching your neck/head forward shifts your Center of Mass (\`com_xyz_m\` X) strongly toward your front toes. If \`com_xyz_m\` X crosses past \`support_x_span_m\` max, you will faceplant!

## 2. Joint Space & Mirrored CAD Servo Conventions (14-D Radians)
Standing \`STAND\` baseline values and hardware limits \`[min, max]\` in radians:

### Neck & Head (4 joints — Note: \`neck_pitch\` and \`head_pitch\` have OPPOSITE pitch axes!)
- \`neck_pitch\`  range \`[-1.57, +1.05]\`, STAND = \`+0.349\` (**DECREASING / negative** pitches the neck base **FORWARD & DOWN**; increasing pulls neck UP & BACK)
- \`head_pitch\`  range \`[-1.57, +1.57]\`, STAND = \`+0.349\` (**INCREASING / more positive** e.g. \`+0.8..+1.3\` tilts the beak **DOWN toward the floor**; decreasing tilts beak UP)
- \`head_yaw\`    range \`[-2.97, +2.97]\`, STAND = \`0.000\`  (\`+\` turns beak LEFT toward \`+Y\`; \`-\` turns beak RIGHT toward \`-Y\`)
- \`head_roll\`   range \`[-0.44, +0.44]\`, STAND = \`0.000\`  (tilts head ear-to-shoulder)

### Legs (10 joints — IMPORTANT: Left & Right XL330 Servo Horns Are Mounted Mirrored!)
Because left and right leg servos face opposite outward directions in CAD, **symmetric bilateral leg motions require OPPOSITE signs on \`left_*\` vs \`right_*\` pitch, knee, ankle, and roll**:
- \`left_hip_yaw\`   \`[-0.44, +0.52]\`, STAND = \`0.000\`  | \`right_hip_yaw\`   \`[-0.52, +0.44]\`, STAND = \`0.000\`
- \`left_hip_roll\`  \`[-0.38, +0.38]\`, STAND = \`-0.087\` | \`right_hip_roll\`  \`[-0.38, +0.38]\`, STAND = \`+0.087\`
- \`left_hip_pitch\` \`[-1.57, +1.57]\`, STAND = \`-0.458\` | \`right_hip_pitch\` \`[-1.57, +1.57]\`, STAND = \`+0.458\`
- \`left_knee\`      \`[-1.57, +1.57]\`, STAND = \`0.000\`  | \`right_knee\`      \`[-1.57, +1.57]\`, STAND = \`0.000\`
- \`left_ankle\`     \`[-1.57, +1.57]\`, STAND = \`+0.453\` | \`right_ankle\`     \`[-1.57, +1.57]\`, STAND = \`-0.453\`

**Verified Avian 3-Link Flat-Foot Law (Measured from MuJoCo \`data.xaxis\`):**
MicroDuck has an **avian backward-hock leg** (\`left_hip_pitch\` axis \`+Y\`, \`left_knee\` axis \`-Y\`, \`left_ankle\` axis \`+Y\`), so keeping both soles flat on the floor requires \`left_hip_pitch - left_knee + left_ankle ≈ 0\` (and opposite signs on the right leg):
1. **Vertical Flat-Foot Squat by \`δ > 0\` (e.g. \`δ = 0.20..0.65 rad\` lowers trunk Z straight down with zero tilt):**
   - \`left_hip_pitch = -0.458 + δ\`   | \`right_hip_pitch = +0.458 - δ\`
   - \`left_knee      = +2 * δ\`       | \`right_knee      = -2 * δ\`
   - \`left_ankle     = +0.453 + δ\`   | \`right_ankle     = -0.453 - δ\`
2. **Pelvic CoM Pullback by \`β > 0\` (e.g. \`β = 0.14..0.28 rad\` tilts shins backward \`-X\` to counter-balance bowing the heavy 189 g head):**
   - \`left_hip_pitch = -0.458 + δ - β\` | \`right_hip_pitch = +0.458 - δ + β\`
   - \`left_ankle     = +0.453 + δ + β\` | \`right_ankle     = -0.453 - δ - β\`
   (Scale \`β\` back down to \`0\` in lockstep when retracting the head upright!)

## 3. Closed-Loop Control Strategy
- Prefer **short, incremental action chunks (1–2 waypoints, \`dt\` = 0.35s–0.85s)** when performing delicate balance or deep bowing so you can inspect the resulting \`<observation>\` and check \`mouth_tip_xyz_m\` and \`toe_margin\` in closed loop!
- Always check \`toe_margin\` and \`heel_margin\` before lowering your beak further. Keep \`toe_margin > 0.012 m\` whenever standing on your feet.

## 4. Strict Harness Output Protocol
On EVERY turn, emit EITHER an \`<action_chunk>\` OR \`<done>\`:

Option A — Execute a 14-D servo trajectory chunk (harness stops generation at \`</action_chunk>\`, steps MuJoCo at 50 Hz, and replies with \`<observation>\`):
<action_chunk>
[
  {
    "dt": 0.60,
    "joints": {
      "left_hip_pitch": -0.16,
      "right_hip_pitch": 0.16,
      "left_knee": 0.60,
      "right_knee": -0.60,
      "left_ankle": 0.75,
      "right_ankle": -0.75
    }
  }
]
</action_chunk>

*(Note: Any joint omitted from \`"joints"\` smoothly holds its current target angle.)*

Option B — Declare task completion ONLY AFTER \`<observation>\` confirms all goal criteria (including returning upright if requested) are satisfied:
<done>Concise summary of the completed motion and final state.</done>

## 5. Verified Zero-RL Motion Reference (50 Hz MuJoCo Tested)
- **CRITICAL BOWING & RECOVERY RULE:** NEVER pitch \`neck_pitch < -0.10\` with straight legs (\`knee ≈ 0\`) or snap legs straight before retracting the heavy head!
  1. **Bow Down (1-wp chunk):** \`{"dt": 0.85, "joints": {"left_hip_pitch": -0.12, "right_hip_pitch": 0.12, "left_knee": 1.16, "right_knee": -1.16, "left_ankle": 1.27, "right_ankle": -1.27, "neck_pitch": -0.35, "head_pitch": 1.38}}\`
  2. **Inspect \`<observation>\`**, then **Lockstep Return Upright (2-wp chunk — retract head first as legs unbend!):**
     WP1: \`{"dt": 0.70, "joints": {"left_hip_pitch": -0.25, "right_hip_pitch": 0.25, "left_knee": 0.64, "right_knee": -0.64, "left_ankle": 0.88, "right_ankle": -0.88, "neck_pitch": 0.10, "head_pitch": 0.75}}\`
     WP2: \`{"dt": 0.85, "joints": {"left_hip_pitch": -0.458, "right_hip_pitch": 0.458, "left_knee": 0.0, "right_knee": 0.0, "left_ankle": 0.453, "right_ankle": -0.453, "neck_pitch": 0.349, "head_pitch": 0.349}}\`

- **Forward Waddle Step (~2.5 cm \`+X\` per 5-waypoint \`<action_chunk>\`; emit ONE step chunk per turn and check \`<observation>\` between steps!):**
  - WP1 (\`dt: 0.28\`): \`{"left_hip_roll": 0.19, "right_hip_roll": 0.37, "left_hip_pitch": -0.23, "left_knee": 0.20, "left_ankle": 0.42, "right_hip_pitch": 0.30, "right_knee": -0.76, "right_ankle": -1.05, "neck_pitch": 0.35, "head_pitch": 0.35, "head_yaw": 0.35, "head_roll": -0.25}\`
  - WP2 (\`dt: 0.28\`): \`{"left_hip_roll": -0.09, "right_hip_roll": 0.09, "left_hip_pitch": -0.18, "left_knee": 0.30, "left_ankle": 0.47, "right_hip_pitch": 0.49, "right_knee": -0.30, "right_ankle": -0.78, "head_yaw": 0.0, "head_roll": 0.0}\`
  - WP3 (\`dt: 0.28\`): \`{"left_hip_roll": -0.37, "right_hip_roll": -0.19, "left_hip_pitch": -0.30, "left_knee": 0.76, "left_ankle": 1.05, "right_hip_pitch": 0.23, "right_knee": -0.20, "right_ankle": -0.42, "head_yaw": -0.35, "head_roll": 0.25}\`
  - WP4 (\`dt: 0.28\`): \`{"left_hip_roll": -0.09, "right_hip_roll": 0.09, "left_hip_pitch": -0.49, "left_knee": 0.30, "left_ankle": 0.78, "right_hip_pitch": 0.18, "right_knee": -0.30, "right_ankle": -0.47, "head_yaw": 0.0, "head_roll": 0.0}\`
  - WP5 (\`dt: 0.30\`): \`{"left_hip_roll": -0.09, "right_hip_roll": 0.09, "left_hip_pitch": -0.36, "left_knee": 0.20, "left_ankle": 0.55, "right_hip_pitch": 0.36, "right_knee": -0.20, "right_ankle": -0.55, "head_yaw": 0.0, "head_roll": 0.0}\`
  *(For a backward step, keep the same rock/knee sequence but swap swing/stance hip_pitch & ankle forward/back offsets.)*

- **Turn In Place (~15° per 4-waypoint \`<action_chunk>\` — note both \`left_hip_yaw\` and \`right_hip_yaw\` share world axis \`-Z\`):**
  - **Turn LEFT (\`+yaw\`):**
    WP1 (\`dt: 0.26\`): \`{"right_hip_roll": -0.19, "right_hip_pitch": 0.36, "right_knee": -0.20, "right_ankle": -0.55, "right_hip_yaw": 0.0, "left_hip_roll": -0.37, "left_hip_pitch": -0.08, "left_knee": 0.76, "left_ankle": 0.83, "left_hip_yaw": 0.32, "head_yaw": -0.30, "head_roll": 0.25}\`
    WP2 (\`dt: 0.26\`): \`{"right_hip_roll": -0.13, "right_hip_pitch": 0.34, "right_knee": -0.24, "right_ankle": -0.57, "right_hip_yaw": -0.32, "left_hip_roll": -0.09, "left_hip_pitch": -0.31, "left_knee": 0.30, "left_ankle": 0.60, "left_hip_yaw": 0.32, "head_yaw": 0.0, "head_roll": 0.0}\`
    WP3 (\`dt: 0.26\`): \`{"right_hip_roll": 0.37, "right_hip_pitch": 0.08, "right_knee": -0.76, "right_ankle": -0.83, "right_hip_yaw": 0.0, "left_hip_roll": 0.19, "left_hip_pitch": -0.36, "left_knee": 0.20, "left_ankle": 0.55, "left_hip_yaw": -0.16, "head_yaw": 0.30, "head_roll": -0.25}\`
    WP4 (\`dt: 0.30\`): \`{"left_hip_yaw": 0.0, "right_hip_yaw": 0.0, "left_hip_roll": -0.09, "right_hip_roll": 0.09, "left_hip_pitch": -0.36, "right_hip_pitch": 0.36, "left_knee": 0.20, "right_knee": -0.20, "left_ankle": 0.55, "right_ankle": -0.55, "head_yaw": 0.0, "head_roll": 0.0}\`
  - **Turn RIGHT (\`-yaw\`):**
    WP1 (\`dt: 0.26\`): \`{"left_hip_roll": 0.19, "left_hip_pitch": -0.36, "left_knee": 0.20, "left_ankle": 0.55, "left_hip_yaw": 0.0, "right_hip_roll": 0.37, "right_hip_pitch": 0.08, "right_knee": -0.76, "right_ankle": -0.83, "right_hip_yaw": -0.32, "head_yaw": 0.30, "head_roll": -0.25}\`
    WP2 (\`dt: 0.26\`): \`{"left_hip_roll": 0.13, "left_hip_pitch": -0.34, "left_knee": 0.24, "left_ankle": 0.57, "left_hip_yaw": 0.32, "right_hip_roll": 0.09, "right_hip_pitch": 0.31, "right_knee": -0.30, "right_ankle": -0.60, "right_hip_yaw": -0.32, "head_yaw": 0.0, "head_roll": 0.0}\`
    WP3 (\`dt: 0.26\`): \`{"left_hip_roll": -0.37, "left_hip_pitch": -0.08, "left_knee": 0.76, "left_ankle": 0.83, "left_hip_yaw": 0.0, "right_hip_roll": -0.19, "right_hip_pitch": 0.36, "right_knee": -0.20, "right_ankle": -0.55, "right_hip_yaw": 0.16, "head_yaw": -0.30, "head_roll": 0.25}\`
    WP4 (\`dt: 0.30\`): \`{"left_hip_yaw": 0.0, "right_hip_yaw": 0.0, "left_hip_roll": -0.09, "right_hip_roll": 0.09, "left_hip_pitch": -0.36, "right_hip_pitch": 0.36, "left_knee": 0.20, "right_knee": -0.20, "left_ankle": 0.55, "right_ankle": -0.55, "head_yaw": 0.0, "head_roll": 0.0}\`

## 6. Interactive Session Behavior
- A human user is interacting with you live across multiple \`<instruction>\` messages while watching your physical MicroDuck body in real time.
- Start from your **actual current pose** reported in the latest \`<observation>\`!
- If the user asks you to **hold** a pose (e.g. "squat down and hold", "look left"), stay in that final pose and emit \`<done>\` once \`<observation>\` confirms it. If they ask for a transient gesture (e.g. "do a bow", "nod twice", "shake your head"), return smoothly to an upright stance before emitting \`<done>\`.
- If the user asks a pure status/proprioception question requiring zero motion (e.g. "where is your beak right now?"), answer directly inside \`<done>...</done>\` using the latest \`<observation>\`.
- Write a friendly, concise, telemetry-grounded message inside \`<done>...</done>\` summarizing what you physically did (mentioning relevant physical measurements like beak height, yaw, or distance moved).`;

export function parseActionChunk(rawBody) {
  let cleaned = String(rawBody || "").trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, "").replace(/\s*```$/, "").trim();
  }
  let parsed;
  try {
    parsed = JSON.parse(cleaned);
  } catch (err) {
    throw new Error(`JSON parse error: ${err.message}`);
  }
  if (!Array.isArray(parsed) || parsed.length === 0) {
    throw new Error("Expected a non-empty JSON array of waypoint objects.");
  }
  if (parsed.length > 12) {
    throw new Error(`Too many waypoints in one chunk (${parsed.length} > 12).`);
  }

  const waypoints = [];
  parsed.forEach((item, idx) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      throw new Error(`Waypoint [${idx}] must be an object with 'dt' and 'joints'.`);
    }
    const dt = Number(item.dt ?? 0.5);
    if (!Number.isFinite(dt) || dt <= 0) {
      throw new Error(`Waypoint [${idx}] has invalid dt=${item.dt}.`);
    }
    const jointsRaw = item.joints;
    if (!jointsRaw || typeof jointsRaw !== "object" || Array.isArray(jointsRaw)) {
      throw new Error(`Waypoint [${idx}] must include a 'joints' object.`);
    }
    const joints = {};
    for (const [k, v] of Object.entries(jointsRaw)) {
      if (!JOINT_SET.has(k)) {
        throw new Error(`Waypoint [${idx}] has unknown joint '${k}'. Valid joints: ${JOINT_NAMES.join(", ")}`);
      }
      const rad = Number(v);
      if (!Number.isFinite(rad)) {
        throw new Error(`Waypoint [${idx}] joint '${k}' value '${v}' is not a finite number.`);
      }
      joints[k] = rad;
    }
    waypoints.push({ dt, joints });
  });
  return waypoints;
}

export function reattachStopTag(text, stopSequences, explicitStop = null) {
  const trimmed = text.trimEnd();
  for (const seq of stopSequences) {
    if (trimmed.endsWith(seq)) {
      return { text: trimmed, matchedStop: seq };
    }
  }
  if (explicitStop && stopSequences.includes(explicitStop)) {
    return { text: text + explicitStop, matchedStop: explicitStop };
  }
  for (const seq of stopSequences) {
    if (seq.startsWith("</") && seq.endsWith(">")) {
      const openTag = "<" + seq.slice(2, -1);
      const lastOpen = text.lastIndexOf(openTag);
      const lastClose = text.lastIndexOf(seq);
      if (lastOpen !== -1 && lastClose < lastOpen) {
        return { text: text + seq, matchedStop: seq };
      }
    }
  }
  return { text, matchedStop: null };
}

/* Direct Browser -> Google Gemini API Client (API key never touches any third-party server) */

const PREFERRED_MODELS = [
  "gemini-3.5-flash",
  "gemini-flash-latest",
  "gemini-3.8-flash",
  "gemini-pro-latest",
  "gemini-3-flash-preview",
];

export async function listGeminiModelsDirect(apiKey) {
  const resp = await fetch("https://generativelanguage.googleapis.com/v1beta/models", {
    method: "GET",
    mode: "cors",
    credentials: "omit",
    referrerPolicy: "no-referrer",
    headers: { "x-goog-api-key": apiKey },
  });
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`;
    try {
      const errJson = await resp.json();
      detail = (errJson.error && errJson.error.message) || detail;
    } catch {
      /* ignore */
    }
    throw new Error(`Invalid key or Gemini error (${resp.status}): ${detail}`);
  }
  const data = await resp.json();
  const rawModels = (data && data.models) || [];
  const textModels = rawModels
    .filter(
      (m) =>
        Array.isArray(m.supportedGenerationMethods) &&
        m.supportedGenerationMethods.includes("generateContent") &&
        typeof m.name === "string" &&
        m.name.includes("gemini") &&
        !/tts|image|transcribe|embedding|imagen|veo/i.test(m.name)
    )
    .map((m) => m.name.replace(/^models\//, ""));

  const bestDefault = PREFERRED_MODELS.find((p) => textModels.includes(p)) || textModels[0] || "gemini-flash-latest";
  return { models: textModels, bestDefault };
}

export async function callGeminiDirect({ apiKey, model, systemPrompt, messages, stopSequences = STOP_SEQUENCES, signal }) {
  const t0 = performance.now();
  let modelId = String(model || "gemini-3.5-flash").replace(/^models\//, "").trim();

  const contents = messages.map((m) => ({
    role: m.role === "assistant" ? "model" : "user",
    parts: [{ text: m.content }],
  }));

  function extractVisibleText(data) {
    const cands = (data && data.candidates) || [];
    if (!cands.length) return "";
    const parts = (cands[0].content && cands[0].content.parts) || [];
    return parts
      .filter((p) => !p.thought)
      .map((p) => p.text || "")
      .join("");
  }

  async function postGenerate(targetModel, genConfig) {
    const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(targetModel)}:generateContent`;
    return fetch(url, {
      method: "POST",
      mode: "cors",
      credentials: "omit",
      referrerPolicy: "no-referrer",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": apiKey,
      },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: systemPrompt }] },
        contents,
        generationConfig: genConfig,
      }),
      signal,
    });
  }

  async function postWithModelFallback(genConfig) {
    let resp = await postGenerate(modelId, genConfig);
    if (resp.status === 404) {
      const discovered = await listGeminiModelsDirect(apiKey);
      if (discovered.bestDefault && discovered.bestDefault !== modelId) {
        modelId = discovered.bestDefault;
        resp = await postGenerate(modelId, genConfig);
      }
    }
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const errJson = await resp.json();
        detail = (errJson.error && errJson.error.message) || detail;
      } catch {
        /* ignore */
      }
      throw new Error(`Gemini API error (${resp.status}): ${detail}`);
    }
    return resp.json();
  }

  let data = await postWithModelFallback({
    temperature: 0.2,
    maxOutputTokens: 8192,
    stopSequences,
  });
  let rawText = extractVisibleText(data);

  // Fallback if a thinking model matched a stop delimiter inside internal thought parts
  if (stopSequences.length > 0 && !rawText.trim()) {
    data = await postWithModelFallback({
      temperature: 0.2,
      maxOutputTokens: 8192,
    });
    rawText = extractVisibleText(data);
  }

  let earliestIdx = null;
  let earliestSeq = null;
  for (const seq of stopSequences) {
    const pos = rawText.indexOf(seq);
    if (pos !== -1 && (earliestIdx === null || pos < earliestIdx)) {
      earliestIdx = pos;
      earliestSeq = seq;
    }
  }
  if (earliestIdx !== null && earliestSeq !== null) {
    rawText = rawText.slice(0, earliestIdx + earliestSeq.length);
  }

  const { text: fullText, matchedStop } = reattachStopTag(rawText, stopSequences, earliestSeq);
  return {
    text: fullText,
    matchedStop,
    latencyS: (performance.now() - t0) / 1000,
    model: modelId,
  };
}

/* Offline Closed-Loop Demo Policy */

function avianStance(deltaSquat, betaPullback = 0.0) {
  const r = (x) => Number(x.toFixed(4));
  return {
    left_hip_yaw: 0.0,
    right_hip_yaw: 0.0,
    left_hip_roll: -0.0873,
    right_hip_roll: 0.0873,
    left_hip_pitch: r(-0.4579 + deltaSquat - betaPullback),
    right_hip_pitch: r(0.4579 - deltaSquat + betaPullback),
    left_knee: r(2.0 * deltaSquat),
    right_knee: r(-2.0 * deltaSquat),
    left_ankle: r(0.453 + deltaSquat + betaPullback),
    right_ankle: r(-0.453 - deltaSquat - betaPullback),
  };
}

function waddleLeg(side, delta, stride = 0.0, roll = 0.0, yaw = 0.0) {
  const r2 = (x) => Number(x.toFixed(2));
  if (side === "left") {
    return {
      left_hip_yaw: r2(yaw),
      left_hip_roll: r2(-0.087 + roll),
      left_hip_pitch: r2(-0.458 + delta - stride),
      left_knee: r2(2.0 * delta),
      left_ankle: r2(0.453 + delta + stride),
    };
  }
  return {
    right_hip_yaw: r2(yaw),
    right_hip_roll: r2(0.087 + roll),
    right_hip_pitch: r2(0.458 - delta + stride),
    right_knee: r2(-2.0 * delta),
    right_ankle: r2(-0.453 - delta - stride),
  };
}

function parseLatestObs(messages) {
  let latestXml = "";
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user" && messages[i].content.includes("<observation")) {
      latestXml = messages[i].content;
      break;
    }
  }
  const vec3 = (name, def) => {
    const m = latestXml.match(new RegExp(`${name}:\\s*\\[([+-]?[0-9.]+),\\s*([+-]?[0-9.]+),\\s*([+-]?[0-9.]+)\\]`));
    return m ? [Number(m[1]), Number(m[2]), Number(m[3])] : def;
  };
  const toeM = latestXml.match(/toe_margin=([+-]?[0-9.]+)m/);
  const rpyM = latestXml.match(/trunk_rpy_deg:\s*\[roll=([+-]?[0-9.]+),\s*pitch=([+-]?[0-9.]+),\s*yaw=([+-]?[0-9.]+)\]/);
  return {
    trunkPos: vec3("trunk_pos_m", [0, 0, 0.116]),
    mouthTip: vec3("mouth_tip_xyz_m", [0.074, 0, 0.218]),
    toeMargin: toeM ? Number(toeM[1]) : 0.03,
    yawDeg: rpyM ? Number(rpyM[3]) : 0.0,
  };
}

export function generateDemoStep(messages) {
  const t0 = performance.now();
  const obs = parseLatestObs(messages);
  let instr = "";
  let doneChunks = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role === "assistant" && m.content.includes("<action_chunk>")) doneChunks += 1;
    if (m.role === "user") {
      const im = m.content.match(/<(?:instruction|goal)>([\s\S]*?)<\/(?:instruction|goal)>/);
      if (im) {
        instr = im[1].trim();
        break;
      }
    }
  }

  const low = instr.toLowerCase();
  const [tx, ty, tz] = obs.trunkPos;
  const [, , mz] = obs.mouthTip;
  let nReq = 1;
  if (["three", "3"].some((w) => low.includes(w))) nReq = 3;
  else if (["two", "twice", "couple", "2"].some((w) => low.includes(w))) nReq = 2;

  const mkChunk = (wps) => ({
    text: `<action_chunk>\n${JSON.stringify(wps, null, 2)}\n</action_chunk>`,
    matchedStop: "</action_chunk>",
    latencyS: (performance.now() - t0) / 1000,
    model: "closed-loop-demo",
  });
  const mkDone = (summary) => ({
    text: `<done>${summary}</done>`,
    matchedStop: "</done>",
    latencyS: (performance.now() - t0) / 1000,
    model: "closed-loop-demo",
  });

  if (low.includes("bow")) {
    if (doneChunks === 0) {
      const legs = avianStance(0.58, 0.24);
      return mkChunk([{ dt: 0.85, joints: { ...legs, neck_pitch: -0.35, head_pitch: 1.38 } }]);
    }
    if (doneChunks === 1) {
      const mid = avianStance(0.28, 0.10);
      return mkChunk([
        { dt: 0.65, joints: { ...mid, neck_pitch: 0.10, head_pitch: 0.75 } },
        { dt: 0.75, joints: STAND_RAW },
      ]);
    }
    return mkDone(
      `Completed a counter-balanced deep bow and returned cleanly to STAND (beak z=${mz.toFixed(3)} m, toe_margin=${Math.round(obs.toeMargin * 1000)} mm).`
    );
  }

  if (["waddle", "walk", "forward", "ahead", "step"].some((w) => low.includes(w))) {
    const back = ["back", "backward", "reverse"].some((w) => low.includes(w));
    const s = back ? -0.18 : 0.22;
    const steps = Math.max(nReq, low.includes("walk") || low.includes("waddle") ? 2 : 1);
    if (doneChunks < steps) {
      return mkChunk([
        { dt: 0.28, joints: { ...waddleLeg("left", 0.10, -s * 0.6, +0.28), ...waddleLeg("right", 0.38, +s, +0.28), neck_pitch: 0.35, head_pitch: 0.35, head_yaw: 0.35, head_roll: -0.25 } },
        { dt: 0.28, joints: { ...waddleLeg("left", 0.15, -s * 0.6, 0.0), ...waddleLeg("right", 0.15, +s * 0.8, 0.0), head_yaw: 0.0, head_roll: 0.0 } },
        { dt: 0.28, joints: { ...waddleLeg("left", 0.38, +s, -0.28), ...waddleLeg("right", 0.10, -s * 0.6, -0.28), head_yaw: -0.35, head_roll: +0.25 } },
        { dt: 0.28, joints: { ...waddleLeg("left", 0.15, +s * 0.8, 0.0), ...waddleLeg("right", 0.15, -s * 0.6, 0.0), head_yaw: 0.0, head_roll: 0.0 } },
        { dt: 0.30, joints: { ...waddleLeg("left", 0.10), ...waddleLeg("right", 0.10), head_yaw: 0.0, head_roll: 0.0 } },
      ]);
    }
    const dir = back ? "backward" : "forward";
    return mkDone(
      `Waddled ${steps} step(s) ${dir} cleanly — trunk is at [${tx >= 0 ? "+" : ""}${tx.toFixed(3)}, ${ty >= 0 ? "+" : ""}${ty.toFixed(3)}, ${tz.toFixed(3)}] m (yaw ${obs.yawDeg >= 0 ? "+" : ""}${obs.yawDeg.toFixed(1)}°).`
    );
  }

  if (["turn", "rotate", "spin"].some((w) => low.includes(w))) {
    const goRight = low.includes("right") && !low.includes("left");
    if (doneChunks < nReq) {
      if (goRight) {
        return mkChunk([
          { dt: 0.26, joints: { ...waddleLeg("left", 0.10, 0, +0.28, 0.0), ...waddleLeg("right", 0.38, 0, +0.28, -0.32), head_yaw: 0.30, head_roll: -0.25 } },
          { dt: 0.26, joints: { ...waddleLeg("left", 0.12, 0, +0.22, +0.32), ...waddleLeg("right", 0.15, 0, 0.0, -0.32), head_yaw: 0.0, head_roll: 0.0 } },
          { dt: 0.26, joints: { ...waddleLeg("left", 0.38, 0, -0.28, 0.0), ...waddleLeg("right", 0.10, 0, -0.28, +0.16), head_yaw: -0.30, head_roll: +0.25 } },
          { dt: 0.30, joints: { ...waddleLeg("left", 0.10), ...waddleLeg("right", 0.10), head_yaw: 0.0, head_roll: 0.0 } },
        ]);
      }
      return mkChunk([
        { dt: 0.26, joints: { ...waddleLeg("right", 0.10, 0, -0.28, 0.0), ...waddleLeg("left", 0.38, 0, -0.28, +0.32), head_yaw: -0.30, head_roll: +0.25 } },
        { dt: 0.26, joints: { ...waddleLeg("right", 0.12, 0, -0.22, -0.32), ...waddleLeg("left", 0.15, 0, 0.0, +0.32), head_yaw: 0.0, head_roll: 0.0 } },
        { dt: 0.26, joints: { ...waddleLeg("right", 0.38, 0, +0.28, 0.0), ...waddleLeg("left", 0.10, 0, +0.28, -0.16), head_yaw: +0.30, head_roll: -0.25 } },
        { dt: 0.30, joints: { ...waddleLeg("left", 0.10), ...waddleLeg("right", 0.10), head_yaw: 0.0, head_roll: 0.0 } },
      ]);
    }
    return mkDone(`Turned ${goRight ? "right" : "left"} in place — heading yaw is now ${obs.yawDeg >= 0 ? "+" : ""}${obs.yawDeg.toFixed(1)}° with both feet flat.`);
  }

  if (low.includes("squat") || low.includes("crouch")) {
    if (doneChunks === 0) {
      const legs = avianStance(0.48, 0.06);
      return mkChunk([{ dt: 0.70, joints: { ...legs, neck_pitch: 0.25, head_pitch: 0.45 } }]);
    }
    return mkDone(`Holding a balanced flat-foot avian squat at trunk z=${tz.toFixed(3)} m (toe_margin=+${Math.round(obs.toeMargin * 1000)} mm).`);
  }

  if (["stand", "upright", "reset"].some((w) => low.includes(w))) {
    if (doneChunks === 0) {
      return mkChunk([{ dt: tz < 0.112 ? 0.75 : 0.55, joints: STAND_RAW }]);
    }
    return mkDone(`Standing tall and centered at trunk z=${tz.toFixed(3)} m, beak z=${mz.toFixed(3)} m.`);
  }

  if (["look", "nod", "shake", "tilt", "wave", "hello", "hi"].some((w) => low.includes(w))) {
    if (doneChunks === 0) {
      const brace = avianStance(0.16, 0.05);
      return mkChunk([
        { dt: 0.55, joints: { ...brace, head_yaw: 0.55, head_roll: 0.12 } },
        { dt: 0.65, joints: { ...brace, head_yaw: -0.55, head_roll: -0.12 } },
        { dt: 0.45, joints: { ...brace, head_yaw: 0.0, head_roll: 0.0, head_pitch: 0.85 } },
        { dt: 0.45, joints: { ...brace, head_pitch: 0.25 } },
        { dt: 0.50, joints: STAND_RAW },
      ]);
    }
    return mkDone(`Looked left and right, nodded, and centered gaze (beak z=${mz.toFixed(3)} m).`);
  }

  return mkDone(
    `Ready (Demo mode — connect a Gemini API key for open-ended instructions). Trunk at [${tx >= 0 ? "+" : ""}${tx.toFixed(3)}, ${ty >= 0 ? "+" : ""}${ty.toFixed(3)}, ${tz.toFixed(3)}] m, beak z=${mz.toFixed(3)} m.`
  );
}
