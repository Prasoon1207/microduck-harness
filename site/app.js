/* Static Browser-Native MicroDuck Harness Controller (RAM-only BYOK + 50 Hz WASM MuJoCo) */

import {
  ACTION_RE,
  DONE_RE,
  INTERACTIVE_SYSTEM_PROMPT,
  STOP_SEQUENCES,
  callGeminiDirect,
  generateDemoStep,
  listGeminiModelsDirect,
  parseActionChunk,
} from "./harness.js";
import { MicroDuckWasmSim } from "./sim.js";

(() => {
  "use strict";

  // Private volatile closure variable — never exposed on window or written to Web Storage / cookies.
  let _ephemeralGeminiKey = "";
  let _authMode = "none"; // "none" | "gemini" | "demo"

  const el = {
    simStatusText: document.getElementById("sim-status-text"),
    keyBadge: document.getElementById("key-badge"),
    btnKeyManage: document.getElementById("btn-key-manage"),
    modelInput: document.getElementById("model-name-input"),
    modelSuggestions: document.getElementById("model-suggestions"),
    btnStand: document.getElementById("btn-reset-stand"),
    btnSit: document.getElementById("btn-reset-sit"),
    btnClear: document.getElementById("btn-clear-chat"),
    btnRecover: document.getElementById("btn-recover-stand"),
    fallenBanner: document.getElementById("fallen-overlay-banner"),
    viewportWrap: document.getElementById("viewport-drag-surface"),
    simCanvas: document.getElementById("sim-canvas"),
    splitDivider: document.getElementById("split-divider"),
    orbitHint: document.getElementById("orbit-hint-badge"),
    camButtons: Array.from(document.querySelectorAll(".cam-tab")),
    telemTime: document.getElementById("telem-time"),
    telemTrunk: document.getElementById("telem-trunk"),
    telemBeak: document.getElementById("telem-beak"),
    telemYaw: document.getElementById("telem-yaw"),
    telemMargins: document.getElementById("telem-margins"),
    chatFeed: document.getElementById("chat-feed"),
    keyGate: document.getElementById("key-gate"),
    keyGateForm: document.getElementById("key-gate-form"),
    apiKeyInput: document.getElementById("api-key-input"),
    btnSaveKey: document.getElementById("btn-save-key"),
    btnUseDemo: document.getElementById("btn-use-demo"),
    btnWipeKey: document.getElementById("btn-wipe-key"),
    keyError: document.getElementById("key-error"),
    emptyWelcome: document.getElementById("empty-welcome"),
    starterBtns: Array.from(document.querySelectorAll(".example-btn")),
    form: document.getElementById("chat-composer-form"),
    promptInput: document.getElementById("chat-prompt-input"),
    btnSend: document.getElementById("btn-send-prompt"),
    btnStop: document.getElementById("btn-stop-turn"),
  };

  const state = {
    simReady: false,
    busy: false,
    cameraMode: "split",
    lastSimState: "LOADING",
    transientErrorUntil: 0,
    entrySeq: 0,
    cards: new Map(),
    llmMessages: [],
    abortCtrl: null,
    cancelRequested: false,
  };

  function escHtml(str) {
    return String(str ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function fmtSigned(n, digits = 3) {
    const v = Number(n || 0);
    return (v >= 0 ? "+" : "") + v.toFixed(digits);
  }

  function showInlineError(msg) {
    state.transientErrorUntil = Date.now() + 4500;
    el.simStatusText.classList.remove("is-executing");
    el.simStatusText.classList.add("is-fallen");
    el.simStatusText.textContent = String(msg || "Error");
  }

  function syncComposerAvailability() {
    const unlocked = state.simReady && _authMode !== "none";
    el.promptInput.disabled = !unlocked || state.busy;
    el.btnSend.disabled = !unlocked || state.busy;
    el.btnSend.hidden = state.busy;
    el.btnStop.hidden = !state.busy;
    el.modelInput.disabled = state.busy || _authMode === "demo";

    if (!state.simReady) {
      el.promptInput.placeholder = "Loading 50 Hz MuJoCo WASM...";
    } else if (_authMode === "none") {
      el.promptInput.placeholder = "Connect Gemini or click Try Demo above...";
    } else if (_authMode === "demo") {
      el.promptInput.placeholder = "Instruct MicroDuck (Demo mode: bow, waddle, turn, squat, nod)...";
    } else {
      el.promptInput.placeholder = "Instruct MicroDuck...";
    }
  }

  function syncAuthUI(closeGate = false) {
    el.keyBadge.classList.remove("is-ram", "is-demo");
    if (_authMode === "gemini" && _ephemeralGeminiKey) {
      el.keyBadge.textContent = "Key in RAM";
      el.keyBadge.classList.add("is-ram");
      el.btnWipeKey.hidden = false;
    } else if (_authMode === "demo") {
      el.keyBadge.textContent = "Demo mode";
      el.keyBadge.classList.add("is-demo");
      el.btnWipeKey.hidden = !_ephemeralGeminiKey;
    } else {
      el.keyBadge.textContent = "No key";
      el.btnWipeKey.hidden = true;
    }

    if (closeGate && _authMode !== "none") {
      el.keyGate.hidden = true;
      el.emptyWelcome.hidden = state.cards.size > 0;
    } else if (_authMode === "none") {
      el.keyGate.hidden = false;
      el.emptyWelcome.hidden = true;
    }
    syncComposerAvailability();
  }

  function wipeEphemeralKey() {
    _ephemeralGeminiKey = "";
    el.apiKeyInput.value = "";
    if (_authMode === "gemini") {
      _authMode = "none";
    }
    syncAuthUI(false);
  }

  window.addEventListener("pagehide", wipeEphemeralKey);
  window.addEventListener("beforeunload", wipeEphemeralKey);

  el.keyGateForm.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const raw = el.apiKeyInput.value.trim();
    el.apiKeyInput.value = "";
    if (raw.length < 20 || /\s/.test(raw)) {
      el.keyError.hidden = false;
      el.keyError.textContent = "Enter a valid Gemini API key (e.g. AIzaSy...).";
      return;
    }
    el.keyError.hidden = true;
    el.btnSaveKey.disabled = true;
    el.btnSaveKey.textContent = "Verifying...";
    try {
      const { models, bestDefault } = await listGeminiModelsDirect(raw);
      if (models.length > 0 && el.modelSuggestions) {
        el.modelSuggestions.innerHTML = "";
        for (const m of models) {
          const opt = document.createElement("option");
          opt.value = m;
          el.modelSuggestions.appendChild(opt);
        }
        const cur = el.modelInput.value.trim();
        if (!cur || !models.includes(cur)) {
          el.modelInput.value = bestDefault;
        }
      }
      _ephemeralGeminiKey = raw;
      _authMode = "gemini";
      syncAuthUI(true);
      if (state.simReady) el.promptInput.focus();
    } catch (err) {
      el.keyError.hidden = false;
      el.keyError.textContent = err.message || String(err);
    } finally {
      el.btnSaveKey.disabled = false;
      el.btnSaveKey.textContent = "Connect Gemini";
    }
  });

  el.btnUseDemo.addEventListener("click", () => {
    el.keyError.hidden = true;
    el.apiKeyInput.value = "";
    _authMode = "demo";
    syncAuthUI(true);
    if (state.simReady) el.promptInput.focus();
  });

  el.btnWipeKey.addEventListener("click", () => {
    el.keyError.hidden = true;
    wipeEphemeralKey();
  });

  el.btnKeyManage.addEventListener("click", () => {
    el.keyError.hidden = true;
    const willShow = el.keyGate.hidden;
    el.keyGate.hidden = !willShow;
    if (willShow) {
      el.emptyWelcome.hidden = true;
      el.apiKeyInput.focus();
    } else if (_authMode !== "none") {
      el.emptyWelcome.hidden = state.cards.size > 0;
    }
  });

  function updateStatus(simState) {
    state.lastSimState = simState;
    if (Date.now() < state.transientErrorUntil) return;
    el.simStatusText.classList.remove("is-executing", "is-fallen");
    if (!state.simReady) {
      el.simStatusText.textContent = simState;
      return;
    }
    if (simState === "FALLEN") {
      el.simStatusText.classList.add("is-fallen");
      el.simStatusText.textContent = "Fallen";
    } else if (state.busy || simState === "EXECUTING_CHUNK" || simState === "EXECUTING") {
      el.simStatusText.classList.add("is-executing");
      el.simStatusText.textContent = simState === "EXECUTING_CHUNK" ? "Stepping 50 Hz" : "Running";
    } else {
      el.simStatusText.textContent = "Ready";
    }
  }

  function renderTelemetry(t, simState) {
    if (!t) return;
    if (simState) updateStatus(t.fallen ? "FALLEN" : simState);

    el.fallenBanner.hidden = !t.fallen;
    el.telemTime.textContent = `${Number(t.sim_time_s || 0).toFixed(2)}s`;

    const [tx, ty, tz] = t.trunk_pos_m || [0, 0, 0];
    el.telemTrunk.textContent = `${fmtSigned(tx)}, ${fmtSigned(ty)}, ${Number(tz).toFixed(3)}`;

    const mz = (t.mouth_tip_xyz_m || [0, 0, 0])[2];
    el.telemBeak.textContent = `${Number(mz).toFixed(3)}m`;

    const yaw = (t.trunk_rpy_deg || [0, 0, 0])[2];
    el.telemYaw.textContent = `${fmtSigned(yaw, 1)}°`;

    if (t.toe_margin_m != null && t.heel_margin_m != null) {
      const toe = Math.round(t.toe_margin_m * 1000);
      const heel = Math.round(t.heel_margin_m * 1000);
      el.telemMargins.textContent = `${toe >= 0 ? "+" : ""}${toe} / ${heel >= 0 ? "+" : ""}${heel} mm`;
    } else {
      el.telemMargins.textContent = t.balance_hint || "—";
    }
  }

  /* Initialize Browser-Native 50 Hz MuJoCo WASM + Three.js Engine */

  const sim = new MicroDuckWasmSim({
    canvas: el.simCanvas,
    onProgress: (msg) => {
      el.simStatusText.textContent = msg;
    },
    onTelemetry: (snap, simState) => {
      renderTelemetry(snap, simState);
    },
  });

  sim
    .init()
    .then(() => {
      state.simReady = true;
      updateStatus("IDLE");
      syncComposerAvailability();
    })
    .catch((err) => {
      showInlineError(`Physics load failed: ${err.message}`);
    });

  /* Camera & Orbit Controls */

  function setCameraUI(mode) {
    state.cameraMode = mode;
    for (const btn of el.camButtons) {
      btn.classList.toggle("is-active", btn.dataset.mode === mode);
    }
    const isPov = mode === "pov";
    el.viewportWrap.classList.toggle("is-dragging", false);
    el.viewportWrap.classList.toggle("is-pov", isPov);
    el.splitDivider.hidden = mode !== "split";
    el.orbitHint.textContent = isPov ? "Head camera" : "Drag to orbit, scroll to zoom";
    if (state.simReady) sim.setCameraMode(mode);
  }

  for (const btn of el.camButtons) {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      setCameraUI(btn.dataset.mode);
    });
  }

  let dragActive = false;
  let lastPtr = { x: 0, y: 0 };

  el.viewportWrap.addEventListener("pointerdown", (ev) => {
    if (!state.simReady || state.cameraMode === "pov" || ev.target.closest("button")) return;
    dragActive = true;
    lastPtr = { x: ev.clientX, y: ev.clientY };
    el.viewportWrap.classList.add("is-dragging");
    try {
      el.viewportWrap.setPointerCapture(ev.pointerId);
    } catch {
      /* ignore */
    }
  });

  el.viewportWrap.addEventListener("pointermove", (ev) => {
    if (!dragActive || !state.simReady) return;
    const dx = ev.clientX - lastPtr.x;
    const dy = ev.clientY - lastPtr.y;
    lastPtr = { x: ev.clientX, y: ev.clientY };
    sim.orbit(-dx * 0.42, -dy * 0.30);
  });

  function endDrag() {
    if (!dragActive) return;
    dragActive = false;
    el.viewportWrap.classList.remove("is-dragging");
  }

  el.viewportWrap.addEventListener("pointerup", endDrag);
  el.viewportWrap.addEventListener("pointercancel", endDrag);

  el.viewportWrap.addEventListener(
    "wheel",
    (ev) => {
      if (!state.simReady || state.cameraMode === "pov") return;
      ev.preventDefault();
      sim.zoom(Math.sign(ev.deltaY) * 0.032);
    },
    { passive: false }
  );

  /* Chat Cards & Closed-Loop Harness Execution */

  function scrollToBottom() {
    requestAnimationFrame(() => {
      el.chatFeed.scrollTop = el.chatFeed.scrollHeight;
    });
  }

  function createExchangeCard(entryId, userText) {
    el.keyGate.hidden = true;
    el.emptyWelcome.hidden = true;

    const wrap = document.createElement("div");
    wrap.className = "exchange";
    wrap.id = `exchange-${entryId}`;

    const userEl = document.createElement("div");
    userEl.className = "user-msg";
    userEl.textContent = userText;

    const reply = document.createElement("div");
    reply.className = "reply-box";

    const stepsList = document.createElement("div");
    stepsList.className = "steps-list";

    const liveEl = document.createElement("div");
    liveEl.className = "live-line";
    liveEl.textContent = "Running...";

    const doneEl = document.createElement("div");
    doneEl.className = "done-msg";
    doneEl.hidden = true;

    reply.append(stepsList, liveEl, doneEl);
    wrap.append(userEl, reply);
    el.chatFeed.appendChild(wrap);

    const card = { wrap, stepsList, liveEl, doneEl };
    state.cards.set(entryId, card);
    scrollToBottom();
    return card;
  }

  function appendActionStepUI(card, subturn, waypoints, matchedStop, latencyS) {
    const wpJson = JSON.stringify(waypoints, null, 2);
    const wpCount = waypoints.length;
    const dur = waypoints.reduce((acc, w) => acc + Math.max(0.08, Math.min(3.0, Number(w.dt || 0.5))), 0).toFixed(2);

    const node = document.createElement("div");
    node.className = "step-item is-active";
    node.innerHTML = `
      <button class="step-toggle" type="button">
        <span class="step-label"><span class="step-tag">Turn ${subturn}</span><span class="step-desc">${wpCount} wp (${dur}s) stepping...</span></span>
        <span class="step-arrow">+</span>
      </button>
      <div class="step-body">
        <div class="trace-caption">
          <span>action_chunk (${wpCount} wp, ${dur}s)</span>
          <span>stop: ${escHtml(matchedStop || "</action_chunk>")} (${latencyS.toFixed(2)}s)</span>
        </div>
        <pre class="trace-pre">${escHtml(`<action_chunk>\n${wpJson}\n</action_chunk>`)}</pre>
        <div class="obs-slot"></div>
      </div>
    `;
    const arrow = node.querySelector(".step-arrow");
    node.querySelector(".step-toggle").addEventListener("click", () => {
      const open = node.classList.toggle("is-open");
      arrow.textContent = open ? "−" : "+";
    });
    card.stepsList.appendChild(node);
    scrollToBottom();

    return {
      node,
      desc: node.querySelector(".step-desc"),
      obsSlot: node.querySelector(".obs-slot"),
      wpCount,
      dur,
    };
  }

  function completeActionStepUI(rec, snap) {
    rec.node.classList.remove("is-active");
    if (snap.status === "ABORTED_FALL") rec.node.classList.add("is-error");
    const mz = Number(snap.mouth_tip_xyz_m[2]).toFixed(3);
    rec.desc.textContent = `${rec.wpCount} wp (${rec.dur}s) · ${snap.balance_hint || snap.status} · beak z=${mz}m`;
    rec.obsSlot.innerHTML = `
      <div class="trace-caption"><span>observation</span><span>${escHtml(snap.status)}</span></div>
      <pre class="trace-pre">${escHtml(snap.observation_xml)}</pre>
    `;
    scrollToBottom();
  }

  function appendHarnessErrorUI(card, subturn, rawText, errXml) {
    const node = document.createElement("div");
    node.className = "step-item is-error";
    node.innerHTML = `
      <button class="step-toggle" type="button">
        <span class="step-label"><span class="step-tag">Repair ${subturn}</span>schema error caught, retrying</span>
        <span class="step-arrow">+</span>
      </button>
      <div class="step-body">
        <pre class="trace-pre">${escHtml(rawText)}</pre>
        <pre class="trace-pre">${escHtml(errXml)}</pre>
      </div>
    `;
    const arrow = node.querySelector(".step-arrow");
    node.querySelector(".step-toggle").addEventListener("click", () => {
      const open = node.classList.toggle("is-open");
      arrow.textContent = open ? "−" : "+";
    });
    card.stepsList.appendChild(node);
    scrollToBottom();
  }

  function finishCardUI(card, status, text) {
    card.liveEl.hidden = true;
    card.doneEl.hidden = false;
    card.doneEl.className = `done-msg is-${status}`;
    card.doneEl.textContent = text || "";
    scrollToBottom();
  }

  function trimContextWindow() {
    if (state.llmMessages.length <= 24) return;
    for (let i = state.llmMessages.length - 18; i < state.llmMessages.length; i++) {
      const m = state.llmMessages[i];
      if (m.role === "user" && m.content.includes("<instruction>")) {
        state.llmMessages = state.llmMessages.slice(i);
        return;
      }
    }
  }

  function setBusy(busy) {
    state.busy = busy;
    syncComposerAvailability();
    updateStatus(busy ? "EXECUTING" : sim.latestSnap && sim.latestSnap.fallen ? "FALLEN" : "IDLE");
  }

  async function sendInstruction(rawText) {
    const text = String(rawText || "").trim();
    if (!text || state.busy || !state.simReady) return;
    if (_authMode === "none") {
      el.keyGate.hidden = false;
      el.apiKeyInput.focus();
      return;
    }

    el.promptInput.value = "";
    autoResize();

    state.entrySeq += 1;
    const entryId = `msg-${state.entrySeq}`;
    const card = createExchangeCard(entryId, text);

    const initSnap = sim.extractTelemetry("OK");
    if (initSnap.fallen) {
      finishCardUI(card, "fallen", "MicroDuck is fallen on the floor. Click 'Stand' to reset upright first.");
      return;
    }

    state.cancelRequested = false;
    state.abortCtrl = new AbortController();
    setBusy(true);

    trimContextWindow();
    state.llmMessages.push({
      role: "user",
      content: `<instruction>\n${text}\n</instruction>\n\n${initSnap.observation_xml}`,
    });

    const maxChunks = 10;
    let executedChunks = 0;
    let finishedViaDone = false;

    try {
      for (let subturn = 1; subturn <= maxChunks; subturn++) {
        if (state.cancelRequested) {
          finishCardUI(card, "stopped", "Stopped by user.");
          return;
        }

        card.liveEl.hidden = false;
        card.liveEl.textContent = `Turn ${subturn}: generating...`;

        let gen;
        if (_authMode === "demo") {
          await new Promise((r) => setTimeout(r, 80));
          gen = generateDemoStep(state.llmMessages);
        } else {
          if (!_ephemeralGeminiKey) {
            throw new Error("Gemini API key was cleared. Enter your key again.");
          }
          gen = await callGeminiDirect({
            apiKey: _ephemeralGeminiKey,
            model: el.modelInput.value.trim() || "gemini-3.5-flash",
            systemPrompt: INTERACTIVE_SYSTEM_PROMPT,
            messages: state.llmMessages,
            stopSequences: STOP_SEQUENCES,
            signal: state.abortCtrl.signal,
          });
          if (gen.model && el.modelInput.value !== gen.model) {
            el.modelInput.value = gen.model;
          }
        }

        if (state.cancelRequested) {
          finishCardUI(card, "stopped", "Stopped before executing servo chunk.");
          return;
        }

        state.llmMessages.push({ role: "assistant", content: gen.text });

        const doneMatch = DONE_RE.exec(gen.text);
        if (doneMatch && !ACTION_RE.test(gen.text)) {
          const summary = (doneMatch[1] || "Completed.").trim();
          finishCardUI(card, "done", summary);
          finishedViaDone = true;
          break;
        }

        const actMatch = ACTION_RE.exec(gen.text);
        if (!actMatch) {
          const errXml =
            "<harness_error>No `<action_chunk>...</action_chunk>` or `<done>...</done>` tag detected. " +
            "Always emit `<action_chunk>[...]</action_chunk>` or `<done>summary</done>`.</harness_error>";
          state.llmMessages.push({ role: "user", content: errXml });
          appendHarnessErrorUI(card, subturn, gen.text, errXml);
          continue;
        }

        let waypoints;
        try {
          waypoints = parseActionChunk(actMatch[1]);
        } catch (err) {
          const errXml = `<harness_error>Invalid action_chunk schema: ${err.message}</harness_error>`;
          state.llmMessages.push({ role: "user", content: errXml });
          appendHarnessErrorUI(card, subturn, gen.text, errXml);
          continue;
        }

        const rec = appendActionStepUI(card, subturn, waypoints, gen.matchedStop, gen.latencyS);
        card.liveEl.textContent = `Turn ${subturn}: stepping MuJoCo (${rec.dur}s)...`;
        updateStatus("EXECUTING_CHUNK");

        const snap = await sim.stepChunk(waypoints);
        executedChunks += 1;
        completeActionStepUI(rec, snap);
        state.llmMessages.push({ role: "user", content: snap.observation_xml });

        if (snap.status === "INTERRUPTED" || state.cancelRequested) {
          finishCardUI(card, "stopped", "Motion interrupted — holding current joint angles.");
          return;
        }

        if (snap.fallen) {
          finishCardUI(
            card,
            "fallen",
            `Lost balance on chunk ${subturn} (${snap.balance_hint}). Click 'Stand' to reset MicroDuck upright.`
          );
          return;
        }
      }

      if (!finishedViaDone) {
        const endSnap = sim.extractTelemetry("OK");
        const [tx, ty, tz] = endSnap.trunk_pos_m;
        const mz = endSnap.mouth_tip_xyz_m[2];
        finishCardUI(
          card,
          "done",
          `Executed ${executedChunks} closed-loop chunks (max turns reached). Trunk [${fmtSigned(tx)}, ${fmtSigned(ty)}, ${tz.toFixed(3)}] m, beak z=${mz.toFixed(3)} m.`
        );
      }
    } catch (err) {
      if (err && err.name === "AbortError") {
        finishCardUI(card, "stopped", "Stopped by user.");
      } else {
        finishCardUI(card, "error", err.message || String(err));
      }
    } finally {
      state.abortCtrl = null;
      setBusy(false);
      if (_authMode !== "none") el.promptInput.focus();
    }
  }

  /* UI Event Bindings */

  function autoResize() {
    el.promptInput.style.height = "32px";
    const sh = el.promptInput.scrollHeight;
    el.promptInput.style.height = `${Math.min(110, Math.max(32, sh))}px`;
    el.promptInput.style.overflowY = sh > 110 ? "auto" : "hidden";
  }

  el.promptInput.addEventListener("input", autoResize);
  el.promptInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      ev.preventDefault();
      sendInstruction(el.promptInput.value);
    }
  });

  el.form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    sendInstruction(el.promptInput.value);
  });

  for (const btn of el.starterBtns) {
    btn.addEventListener("click", () => sendInstruction(btn.dataset.prompt || btn.textContent));
  }

  el.btnStop.addEventListener("click", () => {
    state.cancelRequested = true;
    if (state.abortCtrl) state.abortCtrl.abort();
    if (state.simReady) sim.cancelChunk();
  });

  function doReset(keyframe, clearChat = false) {
    if (!state.simReady) return;
    state.cancelRequested = true;
    if (state.abortCtrl) state.abortCtrl.abort();
    sim.reset(keyframe);
    state.llmMessages = [];
    if (clearChat) {
      for (const c of state.cards.values()) c.wrap.remove();
      state.cards.clear();
      if (_authMode !== "none" && el.keyGate.hidden) {
        el.emptyWelcome.hidden = false;
      }
    }
  }

  el.btnStand.addEventListener("click", () => doReset("STAND", false));
  el.btnRecover.addEventListener("click", () => doReset("STAND", false));
  el.btnSit.addEventListener("click", () => doReset("SIT", false));
  el.btnClear.addEventListener("click", () => doReset("STAND", true));

  syncAuthUI(false);
})();
