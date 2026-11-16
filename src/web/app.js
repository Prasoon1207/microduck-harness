(() => {
  "use strict";

  const el = {
    simStatusText: document.getElementById("sim-status-text"),
    providerSelect: document.getElementById("provider-select"),
    modelInput: document.getElementById("model-name-input"),
    modelSuggestions: document.getElementById("model-suggestions"),
    btnStand: document.getElementById("btn-reset-stand"),
    btnSit: document.getElementById("btn-reset-sit"),
    btnClear: document.getElementById("btn-clear-chat"),
    btnRecover: document.getElementById("btn-recover-stand"),
    fallenBanner: document.getElementById("fallen-overlay-banner"),
    viewportWrap: document.getElementById("viewport-drag-surface"),
    streamImg: document.getElementById("sim-stream-img"),
    orbitHint: document.getElementById("orbit-hint-badge"),
    camButtons: Array.from(document.querySelectorAll(".cam-tab")),
    telemTime: document.getElementById("telem-time"),
    telemTrunk: document.getElementById("telem-trunk"),
    telemBeak: document.getElementById("telem-beak"),
    telemYaw: document.getElementById("telem-yaw"),
    telemMargins: document.getElementById("telem-margins"),
    chatFeed: document.getElementById("chat-feed"),
    emptyWelcome: document.getElementById("empty-welcome"),
    starterBtns: Array.from(document.querySelectorAll(".example-btn")),
    form: document.getElementById("chat-composer-form"),
    promptInput: document.getElementById("chat-prompt-input"),
    btnSend: document.getElementById("btn-send-prompt"),
    btnStop: document.getElementById("btn-stop-turn"),
  };

  const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8765" : ".";

  function apiUrl(path) {
    const clean = path.startsWith("/") ? path : `/${path}`;
    return `${API_BASE}${clean}`;
  }

  const state = {
    busy: false,
    cameraMode: "split",
    lastSimState: "IDLE",
    providers: [],
    renderedProviderSig: "",
    cards: new Map(),
    visualActiveUntil: Date.now() + 2000,
    transientErrorUntil: 0,
  };

  function bumpVisualActivity(ms = 1600) {
    state.visualActiveUntil = Math.max(state.visualActiveUntil, Date.now() + ms);
  }

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
    el.simStatusText.textContent = String(msg || "Request failed");
  }

  async function postJson(url, body = {}) {
    const resp = await fetch(apiUrl(url), {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await resp.json().catch(() => ({ ok: false, error: `HTTP ${resp.status}` }));
    if (!resp.ok || data.ok === false) {
      const detail = data.error || data.detail || `Request failed (${resp.status})`;
      throw new Error(detail);
    }
    return data;
  }

  function setBusy(busy) {
    const next = Boolean(busy);
    state.busy = next;
    el.btnSend.hidden = next;
    el.btnStop.hidden = !next;
    el.providerSelect.disabled = next;
    el.modelInput.disabled = next;
    if (next) bumpVisualActivity(2500);
    updateStatus(next ? "EXECUTING" : state.lastSimState);
  }

  function updateStatus(simState) {
    state.lastSimState = simState || "IDLE";
    if (Date.now() < state.transientErrorUntil) return;
    el.simStatusText.classList.remove("is-executing", "is-fallen");
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

  function renderProviderPicker(providers, activeProvider, activeModel) {
    state.providers = providers || [];
    const sig = JSON.stringify(state.providers.map((p) => [p.id, p.configured])) + `|${activeProvider}`;
    if (sig !== state.renderedProviderSig) {
      state.renderedProviderSig = sig;
      el.providerSelect.innerHTML = "";
      for (const p of state.providers) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.configured || p.id === "demo" ? p.label : `${p.label} (no key)`;
        if (p.id === activeProvider) opt.selected = true;
        el.providerSelect.appendChild(opt);
      }
      refreshModelDatalist(activeProvider);
    } else if (el.providerSelect.value !== activeProvider && document.activeElement !== el.providerSelect) {
      el.providerSelect.value = activeProvider;
      refreshModelDatalist(activeProvider);
    }

    if (document.activeElement !== el.modelInput && activeModel && el.modelInput.value !== activeModel) {
      el.modelInput.value = activeModel;
    }
  }

  function refreshModelDatalist(providerId) {
    el.modelSuggestions.innerHTML = "";
    const found = state.providers.find((p) => p.id === providerId);
    if (!found) return;
    for (const m of found.suggested_models || []) {
      const opt = document.createElement("option");
      opt.value = m;
      el.modelSuggestions.appendChild(opt);
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

  /* Camera & Orbit */

  function setCameraUI(mode) {
    if (!mode || state.cameraMode === mode) return;
    state.cameraMode = mode;
    for (const btn of el.camButtons) {
      btn.classList.toggle("is-active", btn.dataset.mode === mode);
    }
    const isPov = mode === "pov";
    el.viewportWrap.classList.toggle("is-pov", isPov);
    el.orbitHint.textContent = isPov ? "Head camera" : "Drag to orbit, scroll to zoom";
  }

  for (const btn of el.camButtons) {
    btn.addEventListener("click", async (ev) => {
      ev.stopPropagation();
      const mode = btn.dataset.mode;
      setCameraUI(mode);
      bumpVisualActivity(1800);
      try {
        await postJson("/api/camera", { mode });
        fetchFrameOnce();
      } catch (err) {
        showInlineError(err.message);
      }
    });
  }

  let dragActive = false;
  let lastPtr = { x: 0, y: 0 };
  let accumAz = 0;
  let accumEl = 0;
  let orbitTimer = null;

  function flushOrbit() {
    orbitTimer = null;
    if (accumAz === 0 && accumEl === 0) return;
    const daz = accumAz;
    const del = accumEl;
    accumAz = 0;
    accumEl = 0;
    bumpVisualActivity(1200);
    postJson("/api/camera", { delta_az: daz, delta_el: del })
      .then(() => fetchFrameOnce())
      .catch(() => {});
  }

  el.viewportWrap.addEventListener("pointerdown", (ev) => {
    if (state.cameraMode === "pov" || ev.target.closest("button")) return;
    dragActive = true;
    lastPtr = { x: ev.clientX, y: ev.clientY };
    el.viewportWrap.classList.add("is-dragging");
    bumpVisualActivity(1500);
    try {
      el.viewportWrap.setPointerCapture(ev.pointerId);
    } catch {
      /* ignore if synthetic event */
    }
  });

  el.viewportWrap.addEventListener("pointermove", (ev) => {
    if (!dragActive) return;
    accumAz -= (ev.clientX - lastPtr.x) * 0.42;
    accumEl -= (ev.clientY - lastPtr.y) * 0.30;
    lastPtr = { x: ev.clientX, y: ev.clientY };
    bumpVisualActivity(1200);
    if (!orbitTimer) orbitTimer = setTimeout(flushOrbit, 45);
  });

  function endDrag() {
    if (!dragActive) return;
    dragActive = false;
    el.viewportWrap.classList.remove("is-dragging");
    flushOrbit();
  }

  el.viewportWrap.addEventListener("pointerup", endDrag);
  el.viewportWrap.addEventListener("pointercancel", endDrag);

  let zoomTimer = null;
  let accumDist = 0;
  el.viewportWrap.addEventListener(
    "wheel",
    (ev) => {
      if (state.cameraMode === "pov") return;
      ev.preventDefault();
      accumDist += Math.sign(ev.deltaY) * 0.032;
      bumpVisualActivity(1200);
      if (!zoomTimer) {
        zoomTimer = setTimeout(() => {
          const dd = accumDist;
          accumDist = 0;
          zoomTimer = null;
          postJson("/api/camera", { delta_dist: dd })
            .then(() => fetchFrameOnce())
            .catch(() => {});
        }, 50);
      }
    },
    { passive: false }
  );

  /* Chat & Trace Reconciliation */

  function scrollToBottom() {
    requestAnimationFrame(() => {
      el.chatFeed.scrollTop = el.chatFeed.scrollHeight;
    });
  }

  function ensureCard(entryId, userText) {
    if (state.cards.has(entryId)) return state.cards.get(entryId);
    if (el.emptyWelcome) el.emptyWelcome.hidden = true;

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

    const card = {
      wrap,
      stepsList,
      liveEl,
      doneEl,
      steps: new Map(),
      finishedSig: "",
    };
    state.cards.set(entryId, card);
    scrollToBottom();
    return card;
  }

  function upsertActionStep(card, step) {
    const key = `act-${step.subturn}`;
    let rec = card.steps.get(key);
    const wpJson = JSON.stringify(step.waypoints || [], null, 2);
    const wpCount = (step.waypoints || []).length;
    const dur = Number(step.duration_s || 0).toFixed(2);

    if (!rec) {
      const node = document.createElement("div");
      node.className = "step-item is-active";
      node.innerHTML = `
        <button class="step-toggle" type="button">
          <span class="step-label"><span class="step-tag">Turn ${step.subturn}</span><span class="step-desc"></span></span>
          <span class="step-arrow">+</span>
        </button>
        <div class="step-body">
          <div class="trace-caption">
            <span>action_chunk (${wpCount} wp, ${dur}s)</span>
            <span>stop: ${escHtml(step.matched_stop || "</action_chunk>")} (${Number(step.latency_s || 0).toFixed(2)}s)</span>
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
      rec = {
        node,
        desc: node.querySelector(".step-desc"),
        obsSlot: node.querySelector(".obs-slot"),
        wpCount,
        dur,
        appliedObs: false,
      };
      card.steps.set(key, rec);
      scrollToBottom();
    }

    if (step.observation_xml) {
      if (!rec.appliedObs) {
        rec.appliedObs = true;
        applyObservation(rec, step.observation_xml, step.status, step.balance_hint, step.telemetry);
        scrollToBottom();
      }
    } else if (!rec.appliedObs) {
      rec.desc.textContent = `${wpCount} wp (${dur}s) stepping...`;
    }
  }

  function applyObservation(rec, obsXml, status, balanceHint, telem) {
    rec.node.classList.remove("is-active");
    if (status === "ABORTED_FALL") rec.node.classList.add("is-error");

    let text = `${rec.wpCount} wp (${rec.dur}s) · ${balanceHint || status}`;
    if (telem && telem.mouth_tip_xyz_m) {
      text += ` · beak z=${Number(telem.mouth_tip_xyz_m[2]).toFixed(3)}m`;
    }
    rec.desc.textContent = text;
    rec.obsSlot.innerHTML = `
      <div class="trace-caption"><span>observation</span><span>${escHtml(status)}</span></div>
      <pre class="trace-pre">${escHtml(obsXml)}</pre>
    `;
  }

  function upsertHarnessError(card, step) {
    const key = `err-${step.subturn}`;
    if (card.steps.has(key)) return;
    const node = document.createElement("div");
    node.className = "step-item is-error";
    node.innerHTML = `
      <button class="step-toggle" type="button">
        <span class="step-label"><span class="step-tag">Repair ${step.subturn}</span>schema error caught, retrying</span>
        <span class="step-arrow">+</span>
      </button>
      <div class="step-body">
        <pre class="trace-pre">${escHtml(step.raw_llm_text || "")}</pre>
        <pre class="trace-pre">${escHtml(step.error_xml || "")}</pre>
      </div>
    `;
    const arrow = node.querySelector(".step-arrow");
    node.querySelector(".step-toggle").addEventListener("click", () => {
      const open = node.classList.toggle("is-open");
      arrow.textContent = open ? "−" : "+";
    });
    card.stepsList.appendChild(node);
    card.steps.set(key, { node });
    scrollToBottom();
  }

  function finishCard(card, status, summaryText) {
    const sig = `${status}|${summaryText}`;
    if (card.finishedSig === sig) return;
    card.finishedSig = sig;
    card.liveEl.hidden = true;
    card.doneEl.hidden = false;
    card.doneEl.className = `done-msg is-${status}`;
    card.doneEl.textContent = summaryText || "";
    scrollToBottom();
  }

  function syncHistory(history) {
    const list = Array.isArray(history) ? history : [];
    if (list.length === 0) {
      if (state.cards.size > 0) {
        for (const c of state.cards.values()) c.wrap.remove();
        state.cards.clear();
      }
      if (el.emptyWelcome) el.emptyWelcome.hidden = false;
      return;
    }

    if (el.emptyWelcome) el.emptyWelcome.hidden = true;
    for (const item of list) {
      const card = ensureCard(item.id, item.user_text);
      for (const st of item.steps || []) {
        if (st.kind === "action_chunk") upsertActionStep(card, st);
        else if (st.kind === "harness_error") upsertHarnessError(card, st);
      }
      if (item.status === "running") {
        card.liveEl.hidden = false;
        card.liveEl.textContent = item.live_note || "Running...";
      } else {
        finishCard(card, item.status, item.done_summary || item.error || "");
      }
    }
  }

  /* Stateless State & Frame Sync Loops */

  let stateInFlight = false;
  async function syncStateOnce() {
    if (stateInFlight) return;
    stateInFlight = true;
    try {
      const resp = await fetch(apiUrl("/api/state"), {
        method: "GET",
        credentials: "same-origin",
        cache: "no-store",
      });
      if (!resp.ok) return;
      const st = await resp.json();
      renderProviderPicker(st.providers, st.provider, st.model);
      if (st.camera && st.camera.mode) setCameraUI(st.camera.mode);
      const wasBusy = state.busy;
      setBusy(st.busy);
      renderTelemetry(st.telemetry, st.sim_state);
      syncHistory(st.chat_history);
      if (wasBusy && !st.busy) {
        fetchFrameOnce();
        el.promptInput.focus();
      }
    } catch {
      /* transient network retry on next tick */
    } finally {
      stateInFlight = false;
    }
  }

  function startStateLoop() {
    async function tick() {
      if (!document.hidden) {
        await syncStateOnce();
      }
      const delay = document.hidden ? 1200 : state.busy ? 160 : 450;
      setTimeout(tick, delay);
    }
    tick();
  }

  let lastFrameEtag = "";
  let currentBlobUrl = null;
  let frameInFlight = false;

  async function fetchFrameOnce() {
    if (frameInFlight || document.hidden) return;
    frameInFlight = true;
    try {
      const headers = lastFrameEtag ? { "If-None-Match": lastFrameEtag } : {};
      const resp = await fetch(apiUrl("/api/frame.jpg"), {
        method: "GET",
        credentials: "same-origin",
        headers,
        cache: "no-cache",
      });
      if (resp.status === 304) return;
      if (!resp.ok) return;
      const etag = resp.headers.get("ETag") || "";
      const blob = await resp.blob();
      if (!blob || blob.size === 0) return;
      if (etag) lastFrameEtag = etag;
      const nextUrl = URL.createObjectURL(blob);
      const prevUrl = currentBlobUrl;
      currentBlobUrl = nextUrl;
      el.streamImg.src = nextUrl;
      if (prevUrl) {
        setTimeout(() => URL.revokeObjectURL(prevUrl), 250);
      }
    } catch {
      /* retry on next frame tick */
    } finally {
      frameInFlight = false;
    }
  }

  function startFrameLoop() {
    async function tick() {
      if (!document.hidden) {
        await fetchFrameOnce();
      }
      const active = state.busy || Date.now() < state.visualActiveUntil;
      const delay = document.hidden ? 1000 : active ? 45 : 180;
      setTimeout(tick, delay);
    }
    tick();
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      bumpVisualActivity(1500);
      syncStateOnce();
      fetchFrameOnce();
    }
  });

  /* Controls */

  async function sendInstruction(rawText) {
    const text = String(rawText || "").trim();
    if (!text || state.busy) return;
    el.promptInput.value = "";
    autoResize();
    setBusy(true);
    bumpVisualActivity(3000);
    try {
      const res = await postJson("/api/chat", { message: text });
      if (res.entry) {
        const card = ensureCard(res.entry.id, res.entry.user_text);
        card.liveEl.textContent = res.entry.live_note || "Running...";
      } else if (res.id) {
        ensureCard(res.id, text);
      }
      syncStateOnce();
    } catch (err) {
      setBusy(false);
      showInlineError(err.message);
    }
  }

  function autoResize() {
    el.promptInput.style.height = "auto";
    el.promptInput.style.height = `${Math.min(110, el.promptInput.scrollHeight)}px`;
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
    postJson("/api/stop")
      .then(() => syncStateOnce())
      .catch((err) => showInlineError(err.message));
  });

  async function triggerReset(keyframe, clearChat = false) {
    bumpVisualActivity(1800);
    try {
      const res = await postJson("/api/reset", { keyframe, clear_chat: clearChat });
      if (clearChat) syncHistory([]);
      if (res.telemetry) renderTelemetry(res.telemetry, "IDLE");
      fetchFrameOnce();
      syncStateOnce();
    } catch (err) {
      showInlineError(err.message);
    }
  }

  el.btnStand.addEventListener("click", () => triggerReset("STAND", false));
  el.btnRecover.addEventListener("click", () => triggerReset("STAND", false));
  el.btnSit.addEventListener("click", () => triggerReset("SIT", false));
  el.btnClear.addEventListener("click", () => triggerReset("STAND", true));

  el.providerSelect.addEventListener("change", async () => {
    const provider = el.providerSelect.value;
    refreshModelDatalist(provider);
    try {
      const res = await postJson("/api/config", { provider });
      if (res.config && res.config.model) {
        el.modelInput.value = res.config.model;
      }
      syncStateOnce();
    } catch (err) {
      showInlineError(err.message);
    }
  });

  el.modelInput.addEventListener("change", async () => {
    const model = el.modelInput.value.trim();
    if (!model) return;
    try {
      await postJson("/api/config", { provider: el.providerSelect.value, model });
      syncStateOnce();
    } catch (err) {
      showInlineError(err.message);
    }
  });

  startStateLoop();
  startFrameLoop();
})();
