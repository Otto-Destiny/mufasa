/* MUFASA UI: continuous chat. Live data only. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  const STEPS = [
    { id: "searching", title: "Searching your papers", hint: "Looking through what you have loaded." },
    { id: "gathering", title: "Finding the best sources", hint: "Picking the passages that matter most." },
    { id: "generating", title: "Writing your answer", hint: "Keeping every claim tied to a source." },
    { id: "checking", title: "Double-checking sources", hint: "Making sure the citations line up." },
  ];

  const state = {
    jobId: null,
    lastAnswer: null,
    config: null,
    audio: null,
    busy: false,
    turns: [], // continuous session: [{id, question, stage, payload}]
    activeTurnId: null,
  };

  function toast(msg) {
    const el = $("#toast");
    el.textContent = msg;
    el.classList.add("show");
    clearTimeout(toast._t);
    toast._t = setTimeout(() => el.classList.remove("show"), 2200);
  }

  function friendlyVerdict(verdict, answerable) {
    if (verdict === "grounded") return { cls: "grounded", label: "Backed by your papers" };
    if (verdict === "partly_grounded") return { cls: "partly", label: "Partly backed" };
    if (verdict === "no_matching_evidence" || answerable === false) {
      return { cls: "none", label: "Nothing matched" };
    }
    return { cls: "none", label: "Needs a closer look" };
  }

  function escapeHtml(text) {
    return String(text || "")
      .replace(/\u2014/g, ",")
      .replace(/\u2013/g, "-")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function renderCitations(text, turnId) {
    return escapeHtml(text).replace(/\[E(\d+)\]/g, (_, n) =>
      `<button type="button" class="cite" data-turn="${turnId}" data-tag="E${n}" aria-label="Open source E${n}">E${n}</button>`
    );
  }

  function updateAskHint() {
    const n = state.turns.filter((t) => t.payload).length;
    const hint = $("#ask-hint");
    if (!n) {
      hint.textContent = "Each answer points back to the pages it used.";
    } else {
      hint.textContent = "Ask a follow-up, or tap New research to start fresh.";
    }
    if (!state.busy) refreshPlaceholderTyping();
  }

  function setView(name) {
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
    $$(".nav button[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
    closeDrawer();
    syncLandingClass();
    if (name === "history") loadHistory();
    if (name === "library") loadLibrary();
    if (name === "statistics") loadStats();
    if (name === "compare") renderCompare();
    if (name === "settings") refreshSettings();
  }

  function openDrawer() {
    $("#sidebar").classList.add("open");
    const bd = $("#backdrop");
    bd.hidden = false;
    bd.classList.add("show");
  }
  function closeDrawer() {
    $("#sidebar").classList.remove("open");
    const bd = $("#backdrop");
    bd.classList.remove("show");
    bd.hidden = true;
  }

  const PLACEHOLDERS_HOME = [
    "Ask anything about your papers…",
    "What did the Bosso water study find?",
    "How do the Kainji methods compare?",
    "What causes flooding along River Suka?",
    "Summarize rainwater findings for Hong LGA…",
  ];
  const PLACEHOLDERS_FOLLOW = [
    "Ask a follow-up…",
    "Go deeper on that finding…",
    "Which source supports that?",
    "Compare that with another study…",
  ];

  let placeholderTimer = null;
  let placeholderToken = 0;
  let frozenPlaceholder = "";

  function placeholderPool() {
    return state.turns.length ? PLACEHOLDERS_FOLLOW : PLACEHOLDERS_HOME;
  }

  function stopPlaceholderTyping() {
    placeholderToken += 1;
    if (placeholderTimer) {
      clearTimeout(placeholderTimer);
      placeholderTimer = null;
    }
  }

  function setPlaceholderInstant(text) {
    stopPlaceholderTyping();
    const input = $("#ask-input");
    if (input) {
      input.placeholder = text;
      frozenPlaceholder = text;
    }
  }

  function typePlaceholder(text, { loopIndex = 0 } = {}) {
    const input = $("#ask-input");
    if (!input || state.busy) return;
    const reduce =
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const token = ++placeholderToken;
    if (placeholderTimer) {
      clearTimeout(placeholderTimer);
      placeholderTimer = null;
    }

    if (reduce || document.activeElement === input || input.value.trim()) {
      input.placeholder = text;
      frozenPlaceholder = text;
      return;
    }

    let i = 0;
    input.placeholder = "";

    const tick = () => {
      if (token !== placeholderToken) return;
      if (state.busy) {
        input.placeholder = frozenPlaceholder || text;
        return;
      }
      if (document.activeElement === input || input.value.trim()) {
        input.placeholder = text;
        frozenPlaceholder = text;
        return;
      }
      i += 1;
      input.placeholder = text.slice(0, i);
      if (i < text.length) {
        placeholderTimer = setTimeout(tick, 28 + Math.random() * 22);
        return;
      }
      frozenPlaceholder = text;
      const pool = placeholderPool();
      placeholderTimer = setTimeout(() => {
        if (token !== placeholderToken || state.busy) return;
        if (document.activeElement === input || input.value.trim()) return;
        const next = (loopIndex + 1) % pool.length;
        erasePlaceholder(pool[next], next, token);
      }, 2400);
    };
    placeholderTimer = setTimeout(tick, 40);
  }

  function erasePlaceholder(nextText, nextIndex, token) {
    const input = $("#ask-input");
    if (!input || token !== placeholderToken || state.busy) return;
    const erase = () => {
      if (token !== placeholderToken || state.busy) return;
      if (document.activeElement === input || input.value.trim()) return;
      const cur = input.placeholder || "";
      if (cur.length > 0) {
        input.placeholder = cur.slice(0, -1);
        placeholderTimer = setTimeout(erase, 16);
        return;
      }
      typePlaceholder(nextText, { loopIndex: nextIndex });
    };
    erase();
  }

  function refreshPlaceholderTyping() {
    if (state.busy) {
      stopPlaceholderTyping();
      const input = $("#ask-input");
      if (input && frozenPlaceholder) input.placeholder = frozenPlaceholder;
      return;
    }
    const input = $("#ask-input");
    const pool = placeholderPool();
    if (input && (document.activeElement === input || input.value.trim())) {
      setPlaceholderInstant(pool[0]);
      return;
    }
    typePlaceholder(pool[0], { loopIndex: 0 });
  }

  function setBusy(on) {
    state.busy = on;
    const form = $("#ask-form");
    const input = $("#ask-input");
    const mic = $("#btn-mic");
    form.classList.toggle("busy", on);
    input.disabled = on;
    $("#btn-send").disabled = on;
    if (mic) mic.disabled = on;
    if (on) {
      stopPlaceholderTyping();
      if (!frozenPlaceholder) frozenPlaceholder = input.placeholder || placeholderPool()[0];
      input.placeholder = frozenPlaceholder;
    } else {
      refreshPlaceholderTyping();
    }
  }

  function syncLandingClass() {
    const landing = $("#landing");
    const onSearch = Boolean($("#view-search")?.classList.contains("active"));
    const app = document.querySelector(".app");
    if (!app) return;
    app.classList.toggle("on-search", onSearch);
    app.classList.toggle("is-landing", onSearch && landing && !landing.hidden);
  }

  function showLanding(show) {
    $("#landing").hidden = !show;
    const foot = $("#home-foot");
    if (foot) foot.hidden = !show;
    $("#thread-wrap").hidden = show;
    syncLandingClass();
  }

  function progressHtml(activeId, gatheringHint) {
    const order = STEPS.map((s) => s.id);
    const cur = activeId === "done" ? order.length : Math.max(0, order.indexOf(activeId));
    return `<div class="progress">${STEPS.map((s, i) => {
      let cls = "step pending";
      if (i < cur) cls = "step done";
      else if (i === cur) cls = "step active";
      const check =
        cls.includes("done")
          ? `<svg class="icon sm" viewBox="0 0 24 24"><path d="m5 12 5 5L20 7" stroke="currentColor"/></svg>`
          : "";
      let hint = "";
      if (i === cur) {
        hint = `<p>${s.id === "gathering" && gatheringHint ? gatheringHint : s.hint}</p>`;
      } else if (i < cur && s.id === "gathering" && gatheringHint) {
        hint = `<p>${gatheringHint}</p>`;
      }
      return `<div class="${cls}">
        <div class="step-dot">${check}</div>
        <div><h3>${s.title}</h3>${hint}</div>
      </div>`;
    }).join("")}</div>`;
  }

  function isFailedAnswer(payload) {
    if (!payload) return true;
    if (payload.error) return true;
    if (payload.answerable === false) return true;
    const verdict = payload.verdict || "";
    if (verdict === "no_matching_evidence" || verdict === "ungrounded") return true;
    const ans = String(payload.answer || "").trim();
    if (!ans) return true;
    if (/could not generate an answer/i.test(ans)) return true;
    return false;
  }

  function failureMessage(payload) {
    if (payload?.error) {
      if (typeof payload.error === "string" && payload.error.trim()) {
        const err = payload.error.trim();
        if (!/traceback|exception|gguf|sqlite|http/i.test(err)) return err;
      }
      const fromAnswer = String(payload.answer || "").trim();
      if (fromAnswer) return fromAnswer;
      return "Something went wrong. Please try again.";
    }
    const ans = String(payload?.answer || "").trim();
    if (/could not generate an answer/i.test(ans)) {
      return "Please try again with a different question.";
    }
    if (ans) return ans;
    return (
      payload?.coverage?.message ||
      "Nothing in your library answers this directly."
    );
  }

  function answerHtml(turn) {
    const payload = turn.payload;
    if (!payload) return "";

    if (isFailedAnswer(payload)) {
      const msg = failureMessage(payload);
      const title =
        payload.answerable === false || payload.verdict === "no_matching_evidence"
          ? "Nothing matched"
          : "Could not answer";
      return `<div class="status-msg" role="status">
        <p class="status-msg-title">${escapeHtml(title)}</p>
        <p class="status-msg-body">${escapeHtml(msg)}</p>
      </div>`;
    }

    const v = friendlyVerdict(payload.verdict, payload.answerable);
    const evidence = payload.evidence || [];
    const cited = evidence.length;
    return `<div class="answer-block" data-turn="${turn.id}">
      <div class="answer-head">
        <span class="badge ${v.cls}">${v.label}</span>
      </div>
      <div class="answer-text">${renderCitations(payload.answer || "", turn.id)}</div>
      <div class="actions">
        <button type="button" class="btn-copy" data-turn="${turn.id}">
          <svg class="icon sm" viewBox="0 0 24 24"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>
          Copy
        </button>
        <button type="button" class="btn-speak" data-turn="${turn.id}">
          <svg class="icon sm" viewBox="0 0 24 24"><path d="M11 5 6 9H3v6h3l5 4V5zm7.5 3.5a5 5 0 0 1 0 7M15 9.5a2.5 2.5 0 0 1 0 5"/></svg>
          Listen
        </button>
        <button type="button" class="btn-up" data-turn="${turn.id}" aria-label="Helpful">
          <svg class="icon sm" viewBox="0 0 24 24"><path d="M7 22V11l5-8 2 4h6a2 2 0 0 1 2 2l-1.5 8a2 2 0 0 1-2 1.5H7zM7 11H4a1 1 0 0 0-1 1v8a1 1 0 0 0 1 1h3"/></svg>
        </button>
        <button type="button" class="btn-down" data-turn="${turn.id}" aria-label="Not helpful">
          <svg class="icon sm" viewBox="0 0 24 24"><path d="M17 2v11l-5 8-2-4H4a2 2 0 0 1-2-2l1.5-8A2 2 0 0 1 5.5 5H17zM17 13h3a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1h-3"/></svg>
        </button>
        ${cited ? `<button type="button" class="meta-chip btn-sources" data-turn="${turn.id}">${cited} source${cited === 1 ? "" : "s"}</button>` : ""}
      </div>
    </div>`;
  }

  function renderStream() {
    const stream = $("#chat-stream");
    stream.innerHTML = state.turns
      .map((turn) => {
        const body = turn.payload
          ? answerHtml(turn)
          : progressHtml(turn.stage || "searching", turn.gatheringHint);
        return `<article class="turn" id="turn-${turn.id}" data-turn="${turn.id}">
          <div class="user-q">${escapeHtml(turn.question)}</div>
          ${body}
        </article>`;
      })
      .join("");

    // Wire citations & actions for completed turns
    $$(".cite", stream).forEach((btn) => {
      btn.addEventListener("click", () => {
        const turn = state.turns.find((t) => t.id === btn.dataset.turn);
        if (turn?.payload) showEvidence(turn.payload, btn.dataset.tag);
      });
    });
    $$(".btn-copy", stream).forEach((btn) =>
      btn.addEventListener("click", async () => {
        const turn = state.turns.find((t) => t.id === btn.dataset.turn);
        if (!turn?.payload?.answer) return;
        await navigator.clipboard.writeText(turn.payload.answer);
        toast("Copied");
      })
    );
    $$(".btn-speak", stream).forEach((btn) =>
      btn.addEventListener("click", () => {
        const turn = state.turns.find((t) => t.id === btn.dataset.turn);
        if (turn?.payload) speak(turn.payload.answer);
      })
    );
    $$(".btn-up", stream).forEach((btn) =>
      btn.addEventListener("click", () => {
        const turn = state.turns.find((t) => t.id === btn.dataset.turn);
        if (turn?.payload) feedback(true, turn.payload);
      })
    );
    $$(".btn-down", stream).forEach((btn) =>
      btn.addEventListener("click", () => {
        const turn = state.turns.find((t) => t.id === btn.dataset.turn);
        if (turn?.payload) feedback(false, turn.payload);
      })
    );
    $$(".btn-sources", stream).forEach((btn) =>
      btn.addEventListener("click", () => {
        const turn = state.turns.find((t) => t.id === btn.dataset.turn);
        if (turn?.payload && !isFailedAnswer(turn.payload) && turn.payload.evidence?.length) {
          showEvidence(turn.payload, turn.payload.evidence[0].tag);
        }
      })
    );

    const last = state.turns[state.turns.length - 1];
    if (
      last?.payload &&
      !isFailedAnswer(last.payload) &&
      last.payload.evidence?.length
    ) {
      showEvidence(last.payload, last.payload.evidence[0].tag);
    } else {
      hideEvidence();
    }

    const el = last ? document.getElementById(`turn-${last.id}`) : null;
    if (el) el.scrollIntoView({ behavior: "smooth", block: "end" });
    updateAskHint();
  }

  function updateActiveProgress(stage, gatheringHint) {
    const turn = state.turns.find((t) => t.id === state.activeTurnId);
    if (!turn || turn.payload) return;
    turn.stage = stage;
    if (gatheringHint) turn.gatheringHint = gatheringHint;
    const node = document.getElementById(`turn-${turn.id}`);
    if (!node) return;
    const q = node.querySelector(".user-q");
    node.innerHTML = "";
    node.appendChild(q);
    const wrap = document.createElement("div");
    wrap.innerHTML = progressHtml(stage, turn.gatheringHint);
    node.appendChild(wrap.firstElementChild);
  }

  function showEvidence(payload, focusTag) {
    if (!payload || isFailedAnswer(payload)) {
      hideEvidence();
      return;
    }
    const list = payload.evidence || [];
    const panel = $("#evidence-panel");
    const workspace = document.querySelector(".workspace");
    const count = $("#evidence-count");
    if (!list.length) {
      hideEvidence();
      return;
    }
    panel.hidden = false;
    panel.setAttribute("aria-hidden", "false");
    panel.classList.add("is-open");
    workspace.classList.add("has-evidence");
    if (count) {
      count.hidden = false;
      count.textContent = String(list.length);
    }
    $("#evidence-list").innerHTML = list
      .map((e) => {
        const title = escapeHtml(e.paper?.title || "Source");
        const year = e.paper?.year ? ` · ${e.paper.year}` : "";
        const page = e.page != null ? `Page ${e.page}` : "";
        const section = e.section ? ` · ${escapeHtml(e.section)}` : "";
        const quote = e.quote
          ? `<div class="quote">${escapeHtml(e.quote)}</div>`
          : e.quote_withheld
            ? `<div class="quote">The full excerpt is not available for this paper.</div>`
            : "";
        return `<article class="ev-card" data-tag="${e.tag}" id="ev-${e.tag}">
          <span class="tag">${e.tag}</span>
          <p class="finding">${escapeHtml(e.text || "")}</p>
          ${quote}
          <p class="meta">${title}${year}<br/>${page}${section}</p>
        </article>`;
      })
      .join("");
    $$(".ev-card").forEach((card) => {
      card.addEventListener("click", () => selectEvidence(card.dataset.tag));
    });
    if (focusTag) selectEvidence(focusTag);
  }

  function hideEvidence() {
    const panel = $("#evidence-panel");
    if (!panel) return;
    panel.hidden = true;
    panel.setAttribute("aria-hidden", "true");
    panel.classList.remove("is-open");
    document.querySelector(".workspace")?.classList.remove("has-evidence");
    const count = $("#evidence-count");
    if (count) {
      count.hidden = true;
      count.textContent = "";
    }
    $("#evidence-list").innerHTML = "";
  }

  function selectEvidence(tag) {
    $$(".ev-card").forEach((c) => c.classList.toggle("active", c.dataset.tag === tag));
    $$(".cite").forEach((c) => c.classList.toggle("active", c.dataset.tag === tag));
    const card = $(`.ev-card[data-tag="${tag}"]`);
    if (card) card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function newSession() {
    state.turns = [];
    state.activeTurnId = null;
    state.lastAnswer = null;
    $("#chat-stream").innerHTML = "";
    hideEvidence();
    showLanding(true);
    updateAskHint();
    setView("search");
    fillRecent();
    $("#ask-input").value = "";
    autosize($("#ask-input"));
    $("#ask-input").focus();
  }

  async function ask(question) {
    question = (question || "").trim();
    if (!question || state.busy) return;

    setView("search");
    showLanding(false);

    const turn = {
      id: `t${Date.now().toString(36)}`,
      question,
      stage: "searching",
      gatheringHint: null,
      payload: null,
    };
    state.turns.push(turn);
    state.activeTurnId = turn.id;
    renderStream();

    $("#ask-input").value = "";
    autosize($("#ask-input"));
    setBusy(true);
    state.jobId = null;

    try {
      const res = await fetch(`/api/ask/stream?q=${encodeURIComponent(question)}`);
      if (!res.ok) {
        // Another answer is already running: keep UI disabled, drop this turn quietly.
        if (res.status === 409) {
          state.turns = state.turns.filter((t) => t.id !== turn.id);
          renderStream();
          if (!state.turns.length) showLanding(true);
          setBusy(false);
          return;
        }
        const err = await res.json().catch(() => ({}));
        const detail = typeof err.detail === "string" ? err.detail : "";
        throw new Error(
          /one generation|already in progress|Cancel the running/i.test(detail)
            ? "Please wait for the current answer to finish."
            : detail || "Could not answer just now."
        );
      }
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      let eventName = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          else if (line.startsWith("data:") && eventName) {
            const data = JSON.parse(line.slice(5).trim());
            if (eventName === "job") state.jobId = data.job_id;
            else if (eventName === "stage") {
              const stage = data.stage === "done" ? "checking" : data.stage;
              let gatheringHint = null;
              if (data.sources?.length) {
                gatheringHint = `${data.sources.length} source${data.sources.length === 1 ? "" : "s"} found`;
              }
              updateActiveProgress(stage || "searching", gatheringHint);
            } else if (eventName === "answer") {
              turn.payload = data;
              state.lastAnswer = data;
              setBusy(false);
              renderStream();
            } else if (eventName === "cancelled") {
              setBusy(false);
              toast("Stopped");
              // leave progress as-is or remove incomplete turn
              state.turns = state.turns.filter((t) => t.id !== turn.id || t.payload);
              renderStream();
              if (!state.turns.length) showLanding(true);
            } else if (eventName === "error") {
              throw new Error(data.error || "Something went wrong");
            }
            eventName = null;
          }
        }
      }
      if (!turn.payload) {
        turn.payload = {
          error: true,
          answer: "Could not finish that answer.",
          evidence: [],
          sources: [],
        };
        setBusy(false);
        renderStream();
      }
    } catch (err) {
      setBusy(false);
      const msg = String(err.message || err || "");
      if (/already in progress|one generation|Cancel the running/i.test(msg)) {
        state.turns = state.turns.filter((t) => t.id !== turn.id || t.payload);
        renderStream();
        if (!state.turns.length) showLanding(true);
        return;
      }
      turn.payload = {
        error: true,
        answer: msg || "Something went wrong.",
        evidence: [],
        sources: [],
      };
      state.lastAnswer = null;
      hideEvidence();
      renderStream();
    }
  }

  async function stop() {
    await fetch("/api/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: state.jobId }),
    });
  }

  async function speak(text) {
    if (!text) return;
    if (state.audio) {
      state.audio.pause();
      state.audio = null;
      return;
    }
    const res = await fetch("/api/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) {
      toast("Voice is not set up on this laptop yet");
      return;
    }
    const url = URL.createObjectURL(await res.blob());
    const audio = new Audio(url);
    state.audio = audio;
    audio.onended = () => {
      state.audio = null;
    };
    await audio.play();
  }

  async function feedback(helpful, payload) {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        helpful,
        question: payload.question,
        answer: payload.answer,
      }),
    });
    toast("Thanks, saved");
  }

  async function loadHistory() {
    const res = await fetch("/api/history");
    const data = await res.json().catch(() => ({ entries: [] }));
    const entries = data.entries || [];
    const box = $("#history-list");
    if (!entries.length) {
      box.innerHTML = `<div class="empty-inline"><h3>No history yet</h3><p>Questions you ask will appear here.</p></div>`;
      return;
    }
    box.innerHTML = `<div class="history-list">${entries
      .map(
        (e) => `<button type="button" class="history-row" data-q="${encodeURIComponent(e.question)}">
        <svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="history-q">${escapeHtml(e.question)}</span>
        <span class="history-meta">${e.answerable === false ? "No match" : "Answered"}</span>
      </button>`
      )
      .join("")}</div>`;
    $$("#history-list .history-row").forEach((b) =>
      b.addEventListener("click", () => ask(decodeURIComponent(b.dataset.q)))
    );
  }

  const STARTERS = [
    "How high was benzene reported in Ogale borehole drinking-water wells?",
    "What did the Bosso water study find?",
    "How do the Kainji methods compare?",
  ];

  function isUsefulSuggestion(q) {
    const t = String(q || "").trim();
    if (t.length < 16) return false;
    const words = t.split(/\s+/).filter(Boolean);
    if (words.length < 3) return false;
    if (words.filter((w) => w.length > 2).length < 2) return false;
    return true;
  }

  function uniqueQuestions(questions) {
    const seen = new Set();
    const out = [];
    for (const raw of questions) {
      const q = String(raw || "").trim();
      const key = q.toLowerCase().replace(/\s+/g, " ");
      if (!key || seen.has(key)) continue;
      seen.add(key);
      out.push(q);
    }
    return out;
  }

  async function fillRecent() {
    let entries = [];
    try {
      const res = await fetch("/api/history?limit=20");
      const data = await res.json();
      entries = data.entries || [];
    } catch {
      entries = [];
    }
    const fromHistory = uniqueQuestions(entries.map((e) => e.question)).filter(isUsefulSuggestion);
    const suggestions = uniqueQuestions([...fromHistory, ...STARTERS]).slice(0, 3);
    const box = $("#recent");
    if (!box) return;
    if (!suggestions.length) {
      box.hidden = true;
      box.innerHTML = "";
      return;
    }
    box.hidden = false;
    box.innerHTML = suggestions
      .map(
        (q) =>
          `<button type="button" class="recent-chip" data-q="${encodeURIComponent(q)}">
            <span>${escapeHtml(q)}</span>
          </button>`
      )
      .join("");
    $$("#recent button").forEach((b) =>
      b.addEventListener("click", () => ask(decodeURIComponent(b.dataset.q)))
    );
  }

  async function loadLibrary() {
    const res = await fetch("/api/papers?limit=100");
    const data = await res.json().catch(() => ({ papers: [] }));
    const papers = data.papers || [];
    const box = $("#library-list");
    if (!papers.length) {
      box.innerHTML = `<div class="empty-inline"><h3>No papers yet</h3><p>Load a library to start asking questions.</p></div>`;
      return;
    }
    box.innerHTML = papers
      .map((p) => {
        const bits = [p.year, p.journal].filter(Boolean).join(" · ");
        return `<div class="list-item">
          <strong>${escapeHtml(p.title || "Untitled paper")}</strong>
          <div class="meta">${escapeHtml(bits)}</div>
        </div>`;
      })
      .join("");
  }

  async function loadStats() {
    const [card, stats] = await Promise.all([
      fetch("/api/corpus").then((r) => r.json()),
      fetch("/api/statistics").then((r) => r.json()),
    ]);
    $("#stat-grid").innerHTML = `
      <div class="stat"><div class="n">${card.papers ?? "-"}</div><div class="k">Papers</div></div>
      <div class="stat"><div class="n">${card.claims ?? "-"}</div><div class="k">Findings</div></div>
      <div class="stat"><div class="n">${(card.year_min && card.year_max) ? `${card.year_min}-${card.year_max}` : "-"}</div><div class="k">Years covered</div></div>
      <div class="stat"><div class="n">${stats.withheld_papers ?? 0}</div><div class="k">Restricted papers</div></div>
    `;
    const facets = (stats.facets || [])
      .slice(0, 10)
      .map(
        (f) =>
          `<div class="list-item"><strong>${escapeHtml(f.label || f.facet)}</strong><div class="meta">${f.claims} findings</div></div>`
      )
      .join("");
    $("#coverage-detail").innerHTML = `
      <p class="section-label">Topics in your library</p>
      <div class="list">${facets || '<div class="list-item"><div class="meta">No topics listed yet.</div></div>'}</div>
      <p class="meta coverage-note">A missing topic means it is not in this library, not that the science does not exist.</p>
    `;
  }

  async function renderCompare() {
    const box = $("#compare-body");
    const ids = (state.lastAnswer?.evidence || []).map((e) => e.claim_id).filter(Boolean);
    if (!ids.length) {
      box.innerHTML = `<div class="empty-inline"><h3>Nothing to compare yet</h3><p>Ask a question first. If several studies speak to it, they will show up here.</p></div>`;
      return;
    }
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ claim_ids: ids }),
    });
    const data = await res.json().catch(() => ({ groups: [] }));
    if (!data.groups?.length) {
      box.innerHTML = `<div class="empty-inline"><h3>Not enough overlap</h3><p>The sources for your last answer do not share enough measurements to compare.</p></div>`;
      return;
    }
    box.innerHTML = data.groups
      .map((g) => {
        const rows = (g.rows || [])
          .map(
            (r) => `<tr>
            <td>${escapeHtml(r.paper_title || "Study")}</td>
            <td>${r.value != null ? escapeHtml(String(r.value)) : "-"} ${escapeHtml(r.unit || "")}</td>
            <td class="stance-${r.stance || ""}">${escapeHtml(r.stance || "")}</td>
            <td>${escapeHtml(Object.entries(r.conditions || {}).map(([k, v]) => `${k}: ${v}`).join("; ") || "-")}</td>
          </tr>`
          )
          .join("");
        return `<div style="margin-bottom:24px">
          <h3 style="margin:0 0 6px;font-size:18px;font-weight:600">${escapeHtml(g.subject || "Topic")}</h3>
          <p style="margin:0 0 12px;color:var(--on-surface-variant)">${escapeHtml(g.narrative || "")}</p>
          <div class="table-wrap"><table>
            <thead><tr><th>Study</th><th>Value</th><th>Agreement</th><th>Conditions</th></tr></thead>
            <tbody>${rows}</tbody>
          </table></div>
        </div>`;
      })
      .join("");
  }

  function refreshSettings() {
    const dark = document.documentElement.classList.contains("dark");
    $("#theme-toggle").classList.toggle("on", dark);
    $("#theme-toggle").setAttribute("aria-pressed", String(dark));
    $("#theme-desc").textContent = dark ? "Using dark mode" : "Using light mode";

    const voiceOn = !!(state.config && state.config.tts_enabled && state.config.voice?.present);
    const pill = $("#voice-status");
    if (voiceOn) {
      pill.className = "status-pill ok";
      pill.textContent = "Ready";
      $("#voice-desc").textContent = "Tap Listen under an answer to hear it";
    } else if (state.config?.tts_enabled) {
      pill.className = "status-pill warn";
      pill.textContent = "Unavailable";
      $("#voice-desc").textContent = "Voice is not installed on this laptop";
    } else {
      pill.className = "status-pill";
      pill.textContent = "Off";
      $("#voice-desc").textContent = "Turned off for now";
    }
  }

  async function loadIntegrity() {
    const row = $("#integrity-row");
    row.hidden = false;
    const res = await fetch("/api/integrity");
    const data = await res.json().catch(() => ({}));
    const overall = (data.overall || "").toLowerCase();
    let cls = "warn";
    let label = "Needs attention";
    let fallback = "Checked what is installed on this laptop.";
    if (overall === "verified") {
      cls = "ok";
      label = "All set";
      fallback = "Everything looks ready.";
    } else if (overall === "unverified") {
      cls = "ok";
      label = "Ready";
      fallback = "Your files are present and ready to use.";
    } else if (overall === "incomplete") {
      cls = "warn";
      label = "Something is missing";
      fallback = "One or more pieces needed to answer are missing.";
    } else if (overall === "mismatch") {
      cls = "bad";
      label = "Needs fixing";
      fallback = "Something changed and should be checked.";
    }
    const friendlyHeadline = (() => {
      const h = String(data.headline || "").toLowerCase();
      if (!h) return fallback;
      if (h.includes("hash") || h.includes("manifest") || h.includes("gguf") || h.includes("component")) {
        return fallback;
      }
      return String(data.headline || fallback).replace(/\u2014/g, ",").replace(/\s*—\s*/g, ", ");
    })();
    $("#integrity-summary").innerHTML = `<span class="status-pill ${cls}">${label}</span>
      <p class="integrity-copy">${escapeHtml(friendlyHeadline)}</p>`;

    const comps = data.components || [];
    const friendlyName = (name) => {
      const n = String(name || "").toLowerCase();
      if (n.includes("database") || n.includes("corpus") || n.includes("db") || n.includes("evidence"))
        return "Research library";
      if (n.includes("model") || n.includes("gguf")) return "Answering";
      if (n.includes("voice") || n.includes("piper") || n.includes("tts")) return "Voice";
      if (n.includes("app") || n.includes("code")) return "App";
      return "Part";
    };
    if (comps.length) {
      $("#integrity-items").innerHTML = comps
        .map((c) => {
          const ok = c.present && (c.status === "verified" || c.status === "unverified");
          const label = ok ? (c.status === "verified" ? "Ready" : "Present") : "Not ready";
          return `<div class="list-item"><strong>${escapeHtml(friendlyName(c.name))}</strong><div class="meta">${label}</div></div>`;
        })
        .join("");
    } else {
      $("#integrity-items").innerHTML = "";
    }
  }

  function autosize(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 72) + "px";
  }

  function applyTheme(dark) {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("mufasa-theme", dark ? "dark" : "light");
    const themeBtn = $("#btn-theme");
    const next = dark ? "Switch to light theme" : "Switch to dark theme";
    themeBtn.title = next;
    themeBtn.setAttribute("aria-label", next);
    refreshSettings();
  }

  function setCollapsed(collapsed) {
    const app = document.querySelector(".app");
    app.classList.toggle("sidebar-collapsed", collapsed);
    const btn = $("#btn-collapse");
    const label = collapsed ? "Open sidebar" : "Close sidebar";
    btn.title = label;
    btn.setAttribute("aria-label", label);
    btn.setAttribute("data-tip", label);
    localStorage.setItem("mufasa-sidebar", collapsed ? "collapsed" : "open");
  }

  async function boot() {
    hideEvidence();
    const dark = localStorage.getItem("mufasa-theme") === "dark";
    document.documentElement.classList.toggle("dark", dark);
    const themeBtn = $("#btn-theme");
    const next = dark ? "Switch to light theme" : "Switch to dark theme";
    themeBtn.title = next;
    themeBtn.setAttribute("aria-label", next);
    try {
      const cfg = await fetch("/api/config").then((r) => r.json());
      state.config = cfg;
    } catch {
      /* library status is on Coverage, not the home screen */
    }
    refreshSettings();
    updateAskHint();
    await fillRecent();
  }

  $$(".nav button[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  $("#btn-new").addEventListener("click", newSession);
  $("#brand-home").addEventListener("click", (e) => {
    e.preventDefault();
    newSession();
  });
  $("#btn-menu").addEventListener("click", openDrawer);
  $("#backdrop").addEventListener("click", closeDrawer);
  $("#btn-collapse").addEventListener("click", () => {
    if (window.matchMedia("(max-width: 899px)").matches) {
      closeDrawer();
      return;
    }
    const collapsed = !document.querySelector(".app").classList.contains("sidebar-collapsed");
    setCollapsed(collapsed);
  });
  $("#btn-theme").addEventListener("click", () =>
    applyTheme(!document.documentElement.classList.contains("dark"))
  );
  $("#theme-toggle").addEventListener("click", () =>
    applyTheme(!document.documentElement.classList.contains("dark"))
  );
  $("#ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    ask($("#ask-input").value);
  });
  $("#ask-input").addEventListener("input", (e) => autosize(e.target));
  $("#ask-input").addEventListener("focus", () => {
    if (state.busy) return;
    stopPlaceholderTyping();
    const input = $("#ask-input");
    if (!input.value.trim()) {
      const text = placeholderPool()[0];
      input.placeholder = text;
      frozenPlaceholder = text;
    }
  });
  $("#ask-input").addEventListener("blur", () => {
    if (!state.busy && !$("#ask-input").value.trim()) refreshPlaceholderTyping();
  });
  $("#ask-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      ask($("#ask-input").value);
    }
  });
  $("#btn-stop").addEventListener("click", stop);
  $("#btn-mic").addEventListener("click", () => toast("Voice typing is not available yet"));
  $("#btn-integrity").addEventListener("click", loadIntegrity);
  $("#btn-close-evidence")?.addEventListener("click", hideEvidence);

  if (localStorage.getItem("mufasa-sidebar") === "collapsed" && window.matchMedia("(min-width: 900px)").matches) {
    setCollapsed(true);
  }
  boot();
})();
