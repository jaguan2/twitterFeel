/* Local-only desktop interface. Python calls use the native pywebview bridge. */
(() => {
  "use strict";
  const paths = {
    datasets: '<path d="M3 7h7l2 3h9v10H3zM3 7V4h7l2 3h8v3"/>',
    analysis:
      '<path d="M4 20h16M7 16v-5M12 16V4M17 16V8"/><circle cx="17" cy="5" r="2"/>',
    findings: '<path d="M5 3h14v18H5zM8 8h8M8 12h8M8 16h5"/>',
    overview:
      '<rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/>',
    explore:
      '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v7c0 4 16 4 16 0V5M4 12v7c0 4 16 4 16 0v-7"/>',
    review:
      '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 4V2h6v2M8 12l3 3 5-6"/>',
    lab: '<path d="M9 3h6M10 3v6L4 19a1.5 1.5 0 001.3 2h13.4a1.5 1.5 0 001.3-2L14 9V3M7 15h10"/>',
    experiments: '<path d="M4 20h17M7 16v-5M12 16V5M17 16V8"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    chevron: '<path d="M9 5l7 7-7 7"/>',
    refresh:
      '<path d="M20 7v5h-5M4 17v-5h5M5 8a8 8 0 0113-3l2 2M4 17l2 2a8 8 0 0013-3"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1 1M18 18l1 1M5 19l1-1M18 6l1-1"/>',
    moon: '<path d="M20 14a8 8 0 01-10-10 8 8 0 1010 10z"/>',
    check: '<path d="M5 12l4 4L19 6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7v1"/>',
    close: '<path d="M6 6l12 12M18 6L6 18"/>',
    export: '<path d="M12 3v12M7 10l5 5 5-5M4 16v5h16v-5"/>',
    user: '<circle cx="12" cy="8" r="4"/><path d="M4 22v-2a8 8 0 0116 0v2"/>',
    spark:
      '<path d="M12 3l2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5z"/>',
    clock: '<circle cx="12" cy="12" r="9"/><path d="M12 6v6l4 2"/>',
    brand: '<path d="M3 12h3l3-7 5 14 3-7h4"/>',
  };
  const icon = (name, size = 18) =>
    `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] || paths.info}</svg>`;
  const esc = (value) =>
    String(value ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  const nice = (value) =>
    String(value ?? "")
      .replace(/_/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase());
  const number = (value) => Number(value || 0).toLocaleString();
  const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
  const store = {
    get(key, fallback) {
      try {
        return localStorage.getItem(key) || fallback;
      } catch {
        return fallback;
      }
    },
    set(key, value) {
      try {
        localStorage.setItem(key, value);
      } catch {}
    },
  };
  const state = {
    page: "overview",
    boot: null,
    run: null,
    runId: null,
    review: null,
    reviewId: null,
    reviewer: store.get("tf-reviewer", "researcher"),
    theme: store.get("tf-theme", "system"),
    explorerTab: "posts",
    search: "",
    emotion: "",
    result: "",
    pageNumber: 0,
    reviewSplit: "development",
    reviewFilter: "pending",
    reviewIndex: 0,
    reviewLabels: [],
    reviewStatus: "reviewed",
    labTab: "threshold",
    threshold: 0.5,
    sandbox: null,
    experiment: null,
    evaluation: null,
    source: null,
    sourceId: null,
    trial: null,
    trialText: "I'm excited about moving, but scared to leave everyone behind.",
    trialThreshold: 0.5,
    intake: {
      name: "",
      content: "",
      format: "csv",
      token: null,
      preview: null,
      textColumn: "",
      userColumn: "",
    },
    selectedDataset: "",
    datasetDetail: null,
    analysisName: "",
    analysisThreshold: 0.5,
    emotionModel: "roberta",
    calibrationProfileId: "",
    includeSentiment: true,
    job: null,
    findings: null,
    findingDrafts: {},
    benchmarkResult: null,
    evalReviewId: null,
    glossarySearch: "",
  };
  const pages = {
    overview: ["Overview", "Your research, at a glance."],
    datasets: [
      "Datasets",
      "Bring your own posts into a versioned research library.",
    ],
    analysis: [
      "Run analysis",
      "Label existing posts with emotions and optional sentiment.",
    ],
    explore: ["Data explorer", "Explore the posts behind your experiments."],
    review: ["Label review", "Make independent judgments about the text."],
    lab: ["Model lab", "Explore model behavior before changing your approach."],
    experiments: [
      "Experiments",
      "Keep the settings, evidence, and results together.",
    ],
    findings: [
      "Findings & notes",
      "Turn the results into observations you can explain.",
    ],
  };
  let toastTimer,
    rangeTimer,
    rangeRequest = 0;
  async function api(action, payload = {}) {
    const reply = await window.pywebview.api.call(action, payload);
    if (!reply || !reply.ok)
      throw new Error(
        reply?.error || "The desktop connection returned no result.",
      );
    return reply.data;
  }
  function toast(message, error = false) {
    const el = document.getElementById("toast");
    el.textContent = message;
    el.className = `visible${error ? " error" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => (el.className = ""), error ? 8000 : 5000);
  }
  function pill(label, tone = "purple", dot = false) {
    return `<span class="pill ${tone}">${dot ? '<span class="pill-dot"></span>' : ""}${esc(label)}</span>`;
  }
  function statusBadge(status) {
    return pill(
      {
        labeled: "Emotional",
        neutral: "Neutral",
        uncertain: "Uncertain",
        review_needed: "Needs review",
        invalid: "Invalid",
      }[status] || nice(status),
      {
        labeled: "purple",
        neutral: "slate",
        uncertain: "amber",
        review_needed: "rose",
        invalid: "slate",
      }[status],
    );
  }
  function chips(labels) {
    return labels?.length
      ? `<div class="label-list">${labels.map((x) => `<span class="label-chip ${x === "neutral" ? "neutral" : ""}">${esc(x)}</span>`).join("")}</div>`
      : '<span class="muted small">No label selected</span>';
  }
  function options(items, value, label = (x) => nice(x)) {
    return items
      .map(
        (x) =>
          `<option value="${esc(typeof x === "object" ? x.id : x)}" ${String(typeof x === "object" ? x.id : x) === String(value) ? "selected" : ""}>${esc(label(x))}</option>`,
      )
      .join("");
  }
  function button(label, action, style = "", ico = "") {
    return `<button class="button ${style}" data-action="${action}">${ico ? icon(ico, 15) : ""}${label}</button>`;
  }
  function card(title, body, subtitle = "", extra = "") {
    return `<section class="card"><div class="card-head"><div><h2>${title}</h2>${subtitle ? `<div class="card-sub">${subtitle}</div>` : ""}</div>${extra}</div><div class="card-body">${body}</div></section>`;
  }
  function notice(message, type = "") {
    return `<div class="notice ${type}">${icon("info", 16)}<span>${message}</span></div>`;
  }
  function kpi(label, value, detail, ico = "explore", tone = "") {
    return `<div class="kpi"><div class="kpi-top"><span>${label}</span><span class="kpi-icon ${tone}">${icon(ico, 17)}</span></div><div class="kpi-number">${value}</div><div class="kpi-bottom">${detail}</div></div>`;
  }
  function countLabels(posts) {
    const result = {};
    for (const post of posts || [])
      for (const label of post.labels || [])
        result[label] = (result[label] || 0) + 1;
    return result;
  }
  function countStatus(posts) {
    const result = {};
    for (const row of posts || [])
      result[row.status] = (result[row.status] || 0) + 1;
    return result;
  }
  function bars(items, max = null, score = false) {
    const limit = max || Math.max(1, ...items.map((x) => x[1]));
    return items
      .map(
        ([label, value], i) =>
          `<div class="bar-row"><span class="bar-label">${esc(label)}</span><div class="bar-track"><div class="bar-fill" style="width:${Math.min(100, (value / limit) * 100)}%;opacity:${Math.max(0.45, 1 - i * 0.055)}"></div></div><span class="bar-value">${score ? Number(value).toFixed(3) : number(value)}</span></div>`,
      )
      .join("");
  }
  function pageHead(extra = "") {
    return `<div class="page-head"><div class="page-title"><span class="page-icon">${icon(state.page, 23)}</span><div><h1>${pages[state.page][0]}</h1><p>${pages[state.page][1]}</p></div></div><div class="actions">${extra}</div></div>`;
  }
  const themeMedia = window.matchMedia("(prefers-color-scheme: dark)");
  let themeSaveChain = Promise.resolve();
  function resolvedTheme() {
    return state.theme === "system" ? (themeMedia.matches ? "dark" : "light") : state.theme;
  }
  function applyTheme() {
    const theme = resolvedTheme();
    document.documentElement.dataset.theme = theme;
    document.body.dataset.theme = theme;
    document.querySelectorAll("[data-theme-choice]").forEach(button => {
      button.setAttribute("aria-checked", String(button.dataset.themeChoice === state.theme));
      button.tabIndex = button.dataset.themeChoice === state.theme ? 0 : -1;
    });
    const label = document.getElementById("appearance-current");
    if (label) label.textContent = `${state.theme === "system" ? `Follow Windows · currently ${nice(theme)}` : `${nice(theme)} mode`} · saved on this device`;
    const control = document.querySelector('[data-action="appearance"]');
    if (control) control.innerHTML = `${icon(theme === "dark" ? "moon" : "sun", 15)}<span>Appearance</span>`;
  }
  function showAppearance() {
    const dialog = document.getElementById("dialog");
    const modes = [{id:"light",title:"Light",text:"Bright surfaces. Clear contrast.",icon:"sun"},
      {id:"dark",title:"Dark",text:"Soft contrast for a dimmer room.",icon:"moon"},
      {id:"system",title:"Follow Windows",text:"Match your system appearance.",icon:"explore"}];
    dialog.classList.add("appearance-dialog");
    dialog.setAttribute("aria-label", "Appearance");
    dialog.innerHTML = `<div class="dialog-head"><div><span class="eyebrow">YOUR WORKSPACE</span><h2>Appearance</h2></div><button class="icon-button" data-action="close-dialog" aria-label="Close appearance">${icon("close",16)}</button></div><div class="dialog-body"><p class="small muted">Choose a look that works for you. Your preference is saved on this device.</p><div class="theme-choices" role="radiogroup" aria-label="Color theme">${modes.map(mode=>`<button class="theme-choice" role="radio" aria-checked="${mode.id===state.theme}" data-theme-choice="${mode.id}" data-action="choose-theme-${mode.id}"><span class="theme-preview" data-preview="${mode.id}" aria-hidden="true"><span class="preview-rail"><i></i><i></i><i></i></span><span class="preview-content"><span class="preview-heading"></span><span class="preview-cards"><i></i><i></i><i></i></span><span class="preview-chart"><i></i><i></i><i></i><i></i></span></span></span><span class="theme-choice-name">${icon(mode.icon,16)}${mode.title}<span class="theme-check">${icon("check",14)}</span></span><span class="theme-choice-copy">${mode.text}</span></button>`).join("")}</div><p id="appearance-current" class="micro muted theme-saved" role="status"></p><section class="design-concept"><span class="eyebrow">DESIGN CONCEPT</span><h3>Research desk</h3><p>A calm workspace for turning posts into findings. Clear steps, readable evidence, and a consistent place for every decision.</p><div class="concept-flow"><span><b>01</b> Import</span><span><b>02</b> Analyze</span><span><b>03</b> Review</span><span><b>04</b> Explain</span></div><div class="concept-principles"><span>Focused navigation</span><span>Readable charts</span><span>Light & dark parity</span></div></section></div>`;
    applyTheme();
    if (!dialog.open) dialog.showModal();
  }
  function navigation() {
    const groups = [["Workspace",["overview"]],["Research",["datasets","analysis","explore","review","findings"]],["Validation",["lab","experiments"]]];
    return groups.map(([label, keys])=>`<div class="nav-group"><div class="nav-label">${label}</div>${keys.map(key=>`<button class="nav-item ${state.page===key?"active":""}" data-nav="${key}" aria-current="${state.page===key?"page":"false"}" title="${pages[key][0]}">${icon(key)}<span>${pages[key][0]}</span>${key==="review"?'<span class="nav-count" id="nav-review-count"></span>':""}</button>`).join("")}</div>`).join("");
  }
  function shell() {
    applyTheme();
    document.getElementById("app").innerHTML =
      `<div class="shell"><aside class="sidebar"><div class="brand"><span class="brand-mark">${icon("brand", 23)}</span><div class="brand-copy">TwitterFeel<div class="brand-sub">Research workspace</div></div></div><nav aria-label="Main navigation">${navigation()}</nav><div class="sidebar-context"><label for="active-run">Active experiment</label><select id="active-run" aria-label="Active experiment">${options(state.boot.runs, state.runId, (r) => `${r.name} · ${number(r.rows)}`)}</select><div class="device"><span class="device-dot"></span>Local desktop application</div></div><button class="profile" data-action="profile" title="Change reviewer"><span class="avatar">${esc(state.reviewer.slice(0, 1).toUpperCase())}</span><div><div class="profile-name">${esc(state.reviewer)}</div><div class="profile-meta">Independent reviewer</div></div></button><button class="version-button" data-action="about">TwitterFeel v${esc(state.boot.app?.version || "0.2.0")} ${icon("info", 12)}</button></aside><div class="workspace"><header class="topbar"><div class="breadcrumb">Workspace ${icon("chevron", 12)} <b>${pages[state.page][0]}</b></div><div class="spacer"></div>${pill("Saved on this device", "slate", true)}<button class="appearance-control" data-action="appearance" aria-label="Appearance: choose light, dark, or Windows theme">${icon(resolvedTheme() === "dark" ? "moon" : "sun", 15)}<span>Appearance</span></button><button class="icon-button" data-action="refresh" title="Refresh workspace" aria-label="Refresh workspace">${icon("refresh", 16)}</button></header><main class="main" id="main"></main></div></div>`;
    document
      .getElementById("active-run")
      .addEventListener("change", async (e) => {
        await changeRun(e.target.value);
      });
    renderPage();
  }
  async function bootstrap() {
    state.boot = await api("bootstrap");
    if (state.boot.preferences?.reviewer)
      state.reviewer = state.boot.preferences.reviewer;
    if (["light", "dark", "system"].includes(state.boot.preferences?.theme))
      state.theme = state.boot.preferences.theme;
    state.runId = state.boot.runs.some((r) => r.id === state.runId)
      ? state.runId
      : state.boot.runs.find((r) => r.id.startsWith("runs/analyses/"))?.id ||
        state.boot.runs[0]?.id;
    state.reviewId = state.boot.reviews.some((r) => r.id === state.reviewId)
      ? state.reviewId
      : state.boot.reviews[0]?.id;
    if (state.runId) state.run = await api("load_run", { run_id: state.runId });
    else state.run = { posts: [], config: {}, summary: {} };
    state.threshold = Number(state.run.config.threshold || 0.5);
    state.sandbox = null;
    if (!state.selectedDataset)
      state.selectedDataset = state.boot.library?.[0]?.id || "";
    if (state.reviewId) await fetchReview();
    state.experiment = await api("experiments");
    shell();
    window.__desktop_ready = true;
  }
  async function changeRun(id) {
    try {
      state.runId = id;
      ++rangeRequest;
      clearTimeout(rangeTimer);
      const run = await api("load_run", { run_id: id });
      if (state.runId !== id) return;
      state.run = run;
      state.threshold = Number(state.run.config.threshold || 0.5);
      state.sandbox = null;
      state.pageNumber = 0;
      state.benchmarkResult = null;
      state.findings = null;
      if (state.page === "findings") await fetchFindings();
      renderPage();
    } catch (e) {
      toast(e.message, true);
    }
  }
  async function fetchReview() {
    const id = state.reviewId,
      reviewer = state.reviewer;
    const result = await api("load_review", { review_id: id, reviewer });
    if (state.reviewId === id && state.reviewer === reviewer)
      state.review = result;
  }
  async function navigate(page) {
    if (!pages[page]) return;
    state.page = page;
    try {
      if (page === "review" && state.reviewId) await fetchReview();
      if (page === "experiments") state.experiment = await api("experiments");
      if (page === "findings" && state.runId) await fetchFindings();
      shell();
    } catch (e) {
      toast(e.message, true);
    }
  }
  function renderPage() {
    const main = document.getElementById("main");
    if (!main) return;
    main.innerHTML = {
      overview: renderOverview,
      datasets: renderDatasets,
      analysis: renderAnalysis,
      findings: renderFindings,
      explore: renderExplorer,
      review: renderReview,
      lab: renderLab,
      experiments: renderExperiments,
    }[state.page]();
    document.querySelectorAll("[data-nav]").forEach((el) => {
      el.classList.toggle("active", el.dataset.nav === state.page);
      el.setAttribute(
        "aria-current",
        el.dataset.nav === state.page ? "page" : "false",
      );
    });
    const count = document.getElementById("nav-review-count");
    if (count)
      count.textContent = state.review
        ? number(state.review.counts?.pending || 0)
        : "";
    bindPage();
  }
  function runThresholdLabel(run = state.run) {
    const config = run?.config || run || {};
    const name =
      config.calibration_profile?.name || run?.calibration_profile_name;
    return (
      name ||
      (config.settings?.threshold_mode === "per_label_profile" ||
      run?.threshold_mode === "per_label_profile"
        ? "Per-emotion profile"
        : Number(config.threshold ?? 0.5).toFixed(2))
    );
  }
  function renderOverview() {
    const posts = state.run.posts,
      c = countStatus(posts),
      labels = Object.entries(countLabels(posts))
        .filter(([x]) => x !== "neutral")
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8),
      reviewed = state.review?.counts?.reviewed || 0,
      total = state.review?.rows?.length || 0;
    const colors = {
      labeled: "#7265df",
      neutral: "#a5b0c5",
      uncertain: "#e2b36d",
      review_needed: "#cb8292",
      invalid: "#807e8c",
    };
    let offset = 0;
    const circumference = 2 * Math.PI * 62;
    const segments = Object.entries(c)
      .map(([status, n]) => {
        const length = posts.length ? (n / posts.length) * circumference : 0;
        const svg = `<circle cx="85" cy="85" r="62" fill="none" stroke="${colors[status] || "#aaa"}" stroke-width="15" stroke-dasharray="${length} ${circumference - length}" stroke-dashoffset="${-offset}" transform="rotate(-90 85 85)"/>`;
        offset += length;
        return svg;
      })
      .join("");
    const checks = state.experiment?.diagnostics?.[0]?.report?.findings || [];
    const parity = checks.filter((x) => x.case !== "blank");
    const passed = parity.filter(
      (x) => x.scores_match && x.labels_match && x.status_matches,
    ).length;
    return (
      pageHead(pill("Model-generated labels", "purple", true)) +
      `<div class="workflow-banner"><div><span class="eyebrow">START WITH YOUR OWN DATA</span><h2>Import posts. Run the models. Review the evidence.</h2><p>Emotions describe expressions such as joy or fear. Sentiment describes positive, negative, or neutral tone.</p></div><div class="actions">${button("Import a dataset", "go-datasets", "primary", "datasets")}${button("Run analysis", "go-analysis", "", "arrow")}</div></div>` +
      (state.runId && !state.runId.startsWith("runs/analyses/")
        ? notice(
            "You are viewing an existing project experiment. Import your own dataset to start a new analysis; nothing is imported or analyzed automatically.",
          )
        : "") +
      sentimentSummary(posts) +
      `<div class="kpis">${kpi("Posts in this experiment", number(posts.length), "Exact text and source identity", "explore", "blue")}${kpi("Emotional posts", number(c.labeled || 0), "One or more emotions selected", "spark")}${kpi("Needs a closer look", number((c.uncertain || 0) + (c.review_needed || 0) + (c.invalid || 0)), "Uncertain, conflicting, or invalid", "clock", "amber")}${kpi("Human-reviewed posts", number(reviewed), `${number(total)} in the active review sample`, "review", "green")}</div><div class="grid-two">${card("Emotion distribution", labels.length ? bars(labels) + `<div class="bar-axis"><span>0</span><span>${number(Math.max(...labels.map((x) => x[1])))}</span></div>` : '<div class="empty">No emotional labels in this experiment.</div>', "Most frequent non-neutral labels", pill("Top 8", "slate"))}${card(
        "Review pipeline",
        `<div class="donut-wrap"><svg class="donut" viewBox="0 0 170 170" role="img" aria-label="Model result distribution">${segments}<text x="85" y="83" text-anchor="middle" class="number">${number(posts.length)}</text><text x="85" y="101" text-anchor="middle" class="caption">posts analyzed</text></svg><div class="legend">${Object.entries(
          c,
        )
          .map(
            ([status, n]) =>
              `<div class="legend-row"><span class="legend-dot" style="background:${colors[status]}"></span><span>${{ labeled: "Emotional", neutral: "Neutral", uncertain: "Uncertain", review_needed: "Needs review", invalid: "Invalid" }[status] || nice(status)}</span><span class="count">${number(n)}</span><span class="percent">${Math.round((n / Math.max(1, posts.length)) * 100)}%</span></div>`,
          )
          .join(
            "",
          )}</div></div><div class="review-callout">${icon("review", 21)}<div><h3>Build the human benchmark</h3><p>Independent judgments help us find what the model gets right—and what it misses.</p><div class="progress"><span style="width:${total ? (reviewed / total) * 100 : 0}%"></span></div><div class="split" style="margin-top:10px"><span class="micro muted">${number(reviewed)} of ${number(total)} reviewed</span>${button("Review posts", "go-review", "ghost", "arrow")}</div></div></div>`,
        "Model results at the saved threshold",
      )}</div><div class="grid-three gap">${card("Active experiment", `<div class="detail-row"><span>Model</span><span>RoBERTa · GoEmotions</span></div><div class="detail-row"><span>Vocabulary</span><span>27 emotions + neutral</span></div><div class="detail-row"><span>Label threshold</span><span>${esc(runThresholdLabel())}</span></div><div class="detail-row"><span>Evaluation</span>${pill("Human review pending", "amber")}</div>`)}${card("Recent model checks", `<div class="check-row"><span>App and CLI consistency</span>${pill(parity.length ? `${passed}/${parity.length} agree` : "Not run", parity.length && passed === parity.length ? "green" : "slate")}</div><div class="check-row"><span>Mixed emotions</span>${pill(checks.length ? "Needs review" : "Not run", checks.length ? "amber" : "slate")}</div><div class="check-row"><span>Sarcasm & context</span>${pill(checks.length ? "Needs review" : "Not run", checks.length ? "amber" : "slate")}</div>${button("View evidence", "go-experiments", "ghost", "arrow")}`)}${card("A note on interpretation", `<span class="kpi-icon" style="display:inline-flex">${icon("info", 19)}</span><p class="research-note">A high score means stronger model support for a label. It does not measure emotion intensity or establish how the author feels.</p>`)}</div><p class="footer-note">Prediction counts describe model behavior. Labeling quality requires human-reference evaluation.</p>`
    );
  }
  function sentimentBadge(sentiment) {
    if (!sentiment?.label)
      return '<span class="muted micro">Not analyzed</span>';
    return pill(
      nice(sentiment.label),
      { positive: "green", negative: "rose", neutral: "slate" }[
        sentiment.label
      ] || "slate",
    );
  }
  function sentimentSummary(posts) {
    const counts = {};
    for (const row of posts)
      if (row.sentiment?.label)
        counts[row.sentiment.label] = (counts[row.sentiment.label] || 0) + 1;
    if (!Object.keys(counts).length) return "";
    return `<section class="sentiment-strip"><div><h3>Sentiment, scored separately</h3><p class="small muted">Overall tone can differ from the emotions expressed.</p></div><div class="sentiment-counts">${["positive", "neutral", "negative"].map((label) => `<div>${sentimentBadge({ label })}<strong>${number(counts[label])}</strong></div>`).join("")}</div></section>`;
  }
  function previewTable(preview, columns) {
    if (!preview?.length) return '<div class="empty">No preview rows.</div>';
    return `<div class="table-wrap import-preview"><table><thead><tr>${columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${preview
      .slice(0, 10)
      .map(
        (row) =>
          `<tr>${columns.map((c) => `<td>${esc(row[c])}</td>`).join("")}</tr>`,
      )
      .join("")}</tbody></table></div>`;
  }
  function renderDatasets() {
    const draft = state.intake,
      preview = draft.preview,
      library = state.boot.library || [];
    return (
      pageHead(pill(`${number(library.length)} saved versions`, "slate")) +
      `<div class="steps"><span class="step active"><b>1</b> Add posts</span><span class="step"><b>2</b> Choose the text column</span><span class="step"><b>3</b> Save a dataset version</span></div>` +
      `<div class="intake-layout"><section class="card"><div class="card-head"><div><h2>Add a dataset</h2><div class="card-sub">CSV, TSV, JSONL, or one post per line in TXT · up to 20 MB</div></div>${button("Choose file", "pick-dataset", "", "datasets")}</div><div class="card-body"><div class="field-group"><label for="import-name" class="field-label">Dataset name</label><input id="import-name" maxlength="120" value="${esc(draft.name)}" placeholder="For example, October product feedback"></div><div class="split"><label for="import-format" class="field-label">Paste format</label><select id="import-format">${options(["csv", "tsv", "txt", "jsonl"], draft.format, (x) => x.toUpperCase())}</select></div><div class="field-group"><label for="import-content" class="field-label">Paste your data</label><textarea id="import-content" class="paste-area mono" spellcheck="false" placeholder="Paste a CSV with a header, or select TXT for one post per line.">${esc(draft.content)}</textarea><div class="split micro muted"><span>Use Ctrl+V, or read the clipboard using the button below.</span><span>${number(draft.content.length)} characters</span></div></div><div class="actions">${button("Paste from clipboard", "paste-clipboard", "", "review")}${button("Preview data", "preview-import", "primary", "arrow")}</div></div></section><aside>${card("Your data stays on this computer", '<p class="small muted">Importing saves a copy of your dataset. It does not run the models. You choose the data and settings in Run analysis.</p><div class="detail-row"><span>Original text</span><span>Preserved</span></div><div class="detail-row"><span>Previous versions</span><span>Kept</span></div><div class="detail-row"><span>Model labels in input</span><span>Ignored</span></div><p class="micro muted gap">Every saved version records its source, column mapping, fingerprint, and app version. Blank posts remain visible as unusable text.</p>')}</aside></div>` +
      (preview
        ? `<section class="card gap"><div class="card-head"><div><h2>Check your column mapping</h2><div class="card-sub">${number(preview.rows)} rows · first ${number(preview.preview?.length)} shown</div></div>${pill("Not saved yet", "amber")}</div><div class="card-body"><div class="mapping-grid"><div class="field-group"><label for="import-text-column" class="field-label">Which column contains the post text?</label><select id="import-text-column"><option value="">Choose a text column</option>${options(preview.columns, draft.textColumn, (x) => x)}</select></div><div class="field-group"><label for="import-user-column" class="field-label">User identifier (optional)</label><select id="import-user-column"><option value="">No user identifier</option>${options(preview.columns, draft.userColumn, (x) => x)}</select></div></div>${previewTable(preview.preview, preview.columns)}<div class="review-controls"><p class="small muted">Saving the same name creates another version. Existing versions stay available.</p>${button("Save dataset version", "import-dataset", "primary", "check")}</div></div></section>`
        : "") +
      `<section class="card gap"><div class="card-head"><div><h2>Dataset library</h2><div class="card-sub">Your imported data, with its history intact</div></div></div>${library.length ? `<div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Version</th><th>Posts</th><th>Imported</th><th></th></tr></thead><tbody>${library.map((item) => `<tr><td><strong>${esc(item.name)}</strong><div class="micro muted">App ${esc(item.app_version)}</div></td><td>${pill(`v${item.version}`, "slate")}</td><td>${number(item.rows)}</td><td class="small muted">${esc(item.created_at?.replace("T", " ").slice(0, 16))}</td><td><button class="button ghost" data-dataset="${esc(item.id)}">Inspect ${icon("chevron", 14)}</button></td></tr>`).join("")}</tbody></table></div>` : '<div class="empty"><h3>Your library is ready</h3><p>Choose a file or paste posts above to create your first dataset.</p></div>'}</section>` +
      (state.datasetDetail
        ? card(
            "Dataset details",
            `<div class="split"><p class="small muted">${esc(state.datasetDetail.manifest.name)} · version ${number(state.datasetDetail.manifest.version)} · ${number(state.datasetDetail.rows)} posts</p>${button("Create review sample", "create-review", "", "review")}${button("Analyze this dataset", "analyze-selected", "primary", "arrow")}</div><p class="micro mono muted gap">${esc(state.datasetDetail.manifest.dataset_sha256)}</p>${previewTable(state.datasetDetail.preview, Object.keys(state.datasetDetail.preview?.[0] || {}))}`,
            "Saved text and identity; exact file fingerprints make this version traceable.",
          )
        : "")
    );
  }
  function renderAnalysis() {
    const library = state.boot.library || [],
      dataset = library.find((x) => x.id === state.selectedDataset),
      running = state.job && ["queued", "running"].includes(state.job.status);
    return (
      pageHead() +
      `<div class="grid-two"><section class="card"><div class="card-head"><div><h2>Set up an analysis</h2><div class="card-sub">Choose an imported dataset. Each run is saved separately.</div></div></div><div class="card-body"><div class="field-group"><label for="analysis-dataset" class="field-label">Dataset version</label><select id="analysis-dataset"><option value="">Choose your dataset</option>${options(library, state.selectedDataset, (x) => `${x.name} · v${x.version} · ${number(x.rows)} posts`)}</select>${!library.length ? `<p class="small muted gap">No imported datasets yet. ${button("Add your data", "go-datasets", "ghost")}</p>` : ""}</div><div class="field-group"><label for="analysis-name" class="field-label">Run name (optional)</label><input id="analysis-name" value="${esc(state.analysisName)}" placeholder="For example, first pass at 0.50"></div><div class="model-choice"><span class="model-choice-icon">${icon("spark", 20)}</span><div><strong>Emotion labels</strong><p class="small muted">GoEmotions · 27 emotions plus neutral. Several emotions can appear in one post.</p></div>${pill("Included", "purple")}</div><label class="model-choice checkbox-label"><input type="checkbox" id="include-sentiment" ${state.includeSentiment ? "checked" : ""}><div><strong>Also analyze sentiment</strong><p class="small muted">Twitter RoBERTa · positive, negative, or neutral tone. This runs a separate model.</p></div></label>${renderEmotionModelSettings()}${renderThresholdSettings()}<div class="review-controls"><span class="small muted">${dataset ? `${number(dataset.rows)} posts · saved dataset v${dataset.version}` : "Select a dataset to begin"}</span><button class="button primary" data-action="start-analysis" ${!dataset || running ? "disabled" : ""}>${icon("analysis", 16)} ${running ? "Analysis in progress" : "Run analysis"}</button></div></div></section><aside>${card("What happens when you run it?", '<ol class="explain-list"><li><strong>Read your saved posts.</strong> The original text stays attached to each result.</li><li><strong>Score the labels.</strong> The emotion model considers each emotion independently. The sentiment model scores overall tone.</li><li><strong>Flag difficult cases.</strong> Blank, uncertain, conflicting, or shortened text needs attention.</li><li><strong>Save the evidence.</strong> Results, model versions, settings, and dataset fingerprints stay together.</li></ol><p class="small muted gap">The models are already trained. This step labels existing posts; it does not train a new model or predict a person’s future emotions.</p>')}${notice("These are model judgments. Scores are support for a label, not emotion intensity or proof of how someone feels.")}</aside></div><div id="analysis-progress">${renderJob()}</div>`
    );
  }
  function renderEmotionModelSettings() {
    const models = state.boot.emotion_models || [{id:"roberta",name:"RoBERTa · current baseline",available:true}];
    return `<div class="field-group"><label class="field-label" for="emotion-model">Emotion model</label><select id="emotion-model">${models.map(model=>`<option value="${esc(model.id)}" ${model.id===state.emotionModel?"selected":""} ${model.available?"":"disabled"}>${esc(model.name)}${model.available?"":" · not installed"}</option>`).join("")}</select><p class="micro muted gap">ModernBERT is a research option with improved scores on our published development comparison. Your own posts still need review. Both models run locally.</p></div>`;
  }
  function renderThresholdSettings() {
    const selectedModel = (state.boot.emotion_models || []).find(model=>model.id===state.emotionModel);
    const profiles = (state.boot.calibration_profiles || []).filter(profile=>!selectedModel || profile.model_provenance?.id===selectedModel.model),
      profile = profiles.find((item) => item.id === state.calibrationProfileId);
    return `<div class="field-group threshold-settings"><label for="calibration-profile" class="field-label">Emotion threshold settings</label><select id="calibration-profile"><option value="">Manual shared threshold</option>${options(profiles, state.calibrationProfileId, (item) => `${item.name} · provisional`)}</select><p class="micro muted gap">Choose a saved profile explicitly, or set one threshold for all emotion labels. Sentiment uses its separate model.</p>${
      profile
        ? `<div class="profile-summary gap"><div class="split"><h3>${esc(profile.name)}</h3>${pill("Provisional profile", "amber")}</div><p class="small muted gap">${profile.method === "regularized_per_label" ? "This profile uses a separate cutoff for each emotion, with labels that have little training support using the shared cutoff." : "This profile uses cutoffs selected from the published development reference."} It changes which labels are selected, while preserving the model’s scores. Performance on your own posts still needs human review.</p><details class="glossary gap"><summary>Inspect saved emotion cutoffs</summary><div class="table-wrap"><table><thead><tr><th>Emotion</th><th>Cutoff</th></tr></thead><tbody>${Object.entries(
            profile.thresholds || {},
          )
            .map(
              ([label, threshold]) =>
                `<tr><td>${esc(nice(label))}</td><td>${Number(threshold).toFixed(2)}</td></tr>`,
            )
            .join(
              "",
            )}</tbody></table></div><p class="micro muted">Shared setting: ${profile.shared_threshold == null ? "Not specified" : Number(profile.shared_threshold).toFixed(2)} · method: ${esc(nice(profile.method))}</p></details></div>`
        : state.calibrationProfileId
          ? notice(
              "The selected saved profile is no longer available. Choose a valid profile or Manual shared threshold before running analysis.",
              "warn",
            )
          : `<div class="field-group"><label for="analysis-threshold" class="field-label">Emotion label threshold <span id="analysis-threshold-value">${state.analysisThreshold.toFixed(2)}</span></label><input id="analysis-threshold" type="range" min="0.05" max="0.95" step="0.05" value="${state.analysisThreshold}"><p class="micro muted">An emotion is selected when its support score meets this value. A higher threshold can miss more emotions. Choose settings using human-reviewed development posts.</p></div>`
    }</div>`;
  }
  function renderJob() {
    const job = state.job;
    if (!job) return "";
    const progress = Math.min(
      100,
      Math.max(
        0,
        (100 * Number(job.processed || 0)) /
          Math.max(1, Number(job.total || 0)),
      ),
    );
    return `<section class="card gap"><div class="card-body"><div class="split"><h2>${{ queued: "Waiting to start", running: "Analyzing your posts", complete: "Analysis saved", failed: "Analysis stopped" }[job.status] || esc(nice(job.status))}</h2>${pill(nice(job.status), job.status === "complete" ? "green" : job.status === "failed" ? "rose" : "purple")}</div><div class="progress gap"><span style="width:${job.status === "complete" ? 100 : progress}%"></span></div><p class="small muted gap">${number(job.processed)} of ${number(job.total)} posts processed. ${job.status === "running" ? "You can explore the app while this runs. Keep the application open until it finishes." : ""}</p>${job.error ? notice(esc(job.error), "warn") : ""}${job.status === "complete" ? `<div class="actions gap">${button("Open results", "open-job", "primary", "arrow")}${button("Read findings", "job-findings", "", "findings")}</div>` : ""}</div></section>`;
  }
  async function fetchFindings() {
    const id = state.runId;
    const result = await api("findings", { run_id: id });
    if (state.runId === id) state.findings = result;
  }
  function renderFindings() {
    if (!state.runId)
      return (
        pageHead() +
        notice(
          "Run an analysis to generate a plain-language summary of the results.",
        )
      );
    const data = state.findings,
      draft = state.findingDrafts[state.runId] || { title: "", note: "" };
    if (!data)
      return (
        pageHead() +
        '<div class="empty">Select an analysis to read its findings.</div>'
      );
    const report = data.report || {},
      notes = data.notes || [];
    return (
      pageHead(button("Export report", "export-findings", "", "export")) +
      `<div class="finding-overview"><span class="eyebrow">WHAT THE MODELS FOUND</span><h2>Results in plain language</h2><p>${esc(report.overview)}</p><div class="micro muted">Active run: ${esc(state.runId)}</div></div>` +
      `<div class="finding-grid">${(report.observations || []).map((item) => card(esc(item.title), `<p>${esc(item.text)}</p>${item.evidence ? `<details class="glossary"><summary>Show supporting numbers</summary><pre class="evidence">${esc(typeof item.evidence === "string" ? item.evidence : JSON.stringify(item.evidence, null, 2))}</pre></details>` : ""}`)).join("")}</div>` +
      card(
        "How to read these results",
        `<ul class="limitations">${(report.limitations || []).map((item) => `<li>${esc(item)}</li>`).join("")}</ul><p class="small muted gap">These observations are calculated from the saved model results. They do not establish labeling accuracy or a person’s mental state.</p>`,
      ) +
      `${data.can_save === false ? notice("This is a historical project run. Run an analysis on your imported data to attach new research observations.") : ""}<div class="grid-two gap"><section class="card"><div class="card-head"><div><h2>Add your own observation</h2><div class="card-sub">Separate your interpretation from the automatically counted results.</div></div></div><div class="card-body"><div class="field-group"><label for="finding-title" class="field-label">Title</label><input ${data.can_save === false ? "disabled" : ""} id="finding-title" value="${esc(draft.title)}" maxlength="160" placeholder="For example, excitement and fear appear together"></div><div class="field-group"><label for="finding-note" class="field-label">What did you observe, and what supports it?</label><textarea ${data.can_save === false ? "disabled" : ""} id="finding-note" placeholder="Describe the evidence, uncertainty, and what to check next.">${esc(draft.note)}</textarea></div><div class="review-controls"><span class="small muted">Author: ${esc(state.reviewer)} · saved notes are preserved</span>${data.can_save === false ? '<button class="button primary" disabled>Available for new analyses</button>' : button("Save observation", "save-finding", "primary", "check")}</div></div></section><section class="card"><div class="card-head"><div><h2>Research notes</h2><div class="card-sub">${number(notes.length)} observations attached to this run</div></div></div><div class="card-body">${notes.length ? notes.map((note) => `<article class="research-entry"><h3>${esc(note.title)}</h3><p class="micro muted">${esc(note.author)} · ${esc((note.created_at || note.saved_at || "").replace("T", " ").slice(0, 16))}</p><p class="note-text">${esc(note.note)}</p></article>`).join("") : '<p class="small muted">No authored observations yet. Use the form to record what you noticed and what still needs checking.</p>'}</div></section></div>`
    );
  }
  function renderExplorer() {
    return (
      pageHead() +
      `<div class="tabs"><button class="tab ${state.explorerTab === "posts" ? "active" : ""}" data-tab="explorer:posts">Labeled posts</button><button class="tab ${state.explorerTab === "source" ? "active" : ""}" data-tab="explorer:source">Source data</button></div>${state.explorerTab === "posts" ? `<section class="card"><div class="toolbar"><input type="search" id="post-search" aria-label="Search post text" placeholder="Search post text…" value="${esc(state.search)}"><select id="emotion-filter" aria-label="Filter by emotion"><option value="">All emotions</option>${options(state.boot.labels, state.emotion)}</select><select id="result-filter" aria-label="Filter by model result"><option value="">All results</option>${options(["labeled", "neutral", "uncertain", "review_needed", "invalid"], state.result)}</select></div><div id="post-results">${explorerTable()}</div></section>` : renderSource()}`
    );
  }
  function filteredPosts() {
    return state.run.posts.filter(
      (row) =>
        (!state.search ||
          row.text.toLowerCase().includes(state.search.toLowerCase())) &&
        (!state.emotion || row.labels.includes(state.emotion)) &&
        (!state.result || row.status === state.result),
    );
  }
  function explorerTable() {
    const rows = filteredPosts(),
      pageSize = 25;
    state.pageNumber = Math.min(
      state.pageNumber,
      Math.max(0, Math.ceil(rows.length / pageSize) - 1),
    );
    const start = state.pageNumber * pageSize,
      view = rows.slice(start, start + pageSize);
    return `<div class="table-wrap"><table class="post-table"><thead><tr><th style="width:58%">Post text</th><th>Emotion labels</th><th>Sentiment</th><th>Result</th><th></th></tr></thead><tbody>${view.map((row) => `<tr data-post="${esc(row.post_id)}" tabindex="0" role="button" aria-label="Inspect post"><td class="post-text"><span class="clamp">${esc(row.text)}</span></td><td>${chips(row.labels.slice(0, 3))}</td><td>${sentimentBadge(row.sentiment)}</td><td>${statusBadge(row.status)}</td><td>${icon("chevron", 15)}</td></tr>`).join("") || '<tr><td colspan="5" class="empty">No posts match these filters.</td></tr>'}</tbody></table></div><div class="table-footer"><span>${number(rows.length)} matching posts${rows.length ? ` · ${number(start + 1)}–${number(Math.min(start + pageSize, rows.length))}` : ""}</span><div class="pagination"><button data-action="posts-prev" aria-label="Previous page" ${state.pageNumber === 0 ? "disabled" : ""}>‹</button><span>${state.pageNumber + 1} / ${Math.max(1, Math.ceil(rows.length / pageSize))}</span><button data-action="posts-next" aria-label="Next page" ${start + pageSize >= rows.length ? "disabled" : ""}>›</button></div></div>`;
  }
  function renderSource() {
    const data = state.source;
    return `<div class="card"><div class="toolbar"><label for="source-dataset" class="small muted">Dataset</label><select id="source-dataset"><option value="">Select a source dataset</option>${options(state.boot.datasets, state.sourceId, (x) => x.name)}</select></div><div class="card-body">${data ? `<div class="kpis">${kpi("Export rows", number(data.rows), "Rows before review sampling", "explore")}${kpi("Users", data.users === null ? "—" : number(data.users), "Distinct source users", "user", "blue")}${kpi("Repeated texts", number(data.repeated_texts), "May include legitimate repetition", "clock", "amber")}${kpi("Blank texts", number(data.blank_texts), "Unusable without text", "review", "green")}</div><h3>Post length distribution</h3>${histogram(data.lengths)}<div class="split micro muted"><span>Shorter posts</span><span>Longer posts · characters</span></div><h3 class="gap">Source preview</h3><p class="small muted" style="margin:7px 0 15px">First ${data.preview.length} rows. Original text is preserved.</p><div class="table-wrap"><table><thead><tr><th>Post identifier</th><th>Text</th></tr></thead><tbody>${data.preview.map((row) => `<tr><td class="mono">${esc(row.post_id)}</td><td>${esc(row.text)}</td></tr>`).join("")}</tbody></table></div>` : '<div class="empty">Select a dataset to inspect its size, text quality, and source rows.</div>'}</div></div>`;
  }
  function histogram(values) {
    if (!values?.length) return '<div class="empty">No length data.</div>';
    const max = values.reduce((a, b) => Math.max(a, b), 1),
      bins = Array(24).fill(0);
    values.forEach((n) => bins[Math.min(23, Math.floor((n / max) * 24))]++);
    const top = Math.max(...bins, 1);
    return `<div class="histogram" role="img" aria-label="Post length histogram">${bins.map((n, i) => `<div class="hist-bar" style="height:${(n / top) * 100}%" title="${Math.round((i / 24) * max)}–${Math.round(((i + 1) / 24) * max)} characters: ${n} posts"></div>`).join("")}</div>`;
  }
  function reviewRows() {
    return (state.review?.rows || []).filter(
      (row) =>
        row.split === state.reviewSplit &&
        (state.reviewFilter === "all" ||
          row.review_status === state.reviewFilter),
    );
  }
  function currentReview() {
    const rows = reviewRows();
    state.reviewIndex = Math.min(
      Math.max(0, state.reviewIndex),
      Math.max(0, rows.length - 1),
    );
    return rows[state.reviewIndex];
  }
  function renderReview() {
    const data = state.review,
      row = currentReview(),
      total = data?.rows?.length || 0,
      c = data?.counts || {},
      done = total - (c.pending || 0),
      practice = data?.metadata?.purpose === "calibration";
    if (row) {
      state.reviewLabels = row.human_labels ? row.human_labels.split(";") : [];
      state.reviewStatus = ["reviewed", "unclear", "invalid"].includes(
        row.review_status,
      )
        ? row.review_status
        : "reviewed";
    }
    return (
      pageHead(
        (practice
          ? pill("Practice round", "amber")
          : button(
              "Prepare 30-post practice round",
              "prepare-calibration",
              "",
              "review",
            )) + button("Export my review", "export-review", "", "export"),
      ) +
      (practice
        ? notice(
            "Practice round: use these development posts to learn the rubric and record difficult distinctions. Thirty posts do not establish labeling accuracy; keep this separate from a final benchmark.",
            "warn",
          )
        : "") +
      renderLabelGuide() +
      `<div class="review-layout"><div><section class="card"><div class="toolbar"><select id="review-sample" aria-label="Review sample">${options(state.boot.reviews, state.reviewId, (x) => x.name)}</select><div class="spacer"></div><select id="review-split" aria-label="Review partition">${options(practice ? ["development"] : ["development", "test"], state.reviewSplit)}</select><select id="review-filter" aria-label="Review state">${options(["pending", "all", "reviewed", "unclear", "invalid"], state.reviewFilter)}</select></div><div class="card-body">${state.reviewSplit === "test" ? notice("Reserved test posts: choose model settings using development data before reviewing final test performance.", "warn") : ""}<div class="split small muted"><span>${number(done)} of ${number(total)} judgments saved</span>${pill(practice ? "Practice: model hints hidden" : "Blind review", practice ? "amber" : "green")}</div><div class="progress" style="margin-top:12px"><span style="width:${total ? (done / total) * 100 : 0}%"></span></div>${row ? `<blockquote class="post-quote">${esc(row.text)}</blockquote><div class="post-meta"><span>Post ${number(state.reviewIndex + 1)} of ${number(reviewRows().length)} in this queue</span><span class="mono">${esc(row.post_id)}</span></div><div class="field-group gap"><label class="field-label">Can this text be labeled?</label><div class="segmented" role="group" aria-label="Judgment type">${["reviewed", "unclear", "invalid"].map((x) => `<button class="segment ${state.reviewStatus === x ? "active" : ""}" data-disposition="${x}" aria-pressed="${state.reviewStatus === x}">${{ reviewed: "Select emotions", unclear: "Not enough context", invalid: "Unusable text" }[x]}</button>`).join("")}</div></div><div class="field-group"><label class="field-label">Emotions expressed</label><p class="small muted">Choose every supported emotion. Neutral stands alone.</p><div class="emotion-options" id="emotion-options">${emotionButtons()}</div></div><div class="field-group"><label class="field-label" for="review-notes">Evidence or notes</label><textarea id="review-notes" placeholder="What in the text supports your judgment?" style="min-height:80px">${esc(row.notes)}</textarea></div><div class="review-controls"><div class="pagination"><button data-action="review-prev" aria-label="Previous post" ${state.reviewIndex === 0 ? "disabled" : ""}>‹</button><button data-action="review-next" aria-label="Next post" ${state.reviewIndex + 1 >= reviewRows().length ? "disabled" : ""}>›</button></div>${button("Save judgment", "save-review", "primary", "check")}</div>` : '<div class="empty">No posts in this queue. Change the filter to revisit saved judgments.</div>'}</div></section></div><aside class="review-side">${card("Your review queue", `<div class="queue-item">Pending<span class="queue-count">${number(c.pending)}</span></div><div class="queue-item">Reviewed<span class="queue-count">${number(c.reviewed)}</span></div><div class="queue-item">Unclear<span class="queue-count">${number(c.unclear)}</span></div><div class="queue-item">Invalid<span class="queue-count">${number(c.invalid)}</span></div><p class="small muted" style="margin-top:15px">Reviewer: <strong>${esc(state.reviewer)}</strong></p>`)}<div class="card gap"><div class="card-body"><div class="section-label">${icon("info")}<h3>Keep your judgment independent</h3></div><p class="small muted">Model suggestions are hidden. Label the speaker's expression, rather than the topic or a quoted person's feelings.</p><details class="glossary"><summary>Review guidance</summary><p>Several emotions can coexist. Choose neutral only for understandable text with no clear emotion. Use unclear for missing context. Sarcasm and quotations deserve a closer look.</p><p>For example, “Excited to move, but afraid to leave” may support excitement and fear.</p></details><p class="micro muted" style="margin-top:16px">Each save preserves a revision. Other reviewers start with their own blank judgments.</p></div></div></aside></div>`
    );
  }
  function emotionButtons() {
    return state.boot.labels
      .map(
        (label) =>
          `<button class="emotion-option ${state.reviewLabels.includes(label) ? "active" : ""}" data-emotion="${label}" aria-pressed="${state.reviewLabels.includes(label)}" ${state.reviewStatus !== "reviewed" ? "disabled" : ""}>${label}</button>`,
      )
      .join("");
  }
  function renderLabelGuide() {
    const definitions =
      state.boot.annotation?.definitions || state.review?.definitions || {};
    if (!Object.keys(definitions).length) return "";
    return `<details class="label-guide"><summary>${icon("info", 16)} Emotion definitions and labeling guidance ${pill(`Rubric ${state.boot.annotation?.version || state.review?.metadata?.rubric_version || "1"}`, "slate")}</summary><div class="label-guide-body"><p class="small muted">Use the same definitions throughout your review. Label the speaker’s expressed emotions; a topic alone is not evidence of an emotion.</p><ul class="limitations">${(state.boot.annotation?.instructions || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul><label for="definition-search" class="field-label gap">Find an emotion or distinction</label><input id="definition-search" type="search" placeholder="For example, grief, sadness, or fear" value="${esc(state.glossarySearch)}"><div id="label-definitions">${renderDefinitions()}</div></div></details>`;
  }
  function renderDefinitions() {
    const definitions =
        state.boot.annotation?.definitions || state.review?.definitions || {},
      search = state.glossarySearch.toLowerCase().trim();
    const entries = Object.entries(definitions).filter(([label, value]) =>
      `${label} ${typeof value === "string" ? value : `${value.definition || ""} ${value.distinction || ""}`}`
        .toLowerCase()
        .includes(search),
    );
    return `<div class="definition-grid">${entries.map(([label, value]) => `<article><h3>${esc(nice(label))}</h3><p>${esc(typeof value === "string" ? value : value.definition)}</p>${typeof value === "object" && value.distinction ? `<p class="definition-distinction">${esc(value.distinction)}</p>` : ""}</article>`).join("") || '<p class="small muted">No definitions match your search.</p>'}</div>`;
  }
  function renderLab() {
    return (
      pageHead() +
      `<div class="tabs"><button class="tab ${state.labTab === "threshold" ? "active" : ""}" data-tab="lab:threshold">Threshold sandbox</button><button class="tab ${state.labTab === "try" ? "active" : ""}" data-tab="lab:try">Try a post</button></div>${state.labTab === "threshold" ? `<div class="card"><div class="card-body"><div class="split"><div><h2>Minimum support to select a label</h2><p class="card-sub">Explore a shared-threshold override on saved scores. Original predictions and saved profiles stay unchanged.</p></div>${pill(`Saved run: ${runThresholdLabel()}`, "slate")}</div><div class="range-row gap"><input id="threshold" type="range" min="0.05" max="0.95" step="0.05" value="${state.threshold}" aria-label="Minimum score threshold"><output class="range-value" id="threshold-value">${state.threshold.toFixed(2)}</output></div><div class="range-labels"><span>More labels · lower threshold</span><span>Fewer labels · higher threshold</span></div></div></div><div id="sandbox-results">${renderSandbox()}</div><div class="gap">${notice("Automatic acceptance describes coverage, not correctness. Tune settings using reviewed development posts and freeze them before test evaluation.")}</div>` : `<div class="grid-two"><section class="card"><div class="card-head"><h2>Analyze existing text</h2>${pill("Local inference", "green", true)}</div><div class="card-body"><p class="small muted">Model: ${esc(state.boot.model_name || "SamLowe/roberta-base-go_emotions")}</p><div class="field-group"><label class="field-label" for="trial-text">Post text</label><textarea id="trial-text" rows="5">${esc(state.trialText)}</textarea></div><div class="split"><label class="small muted" for="trial-threshold">Label threshold <select id="trial-threshold">${options([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], state.trialThreshold, (x) => Number(x).toFixed(2))}</select></label>${button("Analyze post", "infer", "primary", "spark")}</div><div id="trial-results">${renderTrial()}</div></div></section><div>${card("What this test tells you", `<p class="small muted">This is a live demonstration of the downloaded model. It is separate from saved experiment predictions.</p><div class="detail-row"><span>Multiple emotions</span><span>Allowed</span></div><div class="detail-row"><span>No label passes</span><span>Uncertain</span></div><div class="detail-row"><span>Neutral + emotion</span><span>Needs review</span></div><p class="research-note">A model can still miss mixed emotions, sarcasm, and context. Inspect the scores and compare against independent judgments.</p>`)}</div></div>`}`
    );
  }
  function renderSandbox() {
    const posts = state.sandbox?.posts || state.run.posts,
      c = countStatus(posts),
      accepted = (c.labeled || 0) + (c.neutral || 0),
      multi = posts.filter(
        (p) => p.labels.filter((x) => x !== "neutral").length > 1,
      ).length,
      before = countLabels(state.run.posts),
      after = countLabels(posts);
    const selected = ["joy", "sadness", "anger", "fear", "excitement", "love"];
    const pairs = selected.map((a) =>
        selected.map(
          (b) =>
            posts.filter((p) => p.labels.includes(a) && p.labels.includes(b))
              .length,
        ),
      ),
      max = Math.max(1, ...pairs.flat());
    return `<div class="lab-kpis">${kpi("Automatic acceptance", pct(accepted / Math.max(posts.length, 1)), "A result without a review flag", "check", "green")}${kpi("Multiple emotions", number(multi), "Two or more non-neutral labels", "spark")}${kpi("Uncertain or conflicting", number((c.uncertain || 0) + (c.review_needed || 0)), "Needs closer inspection", "clock", "amber")}</div><div class="grid-two">${card(
      "Label frequency comparison",
      `<div class="table-wrap"><table><thead><tr><th>Emotion</th><th>Saved run</th><th>Sandbox</th><th>Change</th></tr></thead><tbody>${[
        ...state.boot.labels,
      ]
        .sort((a, b) => (after[b] || 0) - (after[a] || 0))
        .map(
          (label) =>
            `<tr><td>${nice(label)}</td><td>${number(before[label])}</td><td class="strong">${number(after[label])}</td><td class="muted">${(after[label] || 0) - (before[label] || 0) > 0 ? "+" : ""}${(after[label] || 0) - (before[label] || 0)}</td></tr>`,
        )
        .join("")}</tbody></table></div>`,
      "How the threshold changes selected labels",
    )}${card("Emotion co-occurrence", `<div class="heatmap" style="grid-template-columns:70px repeat(6,minmax(0,1fr))"><span></span>${selected.map((x) => `<span class="heat-label">${nice(x)}</span>`).join("")}${selected.map((a, i) => `<span class="heat-label">${nice(a)}</span>${selected.map((b, j) => `<span class="heat-cell" style="background:rgba(115,98,215,${0.06 + (pairs[i][j] / max) * 0.72});color:${pairs[i][j] / max > 0.5 ? "white" : "var(--text)"}" title="${a} and ${b}: ${pairs[i][j]} posts">${pairs[i][j]}</span>`).join("")}`).join("")}</div><p class="small muted" style="margin-top:22px">Counts of posts assigned both labels. The diagonal shows each emotion's total. This chart does not establish emotional similarity.</p>`, "Selected emotion pairs")}</div>`;
  }
  function renderTrial() {
    const result = state.trial;
    if (!result || result.text !== state.trialText) return "";
    return `<div class="result-box"><div class="split"><h3>Selected emotions</h3>${statusBadge(result.status)}</div><div style="margin-top:12px">${chips(result.labels)}</div>${result.truncated ? `<div class="gap">${notice("Text exceeded the token limit and was truncated. Review the result.", "warn")}</div>` : ""}<div class="score-table">${bars(
      Object.entries(result.scores)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 12),
      1,
      true,
    )}</div><p class="micro muted">Independent model support scores; they need not add to 100%.</p></div>`;
  }
  function renderExperiments() {
    const ex = state.experiment || { evaluations: [], diagnostics: [] },
      evals = ex.evaluations || [],
      diags = ex.diagnostics || [];
    const latest = diags[0],
      findings = latest?.report?.findings || [],
      nonblank = findings.filter((x) => x.case !== "blank"),
      passed = nonblank.filter(
        (x) => x.scores_match && x.labels_match && x.status_matches,
      ).length;
    return (
      pageHead() +
      renderModelComparisons() +
      renderRefinements() +
      renderPublishedBenchmarks() +
      card(
        "Saved labeling runs",
        `<div class="table-wrap"><table><thead><tr><th>Experiment</th><th>Posts</th><th>Threshold</th><th>Model</th></tr></thead><tbody>${state.boot.runs.map((run) => `<tr><td class="strong">${esc(run.name)}</td><td>${number(run.rows)}</td><td>${esc(runThresholdLabel(run))}</td><td class="muted">${esc(run.model)}</td></tr>`).join("")}</tbody></table></div><p class="micro muted" style="margin-top:14px">Compare quality on the same posts, human labels, partition, and vocabulary. Prediction counts are not accuracy.</p>`,
      ) +
      `<div class="grid-two gap">${card("Evaluate human judgments", `<label class="field-label" for="eval-review">Review sample</label><select id="eval-review" style="width:100%">${options(state.boot.reviews, state.evalReviewId || state.reviewId, (x) => x.name)}</select><div class="field-group"><label class="field-label" for="eval-split">Evaluation partition</label><select id="eval-split"><option value="development">Development</option><option value="test">Reserved test</option></select></div><p class="small muted" style="margin-bottom:17px">Uses <strong>${esc(state.reviewer)}</strong>'s saved judgments and the active experiment's saved predictions.</p>${button("Evaluate selected run", "evaluate", "primary", "experiments")}<p class="micro muted" style="margin-top:15px">Single-reviewer draft labels are a preliminary reference. Test results should be inspected after settings are frozen.</p>`)}${card("Saved evaluations", evals.length ? `<select id="saved-evaluation" style="width:100%"><option value="">Choose an evaluation</option>${options(evals, state.evaluation?.id, (x) => `${x.metadata?.run_id || x.id} · ${x.metrics?.split || "development"} · ${x.metrics?.reviewed_rows || 0} posts`)}</select><p class="small muted" style="margin-top:17px">Each evaluation keeps a review snapshot, source hashes, settings, and per-label metrics.</p>` : `<div class="empty" style="padding:16px 0 25px">${icon("review", 27)}<p style="margin:12px 0">No human-reference evaluations yet.</p>${button("Review development posts", "go-review", "ghost", "arrow")}</div>`)}</div><div id="evaluation-results">${renderEvaluation()}</div>${renderBenchmark()}<section class="card gap"><div class="card-head"><div><h2>Model execution checks</h2><div class="card-sub">Real model · synthetic examples · app and CLI consistency</div></div>${pill(nonblank.length ? `${passed}/${nonblank.length} agree` : "Not run", nonblank.length && passed === nonblank.length ? "green" : "slate")}</div><div class="card-body">${findings.length ? `<div class="table-wrap"><table><thead><tr><th>Scenario</th><th>Synthetic example</th><th>Model labels</th><th>Result</th></tr></thead><tbody>${findings.map((row) => `<tr><td class="strong">${esc(nice(row.case))}</td><td class="post-text"><span class="clamp">${esc(row.text) || "(blank input)"}</span></td><td>${chips(row.cli_labels)}</td><td>${statusBadge(row.cli_status)}</td></tr>`).join("")}</tbody></table></div><div class="gap">${notice("Execution consistency is not labeling accuracy. Current examples expose missed mixed emotions, a sarcasm error, and ambiguous text labeled neutral.", "warn")}</div>` : '<p class="muted small">No diagnostic run found. Use the check_labeling command to create one.</p>'}</div></section>`
    );
  }
  function renderModelComparisons() {
    const comparisons = state.experiment?.model_comparisons || [];
    if (!comparisons.length) return "";
    const metric = (value) => (value == null ? "—" : Number(value).toFixed(3)),
      signed = (value) =>
        value == null
          ? "—"
          : `${Number(value) > 0 ? "+" : ""}${Number(value).toFixed(3)}`,
      names = {
        "fixed_0.5": "Fixed shared 0.50",
        "fixed_0.3": "Fixed shared 0.30",
        fine_shared: "Tuned shared cutoff",
        regularized_per_label: "Regularized cutoffs by emotion",
        regularized_macro_per_label: "Cutoffs balanced across emotions",
      };
    return (
      card(
        "Emotion model comparison",
        `<p class="small muted">A separate model can change the emotion scores themselves. This comparison evaluates ModernBERT against the RoBERTa baseline on the same published development reference, using the same threshold-selection procedures.</p><p class="small muted gap"><strong>The application default remains RoBERTa.</strong> Choose ModernBERT explicitly in Run analysis. Each model uses its own matching threshold settings.</p>${comparisons
          .map((item) => {
            const report = item.report || {},
              methods = Object.entries(report.methods || {});
            return `<section class="model-comparison-report gap"><div class="comparison-models"><div><span class="eyebrow">BASELINE MODEL</span><h3>${esc(report.baseline_model)}</h3></div><div><span class="eyebrow">RESEARCH CANDIDATE</span><h3>${esc(report.candidate_model)}</h3></div></div><p class="small muted gap">${number(report.rows)} reference posts · ${number(report.unique_texts)} exact text groups · reserved test ${report.test_scored ? "was scored" : "not scored"} · ${report.folds_verified_identical ? "identical development folds verified" : "matching development folds not verified"}.</p>${report.folds_verified_identical ? "" : notice("The report does not verify matching folds. Treat these model differences with caution.", "warn")}<div class="table-wrap gap"><table><thead><tr><th>Threshold method</th><th>RoBERTa F1</th><th>ModernBERT F1</th><th>Change in F1</th><th>95% paired interval</th><th>Macro F1<br>baseline / candidate</th><th>Mixed recovery<br>baseline / candidate</th></tr></thead><tbody>${methods.map(([method, row]) => `<tr><td class="strong">${esc(names[method] || nice(method))}</td><td>${metric(row.baseline?.micro_f1)}</td><td>${metric(row.candidate?.micro_f1)}</td><td>${signed(row.micro_f1_difference)}</td><td class="small">${row.paired_difference_interval ? `${signed(row.paired_difference_interval.lower)} to ${signed(row.paired_difference_interval.upper)}` : "—"}</td><td>${metric(row.baseline?.macro_f1_all_labels)} / ${metric(row.candidate?.macro_f1_all_labels)}</td><td>${row.baseline?.mixed_emotion?.full_label_recovery == null ? "—" : pct(row.baseline.mixed_emotion.full_label_recovery)} / ${row.candidate?.mixed_emotion?.full_label_recovery == null ? "—" : pct(row.candidate.mixed_emotion.full_label_recovery)}</td></tr>`).join("")}</tbody></table></div><div class="refinement-reading gap"><h3>Read the gains and the tradeoffs together</h3><p class="small muted gap">Positive F1 differences favor the alternative. Check macro F1 and mixed-emotion recovery alongside overall F1; a model can improve one measure while worsening another. Paired intervals estimate uncertainty on these same reference posts.</p><p class="small muted gap">Tuned cutoffs were chosen using other development folds. These remain existing models evaluated on GoEmotions; this is not a fresh independent Twitter test, and it does not establish improved accuracy on your own posts.</p></div><details class="glossary gap"><summary>Precision, recall, exact matches, and emotion-only scores</summary><div class="table-wrap gap"><table><thead><tr><th>Threshold method</th><th>Model</th><th>Precision</th><th>Recall</th><th>Exact set match</th><th>Emotion-only F1</th><th>Mixed label recall</th></tr></thead><tbody>${methods.map(([method, row]) => ["baseline", "candidate"].map((key) => `<tr><td>${esc(names[method] || nice(method))}</td><td>${key === "baseline" ? "Current" : "Alternative"}</td><td>${metric(row[key]?.micro_precision)}</td><td>${metric(row[key]?.micro_recall)}</td><td>${row[key]?.exact_match == null ? "—" : pct(row[key].exact_match)}</td><td>${metric(row[key]?.emotion_only_micro_f1)}</td><td>${row[key]?.mixed_emotion?.label_recall == null ? "—" : pct(row[key].mixed_emotion.label_recall)}</td></tr>`).join("")).join("")}</tbody></table></div></details>${(report.warnings || []).length ? `<details class="glossary gap"><summary>Comparison limitations and cautions</summary><ul class="limitations">${report.warnings.map((text) => `<li>${esc(text)}</li>`).join("")}</ul></details>` : ""}<details class="glossary gap"><summary>Model revisions and report provenance</summary><pre class="evidence">${esc(JSON.stringify({ id: item.id, candidate_revision: report.candidate_revision, baseline_model_provenance: report.baseline_model_provenance, candidate_model_provenance: report.candidate_model_provenance, hashes: report.hashes, folds_verified_identical: report.folds_verified_identical, scoring_seconds: report.scoring_seconds, run_id: report.run_id }, null, 2))}</pre></details></section>`;
          })
          .join("")}`,
        "Read-only development diagnostic",
        pill("Original default retained", "slate"),
      ) + '<div class="gap"></div>'
    );
  }
  function renderRefinements() {
    const reports = state.experiment?.refinements || [];
    return (
      card(
        "Threshold refinement",
        `<p class="small muted">Compare ways of selecting emotion cutoffs while holding the trained model fixed. Each development post is scored using cutoffs chosen from the other folds, with exact duplicate texts kept together.</p><p class="small muted gap">This check uses the model’s existing GoEmotions domain. It is not a fresh independent Twitter evaluation, and reserved test labels are not used.</p>${reports.length ? reports.map((item) => renderRefinementReport(item)).join("") : '<div class="empty"><p>No saved refinement comparison is available yet.</p><p class="small muted">New reports will appear here after the development comparison finishes. Manual shared thresholds remain available in Run analysis.</p></div>'}`,
        "Cross-validation estimates for threshold selection · provisional",
      ) + '<div class="gap"></div>'
    );
  }
  function renderRefinementReport(item) {
    const report = item.report || {},
      methods = Object.entries(report.methods || {}),
      names = {
        "fixed_0.5": "Fixed shared 0.50",
        "fixed_0.3": "Fixed shared 0.30",
        fine_shared: "Tuned shared cutoff",
        regularized_per_label: "Regularized cutoffs by emotion",
        regularized_macro_per_label: "Cutoffs balanced across emotions",
      },
      metric = (value) => (value == null ? "—" : Number(value).toFixed(3)),
      signed = (value) =>
        value == null
          ? "—"
          : `${Number(value) > 0 ? "+" : ""}${Number(value).toFixed(3)}`,
      selected = report.methods?.[report.selected_method],
      interval = selected?.paired_difference_interval,
      baseline = report.methods?.["fixed_0.3"];
    return `<section class="refinement-report gap"><div class="split"><h3>Five-fold development comparison</h3>${pill("Provisional · test held out", "amber")}</div><p class="small muted gap">Every row below combines predictions from the five held-out folds. The tuned settings for a post were selected without using that fold’s reference labels. Exact duplicate texts stay in the same fold.</p><div class="table-wrap gap"><table><thead><tr><th>Threshold method</th><th>Micro F1</th><th>Precision</th><th>Recall</th><th>Exact set match</th><th>Mixed: all labels found</th><th>Change vs 0.30</th><th>95% paired interval</th></tr></thead><tbody>${methods.map(([method, row]) => `<tr class="${method === report.selected_method ? "candidate-row" : ""}"><td class="strong">${esc(names[method] || nice(method))}${method === report.selected_method ? '<div class="micro muted">Report selection</div>' : ""}</td><td>${metric(row.micro_f1)}</td><td>${metric(row.micro_precision)}</td><td>${metric(row.micro_recall)}</td><td>${row.exact_match == null ? "—" : pct(row.exact_match)}</td><td>${row.mixed_emotion?.full_label_recovery == null ? "—" : pct(row.mixed_emotion.full_label_recovery)}<div class="micro muted">${number(row.mixed_emotion?.posts)} mixed posts</div></td><td>${signed(row["difference_micro_f1_vs_fixed_0.3"] ?? row.difference_micro_f1_vs_fixed_0_3)}</td><td class="small">${row.paired_difference_interval ? `${signed(row.paired_difference_interval.lower)} to ${signed(row.paired_difference_interval.upper)}` : "—"}</td></tr>`).join("")}</tbody></table></div><div class="refinement-reading gap"><h3>How much evidence is there for an improvement?</h3>${Array.isArray(report.balanced_improvement_candidates) && !report.balanced_improvement_candidates.length ? '<p class="small muted gap"><strong>No tested method was consistently better across the report?s balance criteria.</strong> A saved profile remains an experiment with tradeoffs.</p>' : ""}<p class="small muted gap">The change column compares F1 with the fixed 0.30 baseline; positive values favor that method. The paired interval resamples the same text groups for both methods.${interval ? (Number(interval.lower) <= 0 && Number(interval.upper) >= 0 ? " The saved candidate’s interval includes zero, so this sample does not establish a clear gain over the baseline." : " The interval describes uncertainty on this development reference; it does not guarantee a gain on new posts.") : ""}</p>${selected && baseline ? `<p class="small muted gap"><strong>Check the tradeoff:</strong> mixed-emotion full-label recovery is ${pct(selected.mixed_emotion?.full_label_recovery)} for the saved candidate and ${pct(baseline.mixed_emotion?.full_label_recovery)} for fixed 0.30. Macro F1 across labels is ${metric(selected.macro_f1_all_labels)} versus ${metric(baseline.macro_f1_all_labels)}. A higher overall F1 can still miss more mixed expressions or perform worse for some emotions.</p>` : ""}<p class="small muted gap">These are threshold-selection estimates on published GoEmotions data for an existing GoEmotions-trained model. They do not establish improved accuracy on your Twitter posts. No default has changed.</p></div>${item.profile_id ? `<div class="profile-availability gap"><div><h3>${esc(item.profile_name || "Saved threshold profile")}</h3><p class="small muted">Available as a provisional option in Run analysis. You must choose it explicitly.</p></div>${button("Review analysis settings", "go-analysis", "", "arrow")}</div>` : ""}<details class="glossary gap"><summary>Additional errors and label-level balance</summary><div class="table-wrap gap"><table><thead><tr><th>Method</th><th>Macro F1, all labels</th><th>Emotion-only micro F1</th><th>Mixed label recall</th><th>Missed labels</th><th>Extra labels</th><th>Neutral/emotion conflicts</th></tr></thead><tbody>${methods.map(([method, row]) => `<tr><td>${esc(names[method] || nice(method))}</td><td>${metric(row.macro_f1_all_labels)}</td><td>${metric(row.emotion_only_micro_f1)}</td><td>${row.mixed_emotion?.label_recall == null ? "\u2014" : pct(row.mixed_emotion.label_recall)}</td><td>${number(row.error_counts?.false_negative_labels)}</td><td>${number(row.error_counts?.false_positive_labels)}</td><td>${number(row.error_counts?.neutral_emotion_conflicts)}</td></tr>`).join("")}</tbody></table></div><p class="micro muted gap">Macro F1 gives each label equal weight. Emotion-only F1 excludes neutral, so its score cannot be driven by neutral matches. Conflict counts identify posts where neutral and an emotion both pass their cutoffs.</p></details>${(report.warnings || []).length ? `<details class="glossary gap"><summary>Read comparison limitations and cautions</summary><ul class="limitations">${report.warnings.map((text) => `<li>${esc(text)}</li>`).join("")}</ul></details>` : ""}<details class="glossary gap"><summary>Comparison settings and source provenance</summary><pre class="evidence">${esc(JSON.stringify({ id: item.id, selected_method: report.selected_method, rows: report.rows, unique_texts: report.unique_texts, test_used: report.test_used, settings: report.settings, folds: report.folds, model_provenance: report.model_provenance, hashes: report.hashes, limitations: report.limitations }, null, 2))}</pre></details></section>`;
  }
  function renderPublishedBenchmarks() {
    const reports = state.experiment?.published || [];
    const metric = (value) => (value == null ? "—" : Number(value).toFixed(3));
    return (
      card(
        "Published human-labeled datasets",
        `<p class="small muted">Use existing human annotations for a larger reference check. GoEmotions matches our 28 labels, but our current model was trained on that dataset. Its results here describe an in-domain development comparison; they do not establish accuracy on your posts. Twitter datasets with different categories need an explicit label mapping.</p>${
          reports.length
            ? reports
                .map((item) => {
                  const report = item.report,
                    rows = report.results || [];
                  return `<section class="gap"><div class="split"><h3>GoEmotions · published development reference</h3>${pill("Human annotations", "green")}</div><p class="small muted gap">${number(report.counts?.reviewed_rows)} reference posts · reserved test data ${report.test_scored ? "was scored" : "was not scored"}. Settings below were compared on development data.</p><details class="glossary gap"><summary>How to interpret this benchmark</summary><ul class="limitations">${[...(report.limitations || []), ...(report.warnings || [])].map((text) => `<li>${esc(text)}</li>`).join("")}</ul></details><div class="table-wrap gap"><table><thead><tr><th>Threshold</th><th>Precision</th><th>Recall</th><th>Micro F1</th><th>Exact set match</th><th>Mixed: all reference labels found</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${Number(row.threshold).toFixed(2)}</td><td>${metric(row.micro_precision)}</td><td>${metric(row.micro_recall)}</td><td>${metric(row.micro_f1)}</td><td>${pct(row.exact_match)}</td><td>${row.mixed_emotion?.full_label_recovery == null ? "—" : pct(row.mixed_emotion.full_label_recovery)}</td></tr>`).join("")}</tbody></table></div><details class="glossary gap"><summary>Dataset selection and report provenance</summary><pre class="evidence">${esc(JSON.stringify({ counts: report.counts, source: report.source, hashes: report.hashes, reference_policy: report.reference_policy }, null, 2))}</pre><p class="micro muted">${esc(item.id)}</p></details></section>`;
                })
                .join("")
            : '<p class="small muted gap">No published reference run has completed yet. The setup command and source details are in docs/PUBLIC_DATASETS.md. Your own human-review benchmark remains separate.</p>'
        }`,
        "Existing annotations supply reference labels; model predictions remain separate.",
      ) + '<div class="gap"></div>'
    );
  }
  function renderBenchmark() {
    const result = state.benchmarkResult,
      report = result?.report,
      counts = report?.counts || {};
    const metric = (value) =>
      value === null || value === undefined ? "—" : Number(value).toFixed(3);
    return `<section class="card gap"><div class="card-head"><div><h2>Build a reliable labeling benchmark</h2><div class="card-sub">Practice the rubric, review independently, then compare settings on development posts.</div></div>${pill("Human reference required", "amber")}</div><div class="card-body"><div class="benchmark-intro"><div><h3>Start with a 30-post practice round</h3><p class="small muted">Learn the emotion definitions and note ambiguous cases before labeling the larger sample. Model suggestions stay hidden. A second reviewer can later make independent judgments.</p>${button("Prepare 30-post practice round", "prepare-calibration", "", "review")}</div><div><h3>Compare thresholds using your reviewed posts</h3><p class="small muted">Uses the review sample selected above, <strong>${esc(state.reviewer)}</strong>’s saved development judgments, and the active run’s scores. Reserved test posts are excluded.</p>${button("Compare thresholds on reviewed development posts", "benchmark", "primary", "experiments")}</div></div><p class="micro muted gap">A threshold comparison checks a setting on the same trained model. It does not train a new model, change the default threshold, or prove that the model is accurate.</p><div id="benchmark-results">${report ? `<div class="metric-strip gap"><div><strong>${number(counts.reviewed_rows)}</strong><span>Reviewed development posts</span></div><div><strong>${number(counts.review_status_counts?.pending || 0)}</strong><span>Pending development posts</span></div><div><strong>${number(counts.held_out_rows)}</strong><span>Reserved posts excluded</span></div></div>${(report.warnings || []).map((warning) => notice(esc(warning), "warn")).join("")}${!report.ready ? notice("No comparison is ready yet. Save human judgments on development posts, then run this comparison again.") : `<div class="benchmark-recommendation gap"><h3>Provisional development result</h3><p class="small muted">${report.recommended_threshold == null ? "No threshold recommendation is available." : `Threshold ${Number(report.recommended_threshold).toFixed(2)} had the strongest result under this report’s selection rule.`} This is a candidate to inspect with more independent labels. Your saved settings remain unchanged.</p></div><div class="table-wrap gap"><table><thead><tr><th>Threshold</th><th>Micro F1</th><th>95% interval</th><th>Precision</th><th>Recall</th><th>Exact match</th><th>Mixed: all labels recovered</th><th>Mixed: label recall</th></tr></thead><tbody>${(report.results || []).map((row) => `<tr class="${row.threshold === report.recommended_threshold ? "candidate-row" : ""}"><td class="strong">${Number(row.threshold).toFixed(2)}</td><td>${metric(row.micro_f1)}</td><td class="small">${row.micro_f1_interval ? `${metric(row.micro_f1_interval.lower)}–${metric(row.micro_f1_interval.upper)}` : "—"}</td><td>${metric(row.micro_precision)}</td><td>${metric(row.micro_recall)}</td><td>${row.exact_match == null ? "—" : pct(row.exact_match)}</td><td>${row.mixed_emotion?.full_label_recovery == null ? "—" : pct(row.mixed_emotion.full_label_recovery)}<div class="micro muted">${number(row.mixed_emotion?.posts)} mixed posts</div></td><td>${row.mixed_emotion?.label_recall == null ? "—" : pct(row.mixed_emotion.label_recall)}</td></tr>`).join("")}</tbody></table></div><details class="glossary gap"><summary>What do these numbers mean?</summary><p>Precision asks how often a selected label agrees with your reference. Recall asks how many reference labels were found. Micro F1 balances these across labels. Exact match requires the full label set to agree.</p><p>Mixed-emotion recovery checks posts for which the human reference contains more than one emotion. The interval shows resampling uncertainty; small samples, rare labels, and a single reviewer still limit what we can conclude.</p><p>Resampling unit: ${esc(report.bootstrap_unit || "post")}. Use development results to choose settings, then freeze them before inspecting reserved test performance.</p></details><details class="glossary"><summary>Report provenance</summary><pre class="evidence">${esc(JSON.stringify({ source: report.source, hashes: report.hashes }, null, 2))}</pre></details>`}${result.path ? `<p class="micro muted gap">Saved report: <span class="mono">${esc(result.path)}</span></p>` : ""}` : '<p class="small muted gap">No threshold comparison has been run in this session.</p>'}</div></div></section>`;
  }
  function renderEvaluation() {
    const ev = state.evaluation;
    if (!ev) return "";
    const m = ev.metrics || {},
      labels = Object.entries(m.per_label || {}),
      errors = ev.errors || [];
    return `<section class="card gap"><div class="card-head"><div><h2>Human-reference results</h2><div class="card-sub">${esc(ev.metadata?.reviewer || state.reviewer)} · ${nice(m.split)} · preliminary reference</div></div>${pill("Draft human labels", "amber")}</div><div class="card-body"><div class="metric-strip"><div><strong>${Number(m.micro_f1 || 0).toFixed(3)}</strong><span>Micro F1</span></div><div><strong>${pct(m.exact_match)}</strong><span>Exact label match</span></div><div><strong>${number(m.reviewed_rows)}</strong><span>Reviewed posts</span></div></div><details><summary>Per-emotion precision, recall, and F1</summary><div class="table-wrap gap"><table><thead><tr><th>Emotion</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead><tbody>${labels.map(([label, x]) => `<tr><td>${nice(label)}</td><td>${Number(x.precision).toFixed(3)}</td><td>${Number(x.recall).toFixed(3)}</td><td>${Number(x.f1).toFixed(3)}</td><td>${number(x.support)}</td></tr>`).join("")}</tbody></table></div></details><h3 class="gap">Inspect the mistakes</h3><div class="table-wrap gap"><table><thead><tr><th>Post</th><th>Missed emotions</th><th>Extra labels</th></tr></thead><tbody>${
      errors
        .filter((x) => !x.exact_match)
        .map(
          (x) =>
            `<tr><td class="post-text">${esc(x.text)}</td><td>${chips(x.missed)}</td><td>${chips(x.extra)}</td></tr>`,
        )
        .join("") ||
      '<tr><td colspan="3" class="empty">No mismatched label sets in the evaluated rows.</td></tr>'
    }</tbody></table></div></div></section>`;
  }
  function bindPage() {
    const bind = (id, event, work) =>
      document.getElementById(id)?.addEventListener(event, work);
    if (state.page === "datasets") {
      bind("import-name", "input", (e) => (state.intake.name = e.target.value));
      bind("import-content", "input", (e) => {
        state.intake.content = e.target.value;
        invalidatePreview();
      });
      bind("import-format", "change", (e) => {
        state.intake.format = e.target.value;
        invalidatePreview();
        renderPage();
      });
      bind(
        "import-text-column",
        "change",
        (e) => (state.intake.textColumn = e.target.value),
      );
      bind(
        "import-user-column",
        "change",
        (e) => (state.intake.userColumn = e.target.value),
      );
    }
    if (state.page === "analysis") {
      bind("emotion-model", "change", (e) => {
        state.emotionModel = e.target.value;
        state.calibrationProfileId = "";
        renderPage();
      });
      bind("analysis-dataset", "change", (e) => {
        state.selectedDataset = e.target.value;
        renderPage();
      });
      bind(
        "analysis-name",
        "input",
        (e) => (state.analysisName = e.target.value),
      );
      bind(
        "include-sentiment",
        "change",
        (e) => (state.includeSentiment = e.target.checked),
      );
      bind("calibration-profile", "change", (e) => {
        state.calibrationProfileId = e.target.value;
        renderPage();
      });
      bind("analysis-threshold", "input", (e) => {
        state.analysisThreshold = Number(e.target.value);
        document.getElementById("analysis-threshold-value").textContent =
          state.analysisThreshold.toFixed(2);
      });
    }
    if (state.page === "findings") {
      for (const [id, key] of [
        ["finding-title", "title"],
        ["finding-note", "note"],
      ])
        bind(id, "input", (e) => {
          state.findingDrafts[state.runId] ||= { title: "", note: "" };
          state.findingDrafts[state.runId][key] = e.target.value;
        });
    }
    if (state.page === "explore") {
      for (const [id, key] of [
        ["post-search", "search"],
        ["emotion-filter", "emotion"],
        ["result-filter", "result"],
      ])
        document
          .getElementById(id)
          ?.addEventListener(id === "post-search" ? "input" : "change", (e) => {
            state[key] = e.target.value;
            state.pageNumber = 0;
            document.getElementById("post-results").innerHTML = explorerTable();
          });
      document
        .getElementById("source-dataset")
        ?.addEventListener("change", async (e) => {
          state.sourceId = e.target.value;
          if (!state.sourceId) {
            state.source = null;
            renderPage();
            return;
          }
          try {
            state.source = await api("load_dataset", {
              dataset_id: state.sourceId,
            });
            renderPage();
          } catch (err) {
            toast(err.message, true);
          }
        });
    }
    if (state.page === "review") {
      bind("definition-search", "input", (e) => {
        state.glossarySearch = e.target.value;
        document.getElementById("label-definitions").innerHTML =
          renderDefinitions();
      });
      for (const [id, key] of [
        ["review-split", "reviewSplit"],
        ["review-filter", "reviewFilter"],
      ])
        document.getElementById(id)?.addEventListener("change", (e) => {
          state[key] = e.target.value;
          state.reviewIndex = 0;
          renderPage();
        });
      document
        .getElementById("review-sample")
        ?.addEventListener("change", async (e) => {
          try {
            state.reviewId = e.target.value;
            state.reviewIndex = 0;
            await fetchReview();
            renderPage();
          } catch (err) {
            toast(err.message, true);
          }
        });
    }
    if (state.page === "lab") {
      document.getElementById("threshold")?.addEventListener("input", (e) => {
        state.threshold = Number(e.target.value);
        document.getElementById("threshold-value").textContent =
          state.threshold.toFixed(2);
        clearTimeout(rangeTimer);
        const request = ++rangeRequest;
        rangeTimer = setTimeout(async () => {
          try {
            const value = await api("sandbox", {
              run_id: state.runId,
              threshold: state.threshold,
            });
            if (request !== rangeRequest) return;
            state.sandbox = value;
            const target = document.getElementById("sandbox-results");
            if (target) target.innerHTML = renderSandbox();
          } catch (err) {
            toast(err.message, true);
          }
        }, 150);
      });
      document.getElementById("trial-text")?.addEventListener("input", (e) => {
        state.trialText = e.target.value;
        document.getElementById("trial-results").innerHTML = renderTrial();
      });
      document
        .getElementById("trial-threshold")
        ?.addEventListener("change", (e) => {
          state.trialThreshold = Number(e.target.value);
          state.trial = null;
          document.getElementById("trial-results").innerHTML = "";
        });
    }
    if (state.page === "experiments") {
      bind("eval-review", "change", (e) => {
        state.evalReviewId = e.target.value;
        state.benchmarkResult = null;
        const target = document.getElementById("benchmark-results");
        if (target)
          target.innerHTML =
            '<p class="small muted gap">Review sample changed. Run the comparison for this sample.</p>';
      });
      document
        .getElementById("saved-evaluation")
        ?.addEventListener("change", async (e) => {
          if (!e.target.value) {
            state.evaluation = null;
            document.getElementById("evaluation-results").innerHTML = "";
            return;
          }
          try {
            state.evaluation = await api("load_evaluation", {
              id: e.target.value,
            });
            document.getElementById("evaluation-results").innerHTML =
              renderEvaluation();
          } catch (err) {
            toast(err.message, true);
          }
        });
    }
  }
  function invalidatePreview() {
    state.intake.token = null;
    state.intake.preview = null;
    document
      .querySelector('[data-action="import-dataset"]')
      ?.closest("section")
      ?.remove();
  }
  function acceptPreview(result) {
    if (result.cancelled) return;
    state.intake.token = result.token;
    state.intake.preview = result.preview;
    state.intake.format = result.preview.format || state.intake.format;
    if (!state.intake.name)
      state.intake.name = result.preview.name || "Imported posts";
    state.intake.textColumn = state.intake.format === "txt" ? "text" : "";
    state.intake.userColumn = "";
  }
  let jobTimer;
  async function pollJob(id) {
    clearTimeout(jobTimer);
    try {
      const job = await api("analysis_status", { job_id: id });
      if (state.job?.id !== id) return;
      state.job = job;
      const target = document.getElementById("analysis-progress");
      if (state.page === "analysis" && target) target.innerHTML = renderJob();
      if (["queued", "running"].includes(job.status))
        jobTimer = setTimeout(() => pollJob(id), 1000);
      else {
        state.boot = await api("bootstrap");
        if (state.page === "analysis") {
          const start = document.querySelector(
            '[data-action="start-analysis"]',
          );
          if (start) {
            start.disabled = !state.selectedDataset;
            start.innerHTML = `${icon("analysis", 16)} Run analysis`;
          }
        }
        if (job.status === "complete")
          toast("Analysis saved. Open the results when you are ready.");
        else if (job.status === "failed")
          toast(
            job.error ||
              "Analysis failed. Check the run settings and try again.",
            true,
          );
      }
    } catch (error) {
      toast(`Could not refresh analysis progress: ${error.message}`, true);
      if (state.job?.id === id) jobTimer = setTimeout(() => pollJob(id), 3000);
    }
  }
  function showPost(id) {
    const row = state.run.posts.find((p) => p.post_id === id);
    if (!row) return;
    const dialog = document.getElementById("dialog");
    dialog.innerHTML = `<div class="dialog-head"><h2>Post detail</h2><button class="icon-button" data-action="close-dialog" aria-label="Close post details">${icon("close", 16)}</button></div><div class="dialog-body"><div class="post-quote">${esc(row.text)}</div><div class="split">${chips(row.labels)}${statusBadge(row.status)}</div><p class="small muted" style="margin-top:14px">${esc(row.review_reason || "No model review flag.")}</p>${row.sentiment ? `<h3 class="gap">Sentiment</h3><p class="small muted">Tone is scored separately from emotion labels.</p>${sentimentBadge(row.sentiment)}<div class="score-table">${bars(Object.entries(row.sentiment.scores || {}), 1, true)}</div>` : ""}<h3 class="gap">Emotion support scores</h3><div class="score-table">${bars(
      Object.entries(row.scores).sort((a, b) => b[1] - a[1]),
      1,
      true,
    )}</div><p class="micro muted gap">Scores represent model support, not emotional intensity.</p></div>`;
    dialog.showModal();
  }
  async function busy(button, work) {
    if (button?.disabled) return;
    const previous = button?.innerHTML;
    if (button) {
      button.disabled = true;
      button.innerHTML =
        '<span class="spinner" style="width:14px;height:14px"></span> Working…';
    }
    try {
      await work();
    } catch (e) {
      toast(e.message, true);
    } finally {
      if (button?.isConnected) {
        button.disabled = false;
        button.innerHTML = previous;
      }
    }
  }
  async function action(name, el) {
    if (name === "appearance") { showAppearance(); return; }
    if (name.startsWith("choose-theme-")) return busy(el, () => {
      const mode = name.slice("choose-theme-".length);
      if (!["light","dark","system"].includes(mode)) throw new Error("Choose a supported color theme.");
      themeSaveChain = themeSaveChain.catch(()=>{}).then(async()=>{
        await api("save_preferences", {theme:mode});
        state.theme = mode;
        store.set("tf-theme", mode);
        applyTheme();
      });
      return themeSaveChain;
    });
    if (name === "prepare-calibration")
      return busy(el, async () => {
        const reviewId =
          state.page === "experiments"
            ? document.getElementById("eval-review")?.value
            : state.reviewId;
        if (!reviewId) throw new Error("Select a review sample first.");
        const result = await api("prepare_calibration", {
          review_id: reviewId,
        });
        state.boot = await api("bootstrap");
        state.reviewId = result.review_id;
        state.reviewIndex = 0;
        state.reviewSplit = "development";
        state.reviewFilter = "pending";
        await navigate("review");
        toast(
          `${result.existing ? "Opened" : "Prepared"} a ${number(result.rows)}-post practice round. Model suggestions stay hidden.`,
        );
      });
    if (name === "benchmark")
      return busy(el, async () => {
        const reviewId = document.getElementById("eval-review")?.value,
          runId = state.runId;
        if (!reviewId) throw new Error("Select a review sample first.");
        if (!runId) throw new Error("Select a saved labeling run first.");
        const result = await api("benchmark", {
          run_id: runId,
          review_id: reviewId,
          reviewer: state.reviewer,
        });
        if (
          state.runId !== runId ||
          (state.evalReviewId || state.reviewId) !== reviewId
        )
          return;
        state.benchmarkResult = result;
        if (state.page === "experiments") renderPage();
        toast(
          result.report?.ready
            ? "Development comparison saved. Settings remain unchanged."
            : "Human reference is not ready yet. Review development posts first.",
        );
      });
    if (name === "about") {
      const app = state.boot.app || {},
        changes = app.changes || [],
        dialog = document.getElementById("dialog");
      dialog.innerHTML = `<div class="dialog-head"><h2>About TwitterFeel</h2><button class="icon-button" data-action="close-dialog" aria-label="Close">${icon("close", 16)}</button></div><div class="dialog-body"><span class="brand-mark">${icon("brand", 24)}</span><h2 class="gap">${esc(app.name || "TwitterFeel")} ${pill(`v${app.version || "0.2.0"}`, "purple")}</h2><p class="small muted gap">Released ${esc(app.released || "")}. A local workspace for labeling emotions in existing posts, exploring sentiment, and recording research evidence.</p><h3 class="gap">In this version</h3><ul class="limitations">${changes.map((change) => `<li>${esc(typeof change === "string" ? change : change.text || change.title || JSON.stringify(change))}</li>`).join("")}</ul><p class="small muted gap">Dataset versions and analysis runs record the app version and their own fingerprints. Updating the app does not turn earlier model results into new results.</p></div>`;
      dialog.showModal();
      return;
    }
    if (name === "pick-dataset")
      return busy(el, async () => {
        const result = await api("pick_dataset");
        acceptPreview(result);
        if (!result.cancelled) {
          state.intake.content = "";
          state.intake.name = result.preview.name || state.intake.name;
        }
        if (state.page === "datasets") renderPage();
      });
    if (name === "paste-clipboard")
      return busy(el, async () => {
        const result = await api("clipboard");
        state.intake.content = result.text || "";
        invalidatePreview();
        if (state.page === "datasets") renderPage();
      });
    if (name === "preview-import")
      return busy(el, async () => {
        const snapshot = {
          name: state.intake.name,
          content: state.intake.content,
          format: state.intake.format,
        };
        if (!snapshot.content.trim())
          throw new Error("Paste your data, or choose a file.");
        const result = await api("preview_import", snapshot);
        if (
          snapshot.content !== state.intake.content ||
          snapshot.format !== state.intake.format
        )
          return;
        acceptPreview(result);
        if (state.page === "datasets") renderPage();
      });
    if (name === "import-dataset")
      return busy(el, async () => {
        const draft = state.intake;
        if (!draft.token)
          throw new Error("Preview the current data before saving it.");
        if (!draft.name.trim()) throw new Error("Give this dataset a name.");
        if (!draft.textColumn)
          throw new Error("Choose the column containing the post text.");
        const result = await api("import_dataset", {
          token: draft.token,
          name: draft.name,
          text_column: draft.textColumn,
          user_column: draft.userColumn || null,
        });
        state.selectedDataset = result.id;
        state.boot = await api("bootstrap");
        state.intake.token = null;
        state.intake.preview = null;
        if (state.page === "datasets") renderPage();
        toast(
          `Saved ${result.name}, version ${result.version}. Ready for analysis.`,
        );
      });
    if (name === "create-review")
      return busy(el, async () => {
        const result = await api("create_review", {
          dataset_id: state.selectedDataset,
        });
        state.boot = await api("bootstrap");
        state.reviewId = result.review_id;
        state.reviewIndex = 0;
        state.reviewSplit = "development";
        state.reviewFilter = "pending";
        await navigate("review");
        toast(
          `Created a blind review sample with ${number(result.rows)} posts.`,
        );
      });
    if (name === "analyze-selected") return navigate("analysis");
    if (name === "start-analysis")
      return busy(el, async () => {
        if (!state.selectedDataset)
          throw new Error("Choose an imported dataset version first.");
        if (
          state.calibrationProfileId &&
          !(state.boot.calibration_profiles || []).some(
            (profile) => profile.id === state.calibrationProfileId,
          )
        )
          throw new Error(
            "Choose an available threshold profile or Manual shared threshold.",
          );
        state.job = await api("start_analysis", {
          dataset_id: state.selectedDataset,
          threshold: state.analysisThreshold,
          emotion_model: state.emotionModel,
          calibration_profile_id: state.calibrationProfileId || null,
          include_sentiment: state.includeSentiment,
          name: state.analysisName.trim() || "Dataset analysis",
        });
        if (state.page === "analysis") renderPage();
        pollJob(state.job.id);
      });
    if (name === "open-job" || name === "job-findings")
      return busy(el, async () => {
        const id = state.job?.run_id;
        if (!id)
          throw new Error("This analysis has not produced a saved run yet.");
        state.boot = await api("bootstrap");
        await changeRun(id);
        await navigate(name === "job-findings" ? "findings" : "explore");
      });
    if (name === "save-finding")
      return busy(el, async () => {
        const id = state.runId,
          draft = state.findingDrafts[id] || {};
        if (!draft.title?.trim() || !draft.note?.trim())
          throw new Error(
            "Enter a title and the evidence for your observation.",
          );
        await api("save_finding", {
          run_id: id,
          author: state.reviewer,
          title: draft.title,
          note: draft.note,
        });
        state.findingDrafts[id] = { title: "", note: "" };
        if (state.runId === id) await fetchFindings();
        if (state.page === "findings" && state.runId === id) renderPage();
        toast("Observation saved with this analysis.");
      });
    if (name === "export-findings")
      return busy(el, async () => {
        const result = await api("export_findings", { run_id: state.runId });
        toast(`Report exported to ${result.path}`);
      });
    if (name.startsWith("go-")) return navigate(name.slice(3));
    if (name === "refresh") return busy(el, bootstrap);
    if (name === "posts-prev" || name === "posts-next") {
      state.pageNumber += name.endsWith("next") ? 1 : -1;
      document.getElementById("post-results").innerHTML = explorerTable();
      return;
    }
    if (name === "review-prev" || name === "review-next") {
      state.reviewIndex += name.endsWith("next") ? 1 : -1;
      renderPage();
      return;
    }
    if (name === "close-dialog") {
      document.getElementById("dialog").close();
      return;
    }
    if (name === "profile") {
      const dialog = document.getElementById("dialog");
      dialog.innerHTML = `<div class="dialog-head"><h2>Reviewer identity</h2><button class="icon-button" data-action="close-dialog" aria-label="Close">${icon("close", 16)}</button></div><div class="dialog-body"><p class="small muted">Judgments are kept separately for each reviewer name. This is a local identity, not an account or sign-in.</p><div class="field-group"><label class="field-label" for="reviewer-name">Reviewer name</label><input id="reviewer-name" type="text" maxlength="80" value="${esc(state.reviewer)}" style="width:100%"></div>${button("Save reviewer", "save-profile", "primary")}</div>`;
      dialog.showModal();
      return;
    }
    if (name === "save-profile")
      return busy(el, async () => {
        const value = document.getElementById("reviewer-name").value.trim();
        if (!value) throw new Error("Enter a reviewer name.");
        await api("save_preferences", { theme: state.theme, reviewer: value });
        state.reviewer = value;
        store.set("tf-reviewer", value);
        if (state.reviewId) await fetchReview();
        document.getElementById("dialog").close();
        shell();
      });
    if (name === "save-review")
      return busy(el, async () => {
        const row = currentReview();
        if (!row) return;
        await api("save_review", {
          review_id: state.reviewId,
          reviewer: state.reviewer,
          post_id: row.post_id,
          text_sha256: row.text_sha256,
          labels: state.reviewLabels,
          status: state.reviewStatus,
          notes: document.getElementById("review-notes").value,
        });
        await fetchReview();
        renderPage();
        toast("Judgment saved. Previous revisions are preserved.");
      });
    if (name === "export-review")
      return busy(el, async () => {
        if (!state.reviewId) throw new Error("No review sample is available.");
        const result = await api("export_review", {
          review_id: state.reviewId,
          reviewer: state.reviewer,
        });
        toast(`Review exported to ${result.path}`);
      });
    if (name === "infer")
      return busy(el, async () => {
        const text = document.getElementById("trial-text").value;
        if (!text.trim()) throw new Error("Enter a post to analyze.");
        state.trialText = text;
        state.trial = await api("infer", {
          text,
          threshold: state.trialThreshold,
        });
        const target = document.getElementById("trial-results");
        if (target) target.innerHTML = renderTrial();
      });
    if (name === "evaluate")
      return busy(el, async () => {
        const reviewId = document.getElementById("eval-review").value,
          split = document.getElementById("eval-split").value;
        if (!reviewId)
          throw new Error("Review development posts before evaluating.");
        if (
          split === "test" &&
          !window.confirm(
            "Evaluate reserved test posts? Model settings should already be frozen.",
          )
        )
          return;
        const result = await api("evaluate", {
          run_id: state.runId,
          review_id: reviewId,
          reviewer: state.reviewer,
          split,
        });
        state.experiment = await api("experiments");
        const latest = state.experiment.evaluations?.find((x) =>
          result.path.replace(/\\/g, "/").endsWith(x.id),
        );
        state.evaluation = latest
          ? await api("load_evaluation", { id: latest.id })
          : { metrics: result.metrics, errors: [] };
        renderPage();
        toast("Evaluation saved with a snapshot of your reference labels.");
      });
  }
  document.addEventListener("click", async (e) => {
    const dataset = e.target.closest("[data-dataset]");
    if (dataset) {
      return busy(dataset, async () => {
        const id = dataset.dataset.dataset;
        const detail = await api("library_dataset", { dataset_id: id });
        state.selectedDataset = id;
        state.datasetDetail = detail;
        if (state.page === "datasets") {
          renderPage();
          document.getElementById("main").lastElementChild?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }
      });
    }
    const nav = e.target.closest("[data-nav]");
    if (nav) {
      await navigate(nav.dataset.nav);
      return;
    }
    const tab = e.target.closest("[data-tab]");
    if (tab) {
      const [group, value] = tab.dataset.tab.split(":");
      state[group === "lab" ? "labTab" : "explorerTab"] = value;
      renderPage();
      return;
    }
    const post = e.target.closest("[data-post]");
    if (post) {
      showPost(post.dataset.post);
      return;
    }
    const disposition = e.target.closest("[data-disposition]");
    if (disposition) {
      state.reviewStatus = disposition.dataset.disposition;
      if (state.reviewStatus !== "reviewed") state.reviewLabels = [];
      document.querySelectorAll("[data-disposition]").forEach((x) => {
        x.classList.toggle(
          "active",
          x.dataset.disposition === state.reviewStatus,
        );
        x.setAttribute(
          "aria-pressed",
          x.dataset.disposition === state.reviewStatus,
        );
      });
      document.getElementById("emotion-options").innerHTML = emotionButtons();
      return;
    }
    const emotion = e.target.closest("[data-emotion]");
    if (emotion && !emotion.disabled) {
      const label = emotion.dataset.emotion;
      if (state.reviewLabels.includes(label))
        state.reviewLabels = state.reviewLabels.filter((x) => x !== label);
      else if (label === "neutral") state.reviewLabels = ["neutral"];
      else
        state.reviewLabels = [
          ...state.reviewLabels.filter((x) => x !== "neutral"),
          label,
        ];
      document.getElementById("emotion-options").innerHTML = emotionButtons();
      return;
    }
    const target = e.target.closest("[data-action]");
    if (target) await action(target.dataset.action, target);
  });
  document.addEventListener("keydown", (e) => {
    const themeChoice = e.target.closest("[data-theme-choice]");
    if (themeChoice && ["ArrowLeft","ArrowRight","Home","End"].includes(e.key)) {
      e.preventDefault();
      const choices = [...document.querySelectorAll("[data-theme-choice]")];
      let index = choices.indexOf(themeChoice);
      index = e.key==="Home" ? 0 : e.key==="End" ? choices.length-1 : (index+(e.key==="ArrowRight"?1:-1)+choices.length)%choices.length;
      choices[index].focus(); choices[index].click();
      return;
    }
    if (
      (e.key === "Enter" || e.key === " ") &&
      e.target.matches("[data-post]")
    ) {
      e.preventDefault();
      showPost(e.target.dataset.post);
    }
  });
  document.getElementById("dialog").addEventListener("close",()=>document.getElementById("dialog").classList.remove("appearance-dialog"));
  themeMedia.addEventListener("change",()=>{ if (state.theme==="system") applyTheme(); });
  async function init() {
    try {
      await bootstrap();
    } catch (e) {
      document.getElementById("app").innerHTML =
        `<div class="boot"><div class="boot-mark">tf</div><h1>Could not open the workspace</h1><p>${esc(e.message)}</p>${button("Try again", "refresh", "primary", "refresh")}</div>`;
    }
  }
  window.addEventListener("pywebviewready", init, { once: true });
  if (window.pywebview?.api) init();
})();
