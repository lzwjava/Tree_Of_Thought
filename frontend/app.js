const STORAGE_KEY = "tot-terminal-ui-v1";
const STORAGE_VERSION = 8;
const DEFAULT_TIMEOUT_SECONDS = 60;
const DEFAULT_POLL_INTERVAL_MS = 1_000;
const DEFAULT_PLANNING_MODEL = "qwen3.5-9b-mlx";
const DEFAULT_MODELING_MODEL = "openai/gpt-oss-120b";
const DEFAULT_REVIEW_MODEL = "qwen/qwen3-4b-2507";
const DEFAULT_NON_TERMINAL_EVALUATION_MODEL = "qwen2.5-0.5b-instruct-mlx";
const LEGACY_QWOPUS_MODEL_FAMILY = "qwopus3.5-9b-v3";
const PREVIOUS_DEFAULT_PLANNING_MODEL = "openai/gpt-oss-120b";
const PREVIOUS_DEFAULT_MODELING_MODEL = "qwen3.6-35b-a3b-ud-mlx";
const PREVIOUS_DEFAULT_REVIEW_MODEL = "qwen3.6-35b-a3b-ud-mlx";
const PREVIOUS_DEFAULT_NON_TERMINAL_EVALUATION_MODEL = "qwen/qwen3-1.7b";

const RECOMMENDED_MODEL_PRESET = {
  planningModel: DEFAULT_PLANNING_MODEL,
  modelingModel: DEFAULT_MODELING_MODEL,
  reviewModel: DEFAULT_REVIEW_MODEL,
  nonTerminalEvaluationModel: DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
  timeoutSeconds: "120",
};

const DEFAULT_PROBLEM_CONTEXT = {
  task:
    "Use the modeling model to propose the next reasoning step, then score each step for physical consistency and variable grounding.",
  notes: [
    "The frontend polls the live scheduler state and renders it as an ASCII tree.",
    "Node deletion is routed through the backend review model before the subtree is removed.",
  ],
  known_context: {
    objective: "Derive and prune a useful physical reasoning tree.",
    expected_output: "concise, structured, and physically valid intermediate steps",
  },
};

const DEFAULT_SCHEDULER_CONFIG = {
  max_reflections: 2,
  expansion_budget: 8,
  max_frontier_size: 8,
  max_children_per_expansion: 4,
  max_frontier_per_diversity_key: 2,
  children_key: "children",
};

const uiState = {
  sessionId: "",
  snapshot: null,
  selectedNodeId: null,
  collapsedNodeIds: new Set(),
  treeModel: emptyTreeModel(),
  searchQuery: "",
  searchMatches: [],
  searchCursor: 0,
  pollingEnabled: true,
  pollIntervalMs: DEFAULT_POLL_INTERVAL_MS,
  pollTimer: null,
  busyRefreshTimer: null,
  requestInFlight: false,
  revealSelection: false,
  statusMessage: "SYSTEM READY // no session attached",
  statusTone: "idle",
  lastUpdatedAt: null,
  lastActionTitle: "No request yet.",
  lastActionDetail: "Use Create Session to start a new run or Load Session State to reconnect to an existing session.",
  lastActionTone: "idle",
  statusLogEntries: [],
  statusLogSequence: 0,
};

const dom = {
  statusLine: document.getElementById("statusLine"),
  actionFeedback: document.getElementById("actionFeedback"),
  actionFeedbackTitle: document.getElementById("actionFeedbackTitle"),
  actionFeedbackDetail: document.getElementById("actionFeedbackDetail"),
  statusLog: document.getElementById("statusLog"),
  problemPromptInput: document.getElementById("problemPromptInput"),
  sessionIdInput: document.getElementById("sessionIdInput"),
  attachSessionButton: document.getElementById("attachSessionButton"),
  createSessionButton: document.getElementById("createSessionButton"),
  refreshSessionButton: document.getElementById("refreshSessionButton"),
  runSessionButton: document.getElementById("runSessionButton"),
  dropSessionButton: document.getElementById("dropSessionButton"),
  budgetInput: document.getElementById("budgetInput"),
  runOnCreateToggle: document.getElementById("runOnCreateToggle"),
  searchInput: document.getElementById("searchInput"),
  prevMatchButton: document.getElementById("prevMatchButton"),
  nextMatchButton: document.getElementById("nextMatchButton"),
  searchMeta: document.getElementById("searchMeta"),
  pollingToggle: document.getElementById("pollingToggle"),
  pollIntervalInput: document.getElementById("pollIntervalInput"),
  expandAllButton: document.getElementById("expandAllButton"),
  collapseAllButton: document.getElementById("collapseAllButton"),
  deleteReasonInput: document.getElementById("deleteReasonInput"),
  deleteNodeButton: document.getElementById("deleteNodeButton"),
  applyRecommendedPresetButton: document.getElementById("applyRecommendedPresetButton"),
  baseUrlInput: document.getElementById("baseUrlInput"),
  planningModelInput: document.getElementById("planningModelInput"),
  modelingModelInput: document.getElementById("modelingModelInput"),
  reviewModelInput: document.getElementById("reviewModelInput"),
  nonTerminalEvaluationModelInput: document.getElementById("nonTerminalEvaluationModelInput"),
  timeoutInput: document.getElementById("timeoutInput"),
  problemContextInput: document.getElementById("problemContextInput"),
  schedulerConfigInput: document.getElementById("schedulerConfigInput"),
  treeStats: document.getElementById("treeStats"),
  selectionStats: document.getElementById("selectionStats"),
  treeViewport: document.getElementById("treeViewport"),
  treeLines: document.getElementById("treeLines"),
  detailSummary: document.getElementById("detailSummary"),
  detailBody: document.getElementById("detailBody"),
  frontierList: document.getElementById("frontierList"),
  activityLog: document.getElementById("activityLog"),
};

boot().catch((error) => {
  handleError(error);
});

async function boot() {
  applyDefaultDrafts();
  restoreDraft();
  wireEvents();
  pushStatusLog(
    "UI ready.",
    "No session attached yet. Create a session or load an existing session state to begin.",
    "idle",
  );
  render();

  if (dom.sessionIdInput.value.trim()) {
    await attachSession({ silent: true });
  }
}

function emptyTreeModel() {
  return {
    root: null,
    nodeById: new Map(),
    ancestorsById: new Map(),
    visibleNodes: [],
    allNodes: [],
    maxDepth: 0,
    activeCount: 0,
    prunedCount: 0,
    leafCount: 0,
  };
}

function applyDefaultDrafts() {
  if (!dom.problemContextInput.value.trim()) {
    dom.problemContextInput.value = JSON.stringify(DEFAULT_PROBLEM_CONTEXT, null, 2);
  }
  if (!dom.schedulerConfigInput.value.trim()) {
    dom.schedulerConfigInput.value = JSON.stringify(DEFAULT_SCHEDULER_CONFIG, null, 2);
  }
}

function readModelInput(element, fallback) {
  if (!(element instanceof HTMLInputElement)) {
    return fallback;
  }
  return sanitizeModelName(element.value, fallback);
}

function writeModelInput(element, value, fallback) {
  if (!(element instanceof HTMLInputElement)) {
    return;
  }
  element.value = sanitizeModelName(value, fallback);
}

function applyRecommendedModelPreset() {
  writeModelInput(
    dom.planningModelInput,
    RECOMMENDED_MODEL_PRESET.planningModel,
    DEFAULT_PLANNING_MODEL,
  );
  writeModelInput(
    dom.modelingModelInput,
    RECOMMENDED_MODEL_PRESET.modelingModel,
    DEFAULT_MODELING_MODEL,
  );
  writeModelInput(
    dom.reviewModelInput,
    RECOMMENDED_MODEL_PRESET.reviewModel,
    DEFAULT_REVIEW_MODEL,
  );
  writeModelInput(
    dom.nonTerminalEvaluationModelInput,
    RECOMMENDED_MODEL_PRESET.nonTerminalEvaluationModel,
    DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
  );
  if (dom.timeoutInput instanceof HTMLInputElement) {
    dom.timeoutInput.value = RECOMMENDED_MODEL_PRESET.timeoutSeconds;
  }
  persistDraft();
  setStatus("Recommended preset applied.", "ok");
}

function restoreDraft() {
  let stored = {};
  try {
    stored = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "{}");
  } catch (_error) {
    stored = {};
  }

  const isLegacyDraft = stored.storageVersion !== STORAGE_VERSION;

  const fieldMap = {
    problemPromptInput: "problemPrompt",
    sessionIdInput: "sessionId",
    problemContextInput: "problemContext",
    schedulerConfigInput: "schedulerConfig",
    baseUrlInput: "baseUrl",
    budgetInput: "budget",
    deleteReasonInput: "deleteReason",
    searchInput: "searchQuery",
  };

  Object.entries(fieldMap).forEach(([domKey, storedKey]) => {
    const value = stored[storedKey];
    if (value !== undefined && dom[domKey]) {
      dom[domKey].value = String(value);
    }
  });

  if (isLegacyDraft) {
    dom.schedulerConfigInput.value = migrateLegacySchedulerConfig(dom.schedulerConfigInput.value);
  }

  const restoredTimeout = sanitizeTimeoutSeconds(
    isLegacyDraft ? migrateLegacyTimeoutSeconds(stored.timeout) : stored.timeout
  );
  dom.timeoutInput.value = String(restoredTimeout);

  writeModelInput(
    dom.planningModelInput,
    isLegacyDraft ? migrateLegacyPlanningModel(stored.planningModel) : stored.planningModel,
    DEFAULT_PLANNING_MODEL,
  );
  writeModelInput(
    dom.modelingModelInput,
    isLegacyDraft ? migrateLegacyModelingModel(stored.modelingModel) : stored.modelingModel,
    DEFAULT_MODELING_MODEL,
  );
  writeModelInput(
    dom.reviewModelInput,
    isLegacyDraft ? migrateLegacyReviewModel(stored.reviewModel) : stored.reviewModel,
    DEFAULT_REVIEW_MODEL,
  );
  writeModelInput(
    dom.nonTerminalEvaluationModelInput,
    stored.nonTerminalEvaluationModel === PREVIOUS_DEFAULT_NON_TERMINAL_EVALUATION_MODEL
      ? DEFAULT_NON_TERMINAL_EVALUATION_MODEL
      : stored.nonTerminalEvaluationModel,
    DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
  );

  const restoredPollInterval = sanitizePollInterval(
    isLegacyDraft ? migrateLegacyPollInterval(stored.pollIntervalMs) : stored.pollIntervalMs
  );
  dom.pollIntervalInput.value = String(restoredPollInterval);

  if (typeof stored.runOnCreate === "boolean") {
    dom.runOnCreateToggle.checked = stored.runOnCreate;
  }
  if (typeof stored.pollingEnabled === "boolean") {
    dom.pollingToggle.checked = stored.pollingEnabled;
  }

  if (!dom.problemPromptInput.value.trim()) {
    const migratedProblemPrompt = extractProblemStatementDraft(dom.problemContextInput.value);
    if (migratedProblemPrompt) {
      dom.problemPromptInput.value = migratedProblemPrompt;
    }
  }

  uiState.sessionId = dom.sessionIdInput.value.trim();
  uiState.searchQuery = dom.searchInput.value.trim();
  uiState.pollingEnabled = dom.pollingToggle.checked;
  uiState.pollIntervalMs = restoredPollInterval;
  restartPolling();
  persistDraft();
}

function persistDraft() {
  const payload = {
    storageVersion: STORAGE_VERSION,
    problemPrompt: dom.problemPromptInput.value,
    sessionId: dom.sessionIdInput.value.trim(),
    problemContext: dom.problemContextInput.value,
    schedulerConfig: dom.schedulerConfigInput.value,
    baseUrl: dom.baseUrlInput.value.trim(),
    planningModel: readModelInput(dom.planningModelInput, DEFAULT_PLANNING_MODEL),
    modelingModel: readModelInput(dom.modelingModelInput, DEFAULT_MODELING_MODEL),
    reviewModel: readModelInput(dom.reviewModelInput, DEFAULT_REVIEW_MODEL),
    nonTerminalEvaluationModel: readModelInput(
      dom.nonTerminalEvaluationModelInput,
      DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    ),
    timeout: dom.timeoutInput.value.trim(),
    budget: dom.budgetInput.value.trim(),
    deleteReason: dom.deleteReasonInput.value,
    searchQuery: dom.searchInput.value,
    runOnCreate: dom.runOnCreateToggle.checked,
    pollingEnabled: dom.pollingToggle.checked,
    pollIntervalMs: dom.pollIntervalInput.value.trim(),
  };
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function migrateLegacySchedulerConfig(rawValue) {
  const fallback = JSON.stringify(DEFAULT_SCHEDULER_CONFIG, null, 2);
  if (!rawValue || !String(rawValue).trim()) {
    return fallback;
  }

  try {
    const parsed = JSON.parse(String(rawValue));
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return fallback;
    }
    if (parsed.max_children_per_expansion === 3) {
      parsed.max_children_per_expansion = 4;
    }
    return JSON.stringify(parsed, null, 2);
  } catch (_error) {
    return fallback;
  }
}

function wireEvents() {
  dom.attachSessionButton.addEventListener("click", () => {
    runUiAction(() => attachSession());
  });
  dom.createSessionButton.addEventListener("click", () => {
    runUiAction(() => createSession());
  });
  dom.refreshSessionButton.addEventListener("click", () => {
    runUiAction(() => refreshSession());
  });
  dom.runSessionButton.addEventListener("click", () => {
    runUiAction(() => runSession());
  });
  dom.dropSessionButton.addEventListener("click", () => {
    runUiAction(() => dropSession());
  });
  dom.prevMatchButton.addEventListener("click", () => {
    jumpToSearchMatch(-1);
  });
  dom.nextMatchButton.addEventListener("click", () => {
    jumpToSearchMatch(1);
  });
  dom.expandAllButton.addEventListener("click", () => {
    uiState.collapsedNodeIds.clear();
    uiState.revealSelection = true;
    render();
    setStatus("Expanded all visible branches.", "ok");
  });
  dom.collapseAllButton.addEventListener("click", () => {
    collapseAllDescendants();
  });
  dom.deleteNodeButton.addEventListener("click", () => {
    runUiAction(() => deleteSelectedNode());
  });
  if (dom.applyRecommendedPresetButton instanceof HTMLButtonElement) {
    dom.applyRecommendedPresetButton.addEventListener("click", () => {
      applyRecommendedModelPreset();
    });
  }

  dom.sessionIdInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      runUiAction(() => attachSession());
    }
  });

  dom.searchInput.addEventListener("input", () => {
    uiState.searchQuery = dom.searchInput.value.trim();
    persistDraft();
    render();
  });
  dom.searchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      jumpToSearchMatch(event.shiftKey ? -1 : 1);
    }
    if (event.key === "Escape") {
      event.preventDefault();
      dom.treeViewport.focus();
    }
  });

  dom.pollingToggle.addEventListener("change", () => {
    uiState.pollingEnabled = dom.pollingToggle.checked;
    restartPolling();
    persistDraft();
    setStatus(
      uiState.pollingEnabled ? "Auto refresh enabled." : "Auto refresh disabled.",
      "ok"
    );
    render();
  });

  dom.pollIntervalInput.addEventListener("change", () => {
    uiState.pollIntervalMs = sanitizePollInterval(dom.pollIntervalInput.value);
    dom.pollIntervalInput.value = String(uiState.pollIntervalMs);
    restartPolling();
    persistDraft();
    render();
  });

  [
    dom.problemPromptInput,
    dom.problemContextInput,
    dom.schedulerConfigInput,
    dom.baseUrlInput,
    dom.planningModelInput,
    dom.modelingModelInput,
    dom.reviewModelInput,
    dom.nonTerminalEvaluationModelInput,
    dom.timeoutInput,
    dom.budgetInput,
    dom.deleteReasonInput,
    dom.runOnCreateToggle,
  ].filter(Boolean).forEach((element) => {
    const eventName = element instanceof HTMLInputElement && element.type === "checkbox" ? "change" : "input";
    element.addEventListener(eventName, persistDraft);
  });

  dom.treeLines.addEventListener("click", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (!(target instanceof HTMLElement)) {
      return;
    }
    selectNode(target.dataset.nodeId || "");
    dom.treeViewport.focus();
  });

  dom.treeLines.addEventListener("dblclick", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (!(target instanceof HTMLElement)) {
      return;
    }
    toggleSelectedNodeExpansion(target.dataset.nodeId || "");
    dom.treeViewport.focus();
  });

  dom.frontierList.addEventListener("click", (event) => {
    const target = event.target.closest("[data-node-id]");
    if (!(target instanceof HTMLElement)) {
      return;
    }
    selectNode(target.dataset.nodeId || "");
    dom.treeViewport.focus();
  });

  document.addEventListener("keydown", handleGlobalKeydown);
}

function handleGlobalKeydown(event) {
  if ((event.shiftKey || event.altKey) && event.key.startsWith("Arrow")) {
    event.preventDefault();
    panViewport(event.key);
    return;
  }

  if (isTextEditingTarget(event.target)) {
    if (event.key === "Escape") {
      dom.treeViewport.focus();
    }
    return;
  }

  switch (event.key) {
    case "/":
      event.preventDefault();
      dom.searchInput.focus();
      dom.searchInput.select();
      break;
    case "ArrowUp":
      event.preventDefault();
      moveSelection(-1);
      break;
    case "ArrowDown":
      event.preventDefault();
      moveSelection(1);
      break;
    case "ArrowLeft":
      event.preventDefault();
      navigateLeft();
      break;
    case "ArrowRight":
      event.preventDefault();
      navigateRight();
      break;
    case "Home":
      event.preventDefault();
      if (uiState.treeModel.root) {
        selectNode(uiState.treeModel.root.id);
      }
      break;
    case "End":
      event.preventDefault();
      if (uiState.treeModel.visibleNodes.length) {
        selectNode(uiState.treeModel.visibleNodes[uiState.treeModel.visibleNodes.length - 1].node.id);
      }
      break;
    case "Enter":
      if (uiState.searchQuery) {
        event.preventDefault();
        jumpToSearchMatch(event.shiftKey ? -1 : 1);
      }
      break;
    case "r":
    case "R":
      event.preventDefault();
      runUiAction(() => refreshSession());
      break;
    case "p":
    case "P":
      event.preventDefault();
      dom.pollingToggle.checked = !dom.pollingToggle.checked;
      dom.pollingToggle.dispatchEvent(new Event("change"));
      break;
    case "e":
    case "E":
      event.preventDefault();
      runUiAction(() => runSession());
      break;
    case "Delete":
    case "Backspace":
      event.preventDefault();
      runUiAction(() => deleteSelectedNode());
      break;
    default:
      break;
  }
}

function isTextEditingTarget(target) {
  return target instanceof HTMLElement && (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT" ||
    target.isContentEditable
  );
}

function sanitizePollInterval(rawValue) {
  const parsed = Number.parseInt(String(rawValue || DEFAULT_POLL_INTERVAL_MS), 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_POLL_INTERVAL_MS;
  }
  return Math.max(DEFAULT_POLL_INTERVAL_MS, parsed);
}

function sanitizeTimeoutSeconds(rawValue) {
  const parsed = Number.parseFloat(String(rawValue || DEFAULT_TIMEOUT_SECONDS));
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_TIMEOUT_SECONDS;
  }
  return parsed;
}

function migrateLegacyTimeoutSeconds(rawValue) {
  const parsed = Number.parseFloat(String(rawValue || ""));
  if (!Number.isFinite(parsed) || parsed === 30 || parsed === 120) {
    return DEFAULT_TIMEOUT_SECONDS;
  }
  return parsed;
}

function migrateLegacyPollInterval(rawValue) {
  const parsed = Number.parseInt(String(rawValue || ""), 10);
  if (!Number.isFinite(parsed)) {
    return DEFAULT_POLL_INTERVAL_MS;
  }
  if (parsed === 60_000) {
    return DEFAULT_POLL_INTERVAL_MS;
  }
  return Math.max(DEFAULT_POLL_INTERVAL_MS, parsed);
}

function formatPollInterval(rawValue) {
  const intervalMs = sanitizePollInterval(rawValue);
  if (intervalMs % 60_000 === 0) {
    const minutes = intervalMs / 60_000;
    return `${minutes}min`;
  }
  if (intervalMs % 1000 === 0) {
    return `${intervalMs / 1000}s`;
  }
  return `${intervalMs}ms`;
}

function restartPolling() {
  if (uiState.pollTimer) {
    window.clearInterval(uiState.pollTimer);
    uiState.pollTimer = null;
  }

  if (!uiState.pollingEnabled) {
    return;
  }

  uiState.pollTimer = window.setInterval(() => {
    if (!uiState.sessionId || uiState.requestInFlight) {
      return;
    }
    refreshSession({ silent: true });
  }, uiState.pollIntervalMs);
}

function enableAutoRefreshForSession() {
  if (!uiState.sessionId) {
    return;
  }

  uiState.pollingEnabled = true;
  dom.pollingToggle.checked = true;
  restartPolling();
  persistDraft();

  if (isSessionBusy(uiState.snapshot)) {
    scheduleBusyRefresh();
  }
}

function clearBusyRefresh() {
  if (uiState.busyRefreshTimer) {
    window.clearTimeout(uiState.busyRefreshTimer);
    uiState.busyRefreshTimer = null;
  }
}

function getRunState(snapshot) {
  return snapshot && typeof snapshot.run_state === "object" && snapshot.run_state
    ? snapshot.run_state
    : {};
}

function isSessionBusy(snapshot) {
  return String(getRunState(snapshot).status || "").toLowerCase() === "busy";
}

function scheduleBusyRefresh(delayMs = 250) {
  clearBusyRefresh();
  if (!uiState.sessionId || !uiState.pollingEnabled || !isSessionBusy(uiState.snapshot)) {
    return;
  }

  uiState.busyRefreshTimer = window.setTimeout(() => {
    uiState.busyRefreshTimer = null;
    if (!uiState.requestInFlight && uiState.sessionId && isSessionBusy(uiState.snapshot)) {
      refreshSession({ silent: true });
    }
  }, delayMs);
}

function setStatus(message, tone = "idle") {
  const options = arguments.length > 2 && arguments[2] !== undefined ? arguments[2] : {};
  uiState.statusMessage = message;
  uiState.statusTone = tone;
  if (options.record !== false) {
    pushStatusLog(message, options.detail || "", tone);
  }
  renderStatus();
}

function setActionFeedback(title, detail, tone = "idle", options = {}) {
  uiState.lastActionTitle = String(title || "").trim() || "No request yet.";
  uiState.lastActionDetail = String(detail || "").trim() || "No further details.";
  uiState.lastActionTone = tone;
  if (options.record !== false) {
    pushStatusLog(uiState.lastActionTitle, uiState.lastActionDetail, tone);
  }
  renderActionFeedback();
  renderStatusLog();
}

function pushStatusLog(title, detail, tone = "idle") {
  const normalizedTitle = String(title || "").trim() || "No title";
  const normalizedDetail = String(detail || "").trim();
  const normalizedTone = String(tone || "idle");
  const timestamp = new Date();
  const latestEntry = uiState.statusLogEntries[0] || null;

  if (
    latestEntry &&
    latestEntry.title === normalizedTitle &&
    latestEntry.detail === normalizedDetail &&
    latestEntry.tone === normalizedTone
  ) {
    latestEntry.count += 1;
    latestEntry.at = timestamp.toISOString();
    return;
  }

  uiState.statusLogSequence += 1;
  uiState.statusLogEntries.unshift({
    id: `status-${uiState.statusLogSequence}`,
    title: normalizedTitle,
    detail: normalizedDetail,
    tone: normalizedTone,
    at: timestamp.toISOString(),
    count: 1,
  });
  uiState.statusLogEntries = uiState.statusLogEntries.slice(0, 40);
}

function captureScrollPosition(element) {
  if (!(element instanceof HTMLElement)) {
    return null;
  }
  return {
    top: element.scrollTop,
    left: element.scrollLeft,
  };
}

function restoreScrollPosition(element, state) {
  if (!(element instanceof HTMLElement) || !state) {
    return;
  }
  element.scrollTop = state.top;
  element.scrollLeft = state.left;
}

function captureScrollState() {
  return {
    treeViewport: captureScrollPosition(dom.treeViewport),
    detailBody: captureScrollPosition(dom.detailBody),
    frontierList: captureScrollPosition(dom.frontierList),
    activityLog: captureScrollPosition(dom.activityLog),
    statusLog: captureScrollPosition(dom.statusLog),
  };
}

function restoreScrollState(scrollState, options = {}) {
  if (options.restoreTree !== false) {
    restoreScrollPosition(dom.treeViewport, scrollState.treeViewport);
  }
  restoreScrollPosition(dom.detailBody, scrollState.detailBody);
  restoreScrollPosition(dom.frontierList, scrollState.frontierList);
  restoreScrollPosition(dom.activityLog, scrollState.activityLog);
  restoreScrollPosition(dom.statusLog, scrollState.statusLog);
}

function render() {
  const scrollState = captureScrollState();
  const shouldRevealSelection = uiState.revealSelection;
  recomputeDerivedState();
  renderStatus();
  renderActionFeedback();
  renderStatusLog();
  renderButtons();
  renderTree();
  renderDetailPane();
  renderFrontier();
  renderActivity();
  renderMeta();
  restoreScrollState(scrollState, { restoreTree: !shouldRevealSelection });
}

function recomputeDerivedState() {
  const root = getRenderableRoot(uiState.snapshot);
  if (!root) {
    uiState.treeModel = emptyTreeModel();
    uiState.searchMatches = [];
    uiState.searchCursor = 0;
    return;
  }

  let model = buildTreeModel(root, uiState.collapsedNodeIds);
  Array.from(uiState.collapsedNodeIds).forEach((nodeId) => {
    if (!model.nodeById.has(nodeId)) {
      uiState.collapsedNodeIds.delete(nodeId);
    }
  });

  if (!uiState.selectedNodeId || !model.nodeById.has(uiState.selectedNodeId)) {
    uiState.selectedNodeId = root.id;
    uiState.revealSelection = true;
  }

  const ancestorIds = model.ancestorsById.get(uiState.selectedNodeId) || [];
  ancestorIds.forEach((ancestorId) => uiState.collapsedNodeIds.delete(ancestorId));

  model = buildTreeModel(root, uiState.collapsedNodeIds);
  uiState.treeModel = model;
  uiState.searchMatches = computeSearchMatches(model.allNodes, uiState.searchQuery);

  const currentMatchIndex = uiState.searchMatches.indexOf(uiState.selectedNodeId);
  if (currentMatchIndex >= 0) {
    uiState.searchCursor = currentMatchIndex;
  } else if (uiState.searchMatches.length === 0) {
    uiState.searchCursor = 0;
  } else {
    uiState.searchCursor = Math.min(uiState.searchCursor, uiState.searchMatches.length - 1);
  }
}

function getRenderableRoot(snapshot) {
  const root = snapshot && snapshot.root ? snapshot.root : null;
  if (root) {
    return root;
  }

  const runState = getRunState(snapshot);
  const metaTask = snapshot && typeof snapshot.meta_task === "object" && snapshot.meta_task
    ? snapshot.meta_task
    : {};
  const hasPendingSession = Boolean(
    snapshot
    && (
      Object.keys(runState).length > 0
      || Object.keys(metaTask).length > 0
      || uiState.sessionId
    )
  );
  if (!hasPendingSession) {
    return null;
  }

  const phase = String(runState.phase || "created").trim() || "created";
  const status = String(runState.status || "idle").trim() || "idle";
  const lastError = String(runState.last_error || "").trim();
  const objective = String(metaTask.objective || "").trim();
  const firstStep = String(metaTask.first_step || "").trim();
  const displayStatus = status.toLowerCase() === "error" ? "ERROR" : "PENDING";
  const waitingSummary = status.toLowerCase() === "error"
    ? `Root construction failed before the first real branch was created.${lastError ? ` Last error: ${lastError}.` : ""}`
    : firstStep
      ? `Waiting for the first root branch. Planned first step: ${firstStep}.`
      : `Waiting for the first root branch. Backend phase: ${phase}.`;

  return {
    id: uiState.sessionId ? `session-${uiState.sessionId.slice(0, 8)}` : "session-pending",
    parent_id: null,
    thought_step: objective ? `${objective} ${waitingSummary}`.trim() : waitingSummary,
    equations: [],
    known_vars: {
      synthetic_placeholder_root: true,
      run_status: status,
      run_phase: phase,
      last_error: lastError,
      meta_task_objective: objective,
      meta_task_first_step: firstStep,
    },
    used_models: [],
    quantities: {},
    boundary_conditions: {},
    status: displayStatus,
    fsm_state: "PENDING ROOT",
    score: 0,
    reflection_history: [],
    children: [],
  };
}

function isSyntheticPlaceholderRoot(node) {
  return Boolean(node && node.known_vars && node.known_vars.synthetic_placeholder_root);
}

function buildTreeModel(root, collapsedNodeIds) {
  const model = emptyTreeModel();
  model.root = root;

  function walk(node, depth, ancestorIds, branchGuides, isLast) {
    const children = getChildren(node);
    const isCollapsed = children.length > 0 && collapsedNodeIds.has(node.id);
    const summary = buildNodePresentation(node, children.length, isCollapsed, depth);
    const searchText = buildSearchText(node, depth);

    model.nodeById.set(node.id, node);
    model.ancestorsById.set(node.id, ancestorIds);
    model.allNodes.push({
      node,
      depth,
      ancestorIds,
      searchText,
      childCount: children.length,
    });
    model.visibleNodes.push({
      node,
      depth,
      ancestorIds,
      childCount: children.length,
      isCollapsed,
      summary,
    });

    model.maxDepth = Math.max(model.maxDepth, depth);
    if (children.length === 0) {
      model.leafCount += 1;
    }
    if (String(node.status || "").toUpperCase().startsWith("PRUNED")) {
      model.prunedCount += 1;
    } else {
      model.activeCount += 1;
    }

    if (isCollapsed) {
      return;
    }

    children.forEach((child, index) => {
      walk(
        child,
        depth + 1,
        ancestorIds.concat(node.id),
        branchGuides.concat(index < children.length - 1),
        index === children.length - 1,
      );
    });
  }

  walk(root, 0, [], [], true);
  return model;
}

function getChildren(node) {
  return Array.isArray(node && node.children) ? node.children : [];
}

function buildNodePresentation(node, childCount, isCollapsed, depth) {
  const foldMark = depth === 0 ? "root" : childCount > 0 ? (isCollapsed ? "+" : "-") : "•";
  const visibleResult = displayResultState(node);
  return {
    foldMark,
    title: summarizeThought(node.thought_step),
    meta: buildNodeMeta(node, childCount, depth),
    routeFamily: getNodeRouteFamily(node),
    stepFocus: getNodeStepFocus(node),
    ignoredNoiseCount: getNodeIgnoredNoiseCount(node),
    status: shortStatus(visibleResult),
    statusTone: statusTone(visibleResult),
  };
}

function buildNodeMeta(node, childCount, depth) {
  const parts = [
    node.id,
    `d${depth}`,
    childCount === 0 ? "leaf" : `${childCount} child${childCount === 1 ? "" : "ren"}`,
  ];

  const resultState = displayResultState(node);
  if (resultState) {
    parts.push(String(resultState).toLowerCase());
  }
  if (Number.isFinite(node.score)) {
    parts.push(`score ${formatScore(node.score)}`);
  }
  return parts.join(" · ");
}

function summarizeThought(thoughtStep) {
  const normalized = trimText(String(thoughtStep || "").trim(), 72);
  return normalized || "No thought step recorded.";
}

function getNodeRouteFamily(node) {
  if (!node || !node.known_vars || typeof node.known_vars !== "object") {
    return "";
  }
  const direct = String(node.known_vars.route_family || "").trim();
  if (direct) {
    return direct;
  }
  const task = node.known_vars.orchestrator_task;
  if (task && typeof task === "object") {
    return String(task.selected_route_family || "").trim();
  }
  return "";
}

function getNodeStepFocus(node) {
  if (!node || !node.known_vars || typeof node.known_vars !== "object") {
    return "";
  }
  const task = node.known_vars.orchestrator_task;
  if (!task || typeof task !== "object") {
    return "";
  }
  return trimText(String(task.step_focus || task.selected_task || "").trim(), 26);
}

function getNodeIgnoredNoiseCount(node) {
  if (!node || !node.known_vars || typeof node.known_vars !== "object") {
    return 0;
  }
  let count = 0;
  if (Array.isArray(node.known_vars.ignored_review_rule_violations)) {
    count += node.known_vars.ignored_review_rule_violations.length;
  }
  const hardRuleCheck = node.known_vars.hard_rule_check;
  if (hardRuleCheck && typeof hardRuleCheck === "object" && Array.isArray(hardRuleCheck.ignored_violations)) {
    count += hardRuleCheck.ignored_violations.length;
  }
  return count;
}

function buildSearchText(node, depth) {
  return [
    node.id,
    node.parent_id,
    node.status,
    node.fsm_state,
    node.thought_step,
    depth,
    joinStrings(node.equations),
    safeJson(node.known_vars),
    joinStrings(node.used_models),
    safeJson(node.quantities),
    safeJson(node.boundary_conditions),
    joinStrings(node.reflection_history),
  ]
    .join(" ")
    .toLowerCase();
}

function computeSearchMatches(allNodes, query) {
  const normalizedQuery = String(query || "").trim().toLowerCase();
  if (!normalizedQuery) {
    return [];
  }
  return allNodes
    .filter((entry) => entry.searchText.includes(normalizedQuery))
    .map((entry) => entry.node.id);
}

function renderStatus() {
  const stamp = uiState.lastUpdatedAt ? ` | updated ${formatClock(uiState.lastUpdatedAt)}` : "";
  const session = uiState.sessionId ? ` | session ${uiState.sessionId}` : "";
  const polling = ` | poll ${uiState.pollingEnabled ? formatPollInterval(uiState.pollIntervalMs) : "off"}`;
  dom.statusLine.textContent = `${uiState.statusMessage}${session}${polling}${stamp}`;
  dom.statusLine.className = `status-line status-${uiState.statusTone}`;
}

function renderActionFeedback() {
  dom.actionFeedbackTitle.textContent = uiState.lastActionTitle;
  dom.actionFeedbackDetail.textContent = uiState.lastActionDetail;
  dom.actionFeedback.className = `action-feedback status-${uiState.lastActionTone}`;
}

function renderStatusLog() {
  if (!dom.statusLog) {
    return;
  }

  if (!uiState.statusLogEntries.length) {
    const empty = document.createElement("div");
    empty.className = "status-log-empty";
    empty.textContent = "No status history yet.";
    dom.statusLog.replaceChildren(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  uiState.statusLogEntries.forEach((entry) => {
    const item = document.createElement("div");
    item.className = `status-log-entry status-${entry.tone}`;

    const header = document.createElement("div");
    header.className = "status-log-header";

    const time = document.createElement("span");
    time.className = "status-log-time";
    time.textContent = formatClock(new Date(entry.at));

    const badge = document.createElement("span");
    badge.className = `status-log-badge status-${entry.tone}`;
    badge.textContent = shortUiTone(entry.tone);

    const title = document.createElement("div");
    title.className = "status-log-title";
    title.textContent = entry.title;

    header.append(time, badge, title);

    if (entry.count > 1) {
      const count = document.createElement("span");
      count.className = "status-log-count";
      count.textContent = `x${entry.count}`;
      header.append(count);
    }

    item.append(header);

    if (entry.detail) {
      const detail = document.createElement("div");
      detail.className = "status-log-detail";
      detail.textContent = entry.detail;
      item.append(detail);
    }

    fragment.append(item);
  });

  dom.statusLog.replaceChildren(fragment);
}

function renderButtons() {
  const hasSession = Boolean(uiState.sessionId);
  const hasSelection = Boolean(getSelectedNode());
  const rootNode = uiState.treeModel.root;
  const selectionIsRoot = Boolean(rootNode && uiState.selectedNodeId === rootNode.id);
  const sessionBusy = isSessionBusy(uiState.snapshot);

  [
    dom.attachSessionButton,
    dom.createSessionButton,
    dom.refreshSessionButton,
    dom.runSessionButton,
    dom.dropSessionButton,
    dom.prevMatchButton,
    dom.nextMatchButton,
    dom.expandAllButton,
    dom.collapseAllButton,
    dom.deleteNodeButton,
  ].forEach((button) => {
    button.disabled = uiState.requestInFlight;
  });

  dom.refreshSessionButton.disabled = uiState.requestInFlight || !hasSession;
  dom.runSessionButton.disabled = uiState.requestInFlight || !hasSession || sessionBusy;
  dom.dropSessionButton.disabled = uiState.requestInFlight || !hasSession;
  dom.expandAllButton.disabled = !uiState.snapshot;
  dom.collapseAllButton.disabled = !uiState.snapshot;
  dom.prevMatchButton.disabled = uiState.searchMatches.length === 0;
  dom.nextMatchButton.disabled = uiState.searchMatches.length === 0;
  dom.deleteNodeButton.disabled =
    uiState.requestInFlight || !hasSession || !hasSelection || selectionIsRoot;
}

function renderMeta() {
  const model = uiState.treeModel;
  const frontier = Array.isArray(uiState.snapshot && uiState.snapshot.frontier)
    ? uiState.snapshot.frontier
    : [];
  const remainingBudget = uiState.snapshot && Number.isFinite(uiState.snapshot.remaining_budget)
    ? uiState.snapshot.remaining_budget
    : "-";
  const runState = getRunState(uiState.snapshot);

  if (!model.root) {
    dom.treeStats.textContent = "no tree loaded";
    dom.selectionStats.textContent = "selection: none";
    dom.searchMeta.textContent = uiState.searchQuery ? "0 matches" : "0 matches";
    return;
  }

  if (model.root.known_vars && model.root.known_vars.synthetic_placeholder_root) {
    dom.treeStats.textContent =
      `root pending | status ${String(runState.status || "idle")} | phase ${String(runState.phase || "created")} | frontier ${frontier.length} | remaining ${remainingBudget}`;
    dom.selectionStats.textContent = "selection: pending root";
    dom.searchMeta.textContent = uiState.searchQuery ? "0 matches" : "0 matches";
    return;
  }

  dom.treeStats.textContent =
    `nodes ${model.allNodes.length} | active ${model.activeCount} | pruned ${model.prunedCount} | leaves ${model.leafCount} | depth ${model.maxDepth} | frontier ${frontier.length} | remaining ${remainingBudget}`;

  const selectedNode = getSelectedNode();
  if (selectedNode) {
    const selectedEntry = model.allNodes.find((entry) => entry.node.id === selectedNode.id);
    const selectedMatchIndex = uiState.searchMatches.indexOf(selectedNode.id);
    const childCount = getChildren(selectedNode).length;
    const routeFamily = getNodeRouteFamily(selectedNode);
    const stepFocus = getNodeStepFocus(selectedNode);
    const ignoredNoiseCount = getNodeIgnoredNoiseCount(selectedNode);
    const matchLabel = uiState.searchMatches.length
      ? ` | match ${selectedMatchIndex >= 0 ? selectedMatchIndex + 1 : 0}/${uiState.searchMatches.length}`
      : "";
    const routeLabel = routeFamily ? ` | route ${routeFamily}` : "";
    const stepLabel = stepFocus ? ` | step ${stepFocus}` : "";
    const noiseLabel = ignoredNoiseCount ? ` | noise ${ignoredNoiseCount}` : "";
    dom.selectionStats.textContent =
      `selection ${selectedNode.id} | depth ${selectedEntry ? selectedEntry.depth : 0} | children ${childCount}${routeLabel}${stepLabel}${noiseLabel}${matchLabel}`;
  } else {
    dom.selectionStats.textContent = "selection: none";
  }

  if (!uiState.searchQuery) {
    dom.searchMeta.textContent = "0 matches";
  } else if (uiState.searchMatches.length === 0) {
    dom.searchMeta.textContent = `0 matches for \"${uiState.searchQuery}\"`;
  } else {
    dom.searchMeta.textContent =
      `${uiState.searchMatches.length} matches | current ${uiState.searchCursor + 1}/${uiState.searchMatches.length}`;
  }
}

function renderTree() {
  const frontierIds = new Set(
    Array.isArray(uiState.snapshot && uiState.snapshot.frontier)
      ? uiState.snapshot.frontier.map((entry) => entry.node_id)
      : []
  );

  if (!uiState.treeModel.root) {
    const empty = document.createElement("div");
    empty.className = "tree-line empty-line";
    empty.textContent = "No tree loaded. Create a session or attach to an existing session id.";
    dom.treeLines.replaceChildren(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  let selectedElement = null;

  uiState.treeModel.visibleNodes.forEach((entry, index) => {
    const line = document.createElement("button");
    line.type = "button";
    line.className = "tree-line";
    line.style.setProperty("--tree-depth", String(entry.depth));
    if (entry.depth === 0) {
      line.classList.add("tree-line-root");
    }
    if (entry.node.id === uiState.selectedNodeId) {
      line.classList.add("selected");
      selectedElement = line;
    }
    if (frontierIds.has(entry.node.id)) {
      line.classList.add("frontier");
    }
    if (uiState.searchMatches.includes(entry.node.id)) {
      line.classList.add("match");
    }
    line.dataset.nodeId = entry.node.id;
    line.setAttribute("aria-selected", entry.node.id === uiState.selectedNodeId ? "true" : "false");

    const contentSpan = document.createElement("span");
    contentSpan.className = "tree-line-content";

    const headerSpan = document.createElement("span");
    headerSpan.className = "tree-line-header";

    const glyphSpan = document.createElement("span");
    glyphSpan.className = "tree-line-fold";
    glyphSpan.textContent = entry.summary.foldMark;

    const titleSpan = document.createElement("span");
    titleSpan.className = "tree-line-title";
    titleSpan.textContent = entry.summary.title;

    const metaSpan = document.createElement("span");
    metaSpan.className = "tree-line-meta";
    metaSpan.textContent = `${String(index + 1).padStart(3, "0")} · ${entry.summary.meta}`;

    const chipRow = document.createElement("span");
    chipRow.className = "tree-line-chips";

    const statusBadge = document.createElement("span");
    statusBadge.className = `tree-badge tree-badge-status tree-badge-status-${entry.summary.statusTone}`;
    statusBadge.textContent = entry.summary.status;
    chipRow.append(statusBadge);

    if (entry.summary.routeFamily) {
      const routeBadge = document.createElement("span");
      routeBadge.className = "tree-badge tree-badge-route";
      routeBadge.textContent = entry.summary.routeFamily;
      chipRow.append(routeBadge);
    }

    if (entry.summary.stepFocus) {
      const stepBadge = document.createElement("span");
      stepBadge.className = "tree-badge tree-badge-step";
      stepBadge.textContent = entry.summary.stepFocus;
      chipRow.append(stepBadge);
    }

    if (entry.summary.ignoredNoiseCount) {
      const noiseBadge = document.createElement("span");
      noiseBadge.className = "tree-badge tree-badge-noise";
      noiseBadge.textContent = `noise ${entry.summary.ignoredNoiseCount}`;
      chipRow.append(noiseBadge);
    }

    if (frontierIds.has(entry.node.id)) {
      const frontierBadge = document.createElement("span");
      frontierBadge.className = "tree-badge tree-badge-frontier";
      frontierBadge.textContent = "frontier";
      chipRow.append(frontierBadge);
    }

    if (uiState.searchMatches.includes(entry.node.id)) {
      const matchBadge = document.createElement("span");
      matchBadge.className = "tree-badge tree-badge-match";
      matchBadge.textContent = "match";
      chipRow.append(matchBadge);
    }

    headerSpan.append(glyphSpan, titleSpan);
    contentSpan.append(headerSpan, metaSpan);

    line.append(contentSpan, chipRow);
    fragment.append(line);
  });

  dom.treeLines.replaceChildren(fragment);

  if (uiState.revealSelection && selectedElement) {
    selectedElement.scrollIntoView({ block: "nearest", inline: "nearest" });
    uiState.revealSelection = false;
  }
}

function renderDetailPane() {
  const selectedNode = getSelectedNode();
  const metaTask = uiState.snapshot && typeof uiState.snapshot.meta_task === "object"
    ? uiState.snapshot.meta_task
    : {};
  const runState = getRunState(uiState.snapshot);
  if (!selectedNode) {
    const fragment = document.createDocumentFragment();
    dom.detailSummary.textContent = Object.keys(metaTask).length > 0
      ? "session meta task | no node selected"
      : "no node selected";

    if (Object.keys(metaTask).length > 0) {
      fragment.append(createDetailSection("Session meta task", createDetailPairs(metaTask)));
    }

    fragment.append(
      createDetailSection(
        "Selection",
        createDetailTextBlock("Attach a session or select a node to inspect the current tree.")
      )
    );
    dom.detailBody.replaceChildren(fragment);
    return;
  }

  const selectedEntry = uiState.treeModel.allNodes.find((entry) => entry.node.id === selectedNode.id);
  const depth = selectedEntry ? selectedEntry.depth : 0;
  const childIds = getChildren(selectedNode).map((child) => child.id);
  const isSyntheticRoot = isSyntheticPlaceholderRoot(selectedNode);
  const fragment = document.createDocumentFragment();
  const visibleResult = displayResultState(selectedNode);
  const routeFamily = getNodeRouteFamily(selectedNode);
  const stepFocus = getNodeStepFocus(selectedNode);
  const ignoredNoiseCount = getNodeIgnoredNoiseCount(selectedNode);

  dom.detailSummary.textContent =
    `${selectedNode.id} | depth ${depth} | ${shortStatus(visibleResult)} | children ${childIds.length}${routeFamily ? ` | route ${routeFamily}` : ""}${stepFocus ? ` | step ${stepFocus}` : ""}${ignoredNoiseCount ? ` | noise ${ignoredNoiseCount}` : ""}`;

  fragment.append(
    createDetailFacts([
      ["Node", selectedNode.id],
      ["Parent", selectedNode.parent_id || "-"],
      ["Depth", String(depth)],
      ["Route", routeFamily || "-"],
      ["Step focus", stepFocus || "-"],
      ["Result", shortStatus(visibleResult)],
      ["Status", String(selectedNode.status || "-").toUpperCase()],
      ["Score", formatScore(selectedNode.score)],
      ["Ignored noise", ignoredNoiseCount ? String(ignoredNoiseCount) : "0"],
      ["Children", String(childIds.length)],
    ])
  );

  if (Object.keys(metaTask).length > 0) {
    fragment.append(createDetailSection("Session meta task", createDetailPairs(metaTask)));
  }

  if (isSyntheticRoot) {
    fragment.append(
      createDetailSection(
        "Session run state",
        createDetailFacts([
          ["Run status", String(runState.status || "-")],
          ["Run phase", String(runState.phase || "-")],
          ["Problem context prepared", String(Boolean(runState.problem_context_prepared))],
          ["Auto run requested", String(Boolean(runState.auto_run_requested))],
          ["Last error", String(runState.last_error || "-")],
        ])
      )
    );
  }

  fragment.append(
    createDetailSection(
      "Thought step",
      createDetailTextBlock(formatInlineValue(selectedNode.thought_step) || "No thought step recorded.")
    ),
    createDetailSection("Equations", createDetailList(selectedNode.equations)),
    createDetailSection("Known vars", createDetailPairs(selectedNode.known_vars)),
    createDetailSection("Quantities", createDetailPairs(selectedNode.quantities)),
    createDetailSection("Boundary conditions", createDetailPairs(selectedNode.boundary_conditions)),
    createDetailSection("Used models", createDetailList(selectedNode.used_models)),
    createDetailSection("Reflection history", createDetailList(selectedNode.reflection_history)),
    createDetailSection("Child nodes", createDetailList(childIds)),
  );

  dom.detailBody.replaceChildren(fragment);
}

function renderDetailEmpty(message) {
  const empty = document.createElement("div");
  empty.className = "detail-empty";
  empty.textContent = message;
  dom.detailBody.replaceChildren(empty);
}

function createDetailFacts(items) {
  const grid = document.createElement("div");
  grid.className = "detail-facts";

  items.forEach(([label, value]) => {
    const fact = document.createElement("div");
    fact.className = "detail-fact";

    const labelSpan = document.createElement("div");
    labelSpan.className = "detail-fact-label";
    labelSpan.textContent = label;

    const valueSpan = document.createElement("div");
    valueSpan.className = "detail-fact-value";
    valueSpan.textContent = value;

    fact.append(labelSpan, valueSpan);
    grid.append(fact);
  });

  return grid;
}

function createDetailSection(title, contentNode) {
  const section = document.createElement("section");
  section.className = "detail-section";

  const titleNode = document.createElement("div");
  titleNode.className = "detail-section-title";
  titleNode.textContent = title;

  section.append(titleNode, contentNode);
  return section;
}

function createDetailTextBlock(text) {
  const block = document.createElement("div");
  block.className = "detail-text";
  block.textContent = text;
  return block;
}

function createDetailList(values) {
  const normalized = Array.isArray(values)
    ? values.map((value) => formatInlineValue(value)).filter(Boolean)
    : [];

  if (normalized.length === 0) {
    return createDetailEmptyBlock("None");
  }

  const list = document.createElement("ul");
  list.className = "detail-list";

  normalized.forEach((value) => {
    const item = document.createElement("li");
    item.textContent = value;
    list.append(item);
  });

  return list;
}

function createDetailPairs(value) {
  const entries = normalizeDetailEntries(value);
  if (entries.length === 0) {
    return createDetailEmptyBlock("None");
  }

  const wrapper = document.createElement("div");
  wrapper.className = "detail-pairs";

  entries.forEach(([key, entryValue]) => {
    const row = document.createElement("div");
    row.className = "detail-pair";

    const keyNode = document.createElement("div");
    keyNode.className = "detail-pair-key";
    keyNode.textContent = key;

    const valueNode = document.createElement("div");
    valueNode.className = "detail-pair-value";
    valueNode.textContent = entryValue;

    row.append(keyNode, valueNode);
    wrapper.append(row);
  });

  return wrapper;
}

function createDetailEmptyBlock(text) {
  const empty = document.createElement("div");
  empty.className = "detail-empty";
  empty.textContent = text;
  return empty;
}

function normalizeDetailEntries(value) {
  if (Array.isArray(value)) {
    return value
      .map((entry, index) => [String(index + 1), formatInlineValue(entry)])
      .filter(([, entryValue]) => Boolean(entryValue));
  }
  if (value && typeof value === "object") {
    return Object.entries(value)
      .map(([key, entryValue]) => [key, formatInlineValue(entryValue)])
      .filter(([, entryValue]) => Boolean(entryValue));
  }

  const text = formatInlineValue(value);
  return text ? [["value", text]] : [];
}

function formatInlineValue(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value.trim();
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch (_error) {
    return String(value);
  }
}

function renderFrontier() {
  const frontier = Array.isArray(uiState.snapshot && uiState.snapshot.frontier)
    ? uiState.snapshot.frontier
    : [];

  if (frontier.length === 0) {
    dom.frontierList.innerHTML = "";
    const empty = document.createElement("div");
    empty.className = "meta-line";
    empty.textContent = "frontier empty";
    dom.frontierList.append(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  frontier.forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "frontier-item";
    button.dataset.nodeId = entry.node_id;

    const idLine = document.createElement("div");
    idLine.className = "frontier-item-id";
    idLine.textContent = entry.node_id;

    const metaLine = document.createElement("div");
    metaLine.className = "frontier-item-meta";
    metaLine.textContent =
      `depth ${entry.depth} | priority ${formatScore(entry.priority)} | score ${formatScore(entry.score)} | ${entry.status}`;

    button.append(idLine, metaLine);
    fragment.append(button);
  });

  dom.frontierList.replaceChildren(fragment);
}

function renderActivity() {
  const entries = Array.isArray(uiState.snapshot && uiState.snapshot.expansion_log)
    ? uiState.snapshot.expansion_log
    : [];

  if (entries.length === 0) {
    dom.activityLog.textContent = "No events yet.";
    return;
  }

  const formatted = entries
    .slice(-8)
    .reverse()
    .map((entry, index) => formatActivityEntry(entry, entries.length - index))
    .join("\n\n");
  dom.activityLog.textContent = formatted;
}

function formatActivityEntry(entry, ordinal) {
  const header = `${String(ordinal).padStart(2, "0")} ${String(entry.event || "event").toUpperCase()}`;
  const details = [];
  if (entry.node_id) {
    details.push(`node=${entry.node_id}`);
  }
  if (Array.isArray(entry.deleted_node_ids) && entry.deleted_node_ids.length) {
    details.push(`deleted=${entry.deleted_node_ids.join(",")}`);
  }
  if (entry.frontier_size_after !== undefined) {
    details.push(`frontier=${entry.frontier_size_after}`);
  }
  return `${header}\n${details.join(" | ")}\n${trimText(JSON.stringify(entry), 300)}`;
}

function selectNode(nodeId) {
  if (!nodeId || !uiState.treeModel.nodeById.has(nodeId)) {
    return;
  }
  uiState.selectedNodeId = nodeId;
  uiState.revealSelection = true;
  render();
}

function moveSelection(direction) {
  if (!uiState.treeModel.visibleNodes.length) {
    return;
  }
  const currentIndex = uiState.treeModel.visibleNodes.findIndex(
    (entry) => entry.node.id === uiState.selectedNodeId,
  );
  const startIndex = currentIndex >= 0 ? currentIndex : 0;
  const nextIndex = clamp(startIndex + direction, 0, uiState.treeModel.visibleNodes.length - 1);
  selectNode(uiState.treeModel.visibleNodes[nextIndex].node.id);
}

function navigateLeft() {
  const node = getSelectedNode();
  if (!node) {
    return;
  }
  const children = getChildren(node);
  if (children.length > 0 && !uiState.collapsedNodeIds.has(node.id)) {
    uiState.collapsedNodeIds.add(node.id);
    setStatus(`Collapsed ${node.id}.`, "ok");
    render();
    return;
  }
  if (node.parent_id) {
    selectNode(node.parent_id);
  }
}

function navigateRight() {
  const node = getSelectedNode();
  if (!node) {
    return;
  }
  const children = getChildren(node);
  if (children.length === 0) {
    return;
  }
  if (uiState.collapsedNodeIds.has(node.id)) {
    uiState.collapsedNodeIds.delete(node.id);
    uiState.revealSelection = true;
    setStatus(`Expanded ${node.id}.`, "ok");
    render();
    return;
  }
  selectNode(children[0].id);
}

function toggleSelectedNodeExpansion(nodeId) {
  const node = uiState.treeModel.nodeById.get(nodeId);
  if (!node || getChildren(node).length === 0) {
    return;
  }
  if (uiState.collapsedNodeIds.has(node.id)) {
    uiState.collapsedNodeIds.delete(node.id);
    setStatus(`Expanded ${node.id}.`, "ok");
  } else {
    uiState.collapsedNodeIds.add(node.id);
    setStatus(`Collapsed ${node.id}.`, "ok");
  }
  uiState.revealSelection = true;
  render();
}

function collapseAllDescendants() {
  if (!uiState.treeModel.root) {
    return;
  }
  const descendantsWithChildren = uiState.treeModel.allNodes
    .filter((entry) => entry.depth > 0 && entry.childCount > 0)
    .map((entry) => entry.node.id);
  uiState.collapsedNodeIds = new Set(descendantsWithChildren);
  uiState.revealSelection = true;
  setStatus("Collapsed all descendant branches.", "ok");
  render();
}

function jumpToSearchMatch(direction) {
  if (uiState.searchMatches.length === 0) {
    setStatus("No search matches.", "error");
    return;
  }

  const currentIndex = uiState.searchMatches.indexOf(uiState.selectedNodeId);
  const baseIndex = currentIndex >= 0 ? currentIndex : uiState.searchCursor;
  const nextIndex = modulo(baseIndex + direction, uiState.searchMatches.length);
  const targetNodeId = uiState.searchMatches[nextIndex];
  revealNode(targetNodeId);
  uiState.searchCursor = nextIndex;
  selectNode(targetNodeId);
  setStatus(`Jumped to match ${nextIndex + 1}/${uiState.searchMatches.length}.`, "ok");
}

function revealNode(nodeId) {
  const ancestors = uiState.treeModel.ancestorsById.get(nodeId) || [];
  ancestors.forEach((ancestorId) => uiState.collapsedNodeIds.delete(ancestorId));
}

function panViewport(key) {
  const stepY = 48;
  const stepX = 96;
  switch (key) {
    case "ArrowUp":
      dom.treeViewport.scrollBy({ top: -stepY, behavior: "smooth" });
      break;
    case "ArrowDown":
      dom.treeViewport.scrollBy({ top: stepY, behavior: "smooth" });
      break;
    case "ArrowLeft":
      dom.treeViewport.scrollBy({ left: -stepX, behavior: "smooth" });
      break;
    case "ArrowRight":
      dom.treeViewport.scrollBy({ left: stepX, behavior: "smooth" });
      break;
    default:
      break;
  }
}

function getSelectedNode() {
  return uiState.selectedNodeId ? uiState.treeModel.nodeById.get(uiState.selectedNodeId) || null : null;
}

async function createSession() {
  if (uiState.requestInFlight) {
    return;
  }

  const payload = buildCreateSessionPayload();
  await withRequest("Creating live session...", async () => {
    const response = await apiRequest("/api/tot/sessions", {
      method: "POST",
      body: payload,
    });
    applySessionState(
      response.session_id,
      response.state,
      "Session created.",
      isSessionBusy(response.state) ? "busy" : "ok",
      summarizeSessionAction("create", null, response.state),
    );
    enableAutoRefreshForSession();
  });
}

async function attachSession(options = {}) {
  const sessionId = dom.sessionIdInput.value.trim();
  if (!sessionId) {
    setStatus("Session id is required to attach.", "error");
    return;
  }

  await withRequest(options.silent ? null : "Attaching session...", async () => {
    const response = await apiRequest(`/api/tot/sessions/${encodeURIComponent(sessionId)}`);
    applySessionState(
      response.session_id,
      response.state,
      options.silent ? "Session restored." : "Session attached.",
      "ok",
      summarizeSessionAction("attach", null, response.state),
    );
    enableAutoRefreshForSession();
  }, options);
}

async function refreshSession(options = {}) {
  if (!uiState.sessionId) {
    if (!options.silent) {
      setStatus("No session attached.", "error");
    }
    return;
  }

  await withRequest(options.silent ? null : "Refreshing session...", async () => {
    const previousSnapshot = uiState.snapshot;
    const response = await apiRequest(`/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}`);
    const refreshSummary = summarizeSessionAction("refresh", previousSnapshot, response.state);
    const preserveSilentFeedback = Boolean(
      options.silent
      && !isSessionBusy(response.state)
      && !hasVisibleMetricsChange(
        getSnapshotMetrics(previousSnapshot),
        getSnapshotMetrics(response.state)
      )
    );
    applySessionState(
      response.session_id,
      response.state,
      preserveSilentFeedback
        ? uiState.statusMessage
        : options.silent ? refreshSummary.statusMessageSilent : refreshSummary.statusMessage,
      preserveSilentFeedback ? uiState.statusTone : refreshSummary.tone,
      preserveSilentFeedback
        ? {
            title: uiState.lastActionTitle,
            detail: uiState.lastActionDetail,
            tone: uiState.lastActionTone,
            record: false,
          }
        : refreshSummary,
    );
  }, options);
}

async function runSession() {
  if (!uiState.sessionId) {
    setStatus("Create or attach a session first.", "error");
    return;
  }
  const additionalBudget = parseNonNegativeInteger(dom.budgetInput.value, "run budget");

  await withRequest("Running scheduler...", async () => {
    if (additionalBudget === 0) {
      const previousSnapshot = uiState.snapshot;
      const response = await apiRequest(`/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}/run`, {
        method: "POST",
        body: { additional_budget: 0 },
      });
      const runSummary = summarizeRunAction(previousSnapshot, response.state, 0);
      applySessionState(
        response.session_id,
        response.state,
        runSummary.statusMessage,
        runSummary.tone,
        runSummary,
      );
      return;
    }

    let remainingBudget = additionalBudget;
    let stepIndex = 0;
    while (remainingBudget > 0) {
      stepIndex += 1;
      const previousSnapshot = uiState.snapshot;
      const response = await apiRequest(`/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}/run`, {
        method: "POST",
        body: { additional_budget: 1 },
      });
      remainingBudget -= 1;
      const runSummary = summarizeRunAction(previousSnapshot, response.state, 1);
      const isFinalStep = remainingBudget === 0;
      applySessionState(
        response.session_id,
        response.state,
        isFinalStep ? runSummary.statusMessage : `Run step ${stepIndex}/${additionalBudget} applied.`,
        isFinalStep ? runSummary.tone : isSessionBusy(response.state) ? "busy" : "ok",
        isFinalStep
          ? runSummary
          : {
              title: `Run step ${stepIndex}/${additionalBudget} applied.`,
              detail: `${formatSnapshotSummary(response.state)} ${formatDeltaSummary(
                getSnapshotDeltas(getSnapshotMetrics(previousSnapshot), getSnapshotMetrics(response.state))
              )}`.trim(),
              tone: isSessionBusy(response.state) ? "busy" : "ok",
              record: false,
            },
      );
      if (getSnapshotMetrics(response.state).frontier === 0) {
        break;
      }
    }
  });
}

async function deleteSelectedNode() {
  if (!uiState.sessionId) {
    setStatus("Create or attach a session first.", "error");
    return;
  }
  const selectedNode = getSelectedNode();
  if (!selectedNode) {
    setStatus("Select a node before deleting.", "error");
    return;
  }
  if (!selectedNode.parent_id) {
    setStatus("Root deletion is blocked by the API.", "error");
    return;
  }

  const reason = dom.deleteReasonInput.value.trim();
  if (!reason) {
    setStatus("Deletion reason is required for backend review.", "error");
    return;
  }

  await withRequest(`Submitting delete review for ${selectedNode.id}...`, async () => {
    const response = await apiRequest(
      `/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}/nodes/${encodeURIComponent(selectedNode.id)}`,
      {
        method: "DELETE",
        body: {
          reason,
          requested_by: "frontend-terminal-gui",
        },
      },
    );
    uiState.selectedNodeId = selectedNode.parent_id;
    uiState.revealSelection = true;
    applySessionState(
      response.session_id,
      response.state,
      response.deleted
        ? `Deleted ${response.deleted_node_ids.length} node(s) after review.`
        : `Delete rejected: ${response.review && response.review.reason ? response.review.reason : "not approved"}`,
      response.deleted ? "ok" : "error",
      response.deleted
        ? {
            title: `Delete approved for ${selectedNode.id}.`,
            detail: `Removed ${response.deleted_node_ids.length} node(s). ${formatSnapshotSummary(response.state)}`,
            tone: "ok",
          }
        : {
            title: `Delete rejected for ${selectedNode.id}.`,
            detail: response.review && response.review.reason
              ? String(response.review.reason)
              : "The backend review model did not approve the deletion.",
            tone: "error",
          },
    );
  });
}

async function dropSession() {
  if (!uiState.sessionId) {
    setStatus("No session attached.", "error");
    return;
  }
  const confirmed = window.confirm(`Drop session ${uiState.sessionId}?`);
  if (!confirmed) {
    return;
  }

  await withRequest("Dropping session...", async () => {
    const droppedSessionId = uiState.sessionId;
    await apiRequest(`/api/tot/sessions/${encodeURIComponent(uiState.sessionId)}`, {
      method: "DELETE",
    });
    clearSessionState();
    setStatus("Session dropped.", "ok");
    setActionFeedback(
      `Session ${droppedSessionId} dropped.`,
      "The backend session was deleted and the local tree view was cleared.",
      "ok",
    );
  });
}

function buildCreateSessionPayload() {
  const problemContext = parseJsonText(dom.problemContextInput.value, "problem context", {});
  const scheduler = parseJsonText(dom.schedulerConfigInput.value, "scheduler config", DEFAULT_SCHEDULER_CONFIG);
  const problemStatement = dom.problemPromptInput.value.trim();
  if (!isPlainObject(problemContext)) {
    throw new Error("Problem context JSON must decode to an object.");
  }
  if (!isPlainObject(scheduler)) {
    throw new Error("Scheduler config JSON must decode to an object.");
  }
  if (!problemStatement) {
    dom.problemPromptInput.focus();
    throw new Error("Problem to solve is required before creating a session.");
  }

  const backend = {
    base_url: dom.baseUrlInput.value.trim(),
    planning_model: readModelInput(dom.planningModelInput, DEFAULT_PLANNING_MODEL),
    modeling_model: readModelInput(dom.modelingModelInput, DEFAULT_MODELING_MODEL),
    review_model: readModelInput(dom.reviewModelInput, DEFAULT_REVIEW_MODEL),
    non_terminal_evaluation_model: readModelInput(
      dom.nonTerminalEvaluationModelInput,
      DEFAULT_NON_TERMINAL_EVALUATION_MODEL,
    ),
    timeout: parsePositiveNumber(dom.timeoutInput.value, "timeout"),
  };

  return {
    problem_context: {
      ...problemContext,
      problem_statement: problemStatement,
    },
    backend,
    scheduler,
    run_on_create: dom.runOnCreateToggle.checked,
  };
}

function extractProblemStatementDraft(rawValue) {
  try {
    const parsed = JSON.parse(String(rawValue || "{}"));
    if (!isPlainObject(parsed)) {
      return "";
    }
    const statement = parsed.problem_statement;
    return typeof statement === "string" ? statement.trim() : "";
  } catch (_error) {
    return "";
  }
}

function sanitizeModelName(value, fallback) {
  const normalized = typeof value === "string" ? value.trim() : "";
  return normalized || fallback;
}

function isLegacyQwopusModel(value) {
  const normalized = sanitizeModelName(value, "");
  if (!normalized.startsWith("mlx-")) {
    return false;
  }
  const suffix = normalized.slice(4);
  return suffix === LEGACY_QWOPUS_MODEL_FAMILY || suffix === `${LEGACY_QWOPUS_MODEL_FAMILY}:2`;
}

function migrateLegacyPlanningModel(value) {
  const normalized = sanitizeModelName(value, DEFAULT_PLANNING_MODEL);
  return normalized === PREVIOUS_DEFAULT_PLANNING_MODEL || isLegacyQwopusModel(normalized)
    ? DEFAULT_PLANNING_MODEL
    : normalized;
}

function migrateLegacyModelingModel(value) {
  const normalized = sanitizeModelName(value, DEFAULT_MODELING_MODEL);
  return isLegacyQwopusModel(normalized) || normalized === PREVIOUS_DEFAULT_MODELING_MODEL
    ? DEFAULT_MODELING_MODEL
    : normalized;
}

function migrateLegacyReviewModel(value) {
  const normalized = sanitizeModelName(value, DEFAULT_REVIEW_MODEL);
  return isLegacyQwopusModel(normalized) || normalized === PREVIOUS_DEFAULT_REVIEW_MODEL
    ? DEFAULT_REVIEW_MODEL
    : normalized;
}

function applySessionState(sessionId, snapshot, message, tone = "ok", actionFeedback = null) {
  uiState.sessionId = String(sessionId || "");
  uiState.snapshot = snapshot || null;
  uiState.lastUpdatedAt = new Date();
  dom.sessionIdInput.value = uiState.sessionId;
  persistDraft();
  if (isSessionBusy(snapshot)) {
    scheduleBusyRefresh();
  } else {
    clearBusyRefresh();
  }
  setStatus(message, tone, { record: !actionFeedback });
  if (actionFeedback) {
    setActionFeedback(actionFeedback.title, actionFeedback.detail, actionFeedback.tone || tone, {
      record: actionFeedback.record !== false,
    });
  }
  render();
}

function clearSessionState() {
  clearBusyRefresh();
  uiState.sessionId = "";
  uiState.snapshot = null;
  uiState.selectedNodeId = null;
  uiState.collapsedNodeIds.clear();
  uiState.searchMatches = [];
  uiState.lastUpdatedAt = null;
  dom.sessionIdInput.value = "";
  persistDraft();
  render();
}

async function withRequest(message, operation, options = {}) {
  if (uiState.requestInFlight) {
    if (!options.silent) {
      setStatus("Another request is still running.", "error");
    }
    return;
  }

  uiState.requestInFlight = true;
  if (message) {
    setStatus(message, "busy", { record: false });
    setActionFeedback(message, "Waiting for the backend response...", "busy");
  }
  renderButtons();

  try {
    await operation();
  } catch (error) {
    if (options.silent && error instanceof Error && /Session not found/i.test(error.message)) {
      uiState.pollingEnabled = false;
      dom.pollingToggle.checked = false;
      restartPolling();
      clearSessionState();
      setStatus("Saved session expired. Start a new session.", "idle", { record: false });
      setActionFeedback(
        "Saved session expired.",
        "Auto refresh stopped because the backend no longer recognizes the saved session id.",
        "warn",
      );
      return;
    }
    handleError(error);
  } finally {
    uiState.requestInFlight = false;
    render();
  }
}

async function apiRequest(path, options = {}) {
  const request = {
    method: options.method || "GET",
    headers: {},
  };
  if (options.body !== undefined) {
    request.headers["Content-Type"] = "application/json";
    request.body = JSON.stringify(options.body);
  }

  const response = await window.fetch(path, request);
  const text = await response.text();
  const payload = parseResponsePayload(text);

  if (!response.ok) {
    if (payload && typeof payload === "object" && payload.detail) {
      throw new Error(String(payload.detail));
    }
    throw new Error(`HTTP ${response.status}`);
  }

  return payload;
}

function parseResponsePayload(text) {
  if (!text) {
    return null;
  }
  try {
    return JSON.parse(text);
  } catch (_error) {
    return text;
  }
}

function handleError(error) {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(message, "error", { record: false });
  setActionFeedback("Request failed.", message, "error");
}

function summarizeSessionAction(action, previousSnapshot, nextSnapshot) {
  const nextMetrics = getSnapshotMetrics(nextSnapshot);
  const previousMetrics = getSnapshotMetrics(previousSnapshot);
  const deltas = getSnapshotDeltas(previousMetrics, nextMetrics);
  const runState = getRunState(nextSnapshot);
  const sessionBusy = isSessionBusy(nextSnapshot);

  if (action === "create") {
    return {
      title: sessionBusy
        ? `Session created: ${nextMetrics.sessionHint}.`
        : `Session ready: ${nextMetrics.rootId || "new root"}.`,
      detail: sessionBusy
        ? `Session code issued before meta-task preparation. Background phase: ${runState.phase || "queued"}. ${formatSnapshotSummary(nextSnapshot)}`.trim()
        : `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim(),
      tone: sessionBusy ? "busy" : "ok",
      statusMessage: sessionBusy ? "Session created. Background initialization started." : "Session created.",
      statusMessageSilent: sessionBusy ? "Background initialization started." : "Session created.",
    };
  }

  if (action === "attach") {
    return {
      title: `Loaded session ${nextMetrics.sessionHint}.`,
      detail: sessionBusy
        ? `Background phase: ${runState.phase || "busy"}. ${formatSnapshotSummary(nextSnapshot)}`
        : `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim(),
      tone: sessionBusy ? "busy" : "ok",
      statusMessage: sessionBusy ? "Session attached. Background run still active." : "Session attached.",
      statusMessageSilent: sessionBusy ? "Background run still active." : "Session restored.",
    };
  }

  const changed = hasVisibleMetricsChange(previousMetrics, nextMetrics);
  if (sessionBusy) {
    return {
      title: changed ? "Background run advanced the tree." : "Background run still in progress.",
      detail: `${formatSnapshotSummary(nextSnapshot)} phase ${runState.phase || "busy"}. ${formatDeltaSummary(deltas)}`.trim(),
      tone: "busy",
      statusMessage: changed ? "Background run advanced the tree." : "Background run still in progress.",
      statusMessageSilent: changed ? "Background run advanced the tree." : "Background run still in progress.",
    };
  }

  return {
    title: changed ? "Session state refreshed." : "Refresh succeeded with no tree change.",
    detail: changed
      ? `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim()
      : `${formatSnapshotSummary(nextSnapshot)} No nodes, frontier entries, or expansion counters changed since the previous snapshot.`,
    tone: changed ? "ok" : "warn",
    statusMessage: changed ? "Session refreshed." : "Refresh completed with no visible change.",
    statusMessageSilent: changed ? "Auto refreshed." : "Auto refresh found no change.",
  };
}

function runUiAction(action) {
  Promise.resolve()
    .then(action)
    .catch((error) => {
      handleError(error);
      render();
    });
}

function summarizeRunAction(previousSnapshot, nextSnapshot, additionalBudget) {
  const previousMetrics = getSnapshotMetrics(previousSnapshot);
  const nextMetrics = getSnapshotMetrics(nextSnapshot);
  const deltas = getSnapshotDeltas(previousMetrics, nextMetrics);

  if (additionalBudget === 0) {
    return {
      title: "Run request completed with +0 budget.",
      detail: `${formatSnapshotSummary(nextSnapshot)} No new scheduler budget was granted, so no expansion was expected.`,
      tone: "warn",
      statusMessage: "Run completed with +0 budget.",
    };
  }

  if (deltas.expansions > 0 || deltas.nodes > 0) {
    return {
      title: `Run advanced the tree with +${additionalBudget} budget.`,
      detail: `${formatSnapshotSummary(nextSnapshot)} ${formatDeltaSummary(deltas)}`.trim(),
      tone: "ok",
      statusMessage: `Run completed: +${Math.max(0, deltas.expansions)} expansion(s), +${Math.max(0, deltas.nodes)} node(s).`,
    };
  }

  const stallReason = inferRunStallReason(nextMetrics);
  return {
    title: stallReason.title,
    detail: `${stallReason.detail} ${formatSnapshotSummary(nextSnapshot)}`.trim(),
    tone: "warn",
    statusMessage: stallReason.statusMessage,
  };
}

function inferRunStallReason(metrics) {
  if (metrics.remainingBudget === 0) {
    return {
      title: "Run stopped: scheduler budget exhausted.",
      detail: "The backend accepted the run request, but the session has no remaining global expansion budget.",
      statusMessage: "Run stopped: budget exhausted.",
    };
  }
  if (metrics.frontier === 0) {
    return {
      title: "Run stopped: no expandable frontier remained.",
      detail: "There were no frontier nodes left that the scheduler could expand.",
      statusMessage: "Run stopped: frontier empty.",
    };
  }
  return {
    title: "Run completed with no visible tree change.",
    detail: "The backend returned successfully, but the snapshot counters did not change. This usually means the scheduler revisited state without finding an expandable branch.",
    statusMessage: "Run completed with no visible change.",
  };
}

function getSnapshotMetrics(snapshot) {
  const root = snapshot && snapshot.root ? snapshot.root : null;
  const frontier = Array.isArray(snapshot && snapshot.frontier) ? snapshot.frontier.length : 0;
  const expansionLog = Array.isArray(snapshot && snapshot.expansion_log) ? snapshot.expansion_log : [];
  const expansionsUsed = Number.isFinite(snapshot && snapshot.expansions_used)
    ? Number(snapshot.expansions_used)
    : 0;
  const remainingBudget = Number.isFinite(snapshot && snapshot.remaining_budget)
    ? Number(snapshot.remaining_budget)
    : null;
  const allNodes = root ? flattenNodes(root) : [];
  const runState = getRunState(snapshot);

  return {
    rootId: root && root.id ? String(root.id) : "",
    sessionHint: uiState.sessionId || dom.sessionIdInput.value.trim() || "current session",
    nodes: allNodes.length,
    active: allNodes.filter((node) => String(node.status || "") === "ACTIVE").length,
    frontier,
    expansionsUsed,
    remainingBudget,
    runStatus: String(runState.status || "idle"),
    runPhase: String(runState.phase || "created"),
    lastEvent: expansionLog.length ? expansionLog[expansionLog.length - 1] : null,
  };
}

function getSnapshotDeltas(previousMetrics, nextMetrics) {
  return {
    nodes: nextMetrics.nodes - previousMetrics.nodes,
    frontier: nextMetrics.frontier - previousMetrics.frontier,
    expansions: nextMetrics.expansionsUsed - previousMetrics.expansionsUsed,
  };
}

function hasVisibleMetricsChange(previousMetrics, nextMetrics) {
  const deltas = getSnapshotDeltas(previousMetrics, nextMetrics);
  return (
    deltas.nodes !== 0
    || deltas.frontier !== 0
    || deltas.expansions !== 0
    || previousMetrics.runStatus !== nextMetrics.runStatus
    || previousMetrics.runPhase !== nextMetrics.runPhase
  );
}

function formatSnapshotSummary(snapshot) {
  const metrics = getSnapshotMetrics(snapshot);
  const remainingBudget = metrics.remainingBudget === null ? "-" : metrics.remainingBudget;
  const runState = metrics.runStatus && metrics.runStatus !== "idle"
    ? ` | ${metrics.runStatus.toLowerCase()} ${metrics.runPhase.toLowerCase()}`
    : "";
  return `nodes ${metrics.nodes} | active ${metrics.active} | frontier ${metrics.frontier} | expansions ${metrics.expansionsUsed} | remaining ${remainingBudget}${runState}`;
}

function formatDeltaSummary(deltas) {
  const parts = [];
  if (deltas.nodes !== 0) {
    parts.push(`${signedCount(deltas.nodes)} node(s)`);
  }
  if (deltas.frontier !== 0) {
    parts.push(`${signedCount(deltas.frontier)} frontier`);
  }
  if (deltas.expansions !== 0) {
    parts.push(`${signedCount(deltas.expansions)} expansion(s)`);
  }
  return parts.length ? `Change: ${parts.join(" | ")}.` : "";
}

function signedCount(value) {
  return value > 0 ? `+${value}` : String(value);
}

function shortUiTone(tone) {
  const normalized = String(tone || "idle").toUpperCase();
  if (normalized === "OK") {
    return "OK";
  }
  if (normalized === "ERROR") {
    return "ERROR";
  }
  if (normalized === "WARN") {
    return "WARN";
  }
  if (normalized === "BUSY") {
    return "BUSY";
  }
  return "INFO";
}

function flattenNodes(root) {
  if (!root) {
    return [];
  }
  const nodes = [];
  const stack = [root];
  while (stack.length) {
    const current = stack.pop();
    if (!current || typeof current !== "object") {
      continue;
    }
    nodes.push(current);
    const children = Array.isArray(current.children) ? current.children : [];
    for (let index = children.length - 1; index >= 0; index -= 1) {
      stack.push(children[index]);
    }
  }
  return nodes;
}

function parseJsonText(rawValue, label, fallbackValue) {
  const text = String(rawValue || "").trim();
  if (!text) {
    return fallbackValue;
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new Error(`${label} JSON is invalid: ${error instanceof Error ? error.message : error}`);
  }
}

function parsePositiveNumber(rawValue, label) {
  const parsed = Number(rawValue);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${label} must be a positive number.`);
  }
  return parsed;
}

function parseNonNegativeInteger(rawValue, label) {
  const parsed = Number.parseInt(String(rawValue || "0"), 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
    throw new Error(`${label} must be a non-negative integer.`);
  }
  return parsed;
}

function shortStatus(status) {
  const normalized = String(status || "UNKNOWN").toUpperCase();
  if (normalized === "PASS") {
    return "PASS";
  }
  if (normalized === "DROP") {
    return "DROP";
  }
  if (normalized === "FINALIZE") {
    return "FINAL";
  }
  if (normalized === "ACTIVE") {
    return "ACT";
  }
  if (normalized === "PRUNED_BY_RULE") {
    return "RULE";
  }
  if (normalized === "PRUNED_BY_SLM") {
    return "SLM";
  }
  return normalized;
}

function statusTone(status) {
  const normalized = String(status || "UNKNOWN").toUpperCase();
  if (normalized === "PASS") {
    return "active";
  }
  if (normalized === "DROP") {
    return "rule";
  }
  if (normalized === "FINALIZE") {
    return "solved";
  }
  if (normalized === "ACTIVE") {
    return "active";
  }
  if (normalized === "PRUNED_BY_RULE") {
    return "rule";
  }
  if (normalized === "PRUNED_BY_SLM") {
    return "slm";
  }
  if (normalized === "SOLVED") {
    return "solved";
  }
  return "unknown";
}

function formatScore(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "-";
  }
  return numeric.toFixed(2);
}

function trimText(text, maxLength) {
  const normalized = String(text || "");
  if (normalized.length <= maxLength) {
    return normalized;
  }
  return `${normalized.slice(0, Math.max(0, maxLength - 3))}...`;
}

function safeJson(value) {
  try {
    return JSON.stringify(value || {});
  } catch (_error) {
    return String(value);
  }
}

function joinStrings(value) {
  return Array.isArray(value) ? value.join(" ") : "";
}

function displayResultState(node) {
  const explicit = String(node && node.result_state ? node.result_state : "").trim().toUpperCase();
  if (explicit) {
    return explicit;
  }
  const normalizedStatus = String(node && node.status ? node.status : "UNKNOWN").trim().toUpperCase();
  if (normalizedStatus === "SOLVED") {
    return "FINALIZE";
  }
  if (normalizedStatus.startsWith("PRUNED")) {
    return "DROP";
  }
  if (normalizedStatus === "ACTIVE") {
    return "PASS";
  }
  return normalizedStatus || "UNKNOWN";
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function modulo(value, divisor) {
  return ((value % divisor) + divisor) % divisor;
}

function formatClock(date) {
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function isPlainObject(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}