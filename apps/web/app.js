const state = {
  apiKey: localStorage.getItem("xm.apiKey") || "demo-key-alpha",
  sessionId: "",
  channel: "amazon",
  busy: false,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

function apiHeaders(extra = {}) {
  return { "Content-Type": "application/json", "X-API-Key": state.apiKey, ...extra };
}

async function api(path, options = {}) {
  const response = await fetch(`/api/v1${path}`, {
    ...options,
    headers: apiHeaders(options.headers || {}),
  });
  if (!response.ok) {
    let detail = `${response.status}`;
    try {
      const payload = await response.json();
      detail = payload.error?.message || detail;
    } catch (_) { /* keep status code */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ------------------------------------------------------------------ boot */
function boot() {
  $("#apiKey").value = state.apiKey;
  $("#apiKey").addEventListener("change", (event) => {
    state.apiKey = event.target.value.trim();
    localStorage.setItem("xm.apiKey", state.apiKey);
    refreshStatus();
  });
  $("#channel").addEventListener("change", (event) => { state.channel = event.target.value; });

  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.view));
  });
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      $("#message").value = chip.textContent;
      $("#message").focus();
    });
  });

  $("#composer").addEventListener("submit", (event) => {
    event.preventDefault();
    send();
  });
  $("#message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  $("#refreshTasks").addEventListener("click", loadTasks);
  $("#refreshOps").addEventListener("click", loadOps);
  $("#kbSearch").addEventListener("click", searchKb);
  $("#kbQuery").addEventListener("keydown", (event) => {
    if (event.key === "Enter") searchKb();
  });

  addSystem("已就绪：默认使用离线确定性模型，无需任何 API Key 即可完整演示。");
  refreshStatus();
}

async function refreshStatus() {
  try {
    const ready = await fetch("/readyz").then((r) => r.json());
    const mode = ready.offline ? "离线模式（mock）" : "在线模型";
    $("#status").textContent = `${mode} · 知识块 ${ready.chunks}`;
  } catch (error) {
    $("#status").textContent = "服务不可达";
  }
}

function switchView(view) {
  document.querySelectorAll(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  if (view === "tasks") loadTasks();
  if (view === "kb") loadKbStats();
  if (view === "ops") loadOps();
}

/* ------------------------------------------------------------------ chat */
function addSystem(text) {
  const wrap = el("div", "msg");
  wrap.append(el("div", "avatar", "CM"));
  const bubble = el("div", "bubble", text);
  bubble.style.color = "var(--muted)";
  wrap.append(bubble);
  $("#chatLog").append(wrap);
  $("#chatLog").scrollTop = 1e9;
}

function addUser(text) {
  const wrap = el("div", "msg user");
  wrap.append(el("div", "avatar", "你"));
  wrap.append(el("div", "bubble", text));
  $("#chatLog").append(wrap);
  $("#chatLog").scrollTop = 1e9;
}

function addAssistant() {
  const wrap = el("div", "msg");
  wrap.append(el("div", "avatar", "AI"));
  const bubble = el("div", "bubble", "");
  wrap.append(bubble);
  const steps = el("div", "steps");
  bubble.append(steps);
  const body = el("div", "answer-text");
  const stepsHost = el("div", "steps-host");
  bubble.append(body, stepsHost);
  $("#chatLog").append(wrap);
  $("#chatLog").scrollTop = 1e9;
  return { bubble, body, stepsHost };
}

async function send() {
  const input = $("#message");
  const text = input.value.trim();
  if (!text || state.busy) return;
  state.busy = true;
  $("#sendBtn").disabled = true;
  input.value = "";
  addUser(text);
  const ui = addAssistant();

  try {
    const response = await fetch("/api/v1/chat/stream", {
      method: "POST",
      headers: apiHeaders(),
      body: JSON.stringify({
        message: text,
        session_id: state.sessionId || null,
        channel: state.channel,
        locale: "zh-CN",
      }),
    });
    if (!response.ok || !response.body) throw new Error(`stream failed: ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let printTimer = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let index;
      while ((index = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, index);
        buffer = buffer.slice(index + 2);
        handleEvent(chunk, ui);
      }
    }
    if (printTimer) clearTimeout(printTimer);
  } catch (error) {
    ui.body.textContent = `请求失败：${error.message}`;
    ui.body.classList.add("error");
  } finally {
    state.busy = false;
    $("#sendBtn").disabled = false;
    input.focus();
  }
}

function handleEvent(raw, ui) {
  const lines = raw.split("\n");
  let event = "message";
  const dataLines = [];
  for (const line of lines) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return;
  let payload = {};
  try { payload = JSON.parse(dataLines.join("\n")); } catch (_) { return; }

  if (event === "start") {
    state.sessionId = payload.session_id;
    $("#sessionId").value = state.sessionId;
  } else if (event === "step") {
    const row = el("div", "step");
    row.append(el("span", "dot", "•"), el("span", "", `${payload.node} · ${payload.detail || payload.status}`));
    ui.stepsHost.append(row);
  } else if (event === "delta") {
    ui.body.textContent += payload.text || "";
    $("#chatLog").scrollTop = 1e9;
  } else if (event === "done") {
    renderDone(ui, payload);
  } else if (event === "error") {
    ui.body.textContent = `执行失败：${payload.message}`;
    ui.body.classList.add("error");
  }
}

function renderDone(ui, payload) {
  ui.body.textContent = payload.answer || "";
  const meta = el("div", "meta");
  meta.append(
    el("span", "tag", `intent: ${payload.intent}`),
    el("span", "tag", `status: ${payload.status}`),
    el("span", "tag", `trace: ${(payload.trace_id || "").slice(0, 8)}`),
  );
  const usage = payload.usage || {};
  if (usage.prompt_tokens !== undefined) {
    meta.append(el("span", "tag", `tokens: ${(usage.prompt_tokens || 0) + (usage.completion_tokens || 0)}`));
  }
  if (payload.requires_approval) {
    const warn = el("span", "tag warn", "待人工确认");
    warn.title = payload.approval_reason || "";
    meta.append(warn);
  }
  ui.bubble.append(meta);

  const citations = payload.citations || [];
  if (citations.length) {
    const box = el("div", "citations");
    citations.forEach((citation) => {
      const card = el("div", "citation");
      const head = el("b", "", `【${citation.index}】${citation.doc_title}`);
      card.append(head, document.createTextNode(` · ${citation.section || ""} · score ${Number(citation.score).toFixed(3)}\n`));
      card.append(document.createTextNode(citation.text || ""));
      box.append(card);
    });
    ui.bubble.append(box);
  }
  $("#chatLog").scrollTop = 1e9;
}

/* ------------------------------------------------------------------ tasks */
async function loadTasks() {
  const body = $("#taskRows");
  body.innerHTML = "";
  try {
    const tasks = await api("/tasks?limit=30");
    if (!tasks.length) {
      const row = el("tr");
      const cell = el("td", "muted", "还没有任务，去对话控制台发起一次请求。");
      cell.colSpan = 5;
      row.append(cell);
      body.append(row);
      return;
    }
    tasks.forEach((task) => {
      const row = el("tr");
      row.append(el("td", "mono", task.id.slice(0, 10)));
      row.append(el("td", "", task.type));
      const statusCell = el("td");
      statusCell.append(el("span", `tag ${statusTag(task.status)}`, task.status));
      row.append(statusCell);
      row.append(el("td", "mono", (task.created_at || "").replace("T", " ").slice(0, 19)));
      const actions = el("td");
      if (task.status === "awaiting_approval") {
        const approve = el("button", "btn", "通过");
        approve.onclick = () => decide(task.id, "approve");
        const reject = el("button", "btn danger", "驳回");
        reject.style.marginLeft = "6px";
        reject.onclick = () => decide(task.id, "reject");
        actions.append(approve, reject);
      }
      row.append(actions);
      body.append(row);
    });
  } catch (error) {
    body.innerHTML = `<tr><td colspan="5" class="error">${error.message}</td></tr>`;
  }
}

function statusTag(status) {
  if (status === "succeeded") return "good";
  if (status === "awaiting_approval") return "warn";
  if (status === "failed" || status === "rejected") return "bad";
  return "";
}

async function decide(taskId, decision) {
  try {
    await api(`/tasks/${taskId}/approve`, {
      method: "POST",
      body: JSON.stringify({ decision, reviewer: "console", comment: "demo approval" }),
    });
    loadTasks();
  } catch (error) {
    alert(error.message);
  }
}

/* ------------------------------------------------------------------ kb */
async function loadKbStats() {
  try {
    const stats = await api("/kb/stats");
    const box = $("#kbStats");
    box.innerHTML = "";
    box.append(el("div", "", `文档 ${stats.documents} 篇 · 知识块 ${stats.chunks} 个 · 向量维度 ${stats.dim}`));
    (stats.sources || []).forEach((doc) => {
      box.append(el("div", "trace", `${doc.doc_id} · ${doc.title} · ${doc.type}`));
    });
  } catch (error) {
    $("#kbStats").textContent = error.message;
  }
}

async function searchKb() {
  const query = $("#kbQuery").value.trim();
  if (!query) return;
  const box = $("#kbHits");
  box.textContent = "检索中…";
  try {
    const hits = await api(`/kb/search?q=${encodeURIComponent(query)}&top_k=5`);
    box.innerHTML = "";
    hits.forEach((hit) => {
      const card = el("div", "hit");
      const head = el("div", "head");
      head.append(el("span", "", `${hit.doc_id} · ${hit.doc_title} · ${hit.section || "正文"}`));
      head.append(el("span", "", `score ${hit.score.toFixed(3)}`));
      card.append(head, el("div", "text", hit.text.slice(0, 220)));
      const bar = el("div", "bar");
      const fill = el("i");
      fill.style.width = `${Math.min(100, Math.max(4, hit.score * 100))}%`;
      bar.append(fill);
      card.append(bar);
      box.append(card);
    });
    if (!hits.length) box.textContent = "没有命中，说明该问题可能超出知识库范围（会被拒答）。";
  } catch (error) {
    box.innerHTML = `<div class="error">${error.message}</div>`;
  }
}

/* ------------------------------------------------------------------ ops */
async function loadOps() {
  try {
    const summary = await api("/ops/summary");
    const cards = $("#opsCards");
    cards.innerHTML = "";
    const add = (key, value, hint) => {
      const card = el("div", "card");
      card.append(el("div", "k", key));
      const v = el("div", "v", String(value));
      if (hint) v.append(el("small", "", ` ${hint}`));
      card.append(v);
      cards.append(card);
    };
    add("运行模式", summary.offline ? "offline" : "online", summary.llm.model);
    add("知识块", summary.corpus.chunks, `/ ${summary.corpus.documents} 文档`);
    add("请求数", summary.http.requests);
    add("平均延迟", `${summary.http.avg_latency_ms}ms`);
  } catch (error) {
    $("#opsCards").innerHTML = `<div class="error">${error.message}</div>`;
  }

  try {
    const traces = await api("/ops/traces?limit=8");
    const list = $("#traceList");
    list.innerHTML = "";
    traces.forEach((trace) => {
      const item = el("div", "trace");
      const row = el("div", "row");
      row.append(el("span", "mono", `${trace.trace_id.slice(0, 8)} · ${trace.route}`));
      row.append(el("span", "", `${trace.duration_ms.toFixed(1)}ms`));
      item.append(row);
      const spans = el("div", "spans");
      trace.spans.slice(0, 6).forEach((span) => spans.append(el("span", "tag", `${span.name} ${span.duration_ms.toFixed(1)}ms`)));
      item.append(spans);
      list.append(item);
    });
    if (!traces.length) list.textContent = "还没有请求。";
  } catch (error) {
    $("#traceList").textContent = error.message;
  }

  try {
    const report = await api("/eval/latest");
    const box = $("#evalBox");
    box.innerHTML = "";
    const m = report.metrics;
    const rows = [
      [`hit@${report.top_k}`, m.hit_at_k.toFixed(4)],
      ["MRR", m.mrr.toFixed(4)],
      ["拒答正确率", m.refusal_accuracy.toFixed(4)],
      ["引用准确率", m.citation_precision.toFixed(4)],
      ["忠实度", m.faithfulness.toFixed(4)],
      ["要点覆盖率", m.key_point_coverage.toFixed(4)],
      ["平均延迟", `${m.avg_latency_ms.toFixed(1)}ms`],
    ];
    rows.forEach(([key, value]) => {
      const row = el("div", "eval-row");
      row.append(el("span", "", key), el("b", "", value));
      box.append(row);
    });
  } catch (error) {
    $("#evalBox").textContent = "还没有评测报告：POST /api/v1/eval/run 生成。";
  }
}

boot();
