import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getVersion } from "@tauri-apps/api/app";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

const state = {
  records: [],
  activeId: null,
  busy: false,
};

const STAGES = [
  "解析 demo 与玩家名单",
  "提取 256 tick 击杀窗口",
  "运行 15 个折模型",
  "聚合玩家与复核队列",
];

const TIER_LABEL = { flag: "FLAG", review: "REVIEW", below: "BELOW", insufficient: "无窗口" };

const MAP_NAMES = {
  de_dust2: ["炙热沙城 II", "Dust II"],
  de_mirage: ["荒漠迷城", "Mirage"],
  de_inferno: ["炼狱小镇", "Inferno"],
  de_train: ["列车停放站", "Train"],
  de_nuke: ["核子危机", "Nuke"],
  de_ancient: ["远古遗迹", "Ancient"],
  de_vertigo: ["殒命大厦", "Vertigo"],
  de_anubis: ["阿努比斯", "Anubis"],
  de_overpass: ["死亡游乐园", "Overpass"],
  cs_office: ["办公室", "Office"],
  cs_italy: ["意大利小镇", "Italy"],
  de_cache: ["死城之谜", "Cache"],
  de_cobblestone: ["古堡激战", "Cobblestone"],
  de_aztec: ["雨林遗迹", "Aztec"],
  de_dust: ["炙热沙城", "Dust"],
  de_basalt: ["", "Basalt"],
  de_edin: ["", "Edin"],
  de_thera: ["", "Thera"],
  de_mills: ["", "Mills"],
};

function mapInfo(code) {
  const raw = String(code || "").trim();
  const entry = MAP_NAMES[raw];
  if (entry) {
    return {
      primary: entry[0] || entry[1],
      secondary: entry[0] ? entry[1] : "",
      code: raw,
    };
  }
  const pretty = raw
    .replace(/^(de|cs)_/i, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
  return { primary: pretty || "未知地图", secondary: "", code: raw };
}

const REASON_TEXT = {
  HS: { label: "HS", title: "爆头率 ≥ 80%" },
  EFF: { label: "Eff", title: "每击杀射击数 ≤ 1.5" },
  SNIPER_HS: { label: "Sniper HS", title: "狙击爆头" },
  WALL: { label: "Wall", title: "穿墙击杀 ≥ 20%" },
  SMOKE: { label: "Smoke", title: "穿烟击杀 ≥ 20%" },
  MULTI4: { label: "Multi4", title: "4 秒内 ≥ 3 杀" },
  BURST: { label: "Burst", title: "连杀间隔 ≤ 4 秒的比例 ≥ 30%" },
};

const WEAPON_TEXT = {
  knife: "刀", rifle: "步枪", sniper: "狙击", pistol: "手枪",
  grenade: "手雷", smg: "冲锋枪", shotgun: "霰弹枪", other: "其他",
};

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function pct(value) {
  return value == null ? "—" : `${(value * 100).toFixed(0)}%`;
}

function num(value, digits = 3) {
  return value == null ? "—" : Number(value).toFixed(digits);
}

function baseName(path) {
  const cut = Math.max(path.lastIndexOf("/"), path.lastIndexOf("\\"));
  return cut === -1 ? path : path.slice(cut + 1);
}

function activeRecord() {
  return state.records.find((record) => record.id === state.activeId) || null;
}

function tierBadge(tier) {
  return `<span class="badge ${esc(tier)}"><i class="dot"></i>${esc(TIER_LABEL[tier] || tier)}</span>`;
}

let toastTimer = 0;

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    node.hidden = true;
  }, 3200);
}

function updateBusy() {
  $("#addBtn").disabled = state.busy;
  $("#dropZone").disabled = state.busy;
}

/* ---------- records ---------- */

function renderRecords() {
  $("#recordEmpty").hidden = state.records.length > 0;
  $("#recordList").innerHTML = state.records.map((record) => {
    const statusLabel = record.status === "done"
      ? "完成"
      : record.status === "failed"
        ? "失败"
        : "分析中";
    const sub = record.status === "done"
      ? `${mapInfo(record.data.map).primary} · ${record.data.summary.flagged} FLAG · ${record.data.summary.players} 人`
      : record.status === "failed"
        ? "解析失败"
        : record.stage;
    return `
      <li class="record${record.id === state.activeId ? " active" : ""}" data-id="${record.id}">
        <button class="record-main" type="button">
          <span class="record-top">
            <span class="record-name" title="${esc(record.path)}">${esc(record.name)}</span>
            <span class="record-state ${record.status}">${statusLabel}</span>
          </span>
          <span class="record-sub">${esc(sub)}</span>
        </button>
      </li>
    `;
  }).join("");
}

function selectRecord(id) {
  const record = state.records.find((item) => item.id === id);
  if (!record) return;
  if (record.id === state.activeId) {
    if (record.status === "done" && record.view !== "results") {
      record.view = "results";
      renderScreen();
      scrollScreenTop();
    }
    return;
  }
  state.activeId = id;
  renderRecords();
  renderScreen();
  scrollScreenTop();
}

function scrollScreenTop() {
  requestAnimationFrame(() => {
    $("#results").scrollTop = 0;
    $("#detail").scrollTop = 0;
  });
}

/* ---------- screens ---------- */

function renderScreen() {
  hideKillTip();
  const record = activeRecord();
  const screens = {
    empty: $("#emptyScreen"),
    working: $("#workingScreen"),
    failed: $("#failedScreen"),
    results: $("#results"),
    detail: $("#detail"),
  };
  Object.values(screens).forEach((node) => {
    node.hidden = true;
  });

  if (!record) {
    screens.empty.hidden = false;
    return;
  }

  if (record.status === "working") {
    $("#workingFile").textContent = record.path;
    $("#workingStage").textContent = record.stage;
    screens.working.hidden = false;
    return;
  }

  if (record.status === "failed") {
    $("#failedFile").textContent = record.path;
    $("#failedReason").textContent = record.error;
    screens.failed.hidden = false;
    return;
  }

  if (record.view === "detail") {
    const player = record.data.players.find((item) => item.steamid === record.detailSteamid);
    if (player) {
      renderDetail(record, player);
      screens.detail.hidden = false;
      return;
    }
    record.view = "results";
  }

  renderResults(record.data);
  screens.results.hidden = false;
}

/* ---------- scoring ---------- */

async function runRecord(record) {
  state.busy = true;
  record.status = "working";
  record.error = null;
  record.stage = STAGES[0];
  updateBusy();
  renderRecords();
  renderScreen();

  let stage = 0;
  const timer = setInterval(() => {
    if (record.status !== "working") return;
    stage = (stage + 1) % STAGES.length;
    record.stage = STAGES[stage];
    renderRecords();
    if (state.activeId === record.id) {
      $("#workingStage").textContent = record.stage;
    }
  }, 1600);

  try {
    const data = await invoke("score_demo", { path: record.path });
    if (!data || data.error) throw new Error(data?.error || "分析失败");
    record.data = data;
    record.status = "done";
    record.stage = "";
  } catch (error) {
    record.status = "failed";
    record.error = error?.message || String(error);
  } finally {
    clearInterval(timer);
    state.busy = false;
    updateBusy();
    renderRecords();
    renderScreen();
    scrollScreenTop();
    await persistRecord(record);
  }
}

async function persistRecord(record) {
  try {
    await invoke("save_record", {
      record: {
        id: record.id,
        path: record.path,
        name: record.name,
        status: record.status,
        error: record.error,
        data: record.data,
        savedAt: new Date().toISOString(),
      },
    });
  } catch (error) {
    console.warn("could not save record", error);
  }
}

async function loadRecords() {
  try {
    const saved = await invoke("load_records");
    if (!Array.isArray(saved)) return;
    state.records = saved
      .filter((record) => record && record.id && record.path)
      .map((record) => ({
        ...record,
        view: "results",
        detailSteamid: null,
      }));
  } catch (error) {
    console.warn("could not load records", error);
  }
  renderRecords();
  renderScreen();
}

async function analyzePath(rawPath) {
  const path = String(rawPath || "").trim().replace(/^"|"$/g, "");
  if (!path) return;
  if (state.busy) {
    toast("正在分析另一局，完成后再添加。");
    return;
  }
  const record = {
    id: String(Date.now()),
    path,
    name: baseName(path),
    status: "working",
    stage: STAGES[0],
    data: null,
    error: null,
    view: "results",
    detailSteamid: null,
  };
  state.records.unshift(record);
  state.activeId = record.id;
  await runRecord(record);
}

async function chooseDemo() {
  if (state.busy) return;
  try {
    const selected = await open({
      multiple: false,
      filters: [{ name: "CS2 demo", extensions: ["dem", "zip"] }],
    });
    if (selected) await analyzePath(selected);
  } catch (error) {
    toast(error?.toString() || "打开文件失败");
  }
}

/* ---------- health / version ---------- */

async function health() {
  try {
    const data = await invoke("health");
    if (!data.ok) throw new Error("model not ready");
    $("#modelText").textContent = `模型 ${data.model}`;
    $("#modelDot").className = "state-dot ok";
    $("#aboutModel").textContent = data.model;
    $("#aboutThreshold").textContent = Number(data.threshold).toFixed(4);
  } catch {
    $("#modelText").textContent = "模型未就绪";
    $("#modelDot").className = "state-dot error";
    $("#aboutModel").textContent = "未加载";
    $("#aboutThreshold").textContent = "—";
  }
}

function setupAbout() {
  $("#aboutBtn").addEventListener("click", () => $("#aboutDialog").showModal());
  $("#aboutClose").addEventListener("click", () => $("#aboutDialog").close());
  $("#aboutDialog").addEventListener("click", (event) => {
    if (event.target === $("#aboutDialog")) $("#aboutDialog").close();
  });

  getVersion()
    .then((version) => {
      $("#versionText").textContent = `ValveGuard ${version}`;
      $("#aboutVersion").textContent = version;
    })
    .catch(() => {
      // The version lives in the window's corner; losing it costs nothing.
    });
}

/* ---------- results ---------- */

function renderResults(data) {
  renderSummary(data);
  renderOverviewChart(data);
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(1, number));
}

function renderOverviewChart(data) {
  const players = [...data.players].sort((a, b) => {
    const left = a.score == null ? -1 : a.score;
    const right = b.score == null ? -1 : b.score;
    return right - left || a.rank - b.rank;
  });
  const map = mapInfo(data.map);
  $("#overviewSub").textContent =
    `${map.primary} · ${players.length} 名玩家 · 按风险分从高到低；悬停查看详情，点击进入玩家页。`;

  const W = 1000;
  const rowH = 40;
  const pad = { left: 232, right: 68, top: 48, bottom: 48 };
  const innerW = W - pad.left - pad.right;
  const innerH = players.length * rowH;
  const H = pad.top + innerH + pad.bottom;
  const x = (value) => pad.left + clamp01(value) * innerW;
  const barH = 14;
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];
  const p95 = clamp01(data.p95);
  const p99 = clamp01(data.threshold);

  const bands = `
    <rect x="${x(p95)}" y="${pad.top - 12}" width="${Math.max(0, x(p99) - x(p95))}" height="${innerH + 12}" fill="rgba(245,181,68,.045)"/>
    <rect x="${x(p99)}" y="${pad.top - 12}" width="${Math.max(0, x(1) - x(p99))}" height="${innerH + 12}" fill="rgba(255,122,122,.05)"/>
  `;

  const grid = ticks.map((tick) => `
    <line class="grid" x1="${x(tick)}" x2="${x(tick)}" y1="${pad.top}" y2="${pad.top + innerH}"/>
    <text class="axis" x="${x(tick)}" y="${pad.top + innerH + 19}" text-anchor="middle">${tick.toFixed(1)}</text>
  `).join("");

  const rows = players.map((player, index) => {
    const top = pad.top + index * rowH;
    const cy = top + rowH / 2;
    const hasScore = player.score != null;
    const key = ["flag", "review", "insufficient"].includes(player.tier) ? player.tier : "below";
    const color = key === "flag" ? "#ff7a7a" : key === "review" ? "#f5b544" : key === "insufficient" ? "#5b6b84" : "#7e9ac2";
    const name = (player.name || player.steamid || "—").slice(0, 12);
    const initial = (player.name || player.steamid || "P").slice(0, 1).toUpperCase();
    const width = hasScore ? Math.max(3, x(player.score) - pad.left) : 0;
    return `
      <g class="ov-row" data-steamid="${esc(player.steamid)}" tabindex="0" role="button"
         aria-label="${esc(player.name)}，风险分 ${hasScore ? num(player.score, 3) : "无"}">
        <rect class="ov-hit" x="6" y="${top + 2}" width="${W - 12}" height="${rowH - 4}" rx="8"/>
        <text class="ov-rank" x="16" y="${cy + 4}">${String(player.rank).padStart(2, "0")}</text>
        <circle class="ov-avatar" cx="54" cy="${cy}" r="14" fill="${color}"/>
        <text class="ov-initial" x="54" y="${cy + 4}" text-anchor="middle">${esc(initial)}</text>
        <text class="ov-name" x="76" y="${cy - 1}">${esc(name)}</text>
        <text class="ov-tier ${key}" x="76" y="${cy + 12}">${esc(TIER_LABEL[player.tier] || player.tier)}</text>
        <rect class="ov-track" x="${pad.left}" y="${cy - barH / 2}" width="${innerW}" height="${barH}" rx="6"/>
        <rect class="ov-bar" x="${pad.left}" y="${cy - barH / 2}" width="${width}" height="${barH}" rx="6" fill="url(#bar-${key})"/>
        <text class="ov-value${hasScore ? "" : " muted"}" x="${pad.left + width + 9}" y="${cy + 4}">${hasScore ? num(player.score, 3) : "无窗口"}</text>
      </g>
    `;
  }).join("");

  $("#overviewChart").innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="本局玩家风险分">
      <defs>
        <linearGradient id="bar-flag" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#ff5f6d"/><stop offset="100%" stop-color="#ff9a9e"/>
        </linearGradient>
        <linearGradient id="bar-review" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#df9727"/><stop offset="100%" stop-color="#ffd27a"/>
        </linearGradient>
        <linearGradient id="bar-below" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#4b6388"/><stop offset="100%" stop-color="#8aa6cf"/>
        </linearGradient>
        <linearGradient id="bar-insufficient" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#39475c"/><stop offset="100%" stop-color="#65758e"/>
        </linearGradient>
      </defs>
      ${bands}
      ${grid}
      <line class="axis-line" x1="${pad.left}" x2="${pad.left}" y1="${pad.top - 6}" y2="${pad.top + innerH}"/>
      <g class="thr-chip p95">
        <rect x="${x(p95) - 39}" y="14" width="78" height="20" rx="10"/>
        <text x="${x(p95)}" y="28" text-anchor="middle">p95 ${num(data.p95, 3)}</text>
      </g>
      <g class="thr-chip p99">
        <rect x="${x(p99) - 39}" y="14" width="78" height="20" rx="10"/>
        <text x="${x(p99)}" y="28" text-anchor="middle">p99 ${num(data.threshold, 3)}</text>
      </g>
      <line class="thr-p95" x1="${x(p95)}" x2="${x(p95)}" y1="${pad.top - 6}" y2="${pad.top + innerH}"/>
      <line class="thr-p99" x1="${x(p99)}" x2="${x(p99)}" y1="${pad.top - 6}" y2="${pad.top + innerH}"/>
      ${rows}
      <text class="axis" x="${pad.left + innerW / 2}" y="${H - 6}" text-anchor="middle">风险分</text>
    </svg>
  `;

  const chart = $("#overviewChart");
  chart.classList.remove("focus");
  $$("#overviewChart .ov-row").forEach((row) => {
    row.addEventListener("click", () => openPlayer(row.dataset.steamid));
  });
  requestAnimationFrame(() => {
    $$("#overviewChart .ov-bar").forEach((bar, index) => {
      bar.style.transitionDelay = `${index * 24}ms`;
      bar.classList.add("in");
    });
    $$("#overviewChart .ov-value").forEach((value, index) => {
      value.style.transitionDelay = `${index * 24 + 220}ms`;
      value.classList.add("in");
    });
  });
}

let tipSteamid = null;

function playerTipHTML(player) {
  const tier = player.tier;
  const hasScore = player.score != null;
  const reasons = player.reasons && player.reasons.length
    ? `<div class="pt-reasons">${player.reasons.map((reason) => {
      const text = REASON_TEXT[reason.code] || { label: reason.code };
      return `<span class="chip">${esc(text.label)}</span>`;
    }).join("")}</div>`
    : `<div class="pt-reasons"><span class="chip none">无行为理由</span></div>`;
  return `
    <div class="pt-head">
      <span class="pt-rank">#${player.rank}</span>
      <span class="pt-name">${esc(player.name)}</span>
      <span class="badge ${esc(tier)}"><i class="dot"></i>${esc(TIER_LABEL[tier] || tier)}</span>
    </div>
    <div class="pt-score">
      <strong>${hasScore ? num(player.score, 3) : "—"}</strong>
      <span>风险分 · ${player.percentile == null ? "—" : `p${num(player.percentile, 1)}`}</span>
    </div>
    <div class="pt-facts">
      <span><b>${player.kill_count}</b> 击杀窗口</span>
      <span><b>${pct(player.behavior.headshot_rate)}</b> 爆头率</span>
      <span><b>${num(player.mean, 3)}</b> 窗口均值</span>
      <span><b>${num(player.max, 3)}</b> 窗口最高</span>
    </div>
    ${reasons}
    <div class="pt-hint">点击查看玩家详情</div>
  `;
}

function showPlayerTip(player, clientX, clientY) {
  const node = $("#playerTip");
  if (tipSteamid !== player.steamid) {
    tipSteamid = player.steamid;
    node.innerHTML = playerTipHTML(player);
  }
  node.hidden = false;
  const gap = 18;
  const width = node.offsetWidth;
  const height = node.offsetHeight;
  let left = clientX + gap;
  let top = clientY - height / 2;
  if (left + width > window.innerWidth - 10) left = clientX - width - gap;
  if (top < 10) top = 10;
  if (top + height > window.innerHeight - 10) top = window.innerHeight - height - 10;
  node.style.left = `${Math.max(10, left)}px`;
  node.style.top = `${top}px`;
}

function hidePlayerTip() {
  $("#playerTip").hidden = true;
  tipSteamid = null;
}

function openPlayer(steamid) {
  const record = activeRecord();
  if (!record || !record.data) return;
  const player = record.data.players.find((item) => item.steamid === steamid);
  if (!player) return;
  record.view = "detail";
  record.detailSteamid = steamid;
  hidePlayerTip();
  renderScreen();
  scrollScreenTop();
}

function setupChartInteractions() {
  const chart = $("#overviewChart");
  chart.addEventListener("mousemove", (event) => {
    const row = event.target.closest(".ov-row");
    if (!row) {
      hidePlayerTip();
      chart.classList.remove("focus");
      delete chart.dataset.active;
      $$("#overviewChart .ov-row").forEach((item) => item.classList.remove("active"));
      return;
    }
    const record = activeRecord();
    if (!record || !record.data) return;
    const player = record.data.players.find((item) => item.steamid === row.dataset.steamid);
    if (!player) return;
    if (chart.dataset.active !== player.steamid) {
      chart.dataset.active = player.steamid;
      chart.classList.add("focus");
      $$("#overviewChart .ov-row").forEach((item) => {
        item.classList.toggle("active", item === row);
      });
    }
    showPlayerTip(player, event.clientX, event.clientY);
  });
  chart.addEventListener("mouseleave", () => {
    hidePlayerTip();
    chart.classList.remove("focus");
    delete chart.dataset.active;
    $$("#overviewChart .ov-row").forEach((row) => row.classList.remove("active"));
  });
  chart.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const row = event.target.closest(".ov-row");
    if (!row) return;
    event.preventDefault();
    openPlayer(row.dataset.steamid);
  });
}

let killTipIndex = null;

function showKillTip(point, clientX, clientY) {
  const node = $("#killTip");
  const index = Number(point.dataset.index);
  if (killTipIndex !== index) {
    killTipIndex = index;
    const flags = [];
    if (point.dataset.hs === "1") flags.push("爆头");
    if (point.dataset.smoke === "1") flags.push("穿烟");
    if (point.dataset.wall === "1") flags.push("穿墙");
    if (point.dataset.midair === "1") flags.push("空中");
    node.innerHTML = `
      <div class="kt-top">
        <span>第 ${index + 1} 个窗口</span>
        <b class="${esc(point.dataset.key)}">${Number(point.dataset.score).toFixed(3)}</b>
      </div>
      <div class="kt-sub">tick ${esc(point.dataset.tick)} · ${esc(point.dataset.weapon)}</div>
      ${flags.length ? `<div class="kt-flags">${flags.map((flag) => `<span class="chip">${flag}</span>`).join("")}</div>` : ""}
    `;
  }
  node.hidden = false;
  const gap = 18;
  const width = node.offsetWidth;
  const height = node.offsetHeight;
  let left = clientX + gap;
  let top = clientY - height / 2;
  if (left + width > window.innerWidth - 10) left = clientX - width - gap;
  if (top < 10) top = 10;
  if (top + height > window.innerHeight - 10) top = window.innerHeight - height - 10;
  node.style.left = `${Math.max(10, left)}px`;
  node.style.top = `${top}px`;
}

function hideKillTip() {
  $("#killTip").hidden = true;
  killTipIndex = null;
  const wrap = $("#sparkWrap");
  if (!wrap) return;
  wrap.classList.remove("hovering");
  $$("#sparkWrap .kill-point.active").forEach((point) => point.classList.remove("active"));
}

function setupDetailInteractions() {
  const wrap = $("#sparkWrap");
  wrap.addEventListener("mousemove", (event) => {
    const points = $$("#sparkWrap .kill-point");
    const svg = wrap.querySelector("svg");
    if (!points.length || !svg) {
      hideKillTip();
      return;
    }
    const rect = svg.getBoundingClientRect();
    if (
      event.clientX < rect.left ||
      event.clientX > rect.right ||
      event.clientY < rect.top ||
      event.clientY > rect.bottom
    ) {
      hideKillTip();
      return;
    }

    const dots = points.map((point) => point.querySelector(".dot"));
    const firstX = Number(dots[0].getAttribute("cx"));
    const lastX = Number(dots[dots.length - 1].getAttribute("cx"));
    const vx = ((event.clientX - rect.left) / rect.width) * 900;
    const ratio = lastX === firstX ? 0 : Math.max(0, Math.min(1, (vx - firstX) / (lastX - firstX)));
    const point = points[Math.round(ratio * (points.length - 1))];
    const dot = point.querySelector(".dot");

    const line = $("#killHoverLine");
    const hoverDot = $("#killHoverDot");
    if (line) {
      line.setAttribute("x1", dot.getAttribute("cx"));
      line.setAttribute("x2", dot.getAttribute("cx"));
    }
    if (hoverDot) {
      hoverDot.setAttribute("cx", dot.getAttribute("cx"));
      hoverDot.setAttribute("cy", dot.getAttribute("cy"));
      hoverDot.setAttribute("fill", dot.getAttribute("fill"));
    }
    if (!point.classList.contains("active")) {
      $$("#sparkWrap .kill-point.active").forEach((item) => item.classList.remove("active"));
      point.classList.add("active");
      wrap.classList.add("hovering");
    }
    showKillTip(point, event.clientX, event.clientY);
  });
  wrap.addEventListener("mouseleave", hideKillTip);
}

function renderSummary(data) {
  const map = mapInfo(data.map);
  const cards = [
    {
      key: "地图",
      value: map.primary,
      sub: map.secondary ? `${map.secondary} · ${data.demo}` : data.demo,
      title: `${map.primary}${map.secondary ? ` · ${map.secondary}` : ""}（${map.code || "unknown"}）`,
    },
    { key: "玩家", value: data.summary.players, sub: `${data.summary.kills} 个击杀窗口` },
    { key: "FLAG", value: data.summary.flagged, sub: "风险分 ≥ 人类 p99", cls: "flag" },
    { key: "REVIEW", value: data.summary.review, sub: "人类 p95–p99 复核带", cls: "review" },
  ];
  $("#summaryGrid").innerHTML = cards.map((card) => `
    <div class="stat ${card.cls || ""}">
      <div class="k">${esc(card.key)}</div>
      <div class="v">${esc(card.value)}</div>
      <div class="s" title="${esc(card.title || card.sub)}">${esc(card.sub)}</div>
    </div>
  `).join("");
}

/* ---------- detail ---------- */

function evidenceRows(player, data) {
  const rows = [];
  if (player.tier === "insufficient") {
    rows.push(["判定", `${tierBadge("insufficient")}<span class="ev-note">没有有效击杀窗口</span>`]);
    rows.push(["风险分", `<b>—</b><span class="ev-note">无法评分</span>`]);
    rows.push(["窗口", `<b>0</b><span class="ev-note">没有击杀，或击杀前不足 223 tick</span>`]);
    rows.push(["说明", `<span class="ev-note">仍保留在名单中，避免 10 人显示成更少的人。</span>`]);
  } else {
    const rule = player.tier === "flag"
      ? `风险分 ≥ 人类 p99 ${num(data.threshold, 4)}`
      : player.tier === "review"
        ? `位于人类 p95 ${num(data.p95, 4)} 与 p99 ${num(data.threshold, 4)} 之间`
        : `低于人类 p95 ${num(data.p95, 4)}`;
    rows.push(["判定", `${tierBadge(player.tier)}<span class="ev-note">${rule}</span>`]);
    rows.push(["风险分", `<b>${num(player.score, 4)}</b><span class="ev-note">击杀窗口分数的收缩均值，k0 = 5，不是概率</span>`]);
    rows.push(["人类阈值", `<span class="ev-pair">p95 <b>${num(data.p95, 4)}</b></span><span class="ev-pair">p99 <b>${num(data.threshold, 4)}</b></span>`]);
    rows.push(["人类百分位", `<b>p${num(player.percentile, 1)}</b>`]);
    rows.push(["击杀窗口", `<b>${player.kill_count}</b><span class="ev-note">${player.reliable ? "证据充足（≥ 3 个窗口）" : "证据不足（< 3 个窗口）"}</span>`]);
    rows.push(["窗口分数", `<span class="ev-pair">均值 <b>${num(player.mean, 3)}</b></span><span class="ev-pair">最高 <b>${num(player.max, 3)}</b></span>`]);
    const reasons = player.reasons.length
      ? player.reasons.map((reason) => {
        const text = REASON_TEXT[reason.code] || { label: reason.code, title: reason.code };
        return `<span class="chip" title="${esc(text.title)}${reason.detail ? "：" + esc(reason.detail) : ""}">${esc(text.label)}</span>`;
      }).join("")
      : `<span class="chip none">无</span>`;
    rows.push(["行为理由", `<div class="ev-chips">${reasons}</div>`]);
  }
  return rows.map(([key, value]) => `
    <div class="ev-row">
      <span class="ev-k">${key}</span>
      <div class="ev-v">${value}</div>
    </div>
  `).join("");
}

function behaviorCards(player) {
  const b = player.behavior;
  const cards = [
    { k: "HS", v: pct(b.headshot_rate), bar: b.headshot_rate || 0, d: "爆头击杀占比" },
    { k: "Shots/K", v: num(b.shots_per_kill, 2), bar: b.shots_per_kill == null ? 0 : Math.min(1, 1.5 / Math.max(0.01, b.shots_per_kill)), d: "每击杀消耗射击数，越低越高效" },
    { k: "4s 连杀", v: b.max_multikill_4s, bar: Math.min(1, (b.max_multikill_4s || 0) / 5), d: "任意 4 秒窗口内最大击杀数" },
    { k: "Burst", v: pct(b.burst_fraction), bar: b.burst_fraction || 0, d: "与上次击杀间隔 ≤4 秒的比例" },
    { k: "击杀间隔", v: b.median_kill_gap_s == null ? "—" : `${num(b.median_kill_gap_s, 1)}s`, bar: Math.min(1, (b.median_kill_gap_s || 0) / 60), d: "击杀间隔中位数" },
    { k: "狙击占比", v: pct(b.sniper_share), bar: b.sniper_share || 0, d: "狙击类武器击杀占比" },
    { k: "穿墙", v: pct(b.wallbang_rate), bar: b.wallbang_rate || 0, d: "穿墙击杀占比" },
    { k: "穿烟", v: pct(b.smoke_rate), bar: b.smoke_rate || 0, d: "穿烟击杀占比" },
  ];
  return cards.map((card) => `
    <div class="behavior-card">
      <div class="k">${esc(card.k)}</div>
      <div class="v">${esc(card.v)}</div>
      <div class="bar"><span data-width="${Math.max(0, Math.min(100, card.bar * 100))}"></span></div>
      <div class="d">${esc(card.d)}</div>
    </div>
  `).join("");
}

function killChart(player, data) {
  const kills = player.kills;
  const W = 900;
  const H = 250;
  const pad = { left: 46, right: 20, top: 20, bottom: 38 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;
  const n = kills.length;
  const x = (index) => pad.left + (n <= 1 ? innerW / 2 : (index / (n - 1)) * innerW);
  const y = (value) => pad.top + (1 - clamp01(value)) * innerH;
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];

  const grid = ticks.map((tick) => `
    <line class="grid" x1="${pad.left}" x2="${pad.left + innerW}" y1="${y(tick)}" y2="${y(tick)}"/>
    <text class="axis" x="${pad.left - 9}" y="${y(tick) + 3.5}" text-anchor="end">${tick.toFixed(1)}</text>
  `).join("");

  const xIndexes = n <= 6
    ? kills.map((_, index) => index)
    : [...new Set([0, Math.round((n - 1) / 3), Math.round((2 * (n - 1)) / 3), n - 1])];
  const xLabels = xIndexes.map((index) => `
    <text class="axis" x="${x(index)}" y="${pad.top + innerH + 18}" text-anchor="middle">${index + 1}</text>
  `).join("");

  const points = kills.map((kill, index) => `${x(index)},${y(kill.score)}`).join(" ");
  const area = `M ${pad.left} ${y(0)} L ${points.replace(/ /g, " L ")} L ${x(n - 1)} ${y(0)} Z`;
  const bands = `
    <rect x="${pad.left}" y="${pad.top}" width="${innerW}" height="${Math.max(0, y(data.threshold) - pad.top)}" fill="rgba(255,122,122,.06)"/>
    <rect x="${pad.left}" y="${y(data.threshold)}" width="${innerW}" height="${Math.max(0, y(data.p95) - y(data.threshold))}" fill="rgba(245,181,68,.06)"/>
  `;
  const dots = kills.map((kill, index) => {
    const key = kill.score >= data.threshold ? "flag" : kill.score >= data.p95 ? "review" : "below";
    const color = key === "flag" ? "#ff7a7a" : key === "review" ? "#f5b544" : "#6b9aff";
    const weapon = WEAPON_TEXT[kill.weapon] || kill.weapon || "—";
    return `
      <g class="kill-point" data-index="${index}" data-tick="${kill.tick}"
         data-score="${kill.score}" data-key="${key}" data-weapon="${esc(weapon)}"
         data-hs="${kill.headshot ? 1 : 0}" data-smoke="${kill.smoke ? 1 : 0}"
         data-wall="${kill.wallbang ? 1 : 0}" data-midair="${kill.midair ? 1 : 0}">
        <circle class="dot" cx="${x(index)}" cy="${y(kill.score)}" r="3.8" fill="${color}"/>
        <circle class="dot-hit" cx="${x(index)}" cy="${y(kill.score)}" r="11"/>
      </g>
    `;
  }).join("");

  return `
    <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(player.name)} 的逐击杀分数">
      <defs>
        <linearGradient id="killFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#6b9aff" stop-opacity=".20"/>
          <stop offset="100%" stop-color="#6b9aff" stop-opacity="0"/>
        </linearGradient>
      </defs>
      ${grid}
      ${bands}
      <path class="area" d="${area}" fill="url(#killFill)"/>
      <line class="thr-p95" x1="${pad.left}" x2="${pad.left + innerW}" y1="${y(data.p95)}" y2="${y(data.p95)}"/>
      <line class="thr-p99" x1="${pad.left}" x2="${pad.left + innerW}" y1="${y(data.threshold)}" y2="${y(data.threshold)}"/>
      <line class="hover-line" id="killHoverLine" x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${pad.top + innerH}"/>
      <polyline class="line" points="${points}"/>
      ${dots}
      <circle class="hover-dot" id="killHoverDot" cx="${pad.left}" cy="${pad.top}" r="5.5"/>
      ${xLabels}
      <text class="axis" x="${pad.left + innerW / 2}" y="${H - 4}" text-anchor="middle">击杀序号（按时间）</text>
    </svg>
  `;
}

function renderDetail(record, player) {
  hideKillTip();
  const data = record.data;
  const map = mapInfo(data.map);
  const hasScore = player.score != null;
  const scoreText = hasScore ? num(player.score, 3) : "—";
  const pctText = player.percentile == null ? "—" : `p${num(player.percentile, 1)}`;
  const initial = (player.name || "P").slice(0, 1).toUpperCase();
  const gaugeR = 52;
  const circumference = 2 * Math.PI * gaugeR;
  const offset = circumference * (1 - (hasScore ? clamp01(player.score) : 0));

  $("#detailHero").innerHTML = `
    <div class="ps-top">
      <div class="ps-identity">
        <div class="ps-avatar ${esc(player.tier)}">${esc(initial)}</div>
        <div class="ps-name-block">
          <div class="ps-title">
            <h2>${esc(player.name)}</h2>
            ${tierBadge(player.tier)}
            <span class="badge ${esc(player.tier)}">${pctText}</span>
            <span class="badge ${player.reliable ? "" : "below"}">${player.reliable ? "证据充足" : "证据不足"}</span>
          </div>
          <div class="ps-meta">${esc(player.steamid)} · ${esc(map.primary)} · 第 ${player.rank} 位 · ${player.kill_count} 个击杀窗口</div>
        </div>
      </div>
      <div class="gauge" style="--c:${circumference}">
        <svg viewBox="0 0 132 132">
          <circle class="track" cx="66" cy="66" r="${gaugeR}"></circle>
          <circle class="value" id="gaugeArc" cx="66" cy="66" r="${gaugeR}"
            stroke-dasharray="${circumference}" stroke-dashoffset="${circumference}"></circle>
        </svg>
        <div class="center"><strong>${scoreText}</strong><span>风险分</span></div>
      </div>
    </div>
    <div class="ps-metrics">
      <div><span>风险分</span><b>${scoreText}</b></div>
      <div><span>人类百分位</span><b>${pctText}</b></div>
      <div><span>窗口均值</span><b>${num(player.mean, 3)}</b></div>
      <div><span>窗口最高</span><b>${num(player.max, 3)}</b></div>
      <div><span>爆头率</span><b>${pct(player.behavior.headshot_rate)}</b></div>
      <div><span>判定阈值</span><b>${num(data.threshold, 3)}</b></div>
    </div>
  `;
  requestAnimationFrame(() => {
    const arc = $("#gaugeArc");
    arc.style.stroke = player.tier === "flag" ? "#ff7a7a" : player.tier === "review" ? "#f5b544" : player.tier === "insufficient" ? "#64748b" : "#6b9aff";
    arc.setAttribute("stroke-dashoffset", offset);
  });

  $("#whyList").innerHTML = evidenceRows(player, data);
  $("#behaviorGrid").innerHTML = behaviorCards(player);
  requestAnimationFrame(() => {
    $$("#behaviorGrid .bar span").forEach((bar) => {
      bar.style.width = `${bar.dataset.width}%`;
    });
  });

  $("#chartStats").innerHTML = `
    <span><b>${player.kill_count}</b> 个窗口</span>
    <span><b>${num(player.mean, 3)}</b> 均值</span>
    <span><b>${num(player.max, 3)}</b> 最高</span>
  `;
  $("#sparkWrap").innerHTML = player.kills.length
    ? killChart(player, data)
    : '<p class="empty">无有效击杀窗口</p>';

  const reviewKills = player.kills.filter((kill) => kill.score >= data.p95 && kill.score < data.threshold).length;
  const highKills = player.kills.filter((kill) => kill.score >= data.threshold).length;
  $("#killStats").innerHTML = `
    <span><b>${player.kill_count}</b> 个窗口</span>
    <span><b>${reviewKills}</b> 复核带</span>
    <span><b>${highKills}</b> 高风险</span>
  `;
  $("#killSub").textContent = "每个窗口为击杀前 223 tick 与击杀后 32 tick 的瞄准序列。";
  $("#killsBody").innerHTML = player.kills.length
    ? player.kills.map((kill, index) => {
      const key = kill.score >= data.threshold ? "flag" : kill.score >= data.p95 ? "review" : "below";
      return `
        <tr>
          <td class="num">${index + 1}</td>
          <td class="num">${kill.tick}</td>
          <td class="num"><b class="score-num ${key}">${num(kill.score, 3)}</b></td>
          <td class="c">${esc(WEAPON_TEXT[kill.weapon] || kill.weapon)}</td>
          <td class="c">${kill.headshot ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td>
          <td class="c">${num(kill.shots, 1)}</td>
          <td class="c">${kill.smoke ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td>
          <td class="c">${kill.wallbang ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td>
          <td class="c">${kill.midair ? '<span class="yes">✓</span>' : '<span class="no">—</span>'}</td>
        </tr>
      `;
    }).join("")
    : '<tr><td colspan="9" class="empty-cell">无有效击杀窗口</td></tr>';
}

/* ---------- tooltips ---------- */

function setupTooltips() {
  const tooltip = $("#tooltip");
  document.addEventListener("mouseover", (event) => {
    const target = event.target.closest("[data-tip]");
    if (!target) return;
    tooltip.textContent = target.dataset.tip;
    tooltip.classList.add("show");
  });
  document.addEventListener("mousemove", (event) => {
    if (!tooltip.classList.contains("show")) return;
    const pad = 12;
    const width = tooltip.offsetWidth;
    const height = tooltip.offsetHeight;
    let left = event.clientX + 14;
    let top = event.clientY + 14;
    if (left + width > window.innerWidth - pad) left = event.clientX - width - 14;
    if (top + height > window.innerHeight - pad) top = event.clientY - height - 14;
    tooltip.style.left = `${Math.max(pad, left)}px`;
    tooltip.style.top = `${Math.max(pad, top)}px`;
  });
  document.addEventListener("mouseout", (event) => {
    const target = event.target.closest("[data-tip]");
    if (target) tooltip.classList.remove("show");
  });
  document.addEventListener("scroll", () => {
    tooltip.classList.remove("show");
    hidePlayerTip();
    hideKillTip();
  }, true);
}

/* ---------- events ---------- */

function setupEvents() {
  $("#addBtn").addEventListener("click", chooseDemo);
  $("#dropZone").addEventListener("click", chooseDemo);

  $("#recordList").addEventListener("click", (event) => {
    const row = event.target.closest(".record[data-id]");
    if (row) selectRecord(row.dataset.id);
  });

  $("#retryBtn").addEventListener("click", () => {
    const record = activeRecord();
    if (record && record.status === "failed" && !state.busy) void runRecord(record);
  });

  $("#backBtn").addEventListener("click", () => {
    const record = activeRecord();
    if (!record || !record.data) return;
    record.view = "results";
    renderScreen();
    scrollScreenTop();
  });
}

/* ---------- drag and drop ---------- */

function setDragging(on) {
  document.body.classList.toggle("dragging", on);
  $("#dropOverlay").hidden = !(on && activeRecord());
}

async function setupDragDrop() {
  window.addEventListener("dragover", (event) => {
    event.preventDefault();
    if (!state.busy) setDragging(true);
  });
  window.addEventListener("dragleave", (event) => {
    if (!event.relatedTarget) setDragging(false);
  });
  window.addEventListener("drop", (event) => {
    event.preventDefault();
    setDragging(false);
  });

  try {
    const webview = getCurrentWebview();
    await webview.onDragDropEvent((event) => {
      const payload = event.payload || {};
      if (payload.type === "enter" || payload.type === "over") {
        if (!state.busy) setDragging(true);
      } else if (payload.type === "leave") {
        setDragging(false);
      } else if (payload.type === "drop") {
        setDragging(false);
        const path = payload.paths && payload.paths[0];
        if (path) void analyzePath(path);
      }
    });
  } catch (error) {
    console.warn("drag-drop unavailable", error);
  }
}

health();
setupAbout();
setupTooltips();
setupEvents();
setupChartInteractions();
setupDetailInteractions();
setupDragDrop();
renderRecords();
renderScreen();
loadRecords();
