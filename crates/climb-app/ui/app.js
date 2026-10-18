/* 攀岩线路高亮 — Tauri 前端逻辑
   零依赖：window.__TAURI__ 全局桥（withGlobalTauri），原生 JS + Canvas/SVG 交互 */

const TAURI = window.__TAURI__;
const invoke = TAURI.core.invoke;
const listen = TAURI.event.listen;
const convertFileSrc = TAURI.core.convertFileSrc;

// ============ 全局状态 ============
const S = {
  folder: "",
  images: [],
  cur: null,                 // 当前图片绝对路径
  summaries: new Map(),      // path -> Summary
  previews: new Map(),       // path -> {orig: assetUrl, result: dataUrl}
  view: "result",            // result | original
  tool: "select",
  overlay: true,
  zoom: 1.0,
  pan: { x: 0, y: 0 },
  lassoPts: [],              // 归一化坐标
  lassoDrawing: false,
  busy: false,
  spaceHeld: false,
};

const $ = (sel) => document.querySelector(sel);
const viewer = $("#viewer");
const inner = $("#viewer-inner");
const viewImg = $("#view-img");
const overlay = $("#overlay");
const scanOverlay = $("#scan-overlay");

const TOOL_HINTS = {
  pan: "拖移视图 · 滚轮缩放 · 双击复位",
  select: "点击岩点 → 保留该颜色线路",
  lasso: "拖圈 = 漏检区域补测并并入所选线路 · 点岩点 = 直接并入所选线路",
  remove: "点击岩点 → 排除/恢复",
};

// ============ 日志 ============
function log(msg) {
  const t = ((performance.now() - S.t0) / 1000).toFixed(1);
  const panel = $("#log-panel");
  const line = document.createElement("div");
  line.className = "log-line";
  line.innerHTML = `<span class="t">[${t}s]</span>${escapeHtml(msg)}`;
  panel.appendChild(line);
  panel.scrollTop = panel.scrollHeight;
  if (panel.children.length > 300) panel.firstChild.remove();
}
S.t0 = performance.now();
function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function msg(m) { $("#st-msg").textContent = m; }

// ============ 视图布局 ============
function curSummary() { return S.cur ? S.summaries.get(S.cur) : null; }

function layout() {
  const sum = curSummary();
  const vw = viewer.clientWidth, vh = viewer.clientHeight;
  let iw, ih;
  if (sum) {
    iw = sum.aw; ih = sum.ah;
  } else if (viewImg.naturalWidth) {
    iw = viewImg.naturalWidth; ih = viewImg.naturalHeight;
  } else { iw = 1; ih = 1; }
  const fit = Math.min(vw / iw, vh / ih);
  const w = iw * fit * S.zoom, h = ih * fit * S.zoom;
  const cx = vw / 2 + S.pan.x, cy = vh / 2 + S.pan.y;
  inner.style.left = `${cx - w / 2}px`;
  inner.style.top = `${cy - h / 2}px`;
  inner.style.width = `${w}px`;
  inner.style.height = `${h}px`;
  viewImg.style.width = `${w}px`;
  viewImg.style.height = `${h}px`;
  overlay.style.width = `${w}px`;
  overlay.style.height = `${h}px`;
  if (sum) overlay.setAttribute("viewBox", `0 0 ${sum.aw} ${sum.ah}`);
  $("#st-zoom").textContent = `${Math.round(S.zoom * 100)}%`;
}

function drawOverlay() {
  const sum = curSummary();
  overlay.innerHTML = "";
  if (!sum || !S.overlay) return;
  const frag = document.createDocumentFragment();
  const svgNS = "http://www.w3.org/2000/svg";
  // 已选线路时清屏：只画所选线路的岩点（框不再杂乱）；未选时全量展示便于浏览
  const showAll = sum.selected == null;
  for (const h of sum.holds) {
    const route = sum.selected != null && h.cluster === sum.selected && !h.excluded;
    if (!showAll && h.cluster !== sum.selected) continue;
    const [x0, y0, x1, y1] = h.bbox;
    if (x1 - x0 < 2 || y1 - y0 < 2) continue;
    const cl = h.cluster != null ? sum.clusters[h.cluster] : null;
    const rx = Math.min(6, (x1 - x0) / 3);
    // 深色衬底描边（灰度结果态下浅色簇框线也能看清）
    const halo = document.createElementNS(svgNS, "rect");
    halo.setAttribute("x", x0); halo.setAttribute("y", y0);
    halo.setAttribute("width", x1 - x0); halo.setAttribute("height", y1 - y0);
    halo.setAttribute("rx", rx);
    halo.setAttribute("class", "hold-rect halo");
    frag.appendChild(halo);
    if (route) {
      const f = document.createElementNS(svgNS, "rect");
      f.setAttribute("x", x0); f.setAttribute("y", y0);
      f.setAttribute("width", x1 - x0); f.setAttribute("height", y1 - y0);
      f.setAttribute("rx", rx);
      f.setAttribute("class", "hold-rect route-fill");
      f.setAttribute("fill", cl ? rgbStr(cl.rgb) : "#888");
      frag.appendChild(f);
    }
    const r = document.createElementNS(svgNS, "rect");
    r.setAttribute("x", x0); r.setAttribute("y", y0);
    r.setAttribute("width", x1 - x0); r.setAttribute("height", y1 - y0);
    r.setAttribute("rx", rx);
    r.setAttribute("class", "hold-rect" + (route ? " route" : "") + (h.excluded ? " excluded" : ""));
    r.setAttribute("stroke", h.excluded ? "#d93636" : (route ? "#fff" : (cl ? rgbStr(cl.rgb) : "#888")));
    if (!route && !h.excluded) r.setAttribute("opacity", "0.75");
    frag.appendChild(r);
  }
  overlay.appendChild(frag);
}
function rgbStr(rgb) { return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`; }

// ============ 图像加载 ============
function showOriginal(path) {
  viewImg.src = convertFileSrc(path);
  drawOverlay();
  layout();
}
function showResult(dataUrl) {
  viewImg.src = dataUrl;
}

function setView(v) {
  S.view = v;
  document.querySelectorAll("#view-toggle .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.view === v));
  if (!S.cur) return;
  const pv = S.previews.get(S.cur);
  const useOrig = v === "original";
  if (useOrig) showOriginal(S.cur);
  else if (pv && pv.result) showResult(pv.result);
  else showOriginal(S.cur);
}

// ============ 分析流程 ============
async function analyzeImage(path, force) {
  if (S.busy) return;
  S.busy = true;
  scanOverlay.hidden = false;
  $("#scan-stage").textContent = "读取图像";
  $("#info-body").textContent = "分析中…";
  msg("分析中…");
  // 立即显示原图
  showOriginal(path);
  layout();
  try {
    const sum = await invoke(force ? "reanalyze" : "analyze", { path });
    S.summaries.set(path, sum);
    scanOverlay.hidden = true;
    // 初始预览 = 原图（未选线时全灰没有意义）
    S.previews.set(path, { orig: convertFileSrc(path), result: null });
    renderSidebars();
    drawOverlay();
    layout();
    setView("original");
    msg(`${sum.holds.length} 个岩点 · ${sum.clusters.length} 条线路 · ${sum.timing_ms.toFixed(0)} ms · 点击岩点或颜色组选线`);
  } catch (e) {
    scanOverlay.hidden = true;
    msg("分析失败: " + e);
    log("分析失败: " + e);
  } finally {
    S.busy = false;
  }
}

async function refreshPreview(path) {
  // 不重新分析，仅渲染当前选择 -> 由 set_selected/toggle_exclude 等返回 dataUrl
  const sum = curSummary();
  if (!sum) return;
  const pv = S.previews.get(path) || { orig: convertFileSrc(path), result: null };
  S.previews.set(path, pv);
}

// ============ 侧栏 ============
function renderSidebars() {
  // 图片列表
  const fl = $("#file-list");
  fl.innerHTML = "";
  $("#img-count").textContent = S.images.length ? `${S.images.length} 张` : "";
  for (const p of S.images) {
    const el = document.createElement("div");
    el.className = "list-item" + (p === S.cur ? " active" : "");
    el.textContent = p.split(/[\\/]/).pop();
    el.title = p;
    el.onclick = () => selectImage(p);
    fl.appendChild(el);
  }
    // 颜色线路
  const cl = $("#cluster-list");
  cl.innerHTML = "";
  const sum = curSummary();
  if (sum) {
    // 列头：点数 · 面积占比
    const head = document.createElement("div");
    head.className = "list-item list-head";
    head.innerHTML = `<span class="swatch" style="visibility:hidden"></span><span class="muted small">线路</span><span class="meta">点数 · 面积%</span>`;
    cl.appendChild(head);
    const none = document.createElement("div");
    none.className = "list-item" + (sum.selected == null ? " active" : "");
    none.innerHTML = `<span class="swatch" style="background:#e8eaed"></span>无（全灰输出）`;
    none.onclick = () => selectCluster(null);
    cl.appendChild(none);
    for (const c of sum.clusters) {
      const el = document.createElement("div");
      el.className = "list-item" + (sum.selected === c.id ? " active" : "");
      el.innerHTML = `<span class="swatch" style="background:${rgbStr(c.rgb)}"></span><span>${escapeHtml(c.name)}</span><span class="meta">${c.count}点 ${c.area_frac.toFixed(1)}%</span>`;
      el.onclick = () => selectCluster(c.id);
      cl.appendChild(el);
    }
  }
  // 检测信息
  if (sum) {
    $("#info-body").textContent =
      `${sum.fw}×${sum.fh}（分析 ${sum.aw}×${sum.ah}）\n岩点 ${sum.holds.length} · 颜色组 ${sum.clusters.length} · ${sum.timing_ms.toFixed(0)} ms\n墙面 V=${sum.wall_v.toFixed(2)} S=${sum.wall_s.toFixed(2)}`;
  } else {
    $("#info-body").textContent = "未分析";
  }
}

function selectImage(path) {
  if (S.cur === path) return;
  S.cur = path;
  S.zoom = 1.0;
  S.pan = { x: 0, y: 0 };
  renderSidebars();
  if (S.summaries.has(path)) {
    const pv = S.previews.get(path);
    if (pv && pv.result) showResult(pv.result); else showOriginal(path);
    drawOverlay();
    layout();
    const sum = curSummary();
    msg(sum.selected == null ? "点击岩点或颜色组选线" : "已加载");
  } else {
    analyzeImage(path, false);
  }
}

// ============ 编辑操作 ============
async function selectCluster(id) {
  if (!S.cur) return;
  try {
    const r = await invoke("set_selected", { path: S.cur, clusterId: id });
    S.summaries.set(S.cur, r.summary);
    const pv = S.previews.get(S.cur) || { orig: convertFileSrc(S.cur), result: null };
    pv.result = r.preview;
    S.previews.set(S.cur, pv);
    setView("result");
    showResult(r.preview);
    renderSidebars();
    drawOverlay();
    const c = id == null ? null : r.summary.clusters[id];
    msg(c ? `保留线路: ${c.name}（标注已保存）` : "输出为全灰");
    log(c ? `选线: ${c.name}` : "清除选择");
  } catch (e) { msg("选线失败: " + e); }
}

async function toggleExclude(holdId) {
  if (!S.cur) return;
  try {
    const r = await invoke("toggle_exclude", { path: S.cur, holdId });
    S.summaries.set(S.cur, r.summary);
    const pv = S.previews.get(S.cur);
    if (pv) { pv.result = r.preview; }
    if (S.view === "result") showResult(r.preview);
    drawOverlay();
    msg("已切换岩点排除状态（已保存）");
  } catch (e) { msg(e); }
}

async function assignHold(holdId, clusterId) {
  if (!S.cur) return;
  try {
    const r = await invoke("assign_to_cluster", { path: S.cur, holdId, clusterId });
    S.summaries.set(S.cur, r.summary);
    const pv = S.previews.get(S.cur);
    if (pv) { pv.result = r.preview; }
    if (S.view === "result") showResult(r.preview);
    renderSidebars();
    drawOverlay();
    msg(`岩点已并入 ${r.summary.clusters[clusterId].name}（已保存）`);
    log(`归组: hold ${holdId} → ${r.summary.clusters[clusterId].name}`);
  } catch (e) { msg(e); }
}

async function undo() {
  if (!S.cur) return;
  try {
    const r = await invoke("undo", { path: S.cur });
    if (r) {
      S.summaries.set(S.cur, r.summary);
      const pv = S.previews.get(S.cur);
      if (pv) pv.result = r.preview;
      if (S.view === "result") showResult(r.preview);
      renderSidebars();
      drawOverlay();
      msg("已撤销");
      log("撤销");
    } else {
      msg("没有可撤销的操作");
    }
  } catch (e) { msg(e); }
}

// ============ 套索 ============
function imgNormPoint(ev) {
  const r = viewImg.getBoundingClientRect();
  const sum = curSummary();
  const iw = sum ? sum.aw : viewImg.naturalWidth;
  const ih = sum ? sum.ah : viewImg.naturalHeight;
  const x = ((ev.clientX - r.left) / r.width) * iw;
  const y = ((ev.clientY - r.top) / r.height) * ih;
  return [x, y];
}
function normPoint(ev) {
  const r = viewImg.getBoundingClientRect();
  return [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
}

overlay.addEventListener("pointerdown", (ev) => {
  if (S.busy || !S.cur) return;
  if (S.tool === "lasso") {
    S.lassoDrawing = true;
    S.lassoPts = [normPoint(ev)];
    overlay.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  } else if (S.tool === "select" || S.tool === "remove") {
    const ip = imgNormPoint(ev);
    const hit = hitTest(ip);
    if (!hit) return;
    if (S.tool === "select") {
      const sum = curSummary();
      if (hit.cluster != null && sum) selectCluster(hit.cluster);
    } else {
      toggleExclude(hit.id);
    }
  }
});

function hitTest(ip) {
  const sum = curSummary();
  if (!sum) return null;
  let best = null;
  for (const h of sum.holds) {
    const [x0, y0, x1, y1] = h.bbox;
    if (ip[0] >= x0 && ip[0] < x1 && ip[1] >= y0 && ip[1] < y1) {
      if (!best || h.area < best.area) best = h;
    }
  }
  return best;
}

overlay.addEventListener("pointermove", (ev) => {
  if (!S.lassoDrawing) return;
  S.lassoPts.push(normPoint(ev));
  // 画笔迹
  const sum = curSummary();
  const w = sum ? sum.aw : viewImg.naturalWidth;
  const h = sum ? sum.ah : viewImg.naturalHeight;
  let path = $("#lasso-path");
  const svgNS = "http://www.w3.org/2000/svg";
  if (!path) {
    path = document.createElementNS(svgNS, "polygon");
    path.id = "lasso-path";
    path.setAttribute("class", "lasso-path");
    overlay.appendChild(path);
  }
  path.setAttribute("points", S.lassoPts.map((p) => `${p[0] * w},${p[1] * h}`).join(" "));
});

overlay.addEventListener("pointerup", async (ev) => {
  if (!S.lassoDrawing) return;
  S.lassoDrawing = false;
  const path = $("#lasso-path");
  if (path) path.remove();
  const pts = S.lassoPts;
  S.lassoPts = [];
  const sum = curSummary();
  // 圈选补测模式下的点击（未拖动）：直接归组 —— 用户已确认该岩点属于当前线路，不做颜色校验
  if (pts.length < 3) {
    if (!sum) return;
    const hit = hitTest(imgNormPoint(ev));
    if (!hit) return;
    if (sum.selected != null) {
      if (hit.cluster !== sum.selected) await assignHold(hit.id, sum.selected);
    } else if (hit.cluster != null) {
      await selectCluster(hit.cluster);
    }
    return;
  }
  try {
    const r = await invoke("lasso_add", {
      path: S.cur,
      points: pts.map((p) => [p[0], p[1]]),
      forceCluster: sum && sum.selected != null ? sum.selected : null,
    });
    S.summaries.set(S.cur, r.summary);
    const pv = S.previews.get(S.cur);
    if (pv) pv.result = r.preview;
    if (S.view === "result") showResult(r.preview);
    renderSidebars();
    drawOverlay();
    msg(`圈选补测：${r.added} 个岩点并入${sum && sum.selected != null ? "所选线路" : "检测"}（已保存）`);
    log(`圈选补测 +${r.added}`);
  } catch (e) { msg(e); }
});

// ============ 缩放/平移 ============
viewer.addEventListener("wheel", (ev) => {
  ev.preventDefault();
  const f = Math.exp(-ev.deltaY / 400);
  const rect = viewer.getBoundingClientRect();
  const cx = ev.clientX - rect.left - rect.width / 2;
  const cy = ev.clientY - rect.top - rect.height / 2;
  S.pan.x = cx - (cx - S.pan.x) * f;
  S.pan.y = cy - (cy - S.pan.y) * f;
  S.zoom = Math.min(40, Math.max(0.05, S.zoom * f));
  layout();
}, { passive: false });

let panning = false;
let panStart = null;
viewer.addEventListener("pointerdown", (ev) => {
  if (S.tool === "pan" || ev.button === 1) {
    panning = true;
    panStart = { x: ev.clientX, y: ev.clientY, px: S.pan.x, py: S.pan.y };
    viewer.setPointerCapture(ev.pointerId);
  }
});
viewer.addEventListener("pointermove", (ev) => {
  if (!panning) return;
  S.pan.x = panStart.px + (ev.clientX - panStart.x);
  S.pan.y = panStart.py + (ev.clientY - panStart.y);
  layout();
});
viewer.addEventListener("pointerup", () => { panning = false; });
viewer.addEventListener("dblclick", () => { S.zoom = 1.0; S.pan = { x: 0, y: 0 }; layout(); });

// ============ 工具栏 ============
document.querySelectorAll(".tool").forEach((b) => {
  b.onclick = () => setTool(b.dataset.tool);
});
function setTool(t) {
  S.tool = t;
  document.querySelectorAll(".tool").forEach((b) => b.classList.toggle("active", b.dataset.tool === t));
  $("#hint-bar").textContent = TOOL_HINTS[t] || "";
  viewer.style.cursor = t === "pan" ? "grab" : t === "lasso" ? "crosshair" : "default";
}

$("#btn-open").onclick = async () => {
  const f = await invoke("pick_folder");
  if (f) await openFolder(f);
};
$("#btn-open-img").onclick = async () => {
  const f = await invoke("pick_image");
  if (!f) return;
  const parent = f.replace(/[\\/][^\\/]*$/, "");
  if (parent !== S.folder) await openFolder(parent);
  selectImage(f);
};
$("#btn-fit").onclick = () => { S.zoom = 1.0; S.pan = { x: 0, y: 0 }; layout(); };
$("#btn-undo").onclick = undo;
$("#btn-export").onclick = async () => {
  if (!S.cur || S.busy) return;
  const r = await invoke("export_current", { path: S.cur });
  if (r) { msg("已导出: " + r); log("导出 " + r); }
};
$("#btn-batch").onclick = async () => {
  if (!S.folder || S.busy) return;
  msg("批量导出中…");
  $("#batch-progress").hidden = false;
  const done = await invoke("batch_export", { folder: S.folder, outDir: null });
  msg(`批量完成 ${done} 张`);
  $("#batch-progress").hidden = true;
};
$("#btn-reset").onclick = async () => {
  if (!S.cur) return;
  await invoke("reset_image", { path: S.cur });
  S.summaries.delete(S.cur);
  S.previews.delete(S.cur);
  await analyzeImage(S.cur, true);
  log("重置当前图片（含已存标注）");
};
$("#log-toggle").onclick = () => {
  const p = $("#log-panel");
  p.hidden = !p.hidden;
};
$("#overlay-toggle").onclick = () => {
  S.overlay = !S.overlay;
  $("#overlay-toggle").classList.toggle("active", S.overlay);
  drawOverlay();
};
document.querySelectorAll("#view-toggle .seg-btn").forEach((b) => {
  b.onclick = () => setView(b.dataset.view);
});

// ============ 参数 ============
function bindParam(id, key, fmt, apply, transform) {
  const el = $(id);
  el.addEventListener("change", async () => {
    const raw = parseFloat(el.value);
    $(id + "-v").textContent = fmt(raw);
    const v = transform ? transform(raw) : raw;
    await invoke("set_params", { patch: { [key]: v } });
    if (S.cur && S.summaries.has(S.cur)) {
      if (apply === "reanalyze") analyzeImage(S.cur, true);
      else {
        const dataUrl = await invoke("refresh_preview", { path: S.cur });
        const pv = S.previews.get(S.cur);
        if (pv) pv.result = dataUrl;
        if (S.view === "result") showResult(dataUrl);
      }
    }
  });
}

// ============ 键盘 ============
document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT") return;
  if (ev.code === "Space") { if (!S.spaceHeld) { S.spaceHeld = true; showOriginal(S.cur); } ev.preventDefault(); return; }
  if (ev.key === "1") setTool("pan");
  if (ev.key === "2") setTool("select");
  if (ev.key === "3") setTool("lasso");
  if (ev.key === "4") setTool("remove");
  if (ev.key === "o" || ev.key === "O") $("#overlay-toggle").click();
  if (ev.key === "f" || ev.key === "F") $("#btn-fit").click();
  if (ev.key === "e" || ev.key === "E") $("#btn-export").click();
  if (ev.key === "ArrowLeft") stepImage(-1);
  if (ev.key === "ArrowRight") stepImage(1);
  if (ev.key === "z" && (ev.ctrlKey || ev.metaKey)) undo();
});
document.addEventListener("keyup", (ev) => {
  if (ev.code === "Space") { S.spaceHeld = false; setView(S.view); }
});

function stepImage(d) {
  if (!S.images.length) return;
  const i = S.images.indexOf(S.cur);
  const n = S.images.length;
  selectImage(S.images[(((i < 0 ? 0 : i) + d) % n + n) % n]);
}

// ============ 事件监听（后端推送） ============
listen("stage", (e) => { $("#scan-stage").textContent = e.payload; });
listen("log", (e) => log(e.payload));
listen("batch", (e) => {
  const p = e.payload;
  $("#batch-text").textContent = `批量 ${p.done}/${p.total}`;
  $("#batch-fill").style.width = `${(p.done / Math.max(1, p.total)) * 100}%`;
});

// ============ 启动 ============
async function openFolder(f) {
  S.folder = f;
  $("#folder-name").textContent = f.split(/[\\/]/).pop() || f;
  S.images = await invoke("list_images", { folder: f });
  S.summaries.clear();
  S.previews.clear();
  renderSidebars();
  log(`打开文件夹: ${f}（${S.images.length} 张）`);
  if (S.images.length) selectImage(S.images[0]);
}

// 图片加载完成后按自然纵横比重排（修复载入期压扁）
viewImg.addEventListener("load", layout);

async function boot() {
  setTool("select");
  $("#overlay-toggle").classList.add("active");
  document.querySelector("#view-toggle .seg-btn[data-view='result']").classList.add("active");
  // 参数初始化
  const params = await invoke("get_params");
  $("#p-wallk").value = params.wall_k;
  $("#p-wallk-v").textContent = params.wall_k.toFixed(1);
  $("#p-minarea").value = Math.log10(params.min_area_frac);
  $("#p-minarea-v").textContent = (params.min_area_frac * 100).toFixed(3) + "%";
  $("#p-feather").value = params.feather;
  $("#p-feather-v").textContent = params.feather.toFixed(1);
  $("#p-bgsat").value = params.bg_sat;
  $("#p-bgsat-v").textContent = params.bg_sat.toFixed(2);
  bindParam("#p-wallk", "wall_k", (v) => v.toFixed(1), "reanalyze");
  bindParam("#p-minarea", "min_area_frac", (v) => (Math.pow(10, v) * 100).toFixed(3) + "%", "reanalyze", (v) => Math.pow(10, v));
  bindParam("#p-feather", "feather", (v) => v.toFixed(1), "render");
  bindParam("#p-bgsat", "bg_sat", (v) => v.toFixed(2), "render");
  const info = await invoke("get_state");
  $("#model-status").textContent = `${info.holds_status} · ${info.sam_status}`;
  await openFolder(info.cwd);
  window.addEventListener("resize", layout);
  msg("打开文件夹或拖入图片 · 点击岩点选线 · 3=圈选补测");
}

boot();
