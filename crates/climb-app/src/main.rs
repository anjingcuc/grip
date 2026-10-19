//! 攀岩线路高亮 — Tauri 后端
//! 职责：文件枚举 / 原生对话框 / 分析调度（异步+阶段事件）/ 预览渲染 / 编辑操作 / 导出

// 发布版不弹控制台黑框（Windows 子系统切换）
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use climb_core as pipeline;
use climb_core::session::{self, SessionState};
use climb_core::{cluster, render, Analysis, Cluster, Hold, HoldKind, Params};
use image::RgbaImage;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_dialog::DialogExt;

// ================= 状态 =================

#[derive(Clone)]
struct Snap {
    sel: Option<usize>,
    excluded: HashSet<usize>,
    holds: Vec<Hold>,
    clusters: Vec<Cluster>,
}

struct ImgEntry {
    full: Arc<RgbaImage>,
    a: Analysis,
    sel: Option<usize>,
    excluded: HashSet<usize>,
    undo: Vec<Snap>,
}

struct AppState {
    params: Mutex<Params>,
    images: Mutex<HashMap<PathBuf, ImgEntry>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            params: Mutex::new(Params::default()),
            images: Mutex::new(HashMap::new()),
        }
    }
}

fn push_undo(e: &mut ImgEntry) {
    e.undo.push(Snap {
        sel: e.sel,
        excluded: e.excluded.clone(),
        holds: e.a.holds.clone(),
        clusters: e.a.clusters.clone(),
    });
    if e.undo.len() > 50 {
        e.undo.remove(0);
    }
}

// ================= 序列化摘要 =================

#[derive(Serialize)]
struct SummaryCluster {
    id: usize,
    name: String,
    count: usize,
    area_frac: f64,
    rgb: [u8; 3],
    hue: f32,
}

#[derive(Serialize)]
struct SummaryHold {
    id: usize,
    cluster: Option<usize>,
    bbox: [u32; 4],
    area: u32,
    lasso: bool,
    excluded: bool,
}

#[derive(Serialize)]
struct Summary {
    path: String,
    aw: u32,
    ah: u32,
    fw: u32,
    fh: u32,
    wall_v: f32,
    wall_s: f32,
    timing_ms: f32,
    holds: Vec<SummaryHold>,
    clusters: Vec<SummaryCluster>,
    selected: Option<usize>,
}

#[derive(Serialize)]
struct AppInfo {
    cwd: String,
    holds_status: String,
    sam_status: String,
}

#[derive(Serialize)]
struct LassoResult {
    added: usize,
    preview: String,
    summary: Summary,
}

/// 编辑类操作的统一返回
#[derive(Serialize)]
struct EditResult {
    preview: String,
    summary: Summary,
}

/// 把当前编辑状态落盘到 <图片目录>/sessions/<文件名>/
fn auto_save(e: &ImgEntry, pb: &Path, params: &Params) {
    let st = SessionState {
        image: pb.display().to_string(),
        fw: e.a.fw,
        fh: e.a.fh,
        aw: e.a.aw,
        ah: e.a.ah,
        wall_v: e.a.wall_v,
        wall_s: e.a.wall_s,
        timing_ms: e.a.timing_ms,
        params: params.clone(),
        selected: e.sel,
        excluded: e.excluded.iter().copied().collect(),
        holds: e.a.holds.clone(),
        clusters: e.a.clusters.clone(),
    };
    let _ = session::save(&session::session_dir_for(pb), &st, &e.a.img);
}

// ================= 工具 =================

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp")
    )
}

fn scan_dir(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|e| e.is_file() && is_image(e))
        .collect();
    v.sort();
    v
}

fn load_dynamic(path: &Path) -> anyhow::Result<image::DynamicImage> {
    Ok(image::ImageReader::open(path)?.with_guessed_format()?.decode()?)
}

fn build_summary(path: &Path, e: &ImgEntry) -> Summary {
    let a = &e.a;
    let total = a.holds.iter().map(|h| h.area as u64).sum::<u64>().max(1) as f64;
    // 同名颜色组消歧（橙×3 这类）
    let names = cluster::disambiguated_names(&a.clusters);
    Summary {
        path: path.display().to_string(),
        aw: a.aw,
        ah: a.ah,
        fw: a.fw,
        fh: a.fh,
        wall_v: a.wall_v,
        wall_s: a.wall_s,
        timing_ms: a.timing_ms,
        holds: a
            .holds
            .iter()
            .map(|h| SummaryHold {
                id: h.id,
                cluster: if h.cluster == usize::MAX { None } else { Some(h.cluster) },
                bbox: h.bbox,
                area: h.area,
                lasso: h.lasso,
                excluded: e.excluded.contains(&h.id),
            })
            .collect(),
        clusters: a
            .clusters
            .iter()
            .enumerate()
            .map(|(ci, c)| SummaryCluster {
                id: ci,
                name: names[ci].clone(),
                count: c.holds.len(),
                area_frac: c.area as f64 / total * 100.0,
                rgb: c.rgb,
                hue: c.hue,
            })
            .collect(),
        selected: e.sel,
    }
}

fn render_preview_url(e: &ImgEntry, params: &Params) -> Result<String, String> {
    let mask = render::keep_mask(&e.a, e.sel, &e.excluded, params.mask_dilate);
    let mask = render::feather(&e.a, &mask, params.feather);
    let fill = render::fill_layer(&e.a, &e.excluded, params.bg_fill);
    let img = render::composite_preview(&e.a, &mask, params.bg_sat, fill.as_ref());
    let mut buf = std::io::Cursor::new(Vec::new());
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 84);
    enc.encode_image(&image::DynamicImage::ImageRgba8(img)).map_err(|e| e.to_string())?;
    Ok(format!("data:image/jpeg;base64,{}", base64_encode(&buf.into_inner())))
}

fn base64_encode(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

/// 偶奇规则光栅化（分析分辨率掩码）
fn raster_poly(pts: &[[f32; 2]], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if pts.len() < 3 {
        return out;
    }
    let minx = pts.iter().map(|p| p[0]).fold(f32::MAX, f32::min).floor().clamp(0.0, w as f32) as usize;
    let miny = pts.iter().map(|p| p[1]).fold(f32::MAX, f32::min).floor().clamp(0.0, h as f32) as usize;
    let maxx = (pts.iter().map(|p| p[0]).fold(f32::MIN, f32::max).ceil().clamp(0.0, w as f32)) as usize;
    let maxy = (pts.iter().map(|p| p[1]).fold(f32::MIN, f32::max).ceil().clamp(0.0, h as f32)) as usize;
    let n = pts.len();
    for y in miny..maxy.min(h) {
        for x in minx..maxx.min(w) {
            let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
            let mut inside = false;
            let mut j = n - 1;
            for i2 in 0..n {
                let (xi, yi) = (pts[i2][0], pts[i2][1]);
                let (xj, yj) = (pts[j][0], pts[j][1]);
                if (yi > py) != (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi {
                    inside = !inside;
                }
                j = i2;
            }
            if inside {
                out[y * w + x] = 255;
            }
        }
    }
    out
}

fn emit_log(app: &AppHandle, msg: impl Into<String>) {
    let _ = app.emit("log", msg.into());
}

// ================= Tauri 命令 =================

#[tauri::command]
fn get_state() -> AppInfo {
    AppInfo {
        cwd: std::env::current_dir().map(|p| p.display().to_string()).unwrap_or_default(),
        holds_status: pipeline::detector::YoloDetect::holds_status().into(),
        sam_status: pipeline::sam::EdgeSam::status().into(),
    }
}

#[tauri::command]
fn list_images(folder: String) -> Vec<String> {
    scan_dir(Path::new(&folder)).iter().map(|p| p.display().to_string()).collect()
}

#[tauri::command]
async fn pick_folder(app: AppHandle) -> Result<Option<String>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog().file().pick_folder(move |p| {
        let _ = tx.send(p);
    });
    let p = rx.recv().map_err(|e| e.to_string())?;
    Ok(p.map(|fp| fp.to_string()))
}

/// 单张图片选择（图片过滤器）
#[tauri::command]
async fn pick_image(app: AppHandle) -> Result<Option<String>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog()
        .file()
        .add_filter("图片", &["png", "jpg", "jpeg", "webp", "bmp"])
        .pick_file(move |p| {
            let _ = tx.send(p);
        });
    let p = rx.recv().map_err(|e| e.to_string())?;
    Ok(p.map(|fp| fp.to_string()))
}

/// 重置某张图：删磁盘会话 + 清内存缓存（下次 analyze 全新跑）
#[tauri::command]
fn reset_image(state: State<'_, AppState>, path: String) -> Result<(), String> {
    let pb = PathBuf::from(&path);
    let dir = session::session_dir_for(&pb);
    if dir.exists() {
        let _ = std::fs::remove_dir_all(&dir);
    }
    state.images.lock().map_err(|_| "lock")?.remove(&pb);
    Ok(())
}

#[tauri::command]
async fn pick_save_file(app: AppHandle, default_name: String) -> Result<Option<String>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    app.dialog().file().set_file_name(default_name).save_file(move |p| {
        let _ = tx.send(p);
    });
    let p = rx.recv().map_err(|e| e.to_string())?;
    Ok(p.map(|fp| fp.to_string()))
}

/// 分析（异步，期间持续 emit "stage" / "log" 事件）
/// 优先级：内存缓存 > 磁盘会话（用户标注）> 全新分析（完后落盘基线会话）
#[tauri::command]
async fn analyze(app: AppHandle, state: State<'_, AppState>, path: String) -> Result<Summary, String> {
    let pb = PathBuf::from(&path);
    {
        let imgs = state.images.lock().map_err(|_| "state lock")?;
        if let Some(e) = imgs.get(&pb) {
            return Ok(build_summary(&pb, e));
        }
    }
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let apph = app.clone();
    let pb2 = pb.clone();
    let (st, a, full) = tauri::async_runtime::spawn_blocking(move || -> Result<(SessionState, Analysis, RgbaImage), String> {
        let dyn_img = load_dynamic(&pb2).map_err(|e| e.to_string())?;
        let full = dyn_img.to_rgba8();
        // 磁盘会话：恢复用户标注
        let sd = pipeline::session::session_dir_for(&pb2);
        if sd.join("state.json").is_file() {
            if let Ok((st, a)) = pipeline::session::load(&sd) {
                emit_log(&apph, "已恢复上次标注");
                return Ok((st, a, full));
            }
        }
        let mut a = pipeline::analyze_with_progress(&dyn_img, &params, &|s| {
            let _ = apph.emit("stage", s.to_string());
            emit_log(&apph, format!("· {s}"));
        });
        pipeline::session::materialize_labels(&mut a);
        let st = SessionState {
            image: pb2.display().to_string(),
            fw: a.fw,
            fh: a.fh,
            aw: a.aw,
            ah: a.ah,
            wall_v: a.wall_v,
            wall_s: a.wall_s,
            timing_ms: a.timing_ms,
            params: params.clone(),
            selected: None,
            excluded: Vec::new(),
            holds: a.holds.clone(),
            clusters: a.clusters.clone(),
        };
        let _ = pipeline::session::save(&sd, &st, &a.img);
        Ok((st, a, full))
    })
    .await
    .map_err(|e| e.to_string())??;

    let entry = ImgEntry {
        full: Arc::new(full),
        a,
        sel: st.selected,
        excluded: st.excluded.into_iter().collect(),
        undo: Vec::new(),
    };
    let mut imgs = state.images.lock().map_err(|_| "lock")?;
    imgs.insert(pb.clone(), entry);
    let e = imgs.get(&pb).unwrap();
    emit_log(&app, format!("分析完成: {} 个岩点 · {} 条颜色线路 · {:.0} ms", e.a.holds.len(), e.a.clusters.len(), e.a.timing_ms));
    Ok(build_summary(&pb, e))
}

/// 参数变化后重分析
#[tauri::command]
async fn reanalyze(app: AppHandle, state: State<'_, AppState>, path: String) -> Result<Summary, String> {
    let pb = PathBuf::from(&path);
    {
        let mut imgs = state.images.lock().map_err(|_| "lock")?;
        imgs.remove(&pb);
    }
    analyze(app, state, path).await
}

/// 设置保留线路（None = 全灰）
#[tauri::command]
fn set_selected(state: State<'_, AppState>, path: String, cluster_id: Option<i64>) -> Result<EditResult, String> {
    let pb = PathBuf::from(&path);
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let mut imgs = state.images.lock().map_err(|_| "lock")?;
    let e = imgs.get_mut(&pb).ok_or("未分析")?;
    push_undo(e);
    e.sel = match cluster_id {
        Some(i) if i >= 0 && (i as usize) < e.a.clusters.len() => Some(i as usize),
        Some(i) => return Err(format!("颜色组 {i} 越界")),
        None => None,
    };
    auto_save(e, &pb, &params);
    let preview = render_preview_url(e, &params)?;
    let summary = build_summary(&pb, e);
    Ok(EditResult { preview, summary })
}

/// 排除/恢复某个岩点
#[tauri::command]
fn toggle_exclude(state: State<'_, AppState>, path: String, hold_id: usize) -> Result<EditResult, String> {
    let pb = PathBuf::from(&path);
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let mut imgs = state.images.lock().map_err(|_| "lock")?;
    let e = imgs.get_mut(&pb).ok_or("未分析")?;
    if !e.a.holds.iter().any(|h| h.id == hold_id) {
        return Err(format!("岩点 {hold_id} 不存在"));
    }
    push_undo(e);
    if e.excluded.contains(&hold_id) {
        e.excluded.remove(&hold_id);
    } else {
        e.excluded.insert(hold_id);
    }
    auto_save(e, &pb, &params);
    let preview = render_preview_url(e, &params)?;
    let summary = build_summary(&pb, e);
    Ok(EditResult { preview, summary })
}

/// 撤销
#[tauri::command]
fn undo(state: State<'_, AppState>, path: String) -> Result<Option<EditResult>, String> {
    let pb = PathBuf::from(&path);
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let mut imgs = state.images.lock().map_err(|_| "lock")?;
    let e = imgs.get_mut(&pb).ok_or("未分析")?;
    match e.undo.pop() {
        Some(s) => {
            e.sel = s.sel;
            e.excluded = s.excluded;
            e.a.holds = s.holds;
            e.a.clusters = s.clusters;
            auto_save(e, &pb, &params);
            let preview = render_preview_url(e, &params)?;
            let summary = build_summary(&pb, e);
            Ok(Some(EditResult { preview, summary }))
        }
        None => Ok(None),
    }
}

/// 把岩点直接并入指定线路（不做颜色校验——用户已确认归属）
#[tauri::command]
fn assign_to_cluster(state: State<'_, AppState>, path: String, hold_id: usize, cluster_id: usize) -> Result<EditResult, String> {
    let pb = PathBuf::from(&path);
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let mut imgs = state.images.lock().map_err(|_| "lock")?;
    let e = imgs.get_mut(&pb).ok_or("未分析")?;
    if cluster_id >= e.a.clusters.len() {
        return Err(format!("颜色组 {cluster_id} 越界"));
    }
    let hp = e.a.holds.iter().position(|h| h.id == hold_id).ok_or_else(|| format!("岩点 {hold_id} 不存在"))?;
    let old = e.a.holds[hp].cluster;
    if old == cluster_id {
        let preview = render_preview_url(e, &params)?;
        let summary = build_summary(&pb, e);
        return Ok(EditResult { preview, summary });
    }
    push_undo(e);
    let area = e.a.holds[hp].area;
    if old != usize::MAX && old < e.a.clusters.len() {
        e.a.clusters[old].holds.retain(|&x| x != hold_id);
        e.a.clusters[old].area = e.a.clusters[old].area.saturating_sub(area);
    }
    e.a.clusters[cluster_id].holds.push(hold_id);
    e.a.clusters[cluster_id].area += area;
    e.a.holds[hp].cluster = cluster_id;
    auto_save(e, &pb, &params);
    let preview = render_preview_url(e, &params)?;
    let summary = build_summary(&pb, e);
    Ok(EditResult { preview, summary })
}

/// 套索补测：归一化坐标点列 -> 新增岩点数 + 更新摘要 + 刷新预览
/// force_cluster：圈住区域里的一切岩点（新检出+已有）直接并入该线路，不做颜色校验
#[tauri::command]
fn lasso_add(state: State<'_, AppState>, path: String, points: Vec<(f32, f32)>, force_cluster: Option<i64>) -> Result<LassoResult, String> {
    let pb = PathBuf::from(&path);
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let mut imgs = state.images.lock().map_err(|_| "lock")?;
    let e = imgs.get_mut(&pb).ok_or("未分析")?;
    if points.len() < 3 {
        return Ok(LassoResult {
            added: 0,
            preview: render_preview_url(e, &params)?,
            summary: build_summary(&pb, e),
        });
    }
    let force = force_cluster.filter(|&fc| fc >= 0 && (fc as usize) < e.a.clusters.len()).map(|fc| fc as usize);
    let (w, h) = (e.a.aw as usize, e.a.ah as usize);
    let pts: Vec<[f32; 2]> = points.iter().map(|&(x, y)| [x * e.a.aw as f32, y * e.a.ah as f32]).collect();
    let region = raster_poly(&pts, w, h);
    let found = pipeline::detect_in_region(&e.a, &params, &region);
    push_undo(e);
    let mut added = 0usize;
    for mut nh in found {
        let r = (nh.area as f32 / std::f32::consts::PI).sqrt();
        let dup = e.a.holds.iter().any(|o| {
            let ro = (o.area as f32 / std::f32::consts::PI).sqrt();
            let dx = o.centroid[0] - nh.centroid[0];
            let dy = o.centroid[1] - nh.centroid[1];
            (dx * dx + dy * dy).sqrt() < (r + ro) * 0.75
        });
        if dup {
            continue;
        }
        nh.id = e.a.holds.len();
        let ci = match force {
            // 用户已确认归属：直接并入所选线路
            Some(fc) => {
                e.a.clusters[fc].holds.push(nh.id);
                e.a.clusters[fc].area += nh.area;
                fc
            }
            None => {
                let nh_ref = nh.clone();
                cluster::assign_hold(&mut e.a.clusters, &nh_ref, params.color_tol_ab, nh.id)
            }
        };
        nh.cluster = ci;
        e.a.holds.push(nh);
        added += 1;
    }
    // 已有岩点被圈进区域：直接改归所选线路
    let mut moved = 0usize;
    if let Some(fc) = force {
        let ids: Vec<usize> = e
            .a
            .holds
            .iter()
            .filter(|hd| {
                hd.cluster != fc
                    && hd.cluster != usize::MAX
                    && !e.excluded.contains(&hd.id)
                    && {
                        let (x, y) = (hd.centroid[0] as usize, hd.centroid[1] as usize);
                        x < w && y < h && region[y * w + x] != 0
                    }
            })
            .map(|hd| hd.id)
            .collect();
        for id in ids {
            let hp = e.a.holds.iter().position(|h| h.id == id).unwrap();
            let old = e.a.holds[hp].cluster;
            let area = e.a.holds[hp].area;
            if old < e.a.clusters.len() {
                e.a.clusters[old].holds.retain(|&x| x != id);
                e.a.clusters[old].area = e.a.clusters[old].area.saturating_sub(area);
            }
            e.a.clusters[fc].holds.push(id);
            e.a.clusters[fc].area += area;
            e.a.holds[hp].cluster = fc;
            moved += 1;
        }
    }
    auto_save(e, &pb, &params);
    Ok(LassoResult {
        added: added + moved,
        preview: render_preview_url(e, &params)?,
        summary: build_summary(&pb, e),
    })
}

/// 导出当前图（原生保存对话框 -> 全分辨率 PNG）
#[tauri::command]
async fn export_current(app: AppHandle, state: State<'_, AppState>, path: String) -> Result<Option<String>, String> {
    let pb = PathBuf::from(&path);
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let stem = pb.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let Some(target) = pick_save_file(app, format!("{stem}_route.png")).await? else {
        return Ok(None);
    };
    let imgs = state.images.lock().map_err(|_| "lock")?;
    let e = imgs.get(&pb).ok_or("未分析")?;
    let mask = render::keep_mask(&e.a, e.sel, &e.excluded, params.mask_dilate);
    let mask = render::feather(&e.a, &mask, params.feather);
    let fill = render::fill_layer(&e.a, &e.excluded, params.bg_fill);
    let img = render::composite_full(&e.full, &e.a, &mask, params.bg_sat, fill.as_ref());
    img.save(&target).map_err(|e2| e2.to_string())?;
    Ok(Some(target))
}

/// 批量导出（每张 emit "batch" 进度事件）
#[tauri::command]
async fn batch_export(app: AppHandle, state: State<'_, AppState>, folder: String, out_dir: Option<String>) -> Result<usize, String> {
    let dir = PathBuf::from(&folder);
    let files = scan_dir(&dir);
    let out = out_dir.map(PathBuf::from).unwrap_or_else(|| dir.join("output"));
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let total = files.len();
    let apph = app.clone();
    let done = tauri::async_runtime::spawn_blocking(move || -> Result<usize, String> {
        let mut done = 0usize;
        for (i, p) in files.iter().enumerate() {
            let _ = apph.emit("batch", serde_json::json!({ "done": done, "total": total, "current": p.display().to_string() }));
            match load_dynamic(p) {
                Ok(dyn_img) => {
                    let a = pipeline::analyze(&dyn_img, &params);
                    let sel = a
                        .clusters
                        .iter()
                        .enumerate()
                        .filter(|(_, c)| c.kind == HoldKind::Colored)
                        .max_by_key(|(_, c)| c.area)
                        .map(|(ci, _)| ci);
                    let mask = render::keep_mask(&a, sel, &Default::default(), params.mask_dilate);
                    let mask = render::feather(&a, &mask, params.feather);
                    let fill = render::fill_layer(&a, &Default::default(), params.bg_fill);
                    let img = render::composite_full(&dyn_img.to_rgba8(), &a, &mask, params.bg_sat, fill.as_ref());
                    let stem = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    if img.save(out.join(format!("{stem}_route.png"))).is_ok() {
                        done += 1;
                    }
                }
                Err(_) => {}
            }
            let _ = apph.emit("log", format!("批量 {}/{}", i + 1, total));
        }
        Ok(done)
    })
    .await
    .map_err(|e| e.to_string())??;
    Ok(done)
}

/// 用当前选择重渲预览（不动 undo 栈）
#[tauri::command]
fn refresh_preview(state: State<'_, AppState>, path: String) -> Result<String, String> {
    let pb = PathBuf::from(&path);
    let params = state.params.lock().map_err(|_| "lock")?.clone();
    let imgs = state.images.lock().map_err(|_| "lock")?;
    let e = imgs.get(&pb).ok_or("未分析")?;
    render_preview_url(e, &params)
}

#[tauri::command]
fn get_params(state: State<'_, AppState>) -> Result<Params, String> {
    Ok(state.params.lock().map_err(|_| "lock")?.clone())
}

#[tauri::command]
fn set_params(state: State<'_, AppState>, patch: serde_json::Value) -> Result<(), String> {
    let mut p = state.params.lock().map_err(|_| "lock")?;
    if let Some(v) = patch.get("wall_k").and_then(|v| v.as_f64()) {
        p.wall_k = v as f32;
    }
    if let Some(v) = patch.get("min_area_frac").and_then(|v| v.as_f64()) {
        p.min_area_frac = v;
    }
    if let Some(v) = patch.get("feather").and_then(|v| v.as_f64()) {
        p.feather = v as f32;
    }
    if let Some(v) = patch.get("bg_sat").and_then(|v| v.as_f64()) {
        p.bg_sat = v as f32;
    }
    Ok(())
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(AppState::new())
        .invoke_handler(tauri::generate_handler![
            get_state,
            list_images,
            pick_folder,
            pick_image,
            reset_image,
            pick_save_file,
            analyze,
            reanalyze,
            set_selected,
            toggle_exclude,
            undo,
            assign_to_cluster,
            lasso_add,
            export_current,
            batch_export,
            refresh_preview,
            get_params,
            set_params,
        ])
        .build(tauri::generate_context!())
        .expect("构建 GRIP 应用失败");
    // 关窗即退出：事件循环请求退出时立即结束整个进程，
    // 连带终止后台分析线程与 WebView2 子进程，杜绝残留。
    // 安全性：所有用户标注在每次编辑时已即时落盘，强退不丢数据。
    app.run(|_app, event| {
        if let tauri::RunEvent::ExitRequested { code, .. } = event {
            std::process::exit(code.unwrap_or(0));
        }
    });
}
