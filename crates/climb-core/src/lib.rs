//! 核心管线：YOLO 岩点检测 -> EdgeSAM 框提示掩码 -> 颜色确认 -> 线路聚类
//!
//! 架构（依据调研报告 §4.3 "检测器自提示 + 轻量 SAM"）：
//!   1. ChalkGPT YOLO11s 检测岩点框（语义检测，免疫镁粉/杂色/阴影干扰）
//!   2. EdgeSAM 以 YOLO 框为提示输出像素级掩码（encoder 一次 + decoder 每框毫秒级）
//!      —— SAM 不可用/小框时回退：框内墙面 Lab 差分；再兜底内切椭圆
//!   3. 颜色确认：掩码内饱和像素众数色相（低饱和的镁粉/胶痕自动忽略）
//!   4. 无模型时回退：整图墙面差分（分块平坦场校正）
//!   5. 色相聚类分线；渲染时整点掩码（含镁粉部分）保色，其余灰度
//!   注：无人物检测、无自动选线 —— 线路由用户在 GUI 点击选择 / CLI --hue 指定

pub mod cluster;
pub mod conn;
pub mod detector;
pub mod hsv;
pub mod morph;
pub mod render;
pub mod sam;
pub mod session;

use image::{DynamicImage, RgbaImage};
use std::path::PathBuf;
use std::time::Instant;

pub use conn::CompStats;

/// 按名字查找模型文件：cwd 及 exe 所在目录的所有上级，逐层找 models/<name> 与 <name>。
/// 覆盖三种运行方式：项目根直接跑（cargo/CLI）、双击 target/release 下的 exe（向上两级到项目根）、
/// exe 与 models/ 同目录分发。
pub(crate) fn find_model(name: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = vec![PathBuf::from(".")];
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut cur = exe.parent().map(|p| p.to_path_buf());
        while let Some(p) = cur {
            roots.push(p.clone());
            cur = p.parent().map(|q| q.to_path_buf());
        }
    }
    for r in &roots {
        let c = r.join("models").join(name);
        if c.is_file() {
            return Some(c);
        }
        let c2 = r.join(name);
        if c2.is_file() {
            return Some(c2);
        }
    }
    None
}

/// 可调参数（GUI 滑杆 / CLI 均可覆盖）
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Params {
    /// 分析分辨率上限（长边）
    pub analysis_max_dim: u32,
    /// 墙面差分灵敏度倍数（阈值 = k * 局部σ；越小检出越多）
    pub wall_k: f32,
    /// 岩点最小面积（占分析图比例）
    pub min_area_frac: f64,
    /// 非彩色岩点（黑/白/灰）最大面积
    pub max_area_frac: f64,
    /// 彩色岩点/体积块最大面积
    pub max_colored_area_frac: f64,
    /// bbox 填充率下限（剔除细长/异形）
    pub min_fill_ratio: f32,
    /// 确认为彩色岩点所需的轮廓内彩色像素比例
    pub min_colored_frac: f32,
    /// 颜色聚类容差（Lab 色度平面 (a*,b*) 欧氏距离，平均链接凝聚聚类）
    pub color_tol_ab: f32,
    /// 保留掩码膨胀半径（分析分辨率 px）
    pub mask_dilate: usize,
    /// 掩码羽化半径（分析分辨率 px）
    pub feather: f32,
    /// 背景保留的饱和度（0 = 完全灰度）
    pub bg_sat: f32,
    /// 墙面净化强度（0 = 关闭；>0 时非岩点区域混向墙面场亮部参考色，先于灰度化）
    /// serde(default)：兼容新增该字段前的老会话 state.json
    #[serde(default)]
    pub bg_fill: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            analysis_max_dim: 1600,
            wall_k: 5.5,
            min_area_frac: 1.2e-4,
            max_area_frac: 0.025,
            max_colored_area_frac: 0.07,
            min_fill_ratio: 0.30,
            min_colored_frac: 0.15,
            color_tol_ab: 14.0,
            mask_dilate: 2,
            feather: 2.5,
            bg_sat: 0.0,
            bg_fill: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum HoldKind {
    Colored,
    Black,
    White,
    Gray,
}

/// 一个岩点（实例掩码 + 颜色属性）
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Hold {
    pub id: usize,
    /// 引用 Analysis.labels 的分量标签（回退路径）；0 表示自包含像素
    pub label: u32,
    pub own_pixels: Vec<u32>,
    pub area: u32,
    /// bbox x0,y0,x1,y1（x1/y1 开区间，分析坐标）
    pub bbox: [u32; 4],
    pub centroid: [f32; 2],
    /// 众数色相（度，用于显示命名）
    pub hue: f32,
    /// 彩色像素的 Lab 色度均值 (a*, b*)（聚类/渲染约束用）
    pub ab: [f32; 2],
    pub sat: f32,
    pub val: f32,
    pub purity: f32,
    pub kind: HoldKind,
    /// 所属颜色组索引；usize::MAX = 未分组
    pub cluster: usize,
    pub lasso: bool,
}

/// 一条颜色线路（岩点集合）
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Cluster {
    pub kind: HoldKind,
    pub hue: f32,
    /// 面积加权 Lab 色度中心 (a*, b*)
    pub ab: [f32; 2],
    pub rgb: [u8; 3],
    pub holds: Vec<usize>,
    pub area: u32,
}

/// 一次完整分析的结果
pub struct Analysis {
    pub aw: u32,
    pub ah: u32,
    pub fw: u32,
    pub fh: u32,
    /// 回退路径的分量标签图；YOLO 主路为空（岩点用 own_pixels）
    pub labels: Vec<u32>,
    pub holds: Vec<Hold>,
    pub clusters: Vec<Cluster>,
    pub img: RgbaImage,
    /// 分析分辨率 HSV（渲染时做逐像素颜色约束，防串色）
    pub hue: Vec<f32>,
    pub sat: Vec<f32>,
    pub val: Vec<f32>,
    pub wall_v: f32,
    pub wall_s: f32,
    pub timing_ms: f32,
}

// ---------------- 墙面平坦场模型（回退路径 / 框内差分共用） ----------------

const TILE: usize = 160;

struct WallField {
    tw: usize,
    th: usize,
    l: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    sigma: Vec<f32>,
}

fn median3(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|x, y| x.partial_cmp(y).unwrap());
    v[v.len() / 2]
}

/// 分块估计墙面颜色与局部噪声 σ，并 3×3 平滑
fn build_wall_field(lab: &[[f32; 3]], sat: &[f32], w: usize, h: usize) -> WallField {
    let tw = w.div_ceil(TILE);
    let th = h.div_ceil(TILE);
    let nt = tw * th;
    let mut fl = vec![0f32; nt];
    let mut fa = vec![0f32; nt];
    let mut fb = vec![0f32; nt];
    let mut fsig = vec![0f32; nt];

    let mut gl: Vec<f32> = Vec::new();
    let mut ga: Vec<f32> = Vec::new();
    let mut gb: Vec<f32> = Vec::new();
    for i in (0..lab.len()).step_by(3) {
        if sat[i] < 0.20 {
            gl.push(lab[i][0]);
            ga.push(lab[i][1]);
            gb.push(lab[i][2]);
        }
    }
    let (gml, gma, gmb) = (median3(&mut gl), median3(&mut ga), median3(&mut gb));
    let mut gd: Vec<f32> = gl
        .iter()
        .zip(&ga)
        .zip(&gb)
        .map(|((&l, &a), &b)| ((l - gml).powi(2) + (a - gma).powi(2) + (b - gmb).powi(2)).sqrt())
        .collect();
    let gsig = (1.4826 * median3(&mut gd)).clamp(2.5, 14.0);

    let mut samples: Vec<Vec<usize>> = vec![Vec::new(); nt];
    let mut i = 0usize;
    for y in 0..h {
        let ty = (y / TILE).min(th - 1);
        for x in 0..w {
            if sat[i] < 0.20 && (x + y) % 2 == 0 {
                let tx = (x / TILE).min(tw - 1);
                samples[ty * tw + tx].push(i);
            }
            i += 1;
        }
    }
    for t in 0..nt {
        if samples[t].len() < 40 {
            fl[t] = gml;
            fa[t] = gma;
            fb[t] = gmb;
            fsig[t] = gsig;
            continue;
        }
        let mut cl: Vec<f32> = samples[t].iter().map(|&i| lab[i][0]).collect();
        let mut ca: Vec<f32> = samples[t].iter().map(|&i| lab[i][1]).collect();
        let mut cb: Vec<f32> = samples[t].iter().map(|&i| lab[i][2]).collect();
        let (ml, ma, mb) = (median3(&mut cl), median3(&mut ca), median3(&mut cb));
        let mut cds: Vec<f32> = samples[t]
            .iter()
            .map(|&i| {
                let p = lab[i];
                ((p[0] - ml).powi(2) + (p[1] - ma).powi(2) + (p[2] - mb).powi(2)).sqrt()
            })
            .collect();
        fl[t] = ml;
        fa[t] = ma;
        fb[t] = mb;
        fsig[t] = (1.4826 * median3(&mut cds)).clamp(2.5, 14.0);
    }
    WallField {
        tw,
        th,
        l: smooth3x3(&fl, tw, th),
        a: smooth3x3(&fa, tw, th),
        b: smooth3x3(&fb, tw, th),
        sigma: smooth3x3(&fsig, tw, th),
    }
}

/// 3×3 均值平滑（边缘钳制复制）
fn smooth3x3(src: &[f32], tw: usize, th: usize) -> Vec<f32> {
    let mut out = vec![0f32; src.len()];
    for y in 0..th {
        for x in 0..tw {
            let mut acc = 0.0;
            let mut cnt = 0.0;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let yy = (y as i32 + dy).clamp(0, th as i32 - 1) as usize;
                    let xx = (x as i32 + dx).clamp(0, tw as i32 - 1) as usize;
                    acc += src[yy * tw + xx];
                    cnt += 1.0;
                }
            }
            out[y * tw + x] = acc / cnt;
        }
    }
    out
}

/// 墙面净化用的墙面场变体（render 填充层的数据源）：与 build_wall_field 同骨架
/// （分块 + 平滑 + 双线性），但统计口径换成"亮部干净色"——块内排除 exclude>0 的像素
/// （岩点并集外扩区），取低饱和墙面像素的 P90 明度与亮部（L ≥ P90）像素的中位数色；
/// 样本不足的块回退全局统计。sigma 置 0（净化不用）。
pub(crate) fn build_fill_field(raw: &[u8], sat: &[f32], exclude: &[u8], w: usize, h: usize) -> WallField {
    let tw = w.div_ceil(TILE);
    let th = h.div_ceil(TILE);
    let nt = tw * th;

    // 分块采样（与 build_wall_field 同口径：低饱和 + 棋盘步进）
    let mut samples: Vec<Vec<usize>> = vec![Vec::new(); nt];
    let mut i = 0usize;
    for y in 0..h {
        let ty = (y / TILE).min(th - 1);
        for x in 0..w {
            if exclude[i] == 0 && sat[i] < 0.20 && (x + y) % 2 == 0 {
                let tx = (x / TILE).min(tw - 1);
                samples[ty * tw + tx].push(i);
            }
            i += 1;
        }
    }

    // 亮部统计：P90 明度 + 亮部像素的中位数 (a*, b*)
    let bright_stats = |idx: &[usize]| -> Option<(f32, f32, f32)> {
        if idx.len() < 40 {
            return None;
        }
        let mut pts: Vec<[f32; 3]> = idx
            .iter()
            .map(|&i| hsv::rgb_to_lab(raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2]))
            .collect();
        pts.sort_by(|x, y| x[0].partial_cmp(&y[0]).unwrap());
        let p90 = pts[((pts.len() * 9) / 10).min(pts.len() - 1)][0];
        let mut ba: Vec<f32> = pts.iter().filter(|p| p[0] >= p90).map(|p| p[1]).collect();
        let mut bb: Vec<f32> = pts.iter().filter(|p| p[0] >= p90).map(|p| p[2]).collect();
        Some((p90, median3(&mut ba), median3(&mut bb)))
    };

    let all: Vec<usize> = samples.iter().flatten().copied().collect();
    let global = bright_stats(&all).unwrap_or((50.0, 0.0, 0.0));

    // 稀疏块回退：先 3×3 邻块联合采样（跟随局部光照），仍不足才用全局
    let neigh_stats = |t: usize| -> Option<(f32, f32, f32)> {
        let (tx, ty) = (t % tw, t / tw);
        let mut idx = Vec::new();
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let (nx, ny) = (tx as i32 + dx, ty as i32 + dy);
                if nx >= 0 && ny >= 0 && (nx as usize) < tw && (ny as usize) < th {
                    idx.extend_from_slice(&samples[ny as usize * tw + nx as usize]);
                }
            }
        }
        bright_stats(&idx)
    };

    let mut fl = vec![0f32; nt];
    let mut fa = vec![0f32; nt];
    let mut fb = vec![0f32; nt];
    for t in 0..nt {
        let (l, a, b) = bright_stats(&samples[t]).or_else(|| neigh_stats(t)).unwrap_or(global);
        fl[t] = l;
        fa[t] = a;
        fb[t] = b;
    }
    WallField {
        tw,
        th,
        l: smooth3x3(&fl, tw, th),
        a: smooth3x3(&fa, tw, th),
        b: smooth3x3(&fb, tw, th),
        sigma: vec![0.0; nt],
    }
}

/// 双线性取墙面参考色与 σ（tile 中心插值）
#[inline]
fn wall_ref(f: &WallField, x: usize, y: usize) -> ([f32; 3], f32) {
    let fx = (x as f32 + 0.5) / TILE as f32 - 0.5;
    let fy = (y as f32 + 0.5) / TILE as f32 - 0.5;
    let cx = fx.max(0.0);
    let cy = fy.max(0.0);
    let tx0 = (cx.floor() as usize).min(f.tw - 1);
    let ty0 = (cy.floor() as usize).min(f.th - 1);
    let tx1 = (tx0 + 1).min(f.tw - 1);
    let ty1 = (ty0 + 1).min(f.th - 1);
    let rx = (cx - tx0 as f32).clamp(0.0, 1.0);
    let ry = (cy - ty0 as f32).clamp(0.0, 1.0);
    let idx = |tx: usize, ty: usize| ty * f.tw + tx;
    let lerp = |v00: f32, v10: f32, v01: f32, v11: f32| {
        (v00 * (1.0 - rx) + v10 * rx) * (1.0 - ry) + (v01 * (1.0 - rx) + v11 * rx) * ry
    };
    let l = lerp(f.l[idx(tx0, ty0)], f.l[idx(tx1, ty0)], f.l[idx(tx0, ty1)], f.l[idx(tx1, ty1)]);
    let a = lerp(f.a[idx(tx0, ty0)], f.a[idx(tx1, ty0)], f.a[idx(tx0, ty1)], f.a[idx(tx1, ty1)]);
    let b = lerp(f.b[idx(tx0, ty0)], f.b[idx(tx1, ty0)], f.b[idx(tx0, ty1)], f.b[idx(tx1, ty1)]);
    let s = lerp(f.sigma[idx(tx0, ty0)], f.sigma[idx(tx1, ty0)], f.sigma[idx(tx0, ty1)], f.sigma[idx(tx1, ty1)]);
    ([l, a, b], s)
}

// ---------------- 颜色确认 ----------------

fn classify_stats(s: &CompStats, min_colored_frac: f32) -> HoldKind {
    let (_mode_hue, purity) = s.mode_hue();
    if s.colored_frac() >= min_colored_frac && purity >= 0.45 {
        HoldKind::Colored
    } else if s.val_mean() < 0.25 {
        HoldKind::Black
    } else if s.val_mean() > 0.70 && s.sat_mean() < 0.30 {
        HoldKind::White
    } else {
        HoldKind::Gray
    }
}

// ---------------- SAM 框提示的触边扩框（大造型欠覆盖修复，issue #7） ----------------

/// 类别感知的取样窗口外扩比例：大体积 0.35，普通岩点 0.18
pub(crate) fn sam_margin(cls: usize) -> f32 {
    if cls == 1 {
        0.35
    } else {
        0.18
    }
}

/// 框 + 外扩比例 -> 取样窗口（分析图像素坐标，裁剪到图内，保证非空）
fn sam_window(b: [f32; 4], ex: f32, aw: u32, ah: u32) -> (usize, usize, usize, usize) {
    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
    let x0 = ((b[0] - bw * ex).max(0.0)) as usize;
    let y0 = ((b[1] - bh * ex).max(0.0)) as usize;
    let x1 = ((b[2] + bw * ex).min(aw as f32)).ceil() as usize;
    let y1 = ((b[3] + bh * ex).min(ah as f32)).ceil() as usize;
    (x0, y0, x1.max(x0 + 1), y1.max(y0 + 1))
}

/// 掩码（logit>0）是否触到窗口内缘：触边 = 物体很可能超出提示框，值得扩框重切
fn mask_touches_edge(emb: &sam::SamEmbedding, grid: &[f32], win: (usize, usize, usize, usize)) -> bool {
    let (x0, y0, x1, y1) = win;
    if x1 - x0 < 8 || y1 - y0 < 8 {
        return false;
    }
    let (xi0, yi0, xi1, yi1) = (x0 + 2, y0 + 2, x1 - 3, y1 - 3);
    let mut hits = 0usize;
    let mut x = xi0;
    while x <= xi1 {
        for y in [yi0, yi1] {
            if emb.sample_logit(grid, x as f32 + 0.5, y as f32 + 0.5) > 0.0 {
                hits += 1;
            }
        }
        x += 2;
    }
    let mut y = yi0;
    while y <= yi1 {
        for x in [xi0, xi1] {
            if emb.sample_logit(grid, x as f32 + 0.5, y as f32 + 0.5) > 0.0 {
                hits += 1;
            }
        }
        y += 2;
    }
    hits >= 3
}

/// 窗口内掩码正采样数（停滞判据：扩框后掩码不再增长说明已覆盖完整）
fn mask_positives(emb: &sam::SamEmbedding, grid: &[f32], win: (usize, usize, usize, usize)) -> usize {
    let (x0, y0, x1, y1) = win;
    let mut cnt = 0usize;
    let mut y = y0;
    while y < y1 {
        let mut x = x0;
        while x < x1 {
            if emb.sample_logit(grid, x as f32 + 0.5, y as f32 + 0.5) > 0.0 {
                cnt += 1;
            }
            x += 3;
        }
        y += 3;
    }
    cnt
}

/// 框四边各扩 pct/2，裁剪到图内
fn grow_box(b: [f32; 4], pct: f32, aw: u32, ah: u32) -> [f32; 4] {
    let dx = (b[2] - b[0]) * pct / 2.0;
    let dy = (b[3] - b[1]) * pct / 2.0;
    [
        (b[0] - dx).max(0.0),
        (b[1] - dy).max(0.0),
        (b[2] + dx).min(aw as f32),
        (b[3] + dy).min(ah as f32),
    ]
}

/// 全自动分析一张图
pub fn analyze(full: &DynamicImage, p: &Params) -> Analysis {
    analyze_with_progress(full, p, &|_| {})
}

/// 带进度回调的分析（GUI 异步 worker 用；cb 收到阶段名）
pub fn analyze_with_progress(full: &DynamicImage, p: &Params, cb: &dyn Fn(&str)) -> Analysis {
    let t0 = Instant::now();
    cb("预处理图像");
    let fw = full.width().max(1);
    let fh = full.height().max(1);
    let scale = (p.analysis_max_dim as f32 / fw.max(fh) as f32).min(1.0);
    let aw = (((fw as f32 * scale).round() as u32).max(32)).min(fw);
    let ah = (((fh as f32 * scale).round() as u32).max(32)).min(fh);
    let img: RgbaImage = if aw == fw && ah == fh {
        full.clone().into_rgba8()
    } else {
        full.resize_exact(aw, ah, image::imageops::FilterType::Triangle).to_rgba8()
    };
    let n = (aw * ah) as usize;
    let (w, h) = (aw as usize, ah as usize);

    cb("计算颜色空间");
    let (hue, sat, val) = hsv::compute_hsv(img.as_raw());

    // ---- 岩点提取 ----
    let min_px = (p.min_area_frac * n as f64) as u32;
    let mut holds: Vec<Hold> = Vec::new();
    let mut labels: Vec<u32> = Vec::new();

    if let Some(det) = detector::YoloDetect::holds() {
        // 主路：ChalkGPT YOLO11s 出框 -> EdgeSAM 框提示掩码 ∩ 局部墙面差分（回退链见下）
        cb("YOLO 检测岩点");
        let boxes = det.detect_holds(&img);
        cb("EdgeSAM 图像编码");
        let (sam, emb) = match sam::EdgeSam::global() {
            Some(s) => (Some(s), s.encode(&img)),
            None => (None, None),
        };
        if sam.is_some() {
            cb("SAM 掩码解码");
        }
        // SAM 路径需要全图 Lab（墙证据判定）
        let lab_opt: Option<Vec<[f32; 3]>> = if sam.is_some() {
            let raw = img.as_raw();
            Some((0..n).map(|i| hsv::rgb_to_lab(raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2])).collect())
        } else {
            None
        };
        // SAM 解码并行化：各框独立，共享 embedding（tract 模型线程安全）。
        // 触边扩框闭环（issue #7）：掩码触到取样窗口边缘 → 扩提示框 25% 重切，
        // 最多 3 次解码；掩码不再增长（停滞）即收敛。大体积框用更大初始窗口（0.35）。
        let sam_grids: Vec<Option<(Vec<f32>, [f32; 4])>> = if let (Some(s), Some(e)) = (sam, emb.as_ref()) {
            use rayon::prelude::*;
            boxes
                .par_iter()
                .map(|(_conf, cls, b)| {
                    let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
                    if bw < 22.0 || bh < 22.0 {
                        return None;
                    }
                    let margin = sam_margin(*cls);
                    let mut bx = *b;
                    let (mut grid, _) = s.mask_for_box(e, bx)?;
                    let mut prev = mask_positives(e, &grid, sam_window(bx, margin, aw, ah));
                    for _ in 0..2 {
                        if !mask_touches_edge(e, &grid, sam_window(bx, margin, aw, ah)) {
                            break;
                        }
                        bx = grow_box(bx, 0.25, aw, ah);
                        let (g2, _) = s.mask_for_box(e, bx)?;
                        let cnt = mask_positives(e, &g2, sam_window(bx, margin, aw, ah));
                        let stagnated = (cnt as f32) < prev as f32 * 1.05;
                        grid = g2;
                        prev = cnt;
                        if stagnated {
                            break;
                        }
                    }
                    Some((grid, bx))
                })
                .collect()
        } else {
            vec![None; boxes.len()]
        };
        for (bi, (_conf, cls, ob)) in boxes.iter().enumerate() {
            // 最终提示框（可能已被触边扩框放大）；无 SAM 时用 YOLO 原框
            let b = sam_grids[bi].as_ref().map(|(_, gb)| *gb).unwrap_or(*ob);
            let (bw, bh) = (b[2] - b[0], b[3] - b[1]);
            if bw < 5.0 || bh < 5.0 {
                continue;
            }
            let ex = sam_margin(*cls);
            let x0 = ((b[0] - bw * ex).max(0.0)) as usize;
            let y0 = ((b[1] - bh * ex).max(0.0)) as usize;
            let x1 = ((b[2] + bw * ex).min(aw as f32)).ceil() as usize;
            let y1 = ((b[3] + bh * ex).min(ah as f32)).ceil() as usize;
            let (cw, chh) = (x1.saturating_sub(x0), y1.saturating_sub(y0));
            if cw < 5 || chh < 5 {
                continue;
            }

            // 1) EdgeSAM 框提示掩码 ∩ 非墙证据，连通域拆分（防条带合并）
            let mut found: Vec<Hold> = Vec::new();
            let mut sam_ok = false;
            if let (Some(s), Some(e), Some(lab)) = (sam, emb.as_ref(), lab_opt.as_deref()) {
                if let Some((grid, _)) = sam_grids[bi].as_ref() {
                    {
                        let wall = local_wall_ref(lab, &sat, w, x0, y0, cw, chh);
                        let t_lo = wall.map(|(_, sg)| (p.wall_k * sg * 0.55).clamp(4.0, 18.0));
                        let mut sam_px: Vec<u32> = Vec::new();
                        let mut sam_bin = vec![0u8; cw * chh];
                        let mut mask_bin = vec![0u8; cw * chh];
                        for y in y0..y1 {
                            for x in x0..x1 {
                                if e.sample_logit(&grid, x as f32 + 0.5, y as f32 + 0.5) > 0.0 {
                                    let gi = (y * w + x) as u32;
                                    sam_px.push(gi);
                                    sam_bin[(y - y0) * cw + (x - x0)] = 255;
                                    let keep = if let (Some((med, _)), Some(t)) = (wall, t_lo) {
                                        let q = lab[gi as usize];
                                        ((q[0] - med[0]).powi(2) + (q[1] - med[1]).powi(2) + (q[2] - med[2]).powi(2)).sqrt() > t
                                    } else {
                                        true
                                    };
                                    if keep {
                                        mask_bin[(y - y0) * cw + (x - x0)] = 255;
                                    }
                                }
                            }
                        }
                        // 连通域拆分：条带掩码按墙缝断开成逐点 blob（窗口局部索引）
                        let (lbl2, cnt2) = conn::label_components(&mask_bin, cw, chh);
                        let mut comps: Vec<Vec<u32>> = vec![Vec::new(); cnt2];
                        for dy in 0..chh {
                            let base = dy * cw;
                            for dx in 0..cw {
                                let l = lbl2[base + dx];
                                if l > 0 {
                                    comps[(l - 1) as usize].push((base + dx) as u32);
                                }
                            }
                        }
                        comps.sort_by_key(|c| std::cmp::Reverse(c.len()));
                        comps.truncate(4);
                        comps.retain(|c| c.len() >= 24);
                        // 测地重构（issue #8）：证据 ∩ SAM 确认的子块为种子，在 SAM 掩码内
                        // 做最近种子扩张——被证据裁掉的连通延伸（纯色尖顶/灰边）收回；
                        // 条带在多种子间剖分不跨域合并；面积上限防掩码过冲被无界吞并
                        let seeds: Vec<&[u32]> = comps.iter().map(|c| c.as_slice()).collect();
                        let caps: Vec<usize> = comps.iter().map(|c| (c.len() * 4).max(c.len() + 2000)).collect();
                        let grown = conn::voronoi_reconstruct(&sam_bin, cw, chh, &seeds, &caps);
                        for gpx in grown {
                            if gpx.len() < 24 {
                                continue;
                            }
                            let cpx: Vec<u32> = gpx
                                .iter()
                                .map(|&i| {
                                    let (lx, ly) = (i as usize % cw, i as usize / cw);
                                    ((y0 + ly) * w + (x0 + lx)) as u32
                                })
                                .collect();
                            if let Some(hf) = hold_from_pixels(&img, w, cpx, p) {
                                found.push(hf);
                                sam_ok = true;
                            }
                        }
                        // 交集为空（岩点颜色≈墙面）：退回完整 SAM 掩码
                        if !sam_ok && sam_px.len() >= 24 {
                            if let Some(hf) = hold_from_pixels(&img, w, sam_px, p) {
                                found.push(hf);
                                sam_ok = true;
                            }
                        }
                    }
                }
            }
            // 2) 回退：框内墙面差分（blob 覆盖 <40% 框面积时与椭圆并集）
            if !sam_ok {
                found = wall_dist_holds_in_crop(
                    &img,
                    w,
                    x0,
                    y0,
                    cw,
                    chh,
                    None,
                    p,
                    ((min_px as f32) * 0.25).max(12.0) as u32,
                    (cw as f64 * chh as f64) as u32,
                );
                if found.is_empty() {
                    if let Some(hf) = hold_from_ellipse(&img, w, x0, y0, cw, chh, p) {
                        found.push(hf);
                    }
                } else {
                    // 只保留中心在检测框内（略放宽）的最大 3 个 blob
                    let cxr = (b[0] - bw * 0.1, b[2] + bw * 0.1);
                    let cyr = (b[1] - bh * 0.1, b[3] + bh * 0.1);
                    found.retain(|hf| {
                        hf.centroid[0] >= cxr.0 && hf.centroid[0] <= cxr.1 && hf.centroid[1] >= cyr.0 && hf.centroid[1] <= cyr.1
                    });
                    found.sort_by_key(|hf| std::cmp::Reverse(hf.area));
                    found.truncate(3);
                    let blob_area: u32 = found.iter().map(|h| h.area).sum();
                    if !found.is_empty() && blob_area * 5 < (cw as u32) * (chh as u32) * 2 {
                        if let Some(ell) = hold_from_ellipse(&img, w, x0, y0, cw, chh, p) {
                            let biggest = &mut found[0];
                            let mut set: std::collections::HashSet<u32> = biggest.own_pixels.iter().copied().collect();
                            for &i in &ell.own_pixels {
                                set.insert(i);
                            }
                            biggest.own_pixels = set.into_iter().collect();
                            biggest.area = biggest.own_pixels.len() as u32;
                            let mut bx = [u32::MAX, u32::MAX, 0u32, 0u32];
                            let mut sx = 0f64;
                            let mut sy = 0f64;
                            for &i in &biggest.own_pixels {
                                let (x, y) = (i as u32 % aw, i as u32 / aw);
                                sx += x as f64;
                                sy += y as f64;
                                if x < bx[0] {
                                    bx[0] = x;
                                }
                                if y < bx[1] {
                                    bx[1] = y;
                                }
                                if x + 1 > bx[2] {
                                    bx[2] = x + 1;
                                }
                                if y + 1 > bx[3] {
                                    bx[3] = y + 1;
                                }
                            }
                            biggest.bbox = bx;
                            biggest.centroid = [(sx / biggest.area as f64) as f32, (sy / biggest.area as f64) as f32];
                        }
                    }
                }
            }
            for mut hf in found {
                let r = (hf.area as f32 / std::f32::consts::PI).sqrt();
                let dup = holds.iter().any(|e| {
                    let re = (e.area as f32 / std::f32::consts::PI).sqrt();
                    let dx = e.centroid[0] - hf.centroid[0];
                    let dy = e.centroid[1] - hf.centroid[1];
                    let centroid_dup = (dx * dx + dy * dy).sqrt() < (r + re) * 0.6;
                    // bbox IoU 去重：滑窗二阶段与主检测框重叠时质心可能偏离，靠质心会漏
                    let ix0 = e.bbox[0].max(hf.bbox[0]) as f32;
                    let iy0 = e.bbox[1].max(hf.bbox[1]) as f32;
                    let ix1 = (e.bbox[2].min(hf.bbox[2])) as f32;
                    let iy1 = (e.bbox[3].min(hf.bbox[3])) as f32;
                    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
                    let ea = ((e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1])) as f32;
                    let fa = ((hf.bbox[2] - hf.bbox[0]) * (hf.bbox[3] - hf.bbox[1])) as f32;
                    let iou = if ea + fa - inter > 1.0 { inter / (ea + fa - inter) } else { 0.0 };
                    centroid_dup || iou > 0.5
                });
                if dup {
                    continue;
                }
                hf.id = holds.len();
                holds.push(hf);
            }
        }
    }

    if holds.is_empty() {
        // 回退：整图墙面差分（无岩点模型时）
        let raw = img.as_raw();
        let mut lab: Vec<[f32; 3]> = vec![[0.0; 3]; n];
        for i in 0..n {
            lab[i] = hsv::rgb_to_lab(raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2]);
        }
        let field = build_wall_field(&lab, &sat, w, h);
        let mut mask = vec![0u8; n];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let (r, sigma) = wall_ref(&field, x, y);
                let q = lab[i];
                let d = ((q[0] - r[0]).powi(2) + (q[1] - r[1]).powi(2) + (q[2] - r[2]).powi(2)).sqrt();
                let t = (p.wall_k * sigma).clamp(6.0, 30.0);
                if d > t {
                    mask[i] = 255;
                }
            }
        }
        let mask = morph::open(&mask, w, h, 2);
        let mut mask = morph::dilate(&mask, w, h, 2);
        let (lbl, cnt) = conn::label_components(&mask, w, h);
        let stats = conn::gather_stats(&lbl, &hue, &sat, &val, w, h, cnt);
        labels = lbl;
        let mut oversized: Vec<&CompStats> = Vec::new();
        for s in &stats {
            if s.area == 0 || (s.area as f64) < p.min_area_frac * n as f64 {
                continue;
            }
            let kind = classify_stats(s, p.min_colored_frac);
            let max_frac_for = if kind == HoldKind::Colored { p.max_colored_area_frac } else { p.max_area_frac };
            let af = s.area as f64 / n as f64;
            if af > max_frac_for {
                if af <= 0.30 && (s.bw() as f64) * (s.bh() as f64) <= 0.36 * n as f64 {
                    oversized.push(s);
                }
                continue;
            }
            if s.fill() < p.min_fill_ratio {
                continue;
            }
            if kind == HoldKind::Black {
                if (s.area as f64) < p.min_area_frac * 1.6 * n as f64 {
                    continue;
                }
                if s.y1 as f64 > 0.95 * h as f64 {
                    continue;
                }
            } else if kind == HoldKind::Gray {
                if s.sat_mean() > 0.13 || s.fill() < (p.min_fill_ratio + 0.12).min(0.6) || s.aspect() > 3.2 {
                    continue;
                }
                if (s.area as f64) < p.min_area_frac * 2.0 * n as f64 {
                    continue;
                }
                if s.y1 as f64 > 0.93 * h as f64 {
                    continue;
                }
            }
            let (mode_hue, purity) = s.mode_hue();
            // Lab 色度中心（回退路径：标签像素直接统计，众数色相 ±15° 内的彩色像素）
            let mut ab = [0.0f32, 0.0f32];
            if kind == HoldKind::Colored {
                let (mut as_, mut bs, mut cw2) = (0f32, 0f32, 0f32);
                for y in s.y0 as usize..s.y1 as usize {
                    for x in s.x0 as usize..s.x1 as usize {
                        let i = y * w + x;
                        if labels[i] == s.label && sat[i] >= 0.22 && val[i] >= 0.10 && hsv::hue_dist(hue[i], mode_hue) <= 15.0 {
                            as_ += lab[i][1];
                            bs += lab[i][2];
                            cw2 += 1.0;
                        }
                    }
                }
                if cw2 >= 6.0 {
                    ab = [as_ / cw2, bs / cw2];
                }
            }
            holds.push(Hold {
                id: holds.len(),
                label: s.label,
                own_pixels: Vec::new(),
                area: s.area,
                bbox: s.bbox_arr(),
                centroid: s.centroid(),
                hue: if kind == HoldKind::Colored { mode_hue } else { 0.0 },
                ab,
                sat: s.sat_mean(),
                val: s.val_mean(),
                purity,
                kind,
                cluster: usize::MAX,
                lasso: false,
            });
        }
        // 超大块内部补测（深色体积上的岩点等）
        oversized.sort_by_key(|s| std::cmp::Reverse(s.area));
        oversized.truncate(8);
        for s in &oversized {
            let mut found = wall_dist_holds_in_crop(
                &img,
                w,
                s.x0 as usize,
                s.y0 as usize,
                s.bw() as usize,
                s.bh() as usize,
                None,
                p,
                (min_px as f32 * 0.5) as u32,
                (p.max_colored_area_frac * 1.2 * n as f64) as u32,
            );
            for mut f in found.drain(..) {
                let r = (f.area as f32 / std::f32::consts::PI).sqrt();
                let dup = holds.iter().any(|e| {
                    let re = (e.area as f32 / std::f32::consts::PI).sqrt();
                    let dx = e.centroid[0] - f.centroid[0];
                    let dy = e.centroid[1] - f.centroid[1];
                    (dx * dx + dy * dy).sqrt() < (r + re) * 0.7
                });
                if dup {
                    continue;
                }
                f.id = holds.len();
                holds.push(f);
            }
        }
    }

    cb("颜色聚类分线");
    let clusters = cluster::build_clusters(&mut holds, p.color_tol_ab);

    // 墙面统计（展示用）
    let mut vs: Vec<f32> = Vec::new();
    let mut ss: Vec<f32> = Vec::new();
    for i in (0..n).step_by(7) {
        if sat[i] < 0.20 {
            vs.push(val[i]);
            ss.push(sat[i]);
        }
    }
    let (wall_v, wall_s) = if vs.is_empty() {
        (0.5, 0.15)
    } else {
        vs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ss.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (vs[vs.len() / 2], ss[ss.len() / 2])
    };

    Analysis {
        aw,
        ah,
        fw,
        fh,
        labels,
        holds,
        clusters,
        img,
        hue,
        sat,
        val,
        wall_v,
        wall_s,
        timing_ms: t0.elapsed().as_secs_f32() * 1000.0,
    }
}

/// 任意像素集合 -> 岩点（颜色统计 + 分类）。SAM 掩码 / 椭圆 / 套索共用
fn hold_from_pixels(img: &RgbaImage, gw: usize, own: Vec<u32>, p: &Params) -> Option<Hold> {
    let raw = img.as_raw();
    let area = own.len() as u32;
    if area < 8 {
        return None;
    }
    let mut hue_bins = [0u32; 72];
    let mut colored = 0u32;
    let mut sat_sum = 0f32;
    let mut val_sum = 0f32;
    let mut sum_x = 0f64;
    let mut sum_y = 0f64;
    let mut bx = [u32::MAX, u32::MAX, 0u32, 0u32];
    for &i in &own {
        let (x, y) = (i as u32 % gw as u32, i as u32 / gw as u32);
        let pi = i as usize * 4;
        let (hh, ss, vv) = hsv::rgb_to_hsv(raw[pi], raw[pi + 1], raw[pi + 2]);
        if ss >= 0.22 && vv >= 0.10 {
            colored += 1;
            hue_bins[(hh / 5.0).floor().rem_euclid(72.0) as usize] += 1;
        }
        sat_sum += ss;
        val_sum += vv;
        sum_x += x as f64;
        sum_y += y as f64;
        if x < bx[0] {
            bx[0] = x;
        }
        if y < bx[1] {
            bx[1] = y;
        }
        if x + 1 > bx[2] {
            bx[2] = x + 1;
        }
        if y + 1 > bx[3] {
            bx[3] = y + 1;
        }
    }
    let mut mode_bin = 0usize;
    for i in 1..72 {
        if hue_bins[i] > hue_bins[mode_bin] {
            mode_bin = i;
        }
    }
    let mut win = 0u32;
    for d in -2i32..=2 {
        win += hue_bins[(mode_bin as i32 + d).rem_euclid(72) as usize];
    }
    let purity = if colored > 0 { win as f32 / colored as f32 } else { 0.0 };
    let colored_frac = colored as f32 / area as f32;
    let sat_mean = sat_sum / area as f32;
    let val_mean = val_sum / area as f32;
    let kind = if colored_frac >= p.min_colored_frac && purity >= 0.45 {
        HoldKind::Colored
    } else if val_mean < 0.25 {
        HoldKind::Black
    } else if val_mean > 0.70 && sat_mean < 0.30 {
        HoldKind::White
    } else {
        HoldKind::Gray
    };
    // Lab 色度中心：只统计众数色相 ±15° 内的彩色像素（抗相邻异色渗入），不足则退回全部彩色像素
    let mode_center = mode_bin as f32 * 5.0 + 2.5;
    let mut ab = [0.0f32, 0.0f32];
    if kind == HoldKind::Colored {
        let (mut as_, mut bs, mut cw) = (0f32, 0f32, 0f32);
        for &i in &own {
            let pi = i as usize * 4;
            let (hh, ss, vv) = hsv::rgb_to_hsv(raw[pi], raw[pi + 1], raw[pi + 2]);
            if ss >= 0.22 && vv >= 0.10 && hsv::hue_dist(hh, mode_center) <= 15.0 {
                let l = hsv::rgb_to_lab(raw[pi], raw[pi + 1], raw[pi + 2]);
                as_ += l[1];
                bs += l[2];
                cw += 1.0;
            }
        }
        if cw >= 6.0 {
            ab = [as_ / cw, bs / cw];
        }
    }
    Some(Hold {
        id: usize::MAX,
        label: 0,
        own_pixels: own,
        area,
        bbox: bx,
        centroid: [(sum_x / area as f64) as f32, (sum_y / area as f64) as f32],
        hue: if kind == HoldKind::Colored { mode_bin as f32 * 5.0 + 2.5 } else { 0.0 },
        ab,
        sat: sat_mean,
        val: val_mean,
        purity,
        kind,
        cluster: usize::MAX,
        lasso: false,
    })
}

/// 椭圆兜底：YOLO 框内墙面差分失败（岩点颜色≈墙面）时，用内切椭圆直接当掩码
#[allow(clippy::too_many_arguments)]
fn hold_from_ellipse(img: &RgbaImage, gw: usize, gx0: usize, gy0: usize, cw: usize, ch: usize, p: &Params) -> Option<Hold> {
    if cw < 3 || ch < 3 {
        return None;
    }
    let (ecx, ecy) = ((cw as f32 - 1.0) * 0.5, (ch as f32 - 1.0) * 0.5);
    let (rx, ry) = (cw as f32 * 0.42, ch as f32 * 0.42);
    let mut own = Vec::new();
    for dy in 0..ch {
        let gy = gy0 + dy;
        for dx in 0..cw {
            let ex = (dx as f32 - ecx) / rx;
            let ey = (dy as f32 - ecy) / ry;
            if ex * ex + ey * ey > 1.0 {
                continue;
            }
            own.push(((gy * gw) + gx0 + dx) as u32);
        }
    }
    hold_from_pixels(img, gw, own, p)
}

/// 裁剪窗口的局部墙面参考色与噪声 σ（低饱和像素的中位数）
fn local_wall_ref(lab: &[[f32; 3]], sat: &[f32], w: usize, x0: usize, y0: usize, cw: usize, ch: usize) -> Option<([f32; 3], f32)> {
    let mut cl: Vec<f32> = Vec::new();
    let mut ca: Vec<f32> = Vec::new();
    let mut cb: Vec<f32> = Vec::new();
    for dy in 0..ch {
        for dx in 0..cw {
            let i = (y0 + dy) * w + x0 + dx;
            if sat[i] < 0.20 {
                cl.push(lab[i][0]);
                ca.push(lab[i][1]);
                cb.push(lab[i][2]);
            }
        }
    }
    if cl.len() < 20 {
        return None;
    }
    let (ml, ma, mb) = (median3(&mut cl), median3(&mut ca), median3(&mut cb));
    let mut ds: Vec<f32> = (0..cl.len())
        .map(|i| {
            let (l, a, b) = (cl[i], ca[i], cb[i]);
            ((l - ml).powi(2) + (a - ma).powi(2) + (b - mb).powi(2)).sqrt()
        })
        .collect();
    Some(([ml, ma, mb], (1.4826 * median3(&mut ds)).clamp(2.0, 14.0)))
}

/// 裁剪窗口内的墙面差分岩点检测（框内掩码提取 / 套索补测 / 超大块内部补测共用）
/// region: 与裁剪窗口等大的 0/255 掩码（None = 全窗口）
#[allow(clippy::too_many_arguments)]
fn wall_dist_holds_in_crop(
    img: &RgbaImage,
    gw: usize,
    gx0: usize,
    gy0: usize,
    cw: usize,
    ch: usize,
    region: Option<&[u8]>,
    p: &Params,
    min_px: u32,
    max_px: u32,
) -> Vec<Hold> {
    if cw < 4 || ch < 4 {
        return Vec::new();
    }
    let raw = img.as_raw();
    let mut lh = vec![0f32; cw * ch];
    let mut ls = vec![0f32; cw * ch];
    let mut lv = vec![0f32; cw * ch];
    let mut llab: Vec<[f32; 3]> = vec![[0.0; 3]; cw * ch];
    let mut sample: Vec<usize> = Vec::new();
    for dy in 0..ch {
        let gy = gy0 + dy;
        for dx in 0..cw {
            let li = dy * cw + dx;
            if let Some(r) = region {
                if r[li] == 0 {
                    continue;
                }
            }
            let gx = gx0 + dx;
            let gi = gy * gw + gx;
            let pi = gi * 4;
            let (hh, ss, vv) = hsv::rgb_to_hsv(raw[pi], raw[pi + 1], raw[pi + 2]);
            lh[li] = hh;
            ls[li] = ss;
            lv[li] = vv;
            llab[li] = hsv::rgb_to_lab(raw[pi], raw[pi + 1], raw[pi + 2]);
            if ss < 0.20 {
                sample.push(li);
            }
        }
    }
    // 局部墙面参考
    let (med, sigma) = if sample.len() >= 30 {
        let mut cl: Vec<f32> = sample.iter().map(|&i| llab[i][0]).collect();
        let mut ca: Vec<f32> = sample.iter().map(|&i| llab[i][1]).collect();
        let mut cb: Vec<f32> = sample.iter().map(|&i| llab[i][2]).collect();
        let (ml, ma, mb) = (median3(&mut cl), median3(&mut ca), median3(&mut cb));
        let mut ds: Vec<f32> = sample
            .iter()
            .map(|&i| {
                let q = llab[i];
                ((q[0] - ml).powi(2) + (q[1] - ma).powi(2) + (q[2] - mb).powi(2)).sqrt()
            })
            .collect();
        ([ml, ma, mb], (1.4826 * median3(&mut ds)).clamp(2.5, 14.0))
    } else {
        ([70.0, 0.0, 0.0], 6.0)
    };
    let t = (p.wall_k * sigma * 0.85).clamp(6.0, 30.0);

    let mut mask = vec![0u8; cw * ch];
    for li in 0..cw * ch {
        if let Some(r) = region {
            if r[li] == 0 {
                continue;
            }
        }
        let q = llab[li];
        let d = ((q[0] - med[0]).powi(2) + (q[1] - med[1]).powi(2) + (q[2] - med[2]).powi(2)).sqrt();
        if d > t {
            mask[li] = 255;
        }
    }
    let mask = morph::open(&mask, cw, ch, 1);
    let mut mask = morph::dilate(&mask, cw, ch, 2);

    // 边缘 1px 裁掉，避免窗口边界截断的假轮廓
    for dx in 0..cw {
        mask[dx] = 0;
        mask[(ch - 1) * cw + dx] = 0;
    }
    for dy in 0..ch {
        mask[dy * cw] = 0;
        mask[dy * cw + cw - 1] = 0;
    }

    let (labels, cnt) = conn::label_components(&mask, cw, ch);
    let stats = conn::gather_stats(&labels, &lh, &ls, &lv, cw, ch, cnt);

    let mut out = Vec::new();
    for s in &stats {
        if s.area == 0 || s.area < min_px || s.area > max_px {
            continue;
        }
        if s.fill() < (p.min_fill_ratio - 0.05).max(0.15) {
            continue;
        }
        let mut own = Vec::with_capacity(s.area as usize);
        for dy in 0..ch {
            let base = dy * cw;
            for dx in 0..cw {
                if labels[base + dx] == s.label {
                    own.push(((gy0 + dy) * gw + (gx0 + dx)) as u32);
                }
            }
        }
        // 统一走 hold_from_pixels（含 Lab 色度中心计算）
        if let Some(h) = hold_from_pixels(img, gw, own, p) {
            out.push(h);
        }
    }
    out
}

/// 套索区域内补测：更宽松阈值重新检测漏掉的岩点（轮廓优先，同主算法）
/// region: 与分析图等大的 0/255 掩码
pub fn detect_in_region(a: &Analysis, p: &Params, region: &[u8]) -> Vec<Hold> {
    let (w, h) = (a.aw as usize, a.ah as usize);
    let n = w * h;
    assert_eq!(region.len(), n);

    let mut x0 = w;
    let mut y0 = h;
    let mut x1 = 0usize;
    let mut y1 = 0usize;
    for y in 0..h {
        for x in 0..w {
            if region[y * w + x] != 0 {
                if x < x0 {
                    x0 = x;
                }
                if y < y0 {
                    y0 = y;
                }
                if x + 1 > x1 {
                    x1 = x + 1;
                }
                if y + 1 > y1 {
                    y1 = y + 1;
                }
            }
        }
    }
    if x1 <= x0 || y1 <= y0 {
        return Vec::new();
    }
    let (cw, ch) = (x1 - x0, y1 - y0);
    // region 是全图掩码，而 wall_dist_holds_in_crop 按裁剪窗口局部坐标索引
    // —— 必须先裁出局部子掩码（曾因直接透传导致套索永远检测为空）
    let mut crop_region = vec![0u8; cw * ch];
    for dy in 0..ch {
        let src_row = (y0 + dy) * w + x0;
        let dst_row = dy * cw;
        for dx in 0..cw {
            crop_region[dst_row + dx] = region[src_row + dx];
        }
    }
    let min_px = ((p.min_area_frac * 0.35 * n as f64).max(30.0)) as u32;
    let max_px = (p.max_colored_area_frac * 2.5 * n as f64) as u32;
    let mut out = wall_dist_holds_in_crop(&a.img, w, x0, y0, cw, ch, Some(&crop_region), p, min_px, max_px);
    for h2 in out.iter_mut() {
        h2.lasso = true;
    }
    out
}
