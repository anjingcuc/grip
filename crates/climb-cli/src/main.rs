//! climb-cli
//!
//! 模式一（批处理，人用）:  climb-cli <目录|图片> [--out output] [--hue 210]
//! 模式二（AI agent 用，stdout 全部为可解析 JSON）:
//!   climb-cli analyze <图片> [--session <目录>]           建会话（检测+聚类），输出摘要
//!   climb-cli edit --session <目录> [--select <组id>]     选线 / 排除 / 补测 / 导出
//!              [--select-name red] [--remove 3,5] [--add x0,y0,x1,y1]
//!              [--out 成品.png] [--debug-out 调试.png]
//!
//! agent 工作流：analyze → (看 debug 图/摘要) → edit --select → edit --remove/--add → edit --out

use anyhow::{bail, Context, Result};
use clap::{Parser as ClapParser, Subcommand};
use climb_core as pipeline;
use climb_core::{cluster, hsv, render, Analysis, Cluster, Hold, HoldKind, Params};
use image::{DynamicImage, GenericImage, Rgba, RgbaImage};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

#[derive(ClapParser)]
#[command(name = "climb-cli", about = "攀岩线路高亮 CLI：批处理 + AI agent 会话模式")]
struct Args {
    /// [批处理模式] 输入图片或目录
    input: Option<PathBuf>,
    #[arg(short, long, default_value = "output")]
    out: PathBuf,
    /// [批处理] 指定保留线路的颜色角（Lab 色度角，度：红25 橙58 黄82 绿150 青200 蓝280 紫315 品红355）
    #[arg(long)]
    hue: Option<f32>,
    #[arg(long, default_value_t = 5.5)]
    k: f32,
    #[arg(long, default_value_t = 0.0)]
    bg_sat: f32,
    #[arg(long, default_value_t = 1600)]
    max_dim: u32,
    #[arg(long)]
    no_compare: bool,
    #[arg(long)]
    no_debug: bool,
    #[command(subcommand)]
    cmd: Option<Cmd>,
}

#[derive(Subcommand)]
enum Cmd {
    /// 分析图片并建立可编辑会话（agent 入口）。stdout 输出 JSON 摘要
    Analyze {
        image: PathBuf,
        /// 会话目录（默认 <图片目录>/sessions/<文件名>/）
        #[arg(long)]
        session: Option<PathBuf>,
        #[arg(long, default_value_t = 5.5)]
        k: f32,
        #[arg(long, default_value_t = 0.0)]
        bg_sat: f32,
        #[arg(long, default_value_t = 1600)]
        max_dim: u32,
    },
    /// 编辑会话：选线 / 排除误检 / 补测区域 / 导出。stdout 输出最新状态 JSON
    Edit {
        #[arg(long)]
        session: PathBuf,
        /// 选择保留的颜色组 id（-1 = 清除选择，输出全灰）
        #[arg(long)]
        select: Option<isize>,
        /// 按颜色名选择（中文或英文均可，取面积最大的匹配组，如 red / 红 / orange）
        #[arg(long)]
        select_name: Option<String>,
        /// 排除的岩点 id（逗号分隔）
        #[arg(long, value_delimiter = ',')]
        remove: Vec<usize>,
        /// 补测区域 "x0,y0,x1,y1"（0-1 归一化坐标，可多次指定）
        #[arg(long = "add")]
        add: Vec<String>,
        /// 导出全分辨率高亮成品
        #[arg(long)]
        out: Option<PathBuf>,
        /// 导出检测调试图（按组着色框，选中组白色粗框）
        #[arg(long)]
        debug_out: Option<PathBuf>,
    },
}

// ---------------- 会话持久化（核心库共享实现） ----------------

use climb_core::session::{self, SessionState};
use climb_core::session::session_dir_for as default_session_dir;
use climb_core::session::materialize_labels;

fn save_session(dir: &Path, st: &SessionState, analysis_img: &RgbaImage) -> Result<()> {
    session::save(dir, st, analysis_img)
}

/// 载入会话 -> (状态, 重建的 Analysis)
fn load_session(dir: &Path) -> Result<(SessionState, Analysis)> {
    session::load(dir)
}

/// 状态摘要 JSON（agent 的主要反馈通道）
fn summary_json(dir: &Path, st: &SessionState) -> Value {
    let total_area: u64 = st.holds.iter().map(|h| h.area as u64).max().unwrap_or(1);
    let _ = total_area;
    let total: u64 = st.holds.iter().map(|h| h.area as u64).sum::<u64>().max(1);
    let names = cluster::disambiguated_names(&st.clusters);
    json!({
        "session": dir.display().to_string(),
        "image": st.image,
        "size": {"full": [st.fw, st.fh], "analysis": [st.aw, st.ah]},
        "timing_ms": st.timing_ms,
        "selected": st.selected,
        "excluded": st.excluded,
        "clusters": st.clusters.iter().enumerate().map(|(ci, c)| json!({
            "id": ci,
            "name": names[ci],
            "count": c.holds.len(),
            "area_frac": (c.area as f64 / total as f64 * 100.0 * 10.0).round() / 10.0,
            "hue": (c.hue * 10.0).round() / 10.0,
            "ab": c.ab,
            "rgb": c.rgb,
            "holds": c.holds,
        })).collect::<Vec<_>>(),
        "holds": st.holds.iter().map(|h| json!({
            "id": h.id,
            "cluster": if h.cluster == usize::MAX { Value::Null } else { json!(h.cluster) },
            "kind": format!("{:?}", h.kind),
            "bbox_norm": [
                (h.bbox[0] as f64 / st.aw as f64 * 1000.0).round() / 1000.0,
                (h.bbox[1] as f64 / st.ah as f64 * 1000.0).round() / 1000.0,
                (h.bbox[2] as f64 / st.aw as f64 * 1000.0).round() / 1000.0,
                (h.bbox[3] as f64 / st.ah as f64 * 1000.0).round() / 1000.0,
            ],
            "area": h.area,
            "lasso": h.lasso,
        })).collect::<Vec<_>>(),
        "hint": "edit --select <cluster_id> 选线; --remove <hold_ids> 排除; --add x0,y0,x1,y1(0-1归一化) 补测; --out 导出成品; --debug-out 导出检测框图",
    })
}

/// 用会话当前状态渲染全分辨率成品
fn render_session(st: &SessionState, a: &Analysis, out: &Path) -> Result<()> {
    let full = image::ImageReader::open(&st.image)
        .with_context(|| format!("打开原图 {}", st.image))?
        .decode()
        .context("解码原图")?
        .to_rgba8();
    let excluded: HashSet<usize> = st.excluded.iter().copied().collect();
    let mask = render::keep_mask(a, st.selected, &excluded, st.params.mask_dilate);
    let mask = render::feather(a, &mask, st.params.feather);
    let img = render::composite_full(&full, a, &mask, st.params.bg_sat);
    img.save(out).with_context(|| format!("保存 {}", out.display()))?;
    Ok(())
}

fn draw_rect(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, thick: i32, c: [u8; 3]) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let (x0, y0, x1, y1) = (x0 as i32, y0 as i32, (x1 as i32).min(w - 1), (y1 as i32).min(h - 1));
    for t in 0..thick {
        for x in x0..=x1 {
            if y0 + t >= 0 && y0 + t < h && x >= 0 && x < w {
                img.put_pixel(x as u32, (y0 + t) as u32, Rgba([c[0], c[1], c[2], 255]));
                img.put_pixel(x as u32, (y1 - t).max(0) as u32, Rgba([c[0], c[1], c[2], 255]));
            }
        }
        for y in y0..=y1 {
            if x0 + t >= 0 && x0 + t < w && y >= 0 && y < h {
                img.put_pixel((x0 + t) as u32, y as u32, Rgba([c[0], c[1], c[2], 255]));
                img.put_pixel((x1 - t).max(0) as u32, y as u32, Rgba([c[0], c[1], c[2], 255]));
            }
        }
    }
}

fn debug_image(st: &SessionState, a: &Analysis) -> RgbaImage {
    let mut dbg = a.img.clone();
    for h in &st.holds {
        let c = st.clusters.get(h.cluster).map(|c| c.rgb).unwrap_or([128, 128, 128]);
        let sel = st.selected == Some(h.cluster) && !st.excluded.contains(&h.id);
        let col = if sel { [255u8, 255, 255] } else { c };
        let t = if sel { 3 } else { 1 };
        draw_rect(&mut dbg, h.bbox[0], h.bbox[1], h.bbox[2], h.bbox[3], t, col);
    }
    dbg
}

// ---------------- agent 子命令 ----------------

fn cmd_analyze(image: &Path, session: Option<&Path>, k: f32, bg_sat: f32, max_dim: u32) -> Result<Value> {
    let mut p = Params::default();
    p.wall_k = k;
    p.bg_sat = bg_sat;
    p.analysis_max_dim = max_dim;
    let dyn_img = image::ImageReader::open(image)
        .with_context(|| format!("打开 {}", image.display()))?
        .with_guessed_format()?
        .decode()
        .with_context(|| format!("解码 {}", image.display()))?;
    let mut a = pipeline::analyze(&dyn_img, &p);
    let timing = a.timing_ms;
    materialize_labels(&mut a);
    let st = SessionState {
        image: image.display().to_string(),
        fw: a.fw,
        fh: a.fh,
        aw: a.aw,
        ah: a.ah,
        wall_v: a.wall_v,
        wall_s: a.wall_s,
        timing_ms: timing,
        params: p,
        selected: None,
        excluded: Vec::new(),
        holds: a.holds.clone(),
        clusters: a.clusters.clone(),
    };
    let dir = session.map(|s| s.to_path_buf()).unwrap_or_else(|| default_session_dir(image));
    save_session(&dir, &st, &a.img)?;
    Ok(summary_json(&dir, &st))
}

fn cmd_edit(
    dir: &Path,
    select: Option<isize>,
    select_name: Option<&str>,
    remove: &[usize],
    add: &[String],
    out: Option<&Path>,
    debug_out: Option<&Path>,
) -> Result<Value> {
    let (mut st, a) = load_session(dir)?;
    let mut actions: Vec<String> = Vec::new();

    // 选线
    if let Some(ci) = select {
        if ci < 0 {
            st.selected = None;
            actions.push("select=none".into());
        } else if (ci as usize) < st.clusters.len() {
            st.selected = Some(ci as usize);
            actions.push(format!("select={ci}({})", cluster::cluster_name(&st.clusters[ci as usize])));
        } else {
            bail!("select {} 超出范围（共 {} 个颜色组）", ci, st.clusters.len());
        }
    }
    if let Some(name) = select_name {
        let nl = name.to_lowercase();
        let best = st
            .clusters
            .iter()
            .enumerate()
            .filter(|(_, c)| cluster::cluster_name(c).to_lowercase().contains(&nl))
            .max_by_key(|(_, c)| c.area);
        match best {
            Some((ci, _)) => {
                st.selected = Some(ci);
                actions.push(format!("select_name={name}->{ci}"));
            }
            None => bail!("未找到匹配 '{name}' 的颜色组"),
        }
    }
    // 排除
    if !remove.is_empty() {
        for &id in remove {
            if st.holds.iter().any(|h| h.id == id) {
                st.excluded.push(id);
            }
        }
        actions.push(format!("remove={}({} 个)", remove.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","), remove.len()));
    }
    // 补测（归一化 bbox -> 分析坐标 -> 区域差分）
    let mut added_total = 0usize;
    for spec in add {
        let v: Result<Vec<f32>> = spec
            .split(',')
            .map(|t| t.trim().parse::<f32>().map_err(|e| anyhow::anyhow!("--add 坐标解析失败 '{spec}': {e}")))
            .collect();
        let v = v?;
        if v.len() != 4 {
            bail!("--add 需要 4 个坐标: x0,y0,x1,y1（0-1 归一化），收到 '{spec}'");
        }
        let (x0, y0, x1, y1) = (v[0], v[1], v[2], v[3]);
        let aw = st.aw as usize;
        let ah = st.ah as usize;
        let px0 = ((x0.clamp(0.0, 1.0) * st.aw as f32) as usize).min(aw - 2);
        let py0 = ((y0.clamp(0.0, 1.0) * st.ah as f32) as usize).min(ah - 2);
        let px1 = (((x1.clamp(0.0, 1.0) * st.aw as f32).ceil() as usize).max(px0 + 2)).min(aw);
        let py1 = (((y1.clamp(0.0, 1.0) * st.ah as f32).ceil() as usize).max(py0 + 2)).min(ah);
        let mut region = vec![0u8; aw * ah];
        for y in py0..py1 {
            for x in px0..px1 {
                region[y * aw + x] = 255;
            }
        }
        let found = pipeline::detect_in_region(&a, &st.params, &region);
        let mut added = 0usize;
        for mut nh in found {
            let r = (nh.area as f32 / std::f32::consts::PI).sqrt();
            let dup = st.holds.iter().any(|e| {
                let re = (e.area as f32 / std::f32::consts::PI).sqrt();
                let dx = e.centroid[0] - nh.centroid[0];
                let dy = e.centroid[1] - nh.centroid[1];
                (dx * dx + dy * dy).sqrt() < (r + re) * 0.75
            });
            if dup {
                continue;
            }
            nh.id = st.holds.len();
            let nh_ref = nh.clone();
            let ci = cluster::assign_hold(&mut st.clusters, &nh_ref, st.params.color_tol_ab, nh.id);
            nh.cluster = ci;
            st.holds.push(nh);
            added += 1;
        }
        added_total += added;
        actions.push(format!("add={spec}(+{added})"));
    }

    save_session(dir, &st, &a.img)?;
    if let Some(o) = out {
        render_session(&st, &a, o)?;
        actions.push(format!("out={}", o.display()));
    }
    if let Some(o) = debug_out {
        debug_image(&st, &a).save(o).with_context(|| format!("保存 {}", o.display()))?;
        actions.push(format!("debug_out={}", o.display()));
    }
    let mut v = summary_json(dir, &st);
    v["actions"] = json!(actions);
    Ok(v)
}

// ---------------- 批处理（原模式） ----------------

fn is_image(p: &Path) -> bool {
    matches!(
        p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
        Some("png" | "jpg" | "jpeg" | "webp" | "bmp")
    )
}

fn collect_images(p: &Path) -> Vec<PathBuf> {
    if p.is_file() {
        return vec![p.to_path_buf()];
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(p)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|e| e.is_file() && is_image(e))
        .collect();
    v.sort();
    v
}

fn main() -> Result<()> {
    let args = Args::parse();

    match &args.cmd {
        Some(Cmd::Analyze { image, session, k, bg_sat, max_dim }) => {
            let v = cmd_analyze(image, session.as_deref(), *k, *bg_sat, *max_dim)?;
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(())
        }
        Some(Cmd::Edit { session, select, select_name, remove, add, out, debug_out }) => {
            let v = cmd_edit(
                session,
                *select,
                select_name.as_deref(),
                remove,
                add,
                out.as_deref(),
                debug_out.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(())
        }
        None => {
            let Some(input) = &args.input else {
                bail!("用法: climb-cli <图片|目录> [批处理] 或 climb-cli analyze|edit [会话模式]，--help 看详情");
            };
            batch_main(&args, input)
        }
    }
}

fn batch_main(args: &Args, input: &Path) -> Result<()> {
    let files = collect_images(input);
    if files.is_empty() {
        bail!("未找到图片: {}", input.display());
    }
    std::fs::create_dir_all(&args.out)?;

    let mut p = Params::default();
    p.wall_k = args.k;
    p.bg_sat = args.bg_sat;
    p.analysis_max_dim = args.max_dim;

    println!(
        "climb-cli: {} 张图片 -> {} | {} | {}",
        files.len(),
        args.out.display(),
        pipeline::detector::YoloDetect::holds_status(),
        pipeline::sam::EdgeSam::status()
    );
    let t_all = std::time::Instant::now();
    let mut ok_route = 0usize;

    for (i, f) in files.iter().enumerate() {
        let stem = f.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let img = image::ImageReader::open(f)
            .with_context(|| format!("打开 {}", f.display()))?
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("解码 {}", f.display()))?;

        let a = pipeline::analyze(&img, &p);

        // 线路选择：--hue 指定颜色角；缺省保留面积最大的彩色组（确定性规则，非自动选线）
        let sel = match args.hue {
            Some(h) => a.clusters.iter().position(|c| pipeline::cluster::hue_match(c, h, 25.0)),
            None => a
                .clusters
                .iter()
                .filter(|c| c.kind == HoldKind::Colored)
                .enumerate()
                .max_by_key(|(_, c)| c.area)
                .map(|(ci, _)| ci),
        };

        println!("\n[{}/{}] {}", i + 1, files.len(), f.display());
        println!(
            "  墙面 V={:.2} S={:.2} | 岩点 {} 个 | 颜色组 {} | 分析 {:.0} ms",
            a.wall_v,
            a.wall_s,
            a.holds.len(),
            a.clusters.len(),
            a.timing_ms
        );
        let n_hold_total = a.holds.iter().map(|h| h.area).sum::<u32>().max(1) as f64;
        let names = pipeline::cluster::disambiguated_names(&a.clusters);
        for (ci, c) in a.clusters.iter().take(10).enumerate() {
            let is_sel = sel == Some(ci);
            let marker = if is_sel { " <= 保留" } else { "" };
            println!(
                "    #{} {} × {:>2} 点  面积{:5.1}%  hue {:>3.0}  ab=({:>3.0},{:>3.0}){}",
                ci,
                names[ci],
                c.holds.len(),
                c.area as f64 / n_hold_total * 100.0,
                c.hue,
                c.ab[0],
                c.ab[1],
                marker
            );
        }

        let mask = render::keep_mask(&a, sel, &HashSet::new(), p.mask_dilate);
        let mask = render::feather(&a, &mask, p.feather);
        let out_img = render::composite_full(&img.to_rgba8(), &a, &mask, p.bg_sat);

        match sel {
            Some(ci) => {
                ok_route += 1;
                let kept = a.clusters[ci].holds.len();
                let colored_px = mask.iter().filter(|&&v| v > 127).count();
                println!(
                    "  输出: 保留 {} 个岩点, 彩色像素 {:.1}% -> {}.route.png",
                    kept,
                    colored_px as f64 / (a.aw * a.ah) as f64 * 100.0,
                    stem
                );
            }
            None => println!("  ⚠ 未找到彩色线路，输出全灰（用 --hue 手动指定）"),
        }

        out_img.save(args.out.join(format!("{}.route.png", stem))).context("保存 route.png")?;

        // 对比图：左原图右结果
        if !args.no_compare {
            let cw = 820u32;
            let ch = (cw as f32 * a.fh as f32 / a.fw as f32).round() as u32;
            let orig = img.resize_exact(cw, ch, image::imageops::FilterType::Triangle).to_rgba8();
            let res = image::imageops::resize(&out_img, cw, ch, image::imageops::FilterType::Triangle);
            let mut canvas = RgbaImage::new(cw * 2 + 6, ch);
            let _ = canvas.copy_from(&orig, 0, 0);
            let _ = canvas.copy_from(&res, cw + 6, 0);
            let cp = args.out.join(format!("{}.compare.jpg", stem));
            let mut fobj = std::fs::File::create(&cp).context("创建 compare.jpg")?;
            let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut fobj, 88);
            enc.encode_image(&DynamicImage::ImageRgba8(canvas)).context("编码 compare.jpg")?;
        }

        // 检测调试图：每个岩点画 bbox，颜色=所属组
        if !args.no_debug {
            let st_fake = SessionState {
                image: f.display().to_string(),
                fw: a.fw,
                fh: a.fh,
                aw: a.aw,
                ah: a.ah,
                wall_v: a.wall_v,
                wall_s: a.wall_s,
                timing_ms: a.timing_ms,
                params: p.clone(),
                selected: sel,
                excluded: Vec::new(),
                holds: a.holds.clone(),
                clusters: a.clusters.clone(),
            };
            let dbg = debug_image(&st_fake, &a);
            dbg.save(args.out.join(format!("{}.debug.png", stem))).context("保存 debug.png")?;
        }
    }

    println!(
        "\n完成: {}/{} 张成功识别线路, 总用时 {:.1}s -> {}",
        ok_route,
        files.len(),
        t_all.elapsed().as_secs_f32(),
        args.out.display()
    );
    Ok(())
}
