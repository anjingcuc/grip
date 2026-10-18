//! 会话持久化：分析结果 + 用户标注（选线/排除/补测岩点）落盘，CLI 与 Tauri GUI 共用
//!
//! 位置：`<图片目录>/sessions/<文件名>/state.json` + `analysis.png`

use super::{Analysis, Cluster, Hold, Params};
use anyhow::{Context, Result};
use image::RgbaImage;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize, Clone)]
pub struct SessionState {
    pub image: String,
    pub fw: u32,
    pub fh: u32,
    pub aw: u32,
    pub ah: u32,
    pub wall_v: f32,
    pub wall_s: f32,
    pub timing_ms: f32,
    pub params: Params,
    pub selected: Option<usize>,
    pub excluded: Vec<usize>,
    pub holds: Vec<Hold>,
    pub clusters: Vec<Cluster>,
}

/// 图片对应的会话目录
pub fn session_dir_for(image: &Path) -> PathBuf {
    let stem = image.file_stem().unwrap_or_default().to_string_lossy().to_string();
    image
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("sessions")
        .join(stem)
}

/// 该图片是否有已保存的会话
pub fn session_exists(image: &Path) -> bool {
    session_dir_for(image).join("state.json").is_file()
}

/// 标签引用物化为自包含像素（会话不再依赖 labels 图）
pub fn materialize_labels(a: &mut Analysis) {
    for h in a.holds.iter_mut() {
        if h.label > 0 {
            h.own_pixels = (0..a.labels.len())
                .filter(|&i| a.labels[i] == h.label)
                .map(|i| i as u32)
                .collect();
            h.label = 0;
        }
    }
}

pub fn save(dir: &Path, st: &SessionState, analysis_img: &RgbaImage) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    analysis_img.save(dir.join("analysis.png")).context("保存 analysis.png")?;
    std::fs::write(dir.join("state.json"), serde_json::to_vec(st).context("序列化 state.json")?).context("写 state.json")?;
    Ok(())
}

/// 载入会话 -> (状态, 重建的 Analysis)
pub fn load(dir: &Path) -> Result<(SessionState, Analysis)> {
    let st: SessionState =
        serde_json::from_slice(&std::fs::read(dir.join("state.json")).context("读 state.json")?).context("解析 state.json")?;
    let img = image::ImageReader::open(dir.join("analysis.png"))
        .context("打开 analysis.png")?
        .decode()
        .context("解码 analysis.png")?
        .to_rgba8();
    let (hue, sat, val) = super::hsv::compute_hsv(img.as_raw());
    let a = Analysis {
        aw: st.aw,
        ah: st.ah,
        fw: st.fw,
        fh: st.fh,
        labels: Vec::new(),
        holds: st.holds.clone(),
        clusters: st.clusters.clone(),
        img,
        hue,
        sat,
        val,
        wall_v: st.wall_v,
        wall_s: st.wall_s,
        timing_ms: st.timing_ms,
    };
    Ok((st, a))
}
