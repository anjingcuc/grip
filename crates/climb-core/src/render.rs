//! 输出合成：保留线路岩点颜色，其余灰度化（羽化边缘、全分辨率渲染）

use super::{Analysis, Hold, HoldKind};
use image::{imageops, RgbaImage};
use rayon::prelude::*;
use std::collections::HashSet;

/// 生成"保留颜色"掩码（分析分辨率）：选中颜色组、未被排除/抹除的岩点像素。
/// 岩点轮廓内部不做颜色校验、统一保色（暗部/污渍/侧光不再被抠成灰黑斑——issue #5）；
/// 但两类"争议像素"仍受颜色约束（Lab 色度 ≤20，低饱和放行）：
///   1) 掩码外扩边缘带——防相邻异色岩点串色；
///   2) 被多个岩点同时覆盖的重叠区（如大体积上附着的异色 screw-on）——独占像素无条件保色，
///      重叠像素须过颜色校验，否则体积掩码里的异色小点会随体积一起保色（#7 扩框后的串色回归）。
pub fn keep_mask(a: &Analysis, sel: Option<usize>, excluded: &HashSet<usize>, erased: &HashSet<usize>, dilate_r: usize) -> Vec<u8> {
    let n = (a.aw * a.ah) as usize;
    let mut core = vec![0u8; n];
    let Some(sel) = sel else { return core };
    if sel >= a.clusters.len() {
        return core;
    }
    let cl = &a.clusters[sel];
    let raw = a.img.as_raw();
    let ok_px = |i: usize| -> bool {
        let (h, s, v) = (a.hue[i], a.sat[i], a.val[i]);
        match cl.kind {
            HoldKind::Colored => {
                if s >= 0.22 && v >= 0.10 {
                    let lab = super::hsv::rgb_to_lab(raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2]);
                    ((lab[1] - cl.ab[0]).powi(2) + (lab[2] - cl.ab[1]).powi(2)).sqrt() <= 20.0
                } else {
                    true // 低饱和（镁粉/胶痕/暗部）随簇保留
                }
            }
            HoldKind::Black => v <= 0.45 || s < 0.22,
            HoldKind::White => v >= 0.55 || s < 0.22,
            HoldKind::Gray => true,
        }
    };
    let visible = |h: &&Hold| !excluded.contains(&h.id) && !erased.contains(&h.id);
    // 引用 labels 的自动岩点（回退路径；labels 每像素唯一，天然无重叠）
    let include_labels: HashSet<u32> = a
        .holds
        .iter()
        .filter(|h| h.cluster == sel && visible(h) && h.label > 0)
        .map(|h| h.label)
        .collect();
    for (i, &l) in a.labels.iter().enumerate() {
        if l > 0 && include_labels.contains(&l) {
            core[i] = 255;
        }
    }
    // 自包含像素路径：先统计每个像素被多少个可见岩点覆盖（重叠区判据）
    let mut claim = vec![0u8; n];
    for h in &a.holds {
        if h.label == 0 && visible(&h) {
            for &i in &h.own_pixels {
                let c = &mut claim[i as usize];
                if *c < 255 {
                    *c += 1;
                }
            }
        }
    }
    // 套索/补测岩点（自包含像素）：独占像素无条件保色，重叠像素过颜色校验
    for h in &a.holds {
        if h.cluster == sel && visible(&h) && h.label == 0 {
            for &i in &h.own_pixels {
                let i = i as usize;
                if claim[i] <= 1 || ok_px(i) {
                    core[i] = 255;
                }
            }
        }
    }
    if dilate_r == 0 {
        return core;
    }
    // 外扩边缘带：逐像素颜色约束（仅环带，不碰轮廓内部）
    let dil = super::morph::dilate(&core, a.aw as usize, a.ah as usize, dilate_r);
    let mut mask = core;
    for i in 0..n {
        if mask[i] == 0 && dil[i] > 0 && ok_px(i) {
            mask[i] = 255;
        }
    }
    mask
}

/// 掩码羽化（柔和边缘）
pub fn feather(a: &Analysis, mask: &[u8], radius: f32) -> Vec<u8> {
    if radius <= 0.0 {
        return mask.to_vec();
    }
    super::morph::blur(mask, a.aw as usize, a.ah as usize, radius)
}

/// 把攀岩者掩码并入保色掩码（issue #11）：person_keep > 0 且检测到人物时，
/// 人物像素随选中线路一起保色（人物与岩点的重叠区取并集）。在羽化前调用。
pub fn keep_person(mask: &mut [u8], a: &Analysis, person_keep: f32) {
    if person_keep > 0.0 {
        if let Some(pp) = &a.person {
            for &i in pp {
                mask[i as usize] = 255;
            }
        }
    }
}

// ---------------- 描边高亮（Outline）：选中线路岩点外缘描 accent 橙边，叠在合成最上层 ----------------

/// accent 橙（与 GUI 设计系统 --accent #ff5a2c 一致）——与岩点颜色无关的强调手段，
/// 解决灰/黑岩点在灰度化墙面上"保色"后不可见的问题（issue #10）
pub const OUTLINE_RGB: [u8; 3] = [255, 90, 44];

/// 描边带宽/柔边（分析分辨率 px）
const OUTLINE_WIDTH: usize = 2;
const OUTLINE_FEATHER: f32 = 1.0;

/// 从羽化后的选中掩码生成描边带 alpha（分析分辨率）：
/// 掩码阈值化 → dilate − 原掩码 = 外向轮廓环带（画在墙面侧，灰底上最醒目）→ 柔边
pub fn outline_band(a: &Analysis, mask: &[u8]) -> Vec<u8> {
    let (w, h) = (a.aw as usize, a.ah as usize);
    let bin: Vec<u8> = mask.iter().map(|&m| if m > 128 { 255 } else { 0 }).collect();
    let dil = super::morph::dilate(&bin, w, h, OUTLINE_WIDTH);
    let band: Vec<u8> = dil.iter().zip(bin.iter()).map(|(&d, &b)| d.saturating_sub(b)).collect();
    feather(a, &band, OUTLINE_FEATHER)
}

/// 分析分辨率描边带双线性放大到全分辨率
pub fn outline_band_full(a: &Analysis, mask: &[u8]) -> Vec<u8> {
    let band = outline_band(a, mask);
    let img = image::GrayImage::from_raw(a.aw, a.ah, band).expect("outline band size mismatch");
    imageops::resize(&img, a.fw, a.fh, imageops::FilterType::Triangle).into_raw()
}

/// 描边染色后处理：band_alpha 必须与 img 同尺寸；strength 0..1
pub fn apply_outline(img: &mut RgbaImage, band_alpha: &[u8], strength: f32) {
    let k = strength.clamp(0.0, 1.0);
    if k <= 0.0 {
        return;
    }
    img.as_mut()
        .par_chunks_exact_mut(4)
        .zip_eq(band_alpha.par_chunks_exact(1))
        .for_each(|(px, b)| {
            let t = (b[0] as f32 / 255.0) * k;
            if t > 0.0 {
                px[0] = (px[0] as f32 + (OUTLINE_RGB[0] as f32 - px[0] as f32) * t) as u8;
                px[1] = (px[1] as f32 + (OUTLINE_RGB[1] as f32 - px[1] as f32) * t) as u8;
                px[2] = (px[2] as f32 + (OUTLINE_RGB[2] as f32 - px[2] as f32) * t) as u8;
            }
        });
}

// ---------------- 墙面净化（Wall Fill）：非岩点区域填墙面场亮部参考色，先于灰度化 ----------------

/// 填充层（分析分辨率）：每像素墙面参考色 + 填充 alpha（已含强度）
pub struct FillLayer {
    pub rgb: Vec<u8>,
    pub alpha: Vec<u8>,
}

/// 填充边界：岩点并集外扩/羽化半径（分析分辨率 px）——外扩吃掉掩码边缘 halo 与镁粉圈
const FILL_DILATE: usize = 3;
const FILL_FEATHER: f32 = 2.0;

/// 按谓词收集岩点并集掩码（label 引用与 own_pixels 两种表示都计入）
fn holds_mask_impl(a: &Analysis, include: impl Fn(&Hold) -> bool) -> Vec<u8> {
    let n = (a.aw * a.ah) as usize;
    let mut m = vec![0u8; n];
    let include_labels: HashSet<u32> = a
        .holds
        .iter()
        .filter(|h| h.label > 0 && include(h))
        .map(|h| h.label)
        .collect();
    if !include_labels.is_empty() {
        for (i, &l) in a.labels.iter().enumerate() {
            if l > 0 && include_labels.contains(&l) {
                m[i] = 255;
            }
        }
    }
    for h in &a.holds {
        if h.label == 0 && include(h) {
            for &i in &h.own_pixels {
                m[i as usize] = 255;
            }
        }
    }
    m
}

/// 岩点并集掩码：抹除（erased）的岩点不并入——抹除的语义是"判定误检，这不是岩点"，
/// 其区域属非岩点区域，净化时填掉。排除（excluded）的岩点**仍并入**——排除只是
/// 颜色误判/不入线路，点还是岩点，受保护不被抹平（走灰度化）。
fn holds_union_mask(a: &Analysis, erased: &HashSet<usize>) -> Vec<u8> {
    holds_mask_impl(a, |h| !erased.contains(&h.id))
}

/// 构造墙面净化填充层。`bg_fill <= 0` 且无抹除点时返回 None（关闭，渲染走原路）。
/// 抹除区域始终满强度抹平（与全局 bg_fill 解耦）：用户对误检点说"这不是岩点"，
/// 期望该区域立即被抹成干净墙面，不受净化滑杆影响。
pub fn fill_layer(a: &Analysis, erased: &HashSet<usize>, bg_fill: f32) -> Option<FillLayer> {
    let k = bg_fill.clamp(0.0, 1.0);
    if k <= 0.0 && erased.is_empty() {
        return None;
    }
    let (w, h) = (a.aw as usize, a.ah as usize);
    let n = w * h;
    let union = holds_union_mask(a, erased);
    let exclude = super::morph::dilate(&union, w, h, FILL_DILATE);
    let field = super::build_fill_field(a.img.as_raw(), &a.sat, &exclude, w, h);

    // 填充 alpha = 墙面掩码（岩点并集外扩区的补集）羽化后 × 强度
    let wall: Vec<u8> = exclude.iter().map(|&e| if e > 0 { 0 } else { 255 }).collect();
    let feathered = feather(a, &wall, FILL_FEATHER);
    let mut alpha: Vec<u8> = feathered.iter().map(|&v| ((v as f32 / 255.0) * k * 255.0).round() as u8).collect();

    // 抹除区域：外扩羽化后满强度（255），压过墙面强度
    if !erased.is_empty() {
        let em = holds_mask_impl(a, |h| erased.contains(&h.id));
        let em = super::morph::dilate(&em, w, h, FILL_DILATE);
        let em = feather(a, &em, FILL_FEATHER);
        for (ap, &e) in alpha.iter_mut().zip(em.iter()) {
            if e > *ap {
                *ap = e;
            }
        }
    }

    // 每像素填充色（墙面场双线性采样 -> RGB）
    let mut rgb = vec![0u8; n * 3];
    for y in 0..h {
        for x in 0..w {
            let (lab, _) = super::wall_ref(&field, x, y);
            let [r, g, b] = super::hsv::lab_to_rgb(lab[0], lab[1], lab[2]);
            let i = (y * w + x) * 3;
            rgb[i] = r;
            rgb[i + 1] = g;
            rgb[i + 2] = b;
        }
    }
    Some(FillLayer { rgb, alpha })
}

#[inline]
fn fill_px(px: &mut [u8], fc: &[u8], t: f32) {
    px[0] = (px[0] as f32 + (fc[0] as f32 - px[0] as f32) * t) as u8;
    px[1] = (px[1] as f32 + (fc[1] as f32 - px[1] as f32) * t) as u8;
    px[2] = (px[2] as f32 + (fc[2] as f32 - px[2] as f32) * t) as u8;
}

#[inline]
fn blend_px(px: &mut [u8], a: f32, bg_sat: f32) {
    let (r, g, b) = (px[0] as f32, px[1] as f32, px[2] as f32);
    let luma = 0.299 * r + 0.587 * g + 0.114 * b;
    // 背景有效灰度：bg_sat>0 时保留一点原色
    let gr = luma + (r - luma) * bg_sat;
    let gg = luma + (g - luma) * bg_sat;
    let gb = luma + (b - luma) * bg_sat;
    px[0] = (gr + (r - gr) * a).clamp(0.0, 255.0) as u8;
    px[1] = (gg + (g - gg) * a).clamp(0.0, 255.0) as u8;
    px[2] = (gb + (b - gb) * a).clamp(0.0, 255.0) as u8;
}

/// 全分辨率合成：mask 为分析分辨率的羽化掩码，双线性放大到原图尺寸；
/// fill 存在时先把非岩点区域混向墙面参考色（净化底图），再走保色/灰度合成
pub fn composite_full(full: &RgbaImage, a: &Analysis, mask: &[u8], bg_sat: f32, fill: Option<&FillLayer>) -> RgbaImage {
    let alpha_small = image::GrayImage::from_raw(a.aw, a.ah, mask.to_vec()).expect("mask size mismatch");
    let alpha = imageops::resize(&alpha_small, a.fw, a.fh, imageops::FilterType::Triangle);
    let mut buf: Vec<u8> = full.as_raw().clone();
    if let Some(f) = fill {
        let frgb_small = image::RgbImage::from_raw(a.aw, a.ah, f.rgb.clone()).expect("fill rgb size mismatch");
        let frgb = imageops::resize(&frgb_small, a.fw, a.fh, imageops::FilterType::Triangle);
        let fa_small = image::GrayImage::from_raw(a.aw, a.ah, f.alpha.clone()).expect("fill alpha size mismatch");
        let fa = imageops::resize(&fa_small, a.fw, a.fh, imageops::FilterType::Triangle);
        buf.par_chunks_exact_mut(4)
            .zip_eq(alpha.as_raw().par_chunks_exact(1))
            .zip_eq(frgb.as_raw().par_chunks_exact(3))
            .zip_eq(fa.as_raw().par_chunks_exact(1))
            .for_each(|(((px, al), fc), fa)| {
                fill_px(px, fc, fa[0] as f32 / 255.0);
                blend_px(px, al[0] as f32 / 255.0, bg_sat);
            });
    } else {
        buf.par_chunks_exact_mut(4)
            .zip_eq(alpha.as_raw().par_chunks_exact(1))
            .for_each(|(px, al)| {
                blend_px(px, al[0] as f32 / 255.0, bg_sat);
            });
    }
    RgbaImage::from_raw(a.fw, a.fh, buf).expect("buffer size mismatch")
}

/// 分析分辨率预览（GUI 用，速度快）；fill 语义同 composite_full
pub fn composite_preview(a: &Analysis, mask: &[u8], bg_sat: f32, fill: Option<&FillLayer>) -> RgbaImage {
    let mut buf: Vec<u8> = a.img.as_raw().clone();
    if let Some(f) = fill {
        buf.par_chunks_exact_mut(4)
            .zip_eq(mask.par_chunks_exact(1))
            .zip_eq(f.rgb.par_chunks_exact(3))
            .zip_eq(f.alpha.par_chunks_exact(1))
            .for_each(|(((px, m), fc), fa)| {
                fill_px(px, fc, fa[0] as f32 / 255.0);
                blend_px(px, m[0] as f32 / 255.0, bg_sat);
            });
    } else {
        buf.par_chunks_exact_mut(4)
            .zip_eq(mask.par_chunks_exact(1))
            .for_each(|(px, m)| {
                blend_px(px, m[0] as f32 / 255.0, bg_sat);
            });
    }
    RgbaImage::from_raw(a.aw, a.ah, buf).expect("buffer size mismatch")
}
