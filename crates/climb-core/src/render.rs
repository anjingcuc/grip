//! 输出合成：保留线路岩点颜色，其余灰度化（羽化边缘、全分辨率渲染）

use super::{Analysis, HoldKind};
use image::{imageops, RgbaImage};
use rayon::prelude::*;
use std::collections::HashSet;

/// 生成"保留颜色"掩码（分析分辨率）：选中颜色组、未被排除的岩点像素。
/// 逐像素颜色约束（Lab 色度平面）：彩色像素须与簇色度中心 (a*,b*) 距离 ≤20，
/// 低饱和像素（镁粉/胶痕/阴影）直接放行——防止相邻异色岩点串色。
pub fn keep_mask(a: &Analysis, sel: Option<usize>, excluded: &HashSet<usize>, dilate_r: usize) -> Vec<u8> {
    let n = (a.aw * a.ah) as usize;
    let mut mask = vec![0u8; n];
    let Some(sel) = sel else { return mask };
    if sel >= a.clusters.len() {
        return mask;
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
    // 引用 labels 的自动岩点（回退路径）
    let include_labels: HashSet<u32> = a
        .holds
        .iter()
        .filter(|h| h.cluster == sel && !excluded.contains(&h.id) && h.label > 0)
        .map(|h| h.label)
        .collect();
    for (i, &l) in a.labels.iter().enumerate() {
        if l > 0 && include_labels.contains(&l) && ok_px(i) {
            mask[i] = 255;
        }
    }
    // 套索/补测岩点（自包含像素）
    for h in &a.holds {
        if h.cluster == sel && !excluded.contains(&h.id) && h.label == 0 {
            for &i in &h.own_pixels {
                if ok_px(i as usize) {
                    mask[i as usize] = 255;
                }
            }
        }
    }
    if dilate_r > 0 {
        super::morph::dilate(&mask, a.aw as usize, a.ah as usize, dilate_r)
    } else {
        mask
    }
}

/// 掩码羽化（柔和边缘）
pub fn feather(a: &Analysis, mask: &[u8], radius: f32) -> Vec<u8> {
    if radius <= 0.0 {
        return mask.to_vec();
    }
    super::morph::blur(mask, a.aw as usize, a.ah as usize, radius)
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

/// 全分辨率合成：mask 为分析分辨率的羽化掩码，双线性放大到原图尺寸
pub fn composite_full(full: &RgbaImage, a: &Analysis, mask: &[u8], bg_sat: f32) -> RgbaImage {
    let alpha_small = image::GrayImage::from_raw(a.aw, a.ah, mask.to_vec()).expect("mask size mismatch");
    let alpha = imageops::resize(&alpha_small, a.fw, a.fh, imageops::FilterType::Triangle);
    let mut buf: Vec<u8> = full.as_raw().clone();
    buf.par_chunks_exact_mut(4)
        .zip_eq(alpha.as_raw().par_chunks_exact(1))
        .for_each(|(px, al)| {
            blend_px(px, al[0] as f32 / 255.0, bg_sat);
        });
    RgbaImage::from_raw(a.fw, a.fh, buf).expect("buffer size mismatch")
}

/// 分析分辨率预览（GUI 用，速度快）
pub fn composite_preview(a: &Analysis, mask: &[u8], bg_sat: f32) -> RgbaImage {
    let mut buf: Vec<u8> = a.img.as_raw().clone();
    buf.par_chunks_exact_mut(4)
        .zip_eq(mask.par_chunks_exact(1))
        .for_each(|(px, m)| {
            blend_px(px, m[0] as f32 / 255.0, bg_sat);
        });
    RgbaImage::from_raw(a.aw, a.ah, buf).expect("buffer size mismatch")
}
