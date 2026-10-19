//! 保色掩码（keep_mask）集成测试：岩点轮廓内部统一保色，颜色约束只在外扩环带生效。
//! 场景：红色岩点右半被阴影/污渍覆盖（色度偏离簇中心 >20）；近邻一个蓝色岩点。
//! 期望（issue #5）：轮廓内的阴影像素照样保色；外扩环带内的蓝色异点像素仍被拒（防串色）。

use climb_core::{hsv, render, Analysis, Cluster, Hold, HoldKind};
use image::{Rgba, RgbaImage};

const W: u32 = 120;
const H: u32 = 120;

fn disc(img: &mut RgbaImage, cx: i32, cy: i32, r: i32, paint: impl Fn(i32, i32) -> [u8; 3]) -> Vec<u32> {
    let mut px = Vec::new();
    for y in (cy - r)..=(cy + r) {
        for x in (cx - r)..=(cx + r) {
            if (x - cx).pow(2) + (y - cy).pow(2) <= r * r && x >= 0 && y >= 0 && (x as u32) < W && (y as u32) < H {
                let c = paint(x, y);
                img.put_pixel(x as u32, y as u32, Rgba([c[0], c[1], c[2], 255]));
                px.push((y as u32 * W + x as u32) as u32);
            }
        }
    }
    px
}

fn hold(id: usize, px: &[u32], cluster: usize, cx: u32, cy: u32, r: u32) -> Hold {
    Hold {
        id,
        label: 0,
        own_pixels: px.to_vec(),
        area: px.len() as u32,
        bbox: [cx - r, cy - r, cx + r + 1, cy + r + 1],
        centroid: [cx as f32, cy as f32],
        hue: 0.0,
        ab: [0.0, 0.0],
        sat: 0.8,
        val: 0.8,
        purity: 1.0,
        kind: HoldKind::Colored,
        cluster,
        lasso: false,
    }
}

fn synth() -> (Analysis, usize) {
    // 灰墙
    let mut img = RgbaImage::new(W, H);
    for px in img.pixels_mut() {
        *px = Rgba([150, 150, 150, 255]);
    }
    // 红色岩点 @ (40,40) r=15：左半正红，右半阴影暗棕（低明度、色度偏离）
    let red_px = disc(&mut img, 40, 40, 15, |x, _| if x < 48 { [200, 45, 40] } else { [110, 50, 30] });
    // 蓝色近邻岩点 @ (63,40) r=6：左缘 x=57，与红点右缘（x=55）仅隔 1px 墙
    let blue_px = disc(&mut img, 63, 40, 6, |_, _| [40, 60, 200]);
    // 构造校验：阴影色与簇红色的 Lab 色度距离确实 >20（否则测试无判别力）
    let lab_red = hsv::rgb_to_lab(200, 45, 40);
    let lab_shadow = hsv::rgb_to_lab(110, 50, 30);
    let d = ((lab_shadow[1] - lab_red[1]).powi(2) + (lab_shadow[2] - lab_red[2]).powi(2)).sqrt();
    assert!(d > 20.0, "阴影色需偏离簇色度中心 >20 才有判别力，实际 {d:.1}");

    let (hue, sat, val) = hsv::compute_hsv(img.as_raw());
    let red = hold(0, &red_px, 0, 40, 40, 15);
    let blue = hold(1, &blue_px, 1, 63, 40, 6);
    let mk_cl = |rgb: [u8; 3], holds: Vec<usize>| {
        let lab = hsv::rgb_to_lab(rgb[0], rgb[1], rgb[2]);
        Cluster { kind: HoldKind::Colored, hue: 0.0, ab: [lab[1], lab[2]], rgb, holds, area: 100 }
    };
    let a = Analysis {
        aw: W,
        ah: H,
        fw: W,
        fh: H,
        labels: Vec::new(),
        holds: vec![red, blue],
        clusters: vec![mk_cl([200, 45, 40], vec![0]), mk_cl([40, 60, 200], vec![1])],
        img,
        hue,
        sat,
        val,
        wall_v: 0.6,
        wall_s: 0.0,
        timing_ms: 0.0,
    };
    (a, 0)
}

fn at(mask: &[u8], x: u32, y: u32) -> u8 {
    mask[(y * W + x) as usize]
}

/// 轮廓内部统一保色：阴影暗棕像素（色度偏离 >20）不再被抠掉
#[test]
fn interior_kept_uniform_despite_shadow() {
    let (a, sel) = synth();
    let none: std::collections::HashSet<usize> = Default::default();
    let mask = render::keep_mask(&a, Some(sel), &none, &none, 2);
    // 阴影区内部点（50,40）：旧行为被颜色约束抠掉，现在必须保色
    assert_eq!(at(&mask, 50, 40), 255, "岩点轮廓内的阴影像素应统一保色");
    assert_eq!(at(&mask, 54, 40), 255, "岩点右缘内侧阴影像素应保色");
    // 左半正红区域照常保色
    assert_eq!(at(&mask, 35, 40), 255);
}

/// 外扩环带仍做颜色约束：环带内的蓝色异点像素被拒，低饱和墙像素放行
#[test]
fn dilated_ring_still_blocks_foreign_color() {
    let (a, sel) = synth();
    let none: std::collections::HashSet<usize> = Default::default();
    let mask = render::keep_mask(&a, Some(sel), &none, &none, 2);
    // 环带内蓝点像素（57,40）：防串色，不保级
    assert_eq!(at(&mask, 57, 40), 0, "外扩环带内的异色（蓝）像素不应保色");
    assert_eq!(at(&mask, 60, 40), 0, "蓝点本体不属于选中线路");
    // 环带内低饱和墙像素（56,40）：放行（抗 halo 的既有行为）
    assert_eq!(at(&mask, 56, 40), 255, "环带内低饱和墙像素应放行");
}
