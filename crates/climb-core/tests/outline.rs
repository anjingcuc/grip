//! 描边高亮（outline）集成测试：选中线路岩点外缘描 accent 橙边（issue #10）
//! 场景：灰墙 + 一个红色圆盘岩点。期望：描边带出现在岩点外侧环带、
//! 内部像素不受影响、强度 0 时输出与现状一致。

use climb_core::{hsv, render, Analysis, Cluster, Hold, HoldKind};
use image::{Rgba, RgbaImage};

const W: u32 = 60;
const H: u32 = 60;

fn synth() -> (Analysis, usize) {
    let mut img = RgbaImage::new(W, H);
    for px in img.pixels_mut() {
        *px = Rgba([150, 150, 150, 255]);
    }
    // 红色岩点圆盘 @ (30,30) r=10
    let mut own = Vec::new();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if (x - 30).pow(2) + (y - 30).pow(2) <= 100 {
                img.put_pixel(x as u32, y as u32, Rgba([200, 45, 40, 255]));
                own.push((y as u32 * W + x as u32) as u32);
            }
        }
    }
    let (hue, sat, val) = hsv::compute_hsv(img.as_raw());
    let lab = hsv::rgb_to_lab(200, 45, 40);
    let h0 = Hold {
        id: 0,
        label: 0,
        own_pixels: own.clone(),
        area: own.len() as u32,
        bbox: [20, 20, 41, 41],
        centroid: [30.0, 30.0],
        hue: 0.0,
        ab: [lab[1], lab[2]],
        sat: 0.8,
        val: 0.8,
        purity: 1.0,
        kind: HoldKind::Colored,
        cluster: 0,
        lasso: false,
    };
    let cl = Cluster { kind: HoldKind::Colored, hue: 0.0, ab: [lab[1], lab[2]], rgb: [200, 45, 40], holds: vec![0], area: h0.area };
    let a = Analysis {
        aw: W,
        ah: H,
        fw: W,
        fh: H,
        labels: Vec::new(),
        holds: vec![h0],
        clusters: vec![cl],
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

fn at(img: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
    img.get_pixel(x, y).0
}

/// 描边带几何：岩点外侧环带 alpha 高、内部与远处为 0
#[test]
fn outline_band_geometry() {
    let (a, sel) = synth();
    let none: std::collections::HashSet<usize> = Default::default();
    let mask = render::keep_mask(&a, Some(sel), &none, &none, 0);
    let band = render::outline_band(&a, &mask);
    let b = |x: u32, y: u32| band[(y * W + x) as usize];
    assert_eq!(b(30, 30), 0, "岩点内部不应有描边");
    assert!(b(41, 30) > 100, "岩点外侧 1px 处应为描边带（实际 {})", b(41, 30));
    assert!(b(43, 30) > 50, "岩点外侧 3px 内仍应有柔边（实际 {})", b(43, 30));
    assert_eq!(b(50, 30), 0, "远离岩点处不应有描边");
}

/// 描边合成：环带像素被染向 accent 橙，内部不变；强度 0 无变化
#[test]
fn outline_colors_band_and_spares_interior() {
    let (a, sel) = synth();
    let none: std::collections::HashSet<usize> = Default::default();
    let mask = render::keep_mask(&a, Some(sel), &none, &none, 0);
    let base = render::composite_preview(&a, &mask, 0.0, None);

    // 强度 0：逐像素一致
    let mut off = base.clone();
    render::apply_outline(&mut off, &render::outline_band(&a, &mask), 0.0);
    assert_eq!(off.as_raw(), base.as_raw(), "outline=0 应与现状一致");

    // 强度 1：环带染橙（R 高 G 中 B 低），内部像素不变
    let mut on = base.clone();
    let band = render::outline_band(&a, &mask);
    render::apply_outline(&mut on, &band, 1.0);
    let edge = at(&on, 41, 30);
    assert!(
        edge[0] > 200 && edge[1] < 140 && edge[2] < 110,
        "描带应呈 accent 橙 #ff5a2c，实际 {:?}",
        edge
    );
    assert_eq!(at(&on, 30, 30), at(&base, 30, 30), "岩点内部像素不应被描边影响");
    assert_eq!(at(&on, 50, 30), at(&base, 50, 30), "远处墙面不应被描边影响");
}
