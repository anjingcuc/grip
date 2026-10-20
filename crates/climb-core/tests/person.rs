//! 攀岩者保色（person_keep）集成测试：人物掩码并入保色（issue #11）
//! 场景：灰墙 + 一个红色岩点 + 一块"人物"区域（肤色/衣服色块）。
//! 期望：person_keep=1 时人物区域保色、=0 时灰度（现状）；无人物时两态一致。

use climb_core::{hsv, render, Analysis, Cluster, Hold, HoldKind};
use image::{Rgba, RgbaImage};

const W: u32 = 80;
const H: u32 = 60;

fn synth() -> (Analysis, usize) {
    let mut img = RgbaImage::new(W, H);
    for px in img.pixels_mut() {
        *px = Rgba([150, 150, 150, 255]);
    }
    // 红色岩点圆盘 @ (20,30) r=8
    let mut own = Vec::new();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            if (x - 20).pow(2) + (y - 30).pow(2) <= 64 {
                img.put_pixel(x as u32, y as u32, Rgba([200, 45, 40, 255]));
                own.push((y as u32 * W + x as u32) as u32);
            }
        }
    }
    // "攀岩者"：右侧蓝色矩形区域 @ x 50..65, y 15..45（不做成岩点）
    let mut person = Vec::new();
    for y in 15..45u32 {
        for x in 50..65u32 {
            img.put_pixel(x, y, Rgba([60, 90, 200, 255]));
            person.push(y * W + x);
        }
    }
    let (hue, sat, val) = hsv::compute_hsv(img.as_raw());
    let lab = hsv::rgb_to_lab(200, 45, 40);
    let h0 = Hold {
        id: 0,
        label: 0,
        own_pixels: own.clone(),
        area: own.len() as u32,
        bbox: [12, 22, 29, 39],
        centroid: [20.0, 30.0],
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
        person: Some(person),
        wall_v: 0.6,
        wall_s: 0.0,
        timing_ms: 0.0,
    };
    (a, 0)
}

fn at(img: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
    img.get_pixel(x, y).0
}

/// person_keep=1：人物区域保色（蓝）；=0：灰度
#[test]
fn person_kept_colored_or_grayed() {
    let (a, sel) = synth();
    let none: std::collections::HashSet<usize> = Default::default();

    let mut m1 = render::keep_mask(&a, Some(sel), &none, &none, 0);
    render::keep_person(&mut m1, &a, 1.0);
    let out1 = render::composite_preview(&a, &m1, 0.0, None);
    let p1 = at(&out1, 57, 30);
    assert!(p1[2] > 150 && p1[0] < 120, "person_keep=1 人物应保色（蓝），实际 {:?}", p1);

    let mut m0 = render::keep_mask(&a, Some(sel), &none, &none, 0);
    render::keep_person(&mut m0, &a, 0.0);
    let out0 = render::composite_preview(&a, &m0, 0.0, None);
    let p0 = at(&out0, 57, 30);
    let (mx, mn) = (p0[0].max(p0[1]).max(p0[2]), p0[0].min(p0[1]).min(p0[2]));
    assert!(mx - mn < 12, "person_keep=0 人物应灰度，实际 {:?}", p0);

    // 岩点保色不受开关影响
    for out in [&out1, &out0] {
        let h = at(out, 20, 30);
        assert!(h[0] > 150 && h[1] < 100, "岩点应保持红色，实际 {:?}", h);
    }
}

/// 未检测到人物（person=None）：两态输出一致
#[test]
fn no_person_two_modes_identical() {
    let (mut a, sel) = synth();
    a.person = None;
    let none: std::collections::HashSet<usize> = Default::default();
    let mut m1 = render::keep_mask(&a, Some(sel), &none, &none, 0);
    render::keep_person(&mut m1, &a, 1.0);
    let out1 = render::composite_preview(&a, &m1, 0.0, None);
    let mut m0 = render::keep_mask(&a, Some(sel), &none, &none, 0);
    render::keep_person(&mut m0, &a, 0.0);
    let out0 = render::composite_preview(&a, &m0, 0.0, None);
    assert_eq!(out1.as_raw(), out0.as_raw(), "无人物时两态输出应一致");
}
