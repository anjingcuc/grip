//! 墙面净化（Wall Fill）集成测试：手工构造合成 Analysis，驱动公共渲染 API。
//! 场景：水平明暗渐变的灰墙 + 深色"螺丝孔"噪点 + 一个红色岩点。
//! 期望：净化关闭时输出与现状一致；全开时孔洞被抹平、填充色跟随渐变、岩点不被污染。

use climb_core::{hsv, render, Analysis, Cluster, Hold, HoldKind};
use image::{Rgba, RgbaImage};

const W: u32 = 360;
const H: u32 = 360;

fn luma(c: [u8; 4]) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
}

fn luma_at(img: &RgbaImage, x: u32, y: u32) -> f32 {
    luma(img.get_pixel(x, y).0)
}

/// 返回 (分析结果, 螺丝孔中心列表, 岩点中心)
fn synth() -> (Analysis, Vec<(u32, u32)>, (u32, u32)) {
    // 灰墙：左亮右暗（v: 200 -> 80），模拟岩馆顶灯渐变
    let mut img = RgbaImage::new(W, H);
    for y in 0..H {
        for x in 0..W {
            let v = (200.0 - 120.0 * x as f32 / W as f32) as u8;
            img.put_pixel(x, y, Rgba([v, v, v, 255]));
        }
    }
    // 螺丝孔：比墙面暗 60 的 3×3 点
    let dots = vec![(60u32, 80u32), (150, 200), (300, 120), (90, 300), (250, 320)];
    for &(cx, cy) in &dots {
        for dy in 0..3 {
            for dx in 0..3 {
                let (x, y) = (cx + dx, cy + dy);
                let v = (luma_at(&img, x, y) - 60.0).max(0.0) as u8;
                img.put_pixel(x, y, Rgba([v, v, v, 255]));
            }
        }
    }
    // 红色岩点：r=18 圆盘 @ (180, 90)
    let (hx, hy, hr) = (180i32, 90i32, 18i32);
    let mut own = Vec::new();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let (dx, dy) = (x - hx, y - hy);
            if dx * dx + dy * dy <= hr * hr {
                img.put_pixel(x as u32, y as u32, Rgba([200, 45, 40, 255]));
                own.push((y as u32 * W + x as u32) as u32);
            }
        }
    }
    // 假阳性"岩点"：暗灰脏斑圆盘 @ (270, 270)——用户排除后应被当墙面填掉
    let (sx, sy, sr) = (270i32, 270i32, 12i32);
    let mut smudge_px = Vec::new();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let (dx, dy) = (x - sx, y - sy);
            if dx * dx + dy * dy <= sr * sr {
                let v = (luma_at(&img, x as u32, y as u32) - 55.0).max(0.0) as u8;
                img.put_pixel(x as u32, y as u32, Rgba([v, v, v, 255]));
                smudge_px.push((y as u32 * W + x as u32) as u32);
            }
        }
    }
    let (hue, sat, val) = hsv::compute_hsv(img.as_raw());
    let lab = hsv::rgb_to_lab(200, 45, 40);
    let hold = Hold {
        id: 0,
        label: 0,
        own_pixels: own.clone(),
        area: own.len() as u32,
        bbox: [(hx - hr) as u32, (hy - hr) as u32, (hx + hr + 1) as u32, (hy + hr + 1) as u32],
        centroid: [hx as f32, hy as f32],
        hue: 0.0,
        ab: [lab[1], lab[2]],
        sat: 0.8,
        val: 0.8,
        purity: 1.0,
        kind: HoldKind::Colored,
        cluster: 0,
        lasso: false,
    };
    let smudge = Hold {
        id: 1,
        label: 0,
        own_pixels: smudge_px.clone(),
        area: smudge_px.len() as u32,
        bbox: [(sx - sr) as u32, (sy - sr) as u32, (sx + sr + 1) as u32, (sy + sr + 1) as u32],
        centroid: [sx as f32, sy as f32],
        hue: 0.0,
        ab: [0.0, 0.0],
        sat: 0.0,
        val: 0.3,
        purity: 0.0,
        kind: HoldKind::Gray,
        cluster: usize::MAX, // 未分组
        lasso: false,
    };
    let cluster = Cluster {
        kind: HoldKind::Colored,
        hue: 0.0,
        ab: [lab[1], lab[2]],
        rgb: [200, 45, 40],
        holds: vec![0],
        area: hold.area,
    };
    let a = Analysis {
        aw: W,
        ah: H,
        fw: W,
        fh: H,
        labels: Vec::new(),
        holds: vec![hold, smudge],
        clusters: vec![cluster],
        img,
        hue,
        sat,
        val,
        wall_v: 0.5,
        wall_s: 0.0,
        timing_ms: 0.0,
    };
    (a, dots, (hx as u32, hy as u32))
}

/// 净化关闭（无填充层）：未选线时输出 = 原图逐像素亮度（现有灰度行为，回归保险）
#[test]
fn fill_off_matches_plain_grayscale() {
    let (a, ..) = synth();
    let mask = render::keep_mask(&a, None, &Default::default(), &Default::default(), 0);
    let out = render::composite_preview(&a, &mask, 0.0, None);
    for (px, orig) in out.pixels().zip(a.img.pixels()) {
        let expect = luma(orig.0) as u8; // blend_px: bg_sat=0、alpha=0 时像素 = 亮度（截断）
        for ch in 0..3 {
            assert!(
                (px.0[ch] as i32 - expect as i32).abs() <= 1,
                "净化关闭时输出偏离现状灰度: {:?} vs 亮度 {}",
                px.0,
                expect
            );
        }
    }
}

/// 净化全开：孔洞抹平、填充色跟随光照渐变、选中岩点保色不被污染
#[test]
fn fill_flattens_wall_follows_gradient_and_spares_hold() {
    let (a, dots, hold_c) = synth();
    let fill = render::fill_layer(&a, &Default::default(), 1.0).expect("bg_fill=1 应产生填充层");
    let mask = render::keep_mask(&a, Some(0), &Default::default(), &Default::default(), 0);
    let out = render::composite_preview(&a, &mask, 0.0, Some(&fill));

    // 1) 螺丝孔被抹平：孔中心亮度 ≈ 邻近墙面亮度（原来暗 60）
    for &(cx, cy) in &dots {
        let dot_l = luma_at(&out, cx + 1, cy + 1);
        let nb_l = luma_at(&out, cx + 14, cy + 1);
        assert!(
            (dot_l - nb_l).abs() < 12.0,
            "螺丝孔 {:?} 未被抹平: 孔心 {:.0} vs 邻墙 {:.0}",
            (cx, cy),
            dot_l,
            nb_l
        );
    }
    // 2) 填充色跟随渐变：左侧填充墙明显亮于右侧（非全局单色）
    let left = luma_at(&out, 20, 240);
    let right = luma_at(&out, 340, 240);
    assert!(
        left > right + 25.0,
        "填充未跟随光照渐变（疑似全局单色）: 左 {:.0} 右 {:.0}",
        left,
        right
    );
    // 3) 选中岩点保色、不被填充污染：中心仍是红
    let p = out.get_pixel(hold_c.0, hold_c.1).0;
    assert!(
        p[0] > 150 && p[1] < 100 && p[2] < 100,
        "岩点被填充污染: {:?}",
        p
    );
}

/// 填充强度 0 等价于无填充层
#[test]
fn fill_strength_zero_is_none() {
    let (a, ..) = synth();
    assert!(render::fill_layer(&a, &Default::default(), 0.0).is_none());
}

/// 抹除 = 判定误检（这不是岩点）：其区域被填平；且与全局 bg_fill 解耦——
/// 即使净化滑杆关闭（bg_fill=0），抹除点也始终满强度抹平
#[test]
fn fill_covers_erased_false_positive() {
    let (a, ..) = synth();
    let erased: std::collections::HashSet<usize> = [1].into_iter().collect(); // 抹除暗灰脏斑
    let none: std::collections::HashSet<usize> = Default::default();

    // bg_fill=1：脏斑中心（270,270）被填平，亮度 ≈ 邻近墙面
    let fill = render::fill_layer(&a, &erased, 1.0).expect("bg_fill=1 应产生填充层");
    let mask = render::keep_mask(&a, Some(0), &none, &erased, 0);
    let out = render::composite_preview(&a, &mask, 0.0, Some(&fill));
    let spot = luma_at(&out, 270, 270);
    let nb = luma_at(&out, 240, 270);
    assert!(
        (spot - nb).abs() < 12.0,
        "被抹除的误检脏斑应被填平: 斑心 {spot:.0} vs 邻墙 {nb:.0}"
    );

    // bg_fill=0：仍产生填充层（仅覆盖抹除区域），脏斑被满强度抹成墙面场参考色
    let fill = render::fill_layer(&a, &erased, 0.0).expect("有抹除点时 bg_fill=0 也应产生填充层");
    // 满强度：斑心填充 alpha = 255（与滑杆解耦的直接证据）
    let ci = (270 * W + 270) as usize;
    assert_eq!(fill.alpha[ci], 255, "抹除区域应满强度填充");
    let out = render::composite_preview(&a, &mask, 0.0, Some(&fill));
    // 斑心 ≈ 填充参考色的亮度（原来暗 ~55 的脏斑消失）；邻墙未被净化（alpha=0）保持原样
    let fr = &fill.rgb[ci * 3..ci * 3 + 3];
    let expect = luma([fr[0], fr[1], fr[2], 255]);
    let spot = luma_at(&out, 270, 270);
    assert!(
        (spot - expect).abs() < 8.0,
        "bg_fill=0 时抹除点应被抹成墙面场参考色: 斑心 {spot:.0} vs 参考色 {expect:.0}"
    );
    assert!(
        spot > luma_at(&a.img, 270, 270) + 40.0,
        "暗脏斑未被抹掉: 抹后 {spot:.0} vs 原始 {:.0}",
        luma_at(&a.img, 270, 270)
    );
}

/// 排除 = 是岩点但颜色误判/不入线路：保留在岩点并集内受保护，
/// 即使 bg_fill=1 也不被抹平（像素保持原样走灰度化）
#[test]
fn fill_spares_excluded_hold() {
    let (a, ..) = synth();
    let excluded: std::collections::HashSet<usize> = [1].into_iter().collect(); // 排除暗灰脏斑
    let none: std::collections::HashSet<usize> = Default::default();
    let fill = render::fill_layer(&a, &none, 1.0).expect("bg_fill=1 应产生填充层");
    let mask = render::keep_mask(&a, Some(0), &excluded, &none, 0);
    let out = render::composite_preview(&a, &mask, 0.0, Some(&fill));

    // 脏斑中心保持原暗度：比被填亮的邻墙暗得多（原暗 ~55）；若被误抹平则两者接近
    let spot = luma_at(&out, 270, 270);
    let nb = luma_at(&out, 240, 270);
    assert!(
        nb - spot > 25.0,
        "被排除的岩点不应被抹平: 斑心 {spot:.0} vs 邻墙 {nb:.0}"
    );
}
