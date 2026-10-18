//! 回归测试：套索补测（detect_in_region）
//! 背景 bug：region 全图掩码被按裁剪局部坐标索引，套索圈在图中部时读到全 0 → 永远返回空。

use climb_core as pipeline;
use climb_core::Params;
use image::{DynamicImage, Rgba, RgbaImage};

fn synth(w: u32, h: u32) -> RgbaImage {
    // 灰墙 + 中部红圆盘（模拟被漏检的岩点）+ 右上蓝圆盘
    let mut img = RgbaImage::from_pixel(w, h, Rgba([122, 120, 116, 255]));
    let disc = |img: &mut RgbaImage, cx: i32, cy: i32, r: i32, col: [u8; 3]| {
        for y in (cy - r)..(cy + r) {
            for x in (cx - r)..(cx + r) {
                let dx = x - cx;
                let dy = y - cy;
                if dx * dx + dy * dy <= r * r && x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                    img.put_pixel(x as u32, y as u32, Rgba([col[0], col[1], col[2], 255]));
                }
            }
        }
    };
    disc(&mut img, (w as i32) / 2, (h as i32) / 2, 45, [200, 45, 40]);
    disc(&mut img, (w as i32) * 3 / 4, (h as i32) / 4, 30, [40, 70, 200]);
    img
}

#[test]
fn lasso_finds_hold_inside_circled_region() {
    let img = synth(400, 400);
    let a = pipeline::analyze(&DynamicImage::ImageRgba8(img), &Params::default());

    // 套索：圈住图像中央的红色圆盘（远离左上角，确保旧 bug 会读到 0）
    let (w, h) = (a.aw as usize, a.ah as usize);
    let mut region = vec![0u8; w * h];
    let (cx, cy, r) = (w as i32 / 2, h as i32 / 2, 80);
    for y in 0..h {
        for x in 0..w {
            let dx = x as i32 - cx;
            let dy = y as i32 - cy;
            if dx * dx + dy * dy <= r * r {
                region[y * w + x] = 255;
            }
        }
    }
    let found = pipeline::detect_in_region(&a, &Params::default(), &region);
    assert!(
        !found.is_empty(),
        "套索补测应至少在圈内找到一个岩点（旧 bug 返回空）"
    );
    assert!(found.iter().all(|h| h.lasso));
    // 找到的岩点应在圈内部（红盘中心附近）
    assert!(
        found
            .iter()
            .any(|h| (h.centroid[0] as i32 - cx).abs() < 30 && (h.centroid[1] as i32 - cy).abs() < 30),
        "补测岩点质心应靠近红盘中心，实际: {:?}",
        found.iter().map(|h| h.centroid).collect::<Vec<_>>()
    );
}

#[test]
fn lasso_empty_region_returns_nothing() {
    let img = synth(300, 300);
    let a = pipeline::analyze(&DynamicImage::ImageRgba8(img), &Params::default());
    let region = vec![0u8; (a.aw * a.ah) as usize];
    let found = pipeline::detect_in_region(&a, &Params::default(), &region);
    assert!(found.is_empty());
}
