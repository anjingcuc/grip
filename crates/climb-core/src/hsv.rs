//! HSV / Lab 颜色空间转换与色相工具

/// RGB(0-255) -> HSV (h: 0..360 度, s: 0..1, v: 0..1)
#[inline]
pub fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let d = max - min;
    let v = max;
    let s = if max <= 1e-6 { 0.0 } else { d / max };
    let h = if d <= 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        60.0 * (((g - b) / d) % 6.0)
    } else if (max - g).abs() < 1e-6 {
        60.0 * (2.0 + (b - r) / d)
    } else {
        60.0 * (4.0 + (r - g) / d)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h, s, v)
}

/// HSV -> RGB(0-255)
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [u8; 3] {
    let h = h.rem_euclid(360.0);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    ]
}

/// sRGB(0-255) -> CIE L*a*b*（D65 白点）
#[inline]
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let lin = |c: f32| -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let r = lin(r as f32 / 255.0);
    let g = lin(g as f32 / 255.0);
    let b = lin(b as f32 / 255.0);
    let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883;
    let f = |t: f32| -> f32 {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// 圆形色相距离（度）
#[inline]
pub fn hue_dist(a: f32, b: f32) -> f32 {
    let d = (a - b).abs();
    d.min(360.0 - d)
}

/// Lab 色度角（度）：atan2(b*, a*)，比暗光下漂移的 HSV 色相更符合感知
/// 锚点（鲜艳 sRGB 参考色的 Lab 角）：红~26 橙~58 黄~82 绿~150 青~200 蓝~280 紫~315 品红~355
#[inline]
pub fn ab_angle(a: f32, b: f32) -> f32 {
    b.atan2(a).to_degrees().rem_euclid(360.0)
}

/// Lab 色度角 -> (中文名, 英文名)。边界按上述锚点标定。
pub fn color_name_ab(a: f32, b: f32) -> (&'static str, &'static str) {
    // 低色度（灰/黑白）不该走到这里；防御性返回灰
    let chroma = (a * a + b * b).sqrt();
    if chroma < 6.0 {
        return ("灰", "gray");
    }
    let ang = ab_angle(a, b);
    match ang {
        x if x >= 345.0 || x < 12.0 => ("品红", "pink"),
        x if x < 38.0 => ("红", "red"),
        x if x < 74.0 => ("橙", "orange"),
        x if x < 96.0 => ("黄", "yellow"),
        x if x < 120.0 => ("黄绿", "lime"),
        x if x < 180.0 => ("绿", "green"),
        x if x < 235.0 => ("青", "cyan"),
        x if x < 295.0 => ("蓝", "blue"),
        _ => ("紫", "purple"),
    }
}

/// Lab -> sRGB(0-255)（取固定 L，用于把簇色度中心渲染成色卡）
pub fn lab_to_rgb(l: f32, a: f32, b: f32) -> [u8; 3] {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let finv = |t: f32| -> f32 {
        let t3 = t.powi(3);
        if t3 > 0.008856 {
            t3
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    let x = finv(fx) * 0.95047;
    let y = finv(fy);
    let z = finv(fz) * 1.08883;
    let lin = |c: f32| -> f32 {
        if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        }
    };
    let r = lin(3.2404542 * x - 1.5371385 * y - 0.4985314 * z);
    let g = lin(-0.969266 * x + 1.8760108 * y + 0.0415560 * z);
    let b2 = lin(0.0556434 * x - 0.2040259 * y + 1.0572252 * z);
    [
        (r * 255.0).clamp(0.0, 255.0) as u8,
        (g * 255.0).clamp(0.0, 255.0) as u8,
        (b2 * 255.0).clamp(0.0, 255.0) as u8,
    ]
}

/// 色相 -> (中文名, 英文名)
pub fn color_name(hue: f32) -> (&'static str, &'static str) {
    let h = hue.rem_euclid(360.0);
    match h {
        h if h < 15.0 || h >= 345.0 => ("红", "red"),
        h if h < 40.0 => ("橙", "orange"),
        h if h < 68.0 => ("黄", "yellow"),
        h if h < 95.0 => ("黄绿", "lime"),
        h if h < 165.0 => ("绿", "green"),
        h if h < 200.0 => ("青", "cyan"),
        h if h < 258.0 => ("蓝", "blue"),
        h if h < 295.0 => ("紫", "purple"),
        _ => ("品红", "magenta"),
    }
}

/// 整幅图 HSV（无分类；岩点候选由 Lab 墙面差分负责）
pub fn compute_hsv(rgba: &[u8]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    use rayon::prelude::*;
    let n = rgba.len() / 4;
    let mut hue = vec![0f32; n];
    let mut sat = vec![0f32; n];
    let mut val = vec![0f32; n];
    hue.par_chunks_exact_mut(1)
        .zip_eq(sat.par_chunks_exact_mut(1))
        .zip_eq(val.par_chunks_exact_mut(1))
        .zip_eq(rgba.par_chunks_exact(4))
        .for_each(|(((h, s), v), px)| {
            let (hh, ss, vv) = rgb_to_hsv(px[0], px[1], px[2]);
            h[0] = hh;
            s[0] = ss;
            v[0] = vv;
        });
    (hue, sat, val)
}
