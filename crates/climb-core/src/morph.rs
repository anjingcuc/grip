//! 二值形态学（开操作去噪、膨胀）与掩码羽化（盒式模糊）

/// 形态学开操作：先腐蚀后膨胀，去除细小噪点
pub fn open(mask: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return mask.to_vec();
    }
    dilate(&erode(mask, w, h, r), w, h, r)
}

pub fn erode(mask: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return mask.to_vec();
    }
    let tmp = hpass(mask, w, h, r, false);
    vpass(&tmp, w, h, r, false)
}

pub fn dilate(mask: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return mask.to_vec();
    }
    let tmp = hpass(mask, w, h, r, true);
    vpass(&tmp, w, h, r, true)
}

/// 水平方向滑窗（二值）：dilate=true 时窗口内非零即置位；否则需全为 1
fn hpass(mask: &[u8], w: usize, h: usize, r: usize, dilate: bool) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if w == 0 {
        return out;
    }
    for y in 0..h {
        let row = y * w;
        let mut cnt = 0usize;
        for x in 0..=r.min(w - 1) {
            cnt += (mask[row + x] != 0) as usize;
        }
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            let window = hi - lo + 1;
            let set = if dilate { cnt > 0 } else { cnt == window };
            out[row + x] = if set { 255 } else { 0 };
            // 滑动到 x+1
            let nlo = (x + 1).saturating_sub(r);
            if nlo > lo {
                cnt -= (mask[row + lo] != 0) as usize;
            }
            let nhi = (x + 1 + r).min(w - 1);
            if nhi > hi {
                cnt += (mask[row + nhi] != 0) as usize;
            }
        }
    }
    out
}

/// 垂直方向滑窗
fn vpass(mask: &[u8], w: usize, h: usize, r: usize, dilate: bool) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    if h == 0 {
        return out;
    }
    for x in 0..w {
        let mut cnt = 0usize;
        for y in 0..=r.min(h - 1) {
            cnt += (mask[y * w + x] != 0) as usize;
        }
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            let window = hi - lo + 1;
            let set = if dilate { cnt > 0 } else { cnt == window };
            out[y * w + x] = if set { 255 } else { 0 };
            let nlo = (y + 1).saturating_sub(r);
            if nlo > lo {
                cnt -= (mask[lo * w + x] != 0) as usize;
            }
            let nhi = (y + 1 + r).min(h - 1);
            if nhi > hi {
                cnt += (mask[nhi * w + x] != 0) as usize;
            }
        }
    }
    out
}

/// 盒式模糊（f32 精度、边缘钳制），用于掩码羽化
pub fn blur(mask: &[u8], w: usize, h: usize, radius: f32) -> Vec<u8> {
    let r = (radius.max(0.5).ceil() as usize).max(1);
    let n = w * h;
    let mut f: Vec<f32> = mask.iter().map(|&v| v as f32).collect();
    let mut tmp = vec![0f32; n];
    // 水平
    for y in 0..h {
        let row = y * w;
        let mut sum = 0f32;
        for x in 0..=r.min(w.saturating_sub(1)) {
            sum += f[row + x.min(w - 1)];
        }
        for x in 0..w {
            let lo = x.saturating_sub(r);
            let hi = (x + r).min(w - 1);
            tmp[row + x] = sum / (hi - lo + 1) as f32;
            let nlo = (x + 1).saturating_sub(r);
            if nlo > lo {
                sum -= f[row + lo];
            }
            let nhi = (x + 1 + r).min(w - 1);
            if nhi > hi {
                sum += f[row + nhi];
            }
        }
    }
    // 垂直
    for x in 0..w {
        let mut sum = 0f32;
        for y in 0..=r.min(h.saturating_sub(1)) {
            sum += tmp[y.min(h - 1) * w + x];
        }
        for y in 0..h {
            let lo = y.saturating_sub(r);
            let hi = (y + r).min(h - 1);
            f[y * w + x] = sum / (hi - lo + 1) as f32;
            let nlo = (y + 1).saturating_sub(r);
            if nlo > lo {
                sum -= tmp[lo * w + x];
            }
            let nhi = (y + 1 + r).min(h - 1);
            if nhi > hi {
                sum += tmp[nhi * w + x];
            }
        }
    }
    f.iter().map(|&v| v.clamp(0.0, 255.0) as u8).collect()
}
