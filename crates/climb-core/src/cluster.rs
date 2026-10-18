//! 岩点颜色聚类 -> 颜色线路
//!
//! 方案（调研报告 §5.2：聚类优先于分类，CIEDE/Lab 距离 + 凝聚聚类）：
//!   彩色岩点在 Lab 色度平面 (a*, b*) 上做**平均链接**凝聚聚类——
//!   红(a↑,b↓) / 橙(a↑,b↑↑) / 粉(a→,b↓↓) 在该平面彼此距离远大于
//!   同一线路内光照造成的漂移，可稳定分开；平均链接按簇质心合并，
//!   对"过渡色桥接链"免疫（一维色相间隙切分的已知失败模式）。
//!   黑/白/灰不走色度聚类，按明度/饱和度特判成组。

use super::hsv::{color_name_ab, hue_dist};
use super::{Cluster, Hold, HoldKind};

struct Agg {
    holds: Vec<usize>,
    /// 面积加权色度中心
    ab: [f32; 2],
    w: f32,
}

#[inline]
fn ab_dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Lab 色度平面平均链接凝聚聚类 + 黑/白/灰特判
pub fn build_clusters(holds: &mut [Hold], tol_ab: f32) -> Vec<Cluster> {
    let mut aggs: Vec<Agg> = Vec::new();
    for (i, h) in holds.iter().enumerate() {
        if h.kind != HoldKind::Colored {
            continue;
        }
        aggs.push(Agg {
            holds: vec![i],
            ab: h.ab,
            w: h.area.max(1) as f32,
        });
    }

    // 平均链接凝聚：反复合并质心距离最近的簇对，直至最近距离 >= tol_ab
    loop {
        if aggs.len() <= 1 {
            break;
        }
        let mut best: Option<(usize, usize, f32)> = None;
        for i in 0..aggs.len() {
            for j in (i + 1)..aggs.len() {
                let d = ab_dist(aggs[i].ab, aggs[j].ab);
                if best.map_or(true, |(_, _, bd)| d < bd) {
                    best = Some((i, j, d));
                }
            }
        }
        let Some((i, j, d)) = best else { break };
        if d >= tol_ab {
            break;
        }
        let mut bj = aggs.swap_remove(j);
        let bi = &mut aggs[i];
        let tw = bi.w + bj.w;
        bi.ab = [
            (bi.ab[0] * bi.w + bj.ab[0] * bj.w) / tw,
            (bi.ab[1] * bi.w + bj.ab[1] * bj.w) / tw,
        ];
        bi.w = tw;
        bi.holds.append(&mut bj.holds);
    }

    let mut clusters: Vec<Cluster> = aggs
        .into_iter()
        .map(|ag| {
            // 展示色相 = Lab 色度角（命名/色卡/--hue 一致）；色卡 = 色度中心的 Lab 反变换
            let hue = super::hsv::ab_angle(ag.ab[0], ag.ab[1]);
            let rgb = display_rgb(ag.ab);
            let area: u32 = ag.holds.iter().map(|&i| holds[i].area).sum();
            Cluster {
                kind: HoldKind::Colored,
                hue,
                ab: ag.ab,
                rgb,
                holds: ag.holds,
                area,
            }
        })
        .collect();

    // 黑 / 白 / 灰：不走色度聚类
    for (kind, rgb) in [
        (HoldKind::Black, [46u8, 46, 52]),
        (HoldKind::White, [235u8, 235, 238]),
        (HoldKind::Gray, [150u8, 150, 155]),
    ] {
        let ids: Vec<usize> = (0..holds.len()).filter(|&i| holds[i].kind == kind).collect();
        if ids.is_empty() {
            continue;
        }
        let area: u32 = ids.iter().map(|&i| holds[i].area).sum();
        clusters.push(Cluster {
            kind,
            hue: -1.0,
            ab: [0.0, 0.0],
            rgb,
            holds: ids,
            area,
        });
    }

    // 按总面积降序
    clusters.sort_by(|a, b| b.area.cmp(&a.area));
    for (ci, c) in clusters.iter().enumerate() {
        for &hi in &c.holds {
            holds[hi].cluster = ci;
        }
    }
    clusters
}

/// 给（套索补测的）新岩点分配颜色组，返回组索引；色度距离过远时新建组
pub fn assign_hold(clusters: &mut Vec<Cluster>, hold: &Hold, tol_ab: f32, hold_id: usize) -> usize {
    if hold.kind != HoldKind::Colored {
        for (ci, c) in clusters.iter_mut().enumerate() {
            if c.kind == hold.kind {
                c.holds.push(hold_id);
                c.area += hold.area;
                return ci;
            }
        }
        let rgb = match hold.kind {
            HoldKind::Black => [46u8, 46, 52],
            HoldKind::White => [235u8, 235, 238],
            HoldKind::Gray => [150u8, 150, 155],
            HoldKind::Colored => unreachable!(),
        };
        clusters.push(Cluster {
            kind: hold.kind,
            hue: -1.0,
            ab: [0.0, 0.0],
            rgb,
            holds: vec![hold_id],
            area: hold.area,
        });
        return clusters.len() - 1;
    }
    let mut best: Option<(usize, f32)> = None;
    for (ci, c) in clusters.iter().enumerate() {
        if c.kind != HoldKind::Colored {
            continue;
        }
        let d = ab_dist(c.ab, hold.ab);
        if best.map_or(true, |(_, bd)| d < bd) {
            best = Some((ci, d));
        }
    }
    if let Some((ci, d)) = best {
        if d <= tol_ab * 1.5 {
            // 重算簇质心、展示色相与色卡
            let c = &mut clusters[ci];
            let tw = c.area as f32 + hold.area as f32;
            c.ab = [
                (c.ab[0] * c.area as f32 + hold.ab[0] * hold.area as f32) / tw,
                (c.ab[1] * c.area as f32 + hold.ab[1] * hold.area as f32) / tw,
            ];
            c.hue = super::hsv::ab_angle(c.ab[0], c.ab[1]);
            c.rgb = display_rgb(c.ab);
            c.holds.push(hold_id);
            c.area += hold.area;
            return ci;
        }
    }
    // 新建颜色组
    let rgb = display_rgb(hold.ab);
    clusters.push(Cluster {
        kind: HoldKind::Colored,
        hue: super::hsv::ab_angle(hold.ab[0], hold.ab[1]),
        ab: hold.ab,
        rgb,
        holds: vec![hold_id],
        area: hold.area,
    });
    clusters.len() - 1
}

/// 颜色组显示名（按 Lab 色度角命名）
pub fn cluster_name(c: &Cluster) -> String {
    match c.kind {
        HoldKind::Colored => {
            let (zh, en) = color_name_ab(c.ab[0], c.ab[1]);
            format!("{} {}", zh, en)
        }
        HoldKind::Black => "黑 black".into(),
        HoldKind::White => "白 white".into(),
        HoldKind::Gray => "灰 gray".into(),
    }
}

/// 色相匹配（CLI --hue 用）
pub fn hue_match(c: &Cluster, hue: f32, window: f32) -> bool {
    c.kind == HoldKind::Colored && hue_dist(c.hue, hue) <= window
}

/// 低色度簇的展示色卡提亮：Lab 色度不足 28 时按比例放大（保持色相族，仅显示用途）
pub fn display_rgb(ab: [f32; 2]) -> [u8; 3] {
    let c = (ab[0] * ab[0] + ab[1] * ab[1]).sqrt();
    if c < 1.0 {
        return [150, 150, 155];
    }
    let scale = if c < 28.0 { 28.0 / c } else { 1.0 };
    super::hsv::lab_to_rgb(65.0, ab[0] * scale, ab[1] * scale)
}

/// 同名组消歧：同名的按面积顺序追加 ①②③…
pub fn disambiguated_names(clusters: &[Cluster]) -> Vec<String> {
    use std::collections::HashMap;
    let base: Vec<String> = clusters.iter().map(cluster_name).collect();
    let mut totals: HashMap<&str, usize> = HashMap::new();
    for n in &base {
        *totals.entry(n.as_str()).or_insert(0) += 1;
    }
    let mut seen: HashMap<&str, usize> = HashMap::new();
    const CIR: [&str; 10] = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"];
    base.iter()
        .map(|n| {
            if totals[n.as_str()] <= 1 {
                return n.clone();
            }
            let k = seen.entry(n.as_str()).or_insert(0);
            *k += 1;
            let tag = CIR.get(*k - 1).copied().unwrap_or("•");
            format!("{n} {tag}")
        })
        .collect()
}
