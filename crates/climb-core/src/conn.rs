//! 连通域标记（两遍 + 并查集，8 连通）与分量统计

/// 单个连通分量的统计信息
pub struct CompStats {
    pub label: u32,
    pub area: u32,
    /// bbox: x0, y0, x1, y1（x1/y1 为开区间）
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub sum_x: f64,
    pub sum_y: f64,
    /// 彩色像素（sat>=0.22 且 v>=0.10）计数，用于颜色确认
    pub colored: u32,
    /// 彩色像素的色相直方图（72 桶，每桶 5°）
    pub hue_bins: [u32; 72],
    pub sat_sum: f32,
    pub val_sum: f32,
}

impl CompStats {
    pub fn centroid(&self) -> [f32; 2] {
        let a = self.area.max(1) as f64;
        [(self.sum_x / a) as f32, (self.sum_y / a) as f32]
    }
    pub fn bbox_arr(&self) -> [u32; 4] {
        [self.x0, self.y0, self.x1, self.y1]
    }
    pub fn bw(&self) -> u32 {
        self.x1.saturating_sub(self.x0).max(1)
    }
    pub fn bh(&self) -> u32 {
        self.y1.saturating_sub(self.y0).max(1)
    }
    pub fn fill(&self) -> f32 {
        self.area as f32 / (self.bw() * self.bh()) as f32
    }
    pub fn aspect(&self) -> f32 {
        let (bw, bh) = (self.bw() as f32, self.bh() as f32);
        bw.max(bh) / bw.min(bh).max(1.0)
    }
    pub fn sat_mean(&self) -> f32 {
        self.sat_sum / self.area.max(1) as f32
    }
    pub fn val_mean(&self) -> f32 {
        self.val_sum / self.area.max(1) as f32
    }
    pub fn colored_frac(&self) -> f32 {
        self.colored as f32 / self.area.max(1) as f32
    }
    /// 众数色相与纯度（众数 ±10° 窗口内像素占彩色像素比例）
    pub fn mode_hue(&self) -> (f32, f32) {
        if self.colored == 0 {
            return (0.0, 0.0);
        }
        let mut best = 0usize;
        for i in 1..72 {
            if self.hue_bins[i] > self.hue_bins[best] {
                best = i;
            }
        }
        let mut win = 0u32;
        for d in -2i32..=2 {
            let idx = (best as i32 + d).rem_euclid(72) as usize;
            win += self.hue_bins[idx];
        }
        (best as f32 * 5.0 + 2.5, win as f32 / self.colored as f32)
    }
}

fn find(parent: &mut [u32], x0: u32) -> u32 {
    let mut x = x0;
    while parent[x as usize] != x {
        let px = parent[x as usize];
        parent[x as usize] = parent[px as usize];
        x = parent[x as usize];
    }
    x
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb as usize] = ra;
    }
}

/// 8 连通连通域标记。返回 (labels, 数量)，labels[i]==0 表示背景
pub fn label_components(mask: &[u8], w: usize, h: usize) -> (Vec<u32>, usize) {
    assert_eq!(mask.len(), w * h);
    let mut labels = vec![0u32; w * h];
    let mut parent: Vec<u32> = vec![0];
    let mut next: u32 = 1;

    for y in 0..h {
        let top = y > 0;
        for x in 0..w {
            let i = y * w + x;
            if mask[i] == 0 {
                continue;
            }
            let mut m: u32 = 0;
            if x > 0 && labels[i - 1] != 0 {
                m = labels[i - 1];
            }
            if top {
                if x > 0 && labels[i - w - 1] != 0 {
                    let l = labels[i - w - 1];
                    if m == 0 {
                        m = l
                    } else if l != m {
                        union(&mut parent, m, l);
                    }
                }
                if labels[i - w] != 0 {
                    let l = labels[i - w];
                    if m == 0 {
                        m = l
                    } else if l != m {
                        union(&mut parent, m, l);
                    }
                }
                if x + 1 < w && labels[i - w + 1] != 0 {
                    let l = labels[i - w + 1];
                    if m == 0 {
                        m = l
                    } else if l != m {
                        union(&mut parent, m, l);
                    }
                }
            }
            if m == 0 {
                labels[i] = next;
                parent.push(next);
                next += 1;
            } else {
                labels[i] = m;
            }
        }
    }

    // 压缩为连续编号
    let mut map = vec![0u32; parent.len()];
    let mut cnt = 0usize;
    for l in 1..parent.len() as u32 {
        let r = find(&mut parent, l);
        if map[r as usize] == 0 {
            cnt += 1;
            map[r as usize] = cnt as u32;
        }
    }
    for i in 0..labels.len() {
        if labels[i] != 0 {
            let r = find(&mut parent, labels[i]);
            labels[i] = map[r as usize];
        }
    }
    (labels, cnt)
}

/// 汇总每个分量的统计（含色相直方图）。返回数组按 label-1 索引
pub fn gather_stats(labels: &[u32], hue: &[f32], sat: &[f32], val: &[f32], w: usize, h: usize, cnt: usize) -> Vec<CompStats> {
    let mut out: Vec<CompStats> = (0..cnt)
        .map(|i| CompStats {
            label: i as u32 + 1,
            area: 0,
            x0: u32::MAX,
            y0: u32::MAX,
            x1: 0,
            y1: 0,
            sum_x: 0.0,
            sum_y: 0.0,
            colored: 0,
            hue_bins: [0; 72],
            sat_sum: 0.0,
            val_sum: 0.0,
        })
        .collect();
    let is_colored = |s: f32, v: f32| s >= 0.22 && v >= 0.10;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let l = labels[i];
            if l == 0 {
                continue;
            }
            let s = &mut out[(l - 1) as usize];
            s.area += 1;
            s.sum_x += x as f64;
            s.sum_y += y as f64;
            if (x as u32) < s.x0 {
                s.x0 = x as u32;
            }
            if (y as u32) < s.y0 {
                s.y0 = y as u32;
            }
            if (x as u32) + 1 > s.x1 {
                s.x1 = x as u32 + 1;
            }
            if (y as u32) + 1 > s.y1 {
                s.y1 = y as u32 + 1;
            }
            if is_colored(sat[i], val[i]) {
                s.colored += 1;
                let b = (hue[i] / 5.0).floor().rem_euclid(72.0) as usize;
                s.hue_bins[b] += 1;
            }
            s.sat_sum += sat[i];
            s.val_sum += val[i];
        }
    }
    out
}

/// 多种子测地重构（8 连通 BFS）：在二值掩码内，把每个掩码像素分配给测地距离
/// 最近的种子块（并列先到先得）。用于把"证据 ∩ SAM"确认的子块扩展回 SAM 掩码的
/// 连通延伸部分（被证据裁掉的纯色尖顶/边缘——issue #8），横跨多块的条带在多种子间
/// 从中剖分、不跨域合并。`caps[i]` 为第 i 个种子块最终面积上限（防掩码过冲被无界吞并）。
/// 输入输出均为窗口局部坐标索引；返回第 i 个种子块重构后的像素集合（含种子本身）。
pub fn voronoi_reconstruct(sam_mask: &[u8], w: usize, h: usize, seeds: &[&[u32]], caps: &[usize]) -> Vec<Vec<u32>> {
    assert_eq!(sam_mask.len(), w * h);
    assert_eq!(seeds.len(), caps.len());
    let mut owner = vec![0u32; w * h];
    let mut q: std::collections::VecDeque<(u32, u32)> = std::collections::VecDeque::new();
    let mut counts: Vec<usize> = Vec::with_capacity(seeds.len());
    for (li, seed) in seeds.iter().enumerate() {
        let l = li as u32 + 1;
        let mut c = 0usize;
        for &i in seed.iter() {
            let i = i as usize;
            if i < w * h && sam_mask[i] > 0 && owner[i] == 0 {
                owner[i] = l;
                q.push_back((i as u32, l));
                c += 1;
            }
        }
        counts.push(c);
    }
    while let Some((i, l)) = q.pop_front() {
        let li = (l - 1) as usize;
        if counts[li] >= caps[li] {
            continue; // 该标签达上限，停止向外扩张
        }
        let (x, y) = (i as usize % w, i as usize / w);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if sam_mask[ni] > 0 && owner[ni] == 0 {
                    owner[ni] = l;
                    counts[li] += 1;
                    q.push_back((ni as u32, l));
                }
            }
        }
    }
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); seeds.len()];
    for (i, &l) in owner.iter().enumerate() {
        if l > 0 {
            out[(l - 1) as usize].push(i as u32);
        }
    }
    out
}
