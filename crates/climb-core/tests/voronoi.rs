//! 测地重构（voronoi_reconstruct）测试：证据确认的子块扩展回掩码连通延伸，
//! 多种子间条带剖分不跨域合并，面积上限防过冲（issue #8）

use climb_core::conn::voronoi_reconstruct;

const W: usize = 40;
const H: usize = 20;

fn rect(m: &mut [u8], x0: usize, y0: usize, x1: usize, y1: usize) {
    for y in y0..y1 {
        for x in x0..x1 {
            m[y * W + x] = 255;
        }
    }
}

fn idxs(x0: usize, y0: usize, x1: usize, y1: usize) -> Vec<u32> {
    let mut v = Vec::new();
    for y in y0..y1 {
        for x in x0..x1 {
            v.push((y * W + x) as u32);
        }
    }
    v
}

/// 单子块经细颈收回连通尖顶（纯色尖顶被证据裁掉的场景）
#[test]
fn single_seed_recovers_tip_through_neck() {
    let mut m = vec![0u8; W * H];
    rect(&mut m, 5, 5, 15, 15); // 主体
    rect(&mut m, 15, 9, 21, 11); // 细颈（2px 高）
    rect(&mut m, 21, 7, 27, 13); // 尖顶
    let seed = idxs(5, 5, 15, 15);
    let out = voronoi_reconstruct(&m, W, H, &[&seed], &[usize::MAX]);
    assert_eq!(out.len(), 1);
    // 尖顶内一点（24,10）应被收回；掩码外一点（30,10）不应出现
    assert!(out[0].contains(&(10 * W as u32 + 24)), "尖顶应经细颈被收回");
    assert!(!out[0].contains(&(10 * W as u32 + 30)), "掩码外像素不应被分配");
    assert_eq!(out[0].len(), 10 * 10 + 6 * 2 + 6 * 6, "应恰好覆盖主体+颈+尖顶");
}

/// 双子块间的条带从中剖分：不跨域合并，不重叠分配
#[test]
fn two_seeds_split_strip_without_merge() {
    let mut m = vec![0u8; W * H];
    rect(&mut m, 4, 6, 10, 14); // A
    rect(&mut m, 26, 6, 32, 14); // B
    rect(&mut m, 10, 9, 26, 11); // 连接条带
    let sa = idxs(4, 6, 10, 14);
    let sb = idxs(26, 6, 32, 14);
    let out = voronoi_reconstruct(&m, W, H, &[&sa, &sb], &[usize::MAX, usize::MAX]);
    let (a, b) = (&out[0], &out[1]);
    // 无重叠
    assert!(!a.iter().any(|i| b.contains(i)), "两种子分配不得重叠");
    // 条带中点靠左归 A、靠右归 B；条带全部被分配（A∪B 覆盖整条带）
    let strip_total = (10..26).map(|x| [9 * W + x, 10 * W + x]).flatten().count();
    assert_eq!(a.len() + b.len(), 6 * 8 + 6 * 8 + strip_total, "A∪B 应覆盖整个掩码");
    assert!(a.contains(&(10 * W as u32 + 12)), "条带左段应归 A");
    assert!(b.contains(&(10 * W as u32 + 23)), "条带右段应归 B");
}

/// 面积上限：小种子不能无界吞并大掩码
#[test]
fn cap_bounds_overgrowth() {
    let mut m = vec![0u8; W * H];
    rect(&mut m, 2, 2, 38, 18); // 大区域
    let seed = idxs(2, 2, 4, 4); // 4px 小种子
    let out = voronoi_reconstruct(&m, W, H, &[&seed], &[6]);
    // 上限 6，允许 BFS 单波 backlog 少量超出（≤7 个邻居）
    assert!(out[0].len() <= 6 + 7, "超出上限过多: {}", out[0].len());
}
