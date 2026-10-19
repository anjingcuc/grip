# AGENTS.md — GRIP（Group Routes In Photos · 攀岩线路高亮）

Rust 工作区：检测攀岩照片中所有岩点 → 按颜色拆分线路 → 选中线路保色、其余灰度。

```
crates/climb-core/    核心算法库（src/lib.rs 主管线 + detector/sam/cluster/conn/hsv/morph/render/session 模块）
crates/climb-cli/     CLI：批处理 + AI agent 会话模式（src/main.rs）
crates/climb-app/     Tauri 2 桌面 GUI（src/main.rs 后端 + ui/ 零构建前端）
crates/climb-core/tests/lasso_detect.rs   套索补测回归测试
models/               ONNX 模型（不进 git，获取方式见 models/README.md）
docs/                 技术调研报告（选型依据）
```

**场景设定**：输入照片**没有攀爬者**。无人物检测、无自动选线——线路由用户点击选择（GUI）或 `--hue` 指定（CLI）。

## 命令

```bash
cargo build --release                          # 全工作区（根目录 target/）
cargo build --release -p climb-cli             # 只编 CLI
cargo build --release -p climb-app             # 只编 Tauri GUI
cargo test --release -p climb-core --test lasso_detect
```

- 改算法后必跑 CLI 全量：完成判据 = 无 panic + cluster 列表合理（无 20+ 点灰/黑巨型簇霸榜）。
- 改套索/补测逻辑后必跑 lasso_detect 测试。
- Tauri GUI 改动后烟测：启动 exe 存活 ~14 秒即过。**UI 视觉验收**：启动时设 `WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222`，再用 `node tools/e2e/shot.js <前缀>`（playwright-core CDP）截屏交 kimi 评审。
- Python 工具（`tools/*.py`）跑之前设 `$env:PYTHONIOENCODING='utf-8'`。

**构建坑**：`cargo update` 会把 kstring 升到 2.0.5（要求 rustc 1.96），锁定 1.95 时执行
`cargo update kstring@2.0.5 --precise 2.0.2`。Cargo.lock 已提交，CI 用 `--locked` 复现。

**Tauri 前端零构建**：`crates/climb-app/ui/` 是原生 HTML/CSS/JS（`window.__TAURI__` 全局桥，无 npm 依赖），改完前端必须重编 exe（asset 嵌入发生在编译期）。`tauri.conf.json` 需要 `icons/icon.ico` 与 tauri `protocol-asset` feature 对应 assetProtocol；`frontendDist` 相对 conf 所在目录（`./ui`）。

## 模型文件（models/，不进 git）

| 文件 | 角色 |
|---|---|
| `chalkgpt_holds.onnx` | 岩点检测（ChalkGPT YOLO11s，2 类：0=岩点 1=大体积） |
| `edge_sam_3x_encoder.onnx` + `edge_sam_3x_decoder.onnx` | EdgeSAM 框提示分割（encoder 输入 1024 letterbox 输出 [1,256,64,64]；decoder 仅收 embedding+points+labels，输出 256×256 logits） |

模型查找用统一的 `climb_core::find_model(name)`：cwd 及 exe 所在目录的**所有上级**逐层找 `models/<name>`。缺 YOLO 回退整图墙面差分；缺 SAM 回退框内墙面差分/椭圆。**回退路径不得删除**。ONNX opset 必须 12——tract 0.21 对高 opset 支持不全。

## 架构（数据流）

```
detector.rs  YOLO 推理（tract，letterbox 640，NMS，conf 0.20）→ 岩点框
             整图 <8 框时 3×3 滑窗二阶段（小点召回）
sam.rs       EdgeSAM：encode() 一次（~1-2s）→ 每框 mask_for_box（rayon 并行解码，单线程慢 3-4 倍）
lib.rs       analyze() 主路：YOLO 框（detect_holds 保留类别：0=岩点 1=大体积）按类别外扩 18%/35%
             → SAM 掩码（触边扩框闭环：掩码触窗口边缘则提示框 +25% 重切，≤3 次解码，掩码增长 <5% 停滞收敛）
             ∩ 非墙证据（局部 Lab 差分 > 0.55k·σ）
             → 连通域拆分（SAM 条带掩码按墙缝断开成逐点 blob，取最大 4 个）
             → 交集为空（岩点≈墙色）时退回完整 SAM 掩码；SAM 不可用/小框（<22px）走墙面差分 blob；最后椭圆兜底
             → hold_from_pixels 颜色确认（众数色相 + colored_frac/purity 门限，黑/白/灰按明度特判）
cluster.rs   Lab 色度平面 (a*,b*) 平均链接凝聚聚类 → 颜色线路（一维色相间隙切分在红/橙/粉卷绕区会被过渡色桥接合并——已验证的失败模式）；命名/色卡/--hue 均用 Lab 色度角
render.rs    keep_mask：岩点轮廓内部统一保色不做颜色校验（issue #5，暗部/污渍不再被抠灰）；
             颜色约束（与簇色度中心 (a*,b*) 距离 ≤20 或低饱和放行——防相邻异点串色）只在外扩环带生效
             → 膨胀/羽化 → 双线性放大回全分辨率 → 灰度混合
session.rs   标注持久化：<图片目录>/sessions/<文件名>/state.json + analysis.png（CLI 与 GUI 共用格式）
```

### 不变量（改代码前先知道）

- **掩码/区域数组的坐标系契约**：`wall_dist_holds_in_crop` 的 `region` 参数按**裁剪窗口局部坐标**索引；`detect_in_region` 的 `region` 参数是**全图掩码**——两者之间必须先裁出子掩码（曾因直接透传导致套索补测永远返回空，回归测试 `lasso_detect.rs`）。
- **标注持久化**：GUI 每次编辑（选线/排除/抹除/圈选/归组/撤销）后自动落盘到 `session.rs` 会话；再次打开图片时 `analyze` 的加载顺序：内存缓存 > 磁盘会话 > 全新分析。`reset_image` 删除磁盘会话。
- **圈选/点击的归组语义**（用户明确指定）：已选线路时，套索圈住或点击的岩点**直接并入所选线路，不做颜色校验**（`lasso_add(force_cluster=)` / `assign_to_cluster`）；未选线路时套索按颜色归组、点击等同选线。别加回颜色校验。
- Tauri 编辑命令统一返回 `EditResult { preview: data_url, summary }`；`lasso_add` 返回 `LassoResult { added, preview, summary }`。前端持有 summary 后框线渲染全部本地进行。
- `Hold.id == holds 数组下标`。undo 快照靠整表 clone，套索补测 push 前必须先赋 `id = holds.len()`。
- 岩点像素双表示：`label > 0` 引用 `Analysis.labels`（仅无 YOLO 回退路径）；`label == 0` 用 `own_pixels`（主路）。`keep_mask` 两条路径都要处理。**会话保存前必须 `materialize_labels`**。
- `Analysis.hue/sat/val`（分析分辨率 HSV）是渲染颜色约束的数据源，删字段会破坏防串色。
- SAM 掩码必须与"非墙证据"求交后再连通域拆分——直接用原始 SAM 掩码会产生横跨多点的条带合并（已验证的失败模式）。
- **颜色判定魔法数联动**：聚类容差 `color_tol_ab`（14，Lab 色度平面欧氏距离，平均链接）；渲染约束 = 像素与簇色度中心距离 ≤20（仅 keep_mask 外扩环带，`render.rs`，低饱和像素放行）；`colored` 门限 sat≥0.22 && v≥0.10（`conn.rs`）；检测置信度 `HOLD_CONF`（0.20）；SAM 起用框 22px（`lib.rs`）；SAM 取样窗口外扩 = 类别感知（岩点 0.18 / 大体积 0.35），触边扩框 +25% ≤3 次解码、掩码增长 <5% 停滞收敛（`lib.rs`，issue #7）。**墙面净化**：`bg_fill`（0=关闭）；填充取样 = 块内低饱和（sat<0.20，同 `build_wall_field` 口径）像素 P90 明度 + 亮部中位数色，稀疏块回退 3×3 邻块→全局；岩点并集外扩 `FILL_DILATE` 3px + 羽化 `FILL_FEATHER` 2px（`render.rs`）。**抹除≠排除（ADR 0002）**：被抹除（erased，误检=非岩点）的点不并入并集，且**始终满强度抹平**（`bg_fill=0` 也抹）；被排除（excluded，颜色误判）的点仍并入并集受保护，只灰度化不抹平。
- **颜色命名/展示色相用 Lab 色度角** `atan2(b*,a*)`（锚点：红26/橙58/黄82/绿150/青200/蓝280/紫315/品红355），不是 HSV 色相——暗光下 HSV 色相漂移会把橙读成红（`color_name_ab`）。CLI `--hue` 用同一角。簇色卡 = `display_rgb(ab)`（低色度提亮到色度 28，仅显示用途）。**同名颜色组用 `disambiguated_names` 追加 ①②③**。
- tract 模型可跨线程共享（rayon par_iter 共用 decoder 已验证）。

### 已定型、不要回头改的决策

- 无人物检测、无自动选线（用户明确移除）。CLI 无 `--hue` 时默认保留面积最大的彩色组——确定性规则，不是"自动选线"。
- 掩码主路 = YOLO 框提示 SAM ∩ 墙面差分（调研报告 §4.3 "检测器自提示"范式的落地），不是端到端分割、不是纯颜色分割。
- 渲染后的边界色争议（橙黄渐变点、暗粉点）属语义模糊，靠 GUI 手动改判，不用阈值强分。

## 性能基准（i9-10900K，900px 分析分辨率）

YOLO ~150ms + SAM encode ~1.5s + 并行解码 ~3-6s + Lab/聚类 ~1s ≈ **6-10 s/张**。若某改动让单张 >15s，先查是否退化了并行解码。

## 视觉验证闭环（改检测/颜色后使用）

本会话主模型无图像输入。视觉核对用 workflow/subagent 指定 `provider: "kimi-coding", model: "k3"`，让它 `read_image` 读 `outputN/*.compare.jpg`（左原图右输出）+ `*.debug.png`（检测框），schema 化返回：覆盖率、掩码完整性（半块/整点）、误保色（串色）、漏检、1-10 分。判据：相比上一版无回归，抽查 ≥3 张。基线：SAM 掩码版均 7 分（整点性 4/4）；Lab 颜色聚类版 8/8/8（红橙粉分离确认）。

选型依据与技术背书：`docs/攀岩岩点识别与线路拆分技术调研报告.md`。

## API 版本备注（编译报错先查这里）

- tract 0.21：输入 `Tensor::from_shape(&[..], vec.as_slice())`；执行 `model.run(tvec!(tensor.into()))`（TValue）；模型类型 `TypedRunnableModel<TypedModel>`；输入 fact 用 `InferenceFact::dt_shape`，类别数在运行时从输出 shape 推（加载期 shape 是符号值）。
- image 0.25：`copy_from` 需要 `use image::GenericImage`；`ImageBuffer` 没有 `as_raw_mut`（克隆 buf 处理后 `from_raw` 重建）。
- Tauri 2：assetProtocol 需要 `tauri` 的 `protocol-asset` feature；`generate_context!` 编译期校验 `frontendDist` 路径存在。

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on `anjingcuc/grip` (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: root `GLOSSARY.md` + `docs/adr/`. See `docs/agents/domain.md`.
