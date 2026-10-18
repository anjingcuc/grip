室内攀岩照片里，同一面墙往往挤着多条颜色各异的线路。本工具把照片拆解成「每条线路一组岩点」，点击任意岩点或颜色即可让该线路保持全彩、其余区域灰度化，用于线路分享、动作复盘与定线沟通。全程本地计算，无网络依赖。

## 演示

https://github.com/user-attachments/assets/1bc89105-815c-40c2-9ccc-42ab5d7b1c70

> 2 分 27 秒功能演示：打开文件夹 → 自动检测 → 点选线路保色 → 圈选补测 → 导出。

## 特性

- **检测不受染色干扰**：镁粉/黑橡胶把岩点染成花色时，同一岩点的彩色部分与污染部分仍连成一个完整轮廓（YOLO11s 语义检测出框 → EdgeSAM 框提示分割 ∩ 局部墙面差分）。
- **相近色分得开**：红/橙/粉/品红等易混色在 Lab 色度平面做凝聚聚类，逐图自适应、不依赖固定色卡。
- **自动 + 精修闭环**：全自动出图；漏检圈选补测、误检一点排除、归组一点改判，所有标注自动保存，重开自动恢复。
- **亮色工具界面**（Tauri 2 + 零构建前端）：扫描动画加载、光标锚定缩放、空格对比原图、批量导出。
- **CLI 双模式**：人用的批处理模式 + 供 AI agent 编程调用的会话模式（JSON 输出）。

## 安装

到 [Releases](https://github.com/anjingcuc/grip/releases) 下载最新构建：

- `GRIP-app-windows-x86_64.zip`：桌面 GUI（解压后运行 `GRIP-app.exe`，模型文件已随包附带）
- `GRIP-cli-windows-x86_64.zip`：命令行工具（模型文件同样附带）

GUI 依赖 Windows 10/11 自带的 WebView2 运行时（绝大多数系统已内置）。

> [!NOTE]
> 模型文件也可从源码自行获取，见 [models/README.md](models/README.md)。

## 使用（GUI）

1. 打开文件夹或单张图片，自动开始分析（约 6-10 秒/张，扫描动画实时显示阶段）。
2. 左栏「颜色线路」列表点击任意颜色组，该线路全部岩点保色，其余灰度。
3. 精修：
   - **圈选补测**（工具 3）：拖圈漏检区域，圈内岩点直接并入当前所选线路；
   - **直接归组**：已选线路时点击任意岩点，立即并入（不做颜色校验）；
   - **移除点**（工具 4）：点击误检岩点排除，再点恢复。
4. `E` 导出全分辨率 PNG；批量导出输出到 `<文件夹>/output/`。

标注（选线/排除/补测/归组）实时写入 `<图片目录>/sessions/<文件名>/`，重开自动恢复。

## 使用（CLI）

### 批处理

```bash
climb-cli .                          # 处理当前目录全部图片 -> ./output
climb-cli photo.jpg --hue 210        # 保留蓝色线路（Lab 色度角：红25 橙58 黄82 绿150 青200 蓝280 紫315 品红355）
```

输出 `<名>.route.png`（成品）、`<名>.compare.jpg`（左右对比）、`<名>.debug.png`（检测框叠加）。无 `--hue` 时保留面积最大的彩色组。

### AI agent 会话模式

stdout 全为可解析 JSON，供有视觉能力的 agent 驱动「识别 → 选线 → 增删 → 导出」全流程：

```bash
# 1. 建会话（分析 + 聚类），返回颜色组与岩点摘要
climb-cli analyze photo.jpg

# 2. 选线（按组 id 或颜色名）
climb-cli edit --session <会话目录> --select-name red

# 3. 排除误检岩点 / 补测漏检区域（归一化 bbox）
climb-cli edit --session <会话目录> --remove 3,5,7
climb-cli edit --session <会话目录> --add 0.42,0.30,0.62,0.50

# 4. 导出成品 + 调试叠加图
climb-cli edit --session <会话目录> --out 成品.png --debug-out 调试.png
```

`analyze` 返回的 JSON 中：

| 字段 | 含义 |
|---|---|
| `clusters[]` | 颜色组：`id` / `name`（含 ①②③ 消歧编号）/ `count` / `area_frac` / `hue`（Lab 色度角）/ `ab`（色度中心）/ `holds`（岩点 id 列表） |
| `holds[]` | 岩点：`id` / `cluster` / `bbox_norm`（0-1 归一化）/ `area` / `kind` / `lasso` |
| `selected` / `excluded` | 当前选择与排除集 |

会话落盘在 `<图片目录>/sessions/<文件名>/`，多次 `edit` 增量修改同一状态，与 GUI 标注互通。

## 从源码构建

```bash
git clone https://github.com/anjingcuc/grip.git
cd line-highlight
# 先放好模型文件（见 models/README.md）
cargo build --release                          # 全工作区
cargo build --release -p climb-cli             # 只编 CLI
cargo build --release -p climb-app             # 只编 GUI（Tauri）
```

要求：Rust 1.95+、Windows（GUI 需 WebView2）。核心管线纯 Rust（image / rayon / tract-onnx），无 C++ 依赖。

## 架构

```
crates/climb-core/   核心算法库（检测 + 分割 + 聚类 + 渲染 + 会话）
crates/climb-cli/    CLI（批处理 + agent 会话模式）
crates/climb-app/    Tauri 2 GUI（src/ 后端 + ui/ 零构建前端）
```

技术选型依据与对照实验记录见 [docs/攀岩岩点识别与线路拆分技术调研报告.md](docs/攀岩岩点识别与线路拆分技术调研报告.md)。面向贡献者与 AI agent 的工程约定见 [AGENTS.md](AGENTS.md)。

## 致谢

- 岩点检测权重：[ChalkGPT](https://github.com/GabiShahmayster/ChalkGPT)（YOLO11s 自训）
- 框提示分割：[EdgeSAM](https://huggingface.co/camenduru/EdgeSAM)（3x ONNX）
- 检测框架背书：调研报告引用的 UCSD / 海德堡 / 斯德哥尔摩大学等公开研究
