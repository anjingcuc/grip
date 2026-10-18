# 模型文件

运行需要两个 ONNX 模型（不进 git，体积太大）。把以下文件放到本目录（`models/`），程序会按 `models/` → exe 同级 → exe 上级目录 逐层自动查找：

| 文件 | 大小 | 来源 |
|---|---|---|
| `chalkgpt_holds.onnx` | ~36MB | ChalkGPT 自训 YOLO11s 岩点检测权重 |
| `edge_sam_3x_encoder.onnx` | ~21MB | EdgeSAM 3x 图像编码器 |
| `edge_sam_3x_decoder.onnx` | ~15MB | EdgeSAM 3x 解码器 |

## 获取方式

**EdgeSAM**（直接下载）：

```
https://huggingface.co/camenduru/EdgeSAM/resolve/main/edge_sam_3x_encoder.onnx
https://huggingface.co/camenduru/EdgeSAM/resolve/main/edge_sam_3x_decoder.onnx
```

**ChalkGPT**（下载 PyTorch 权重后导出 ONNX）：

```bash
# 1. 下载 https://cdn.jsdelivr.net/gh/GabiShahmayster/ChalkGPT@main/weights/holds/v1/weights/best.pt
# 2. 导出（需要 python + ultralytics）：
pip install ultralytics
yolo export model=best.pt format=onnx imgsz=640 opset=12 simplify=True
# 3. 重命名为 chalkgpt_holds.onnx 放入本目录
```

> opset 必须是 12：推理用 tract 0.21，更高 opset 支持不全。

缺模型时程序自动回退到经典算法（整图墙面差分），功能可用但精度下降。
