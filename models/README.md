# 模型文件

运行需要两个 ONNX 模型（不进 git，体积太大）。把以下文件放到本目录（`models/`），程序会按 `models/` → exe 同级 → exe 上级目录 逐层自动查找：

| 文件 | 大小 | 来源 |
|---|---|---|
| `chalkgpt_holds.onnx` | ~36MB | ChalkGPT 自训 YOLO11s 岩点检测权重 |
| `edge_sam_3x_encoder.onnx` | ~21MB | EdgeSAM 3x 图像编码器 |
| `edge_sam_3x_decoder.onnx` | ~15MB | EdgeSAM 3x 解码器 |
| `yolo11n_person.onnx` | ~11MB | （可选）COCO 预训练 YOLO11n，人物检测；缺失时攀岩者保色功能静默关闭 |

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

**人物模型**（YOLO11n COCO 权重导出，person = 第 0 类）：

```bash
# 1. 下载 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
# 2. 导出：
yolo export model=yolo11n.pt format=onnx imgsz=640 opset=12 simplify=True
# 3. 重命名为 yolo11n_person.onnx 放入本目录
```

> opset 必须是 12：推理用 tract 0.21，更高 opset 支持不全。

缺模型时程序自动回退到经典算法（整图墙面差分），功能可用但精度下降。缺人物模型只影响"攀岩者保色"开关，其余功能不受影响。
