# -*- coding: utf-8 -*-
"""验证 ssd_mobilenet_v1_10.onnx 的输入输出格式与人检效果（kimi 真值对照）"""
import sys, json
import numpy as np
from PIL import Image

MODEL = r"models\ssd_mobilenet_v1_10.onnx"
# kimi 提供的攀爬者真值 bbox [x0,y0,x1,y1] (0-1)
GT = {
    "10月11日.png": [0.45, 0.30, 0.87, 0.57],
    "10月14日.png": [0.385, 0.305, 0.60, 0.615],
    "10月16日 (2).png": [0.40, 0.31, 0.70, 0.585],
    "mpv-shot0001.jpg": [0.37, 0.355, 0.69, 0.575],
}

import onnxruntime as ort
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
print("=== inputs ===")
for i in sess.get_inputs():
    print(" ", i.name, i.shape, i.type)
print("=== outputs ===")
for o in sess.get_outputs():
    print(" ", o.name, o.shape, o.type)

inp = sess.get_inputs()[0]
out_names = [o.name for o in sess.get_outputs()]

def iou(a, b):
    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / max(ua, 1e-9)

for fname, gt in GT.items():
    im = Image.open(fname).convert("RGB")
    W, H = im.size
    # 动态分辨率，喂原图即可（uint8, NHWC, 0-255）；过大则缩到长边 800 提速
    if max(W, H) > 800:
        s = 800 / max(W, H)
        im = im.resize((int(W * s), int(H * s)))
    a = np.asarray(im).astype(np.uint8)
    feeds = {inp.name: a[None]}
    try:
        outs = sess.run(out_names, feeds)
    except Exception as e:
        print(fname, "run failed:", e); continue
    boxes = outs[0][0]; labels = outs[1][0]; scores = outs[2][0]
    # 打印所有 score>0.4 的检测
    found = []
    for b, l, s in zip(boxes, labels, scores):
        if s < 0.4:
            continue
        found.append((int(l), float(s), [float(x) for x in b]))
    # person 类聚类
    persons = [f for f in found if f[0] == max(set([x[0] for x in found]), default=-1)]
    best_iou, best = 0.0, None
    for (l, s, b) in found:
        # b 顺序未知，先按 [y0,x0,y1,x1] 试
        bb = [b[1], b[0], b[3], b[2]]
        v = iou(gt, bb)
        if v > best_iou:
            best_iou, best = v, (l, s, bb, b)
    print(f"\n{fname}: 检测数(score>0.4)={len(found)}")
    for l, s, b in found[:10]:
        print(f"  class={l} score={s:.2f} box={b}")
    if best:
        print(f"  与真值最匹配: class={best[0]} score={best[1]:.2f} iou={best_iou:.2f} box={best[2]} (原始顺序 {best[3]})")
