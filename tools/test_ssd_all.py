# -*- coding: utf-8 -*-
"""在全部 12 张上统计 SSD person(=1) 检测召回（对照 kimi 真值）"""
import numpy as np
from PIL import Image
import onnxruntime as ort

GT = {
    "10月11日.png": [0.45, 0.30, 0.87, 0.57],
    "10月13日 (1)(1).png": [0.36, 0.375, 0.56, 0.51],
    "10月13日 (1).png": [0.40, 0.328, 0.57, 0.472],
    "10月13日 (2)(1).png": [0.44, 0.335, 0.78, 0.65],
    "10月13日 (2).png": [0.19, 0.34, 0.69, 0.72],
    "10月14日(1).png": [0.26, 0.352, 0.67, 0.662],
    "10月14日.png": [0.385, 0.305, 0.60, 0.615],
    "10月15日.png": [0.375, 0.358, 0.668, 0.642],
    "10月16日 (1).png": [0.32, 0.33, 0.63, 0.66],
    "10月16日 (2).png": [0.40, 0.31, 0.70, 0.585],
    "10月16日.png": [0.32, 0.32, 0.68, 0.56],
    "mpv-shot0001.jpg": [0.37, 0.355, 0.69, 0.575],
}

sess = ort.InferenceSession(r"models\ssd_mobilenet_v1_10.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
names = [o.name for o in sess.get_outputs()]

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)

print(f"{'图片':<22} {'最好IoU':>7} {'score':>6}  检测(person, thr=0.2)")
good = 0
for fname, gt in GT.items():
    im = Image.open(fname).convert("RGB")
    a = np.asarray(im).astype(np.uint8)
    outs = sess.run(names, {inp.name: a[None]})
    boxes, labels, scores = outs[0][0], outs[1][0], outs[2][0]
    best_iou, best_s = 0.0, 0.0
    for b, l, s in zip(boxes, labels, scores):
        if int(l) != 1 or s < 0.2:
            continue
        bb = [b[1], b[0], b[3], b[2]]  # yx->xy
        v = iou(gt, bb)
        if v > best_iou:
            best_iou, best_s = v, s
    if best_iou >= 0.5:
        good += 1
    print(f"{fname:<22} {best_iou:>7.2f} {best_s:>6.2f}")
print(f"\nIoU>=0.5: {good}/{len(GT)}")
