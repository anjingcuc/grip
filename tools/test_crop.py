# -*- coding: utf-8 -*-
"""测试: (1) 裁剪 GT±50% 后能否检出 person（验证滑窗二阶段方案）(2) yolov8s 对比"""
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

GT_FAIL = {
    "10月13日 (1)(1).png": [0.36, 0.375, 0.56, 0.51],
    "10月13日 (1).png": [0.40, 0.328, 0.57, 0.472],
    "10月14日(1).png": [0.26, 0.352, 0.67, 0.662],
    "10月16日.png": [0.32, 0.32, 0.68, 0.56],
    "mpv-shot0001.jpg": [0.37, 0.355, 0.69, 0.575],
}

def load(model):
    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    return s, s.get_inputs()[0]

def run(sess, inp, im, S=640):
    w, h = im.size
    sc = min(S / w, S / h)
    nw, nh = int(w * sc + 0.5), int(h * sc + 0.5)
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    canvas.paste(im.resize((nw, nh)), ((S - nw) // 2, (S - nh) // 2))
    a = np.asarray(canvas).astype(np.float32) / 255.0
    y = sess.run(None, {inp.name: a.transpose(2, 0, 1)[None]})[0][0]
    boxes, scores = y[:4].T, y[4:].T
    cls, conf = scores.argmax(1), scores.max(1)
    out = []
    for i in range(len(conf)):
        if cls[i] == 0 and conf[i] >= 0.25:
            cx, cy, ww, hh = boxes[i]
            x0 = (cx - ww / 2 - (S - nw) // 2) / sc
            y0 = (cy - hh / 2 - (S - nh) // 2) / sc
            x1 = (cx + ww / 2 - (S - nw) // 2) / sc
            y1 = (cy + hh / 2 - (S - nh) // 2) / sc
            out.append((float(conf[i]), [x0, y0, x1, y1]))
    return out  # 像素坐标

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)

model = sys.argv[1] if len(sys.argv) > 1 else r"models\yolov8n.onnx"
sess, inp = load(model)
print("model:", model)
for fname, gt in GT_FAIL.items():
    im = Image.open(fname).convert("RGB")
    W, H = im.size
    gtp = [gt[0] * W, gt[1] * H, gt[2] * W, gt[3] * H]
    # 裁剪 GT 外扩 60%
    cx, cy = (gtp[0] + gtp[2]) / 2, (gtp[1] + gtp[3]) / 2
    bw, bh = (gtp[2] - gtp[0]) * 2.2, (gtp[3] - gtp[1]) * 2.2
    x0, y0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
    x1, y1 = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
    crop = im.crop((x0, y0, x1, y1))
    dets = run(sess, inp, crop)
    best = max((iou([d[1][0]+x0, d[1][1]+y0, d[1][2]+x0, d[1][3]+y0], gtp), d[0], d[1]) for d in dets) if dets else (0, 0, None)
    b = best[2]
    bs = f"[{(b[0]+x0)/W:.2f},{(b[1]+y0)/H:.2f},{(b[2]+x0)/W:.2f},{(b[3]+y0)/H:.2f}]" if b else "-"
    print(f"{fname:<22} crop_dets={len(dets)} bestIoU={best[0]:.2f} conf={best[1]:.2f} {bs}")
