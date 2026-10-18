# -*- coding: utf-8 -*-
"""ChalkGPT ONNX 岩点模型快速验证：类别含义 / 检测数 / 延迟"""
import time
import numpy as np
from PIL import Image
import onnxruntime as ort

sess = ort.InferenceSession(r"models\chalkgpt_holds.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print("input:", inp.name, inp.shape)
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
S = 640
NC = sess.get_outputs()[0].shape[1] - 4

def letterbox(im):
    w, h = im.size
    s = min(S / w, S / h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    c = Image.new("RGB", (S, S), (114, 114, 114))
    c.paste(im.resize((nw, nh)), ((S - nw) // 2, (S - nh) // 2))
    return c, s, (S - nw) // 2, (S - nh) // 2

def nms(dets, thr=0.45):
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    for d in dets:
        if all(iou(d[1], k[1]) < thr for k in keep):
            keep.append(d)
    return keep

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)

FILES = ["10月11日.png", "10月13日 (2).png", "10月14日(1).png", "10月16日 (2).png", "mpv-shot0001.jpg"]
for f in FILES:
    im = Image.open(f).convert("RGB")
    W, H = im.size
    # 降采样到长边 900（模拟分析分辨率）
    if max(W, H) > 900:
        s2 = 900 / max(W, H)
        im = im.resize((int(W * s2 + 0.5), int(H * s2 + 0.5)))
        W, H = im.size
    lb, sc, dx, dy = letterbox(im)
    a = np.asarray(lb).astype(np.float32) / 255.0
    t0 = time.perf_counter()
    y = sess.run(None, {inp.name: a.transpose(2, 0, 1)[None]})[0][0]
    dt = (time.perf_counter() - t0) * 1000
    boxes, scores = y[:4].T, y[4:].T
    cls, conf = scores.argmax(1), scores.max(1)
    dets = []
    for i in range(len(conf)):
        if conf[i] < 0.25:
            continue
        cx, cy, ww, hh = boxes[i]
        x0 = (cx - ww/2 - dx) / sc; y0 = (cy - hh/2 - dy) / sc
        x1 = (cx + ww/2 - dx) / sc; y1 = (cy + hh/2 - dy) / sc
        dets.append((float(conf[i]), [x0, y0, x1, y1], int(cls[i])))
    kept = nms(dets)
    c0 = sum(1 for d in kept if d[2] == 0)
    c1 = sum(1 for d in kept if d[2] == 1)
    sizes0 = [int(d[1][2]-d[1][0]) for d in kept if d[2] == 0]
    sizes1 = [int(d[1][2]-d[1][0]) for d in kept if d[2] == 1]
    print(f"{f:<20} {dt:5.0f}ms  class0={c0} (宽{min(sizes0) if sizes0 else '-'}~{max(sizes0) if sizes0 else '-'}px)  class1={c1} (宽{min(sizes1) if sizes1 else '-'}~{max(sizes1) if sizes1 else '-'}px)")
    top = sorted(kept, key=lambda d: -(d[1][2]-d[1][0])*(d[1][3]-d[1][1]))[:3]
    for c, b, cl in top:
        print(f"    conf={c:.2f} cls={cl} box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}] {int(b[2]-b[0])}x{int(b[3]-b[1])}px")
