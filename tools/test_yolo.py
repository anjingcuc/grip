# -*- coding: utf-8 -*-
"""测试 yolov8n.onnx（640 letterbox + NMS）在 12 张图的 person 召回（对照 kimi 真值）"""
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

sess = ort.InferenceSession(r"models\yolov8n.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
print("input:", inp.name, inp.shape, inp.type)
print("outputs:", [(o.name, o.shape) for o in sess.get_outputs()])
S = 640

def letterbox(im):
    w, h = im.size
    s = min(S / w, S / h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    im2 = im.resize((nw, nh))
    canvas = Image.new("RGB", (S, S), (114, 114, 114))
    canvas.paste(im2, ((S - nw) // 2, (S - nh) // 2))
    return canvas, s, (S - nw) // 2, (S - nh) // 2

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / max(ua, 1e-9)

def nms(dets, thr=0.45):
    dets = sorted(dets, key=lambda d: -d[0])
    keep = []
    for d in dets:
        if all(iou(d[1], k[1]) < thr for k in keep):
            keep.append(d)
    return keep

good = 0
for fname, gt in GT.items():
    im = Image.open(fname).convert("RGB")
    W, H = im.size
    lb, s, dx, dy = letterbox(im)
    a = np.asarray(lb).astype(np.float32) / 255.0
    x = a.transpose(2, 0, 1)[None]
    outs = sess.run(None, {inp.name: x})
    y = outs[0][0]  # (84, 8400)
    boxes = y[:4].T  # cx,cy,w,h
    scores = y[4:].T  # (8400, 80)
    cls = scores.argmax(1)
    conf = scores.max(1)
    dets = []
    for i in range(len(conf)):
        if cls[i] == 0 and conf[i] > 0.25:  # person
            cx, cy, w, h = boxes[i]
            x0 = (cx - w / 2 - dx) / s / W
            y0 = (cy - h / 2 - dy) / s / H
            x1 = (cx + w / 2 - dx) / s / W
            y1 = (cy + h / 2 - dy) / s / H
            dets.append((float(conf[i]), [x0, y0, x1, y1]))
    dets = nms(dets)
    best_iou, best_s, best_box = 0.0, 0.0, None
    for c, bb in dets:
        v = iou(gt, bb)
        if v > best_iou:
            best_iou, best_s, best_box = v, c, bb
    if best_iou >= 0.5:
        good += 1
    bs = f"{best_s:.2f}" if best_box else "-"
    bb = f"[{best_box[0]:.2f},{best_box[1]:.2f},{best_box[2]:.2f},{best_box[3]:.2f}]" if best_box else "-"
    print(f"{fname:<22} IoU={best_iou:.2f} conf={bs} {bb}  persons={len(dets)}")
print(f"\nIoU>=0.5: {good}/{len(GT)}")
