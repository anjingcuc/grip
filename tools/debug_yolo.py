# -*- coding: utf-8 -*-
"""调试：换权重 + 低阈值 dump 失败图的全部检测"""
import sys
import numpy as np
from PIL import Image
import onnxruntime as ort

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
        "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
        "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
        "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
        "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
        "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
        "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
        "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
        "hair drier","toothbrush"]

GT_FAIL = {
    "10月13日 (1)(1).png": [0.36, 0.375, 0.56, 0.51],
    "10月13日 (1).png": [0.40, 0.328, 0.57, 0.472],
    "10月14日(1).png": [0.26, 0.352, 0.67, 0.662],
    "10月16日.png": [0.32, 0.32, 0.68, 0.56],
    "mpv-shot0001.jpg": [0.37, 0.355, 0.69, 0.575],
}

model = sys.argv[1] if len(sys.argv) > 1 else r"models\yolov8n.onnx"
conf_t = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0]
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

print("model:", model, "conf>=", conf_t)
for fname, gt in GT_FAIL.items():
    im = Image.open(fname).convert("RGB")
    W, H = im.size
    lb, s, dx, dy = letterbox(im)
    a = np.asarray(lb).astype(np.float32) / 255.0
    x = a.transpose(2, 0, 1)[None]
    y = sess.run(None, {inp.name: x})[0][0]
    boxes, scores = y[:4].T, y[4:].T
    cls, conf = scores.argmax(1), scores.max(1)
    rows = []
    for i in range(len(conf)):
        if conf[i] < conf_t:
            continue
        cx, cy, w, h = boxes[i]
        x0 = (cx - w / 2 - dx) / s / W
        y0 = (cy - h / 2 - dy) / s / H
        x1 = (cx + w / 2 - dx) / s / W
        y1 = (cy + h / 2 - dy) / s / H
        rows.append((float(conf[i]), COCO[cls[i]] if cls[i] < len(COCO) else str(cls[i]), [x0, y0, x1, y1]))
    rows.sort(reverse=True, key=lambda r: r[0])
    print(f"\n{fname}  GT={gt}")
    for c, name, bb in rows[:8]:
        print(f"  {name:<12} conf={c:.2f} iou_with_gt={iou(gt, [bb[0], bb[1], bb[2], bb[3]]):.2f} box=[{bb[0]:.2f},{bb[1]:.2f},{bb[2]:.2f},{bb[3]:.2f}]")
