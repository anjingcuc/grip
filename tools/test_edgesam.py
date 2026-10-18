# -*- coding: utf-8 -*-
"""EdgeSAM(YOLO框提示) 掩码质量验证（修正 IO：1024 编码 / 简化 decoder）"""
import time
import numpy as np
from PIL import Image, ImageDraw
import onnxruntime as ort

FILES = [
    r"测试\微信图片_20261017080405_99_151.jpg",
    r"测试\微信图片_20261017080529_105_151.jpg",
    r"测试\微信图片_20261017080541_108_151.jpg",
]

yolo = ort.InferenceSession(r"models\chalkgpt_holds.onnx", providers=["CPUExecutionProvider"])
enc = ort.InferenceSession(r"models\edge_sam_3x_encoder.onnx", providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(r"models\edge_sam_3x_decoder.onnx", providers=["CPUExecutionProvider"])
S = 640

def letterbox(im, S=640):
    w, h = im.size
    s = min(S / w, S / h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    c = Image.new("RGB", (S, S), (114, 114, 114))
    c.paste(im.resize((nw, nh)), ((S - nw) // 2, (S - nh) // 2))
    return c, s, (S - nw) // 2, (S - nh) // 2

def yolo_boxes(im):
    W, H = im.size
    lb, sc, dx, dy = letterbox(im)
    a = np.asarray(lb).astype(np.float32) / 255.0
    y = yolo.run(None, {"images": a.transpose(2, 0, 1)[None]})[0][0]
    boxes, scores = y[:4].T, y[4:].T
    cls, conf = scores.argmax(1), scores.max(1)
    dets = []
    for i in range(len(conf)):
        if conf[i] < 0.20:
            continue
        cx, cy, ww, hh = boxes[i]
        x0 = (cx - ww / 2 - dx) / sc; y0 = (cy - hh / 2 - dy) / sc
        x1 = (cx + ww / 2 - dx) / sc; y1 = (cy + hh / 2 - dy) / sc
        dets.append((float(conf[i]), [x0, y0, x1, y1]))
    dets.sort(key=lambda d: -d[0])
    keep = []
    for d in dets:
        ok = True
        for k in keep:
            ix0, iy0 = max(d[1][0], k[1][0]), max(d[1][1], k[1][1])
            ix1, iy1 = min(d[1][2], k[1][2]), min(d[1][3], k[1][3])
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            ua = (d[1][2]-d[1][0])*(d[1][3]-d[1][1]) + (k[1][2]-k[1][0])*(k[1][3]-k[1][1]) - inter
            if ua > 0 and inter / ua > 0.45:
                ok = False; break
        if ok:
            keep.append(d)
    return keep

mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def sam_prep(im):
    """ResizeLongestSide(1024): 等比缩放 + 右下零填充，返回 (CHW float32, 缩放系数)"""
    w, h = im.size
    s = 1024 / max(w, h)
    nw, nh = int(w * s + 0.5), int(h * s + 0.5)
    im2 = im.resize((nw, nh), Image.BILINEAR)
    canvas = np.zeros((1024, 1024, 3), dtype=np.float32)
    a = np.asarray(im2).astype(np.float32) / 255.0
    canvas[:nh, :nw, :] = a
    x = (canvas - mean) / std
    return x.transpose(2, 0, 1)[None].astype(np.float32), s

for f in FILES:
    im = Image.open(f).convert("RGB")
    W, H = im.size
    if max(W, H) > 900:
        s2 = 900 / max(W, H)
        im = im.resize((int(W * s2 + 0.5), int(H * s2 + 0.5)))
        W, H = im.size
    t0 = time.perf_counter()
    boxes = yolo_boxes(im)
    t_yolo = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    x1024, sscale = sam_prep(im)
    emb = enc.run(None, {"image": x1024})[0]
    t_enc = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    masks = []
    for conf, b in boxes[:80]:
        # 框提示：两个角点, label 2/3，坐标变换到 1024 letterbox 空间
        coords = np.array([[[b[0] * sscale, b[1] * sscale], [b[2] * sscale, b[3] * sscale]]], dtype=np.float32)
        labels = np.array([[2.0, 3.0]], dtype=np.float32)
        outs = dec.run(None, {
            "image_embeddings": emb,
            "point_coords": coords,
            "point_labels": labels,
        })
        scores_out, masks_out = outs[0], outs[1]
        m = masks_out[0, 0] > 0
        masks.append((conf, b, m, float(scores_out[0, 0])))
    t_dec = (time.perf_counter() - t0) * 1000
    print(f"\n{f.split(chr(92))[-1]}  {W}x{H}  masks shape={masks[0][2].shape if masks else '-'}")
    print(f"  yolo={t_yolo:.0f}ms enc={t_enc:.0f}ms dec={len(masks)}个/{t_dec:.0f}ms")
    if masks:
        mh, mw = masks[0][2].shape
        # 256 掩码 ↔ 1024 空间 ↔ 原图（/sscale）
        areas = [int(m.sum()) * 16 for _, _, m, _ in masks]  # 1024空间面积
        box_areas = [int((b[2]-b[0])*(b[3]-b[1]) * sscale * sscale) for _, b, _, _ in masks]
        cover = [a_ / max(ba, 1) for a_, ba in zip(areas, box_areas)]
        ious = [i for _, _, _, i in masks]
        print(f"  掩码/框面积比: 中位{np.median(cover):.2f} min{min(cover):.2f} max{max(cover):.2f}")
        print(f"  SAM iou分: 中位{np.median(ious):.2f} min{min(ious):.2f} max{max(ious):.2f}")
        # 目检可视化：掩码并集描边（红色）+ YOLO框（绿色）
        vis = im.copy().convert("RGB")
        dr = ImageDraw.Draw(vis)
        union = np.zeros((mh, mw), dtype=bool)
        for _, _, m, _ in masks:
            union |= m
        er = np.zeros_like(union)
        er[1:, :] |= union[:-1, :]; er[:-1, :] |= union[1:, :]
        er[:, 1:] |= union[:, :-1]; er[:, :-1] |= union[:, 1:]
        edge = union & ~er
        ys, xs = np.where(edge)
        sx = W / (mw * 4 / sscale) if mw else 1
        sy = H / (mh * 4 / sscale) if mh else 1
        for yy, xx in zip(ys.tolist(), xs.tolist()):
            dr.rectangle([xx * 4 / sscale - 1, yy * 4 / sscale - 1, xx * 4 / sscale + 1, yy * 4 / sscale + 1], fill=(255, 0, 0))
        for _, b, _, _ in masks:
            dr.rectangle(b, outline=(0, 255, 0), width=1)
        out_name = "edgesam_" + f.split(chr(92))[-1].replace(".jpg", ".png")
        vis.save(out_name)
        print(f"  可视化 -> {out_name}")
