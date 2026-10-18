# -*- coding: utf-8 -*-
"""Analyze climbing photos: size, wall luminance, hue distribution of saturated pixels, skin fraction."""
import sys, glob, os
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

files = []
for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.JPG", "*.PNG"):
    files.extend(glob.glob(pat))
files = sorted(set(files))
files = [f for f in files if not os.path.basename(f).startswith("output")]

for f in files:
    im = Image.open(f)
    w, h = im.size
    im2 = im.convert("RGB")
    if max(w, h) > 1400:
        s = 1400 / max(w, h)
        im2 = im2.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
    a = np.asarray(im2).astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(-1); mn = a.min(-1)
    v = mx
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    # hue
    dr, dg, db = mx - r, mx - g, mx - b
    hue = np.zeros_like(mx)
    m = (mx == r) & (mx > mn); hue[m] = ((g - b)[m] / np.maximum(mx - mn, 1e-6)[m]) % 6
    m = (mx == g) & (mx > mn); hue[m] = (2 + (b - r)[m] / np.maximum(mx - mn, 1e-6)[m])
    m = (mx == b) & (mx > mn); hue[m] = (4 + (r - g)[m] / np.maximum(mx - mn, 1e-6)[m])
    hue = hue * 60.0
    n = a.shape[0] * a.shape[1]
    print("=" * 70)
    print(f"{f}  {w}x{h}  analyzed at {im2.size}")
    print(f"  median V={np.median(v):.3f}  median S={np.median(s):.3f}  (wall est)")
    print(f"  sat>0.25: {(s>0.25).mean()*100:.1f}%   sat>0.45: {(s>0.45).mean()*100:.1f}%")
    print(f"  dark(<0.20V, s<0.25): {((v<0.20)&(s<0.25)).mean()*100:.1f}%   bright(>0.85V, s<0.2): {((v>0.85)&(s<0.2)).mean()*100:.1f}%")
    # hue histogram of saturated pixels (30 bins)
    mask = (s > 0.30) & (v > 0.15)
    if mask.sum() > 50:
        hh = hue[mask]
        hist, edges = np.histogram(hh, bins=24, range=(0, 360))
        tot = hist.sum()
        peaks = []
        for i, c in enumerate(hist):
            if c > tot * 0.04 and c > 200:
                peaks.append(f"{int(edges[i])}-{int(edges[i+1])}deg:{c*100//tot}%")
        print(f"  saturated px: {tot*100//n}%  hue peaks: {peaks}")
    # skin fraction (Kovac-ish)
    skin = (r > 95) & (g > 40) & (b > 20) & (mx - mn > 0.15) & (np.abs(r - g) > 0.15) & (r > g) & (r > b)
    print(f"  skin-ish px: {skin.mean()*100:.2f}%")
