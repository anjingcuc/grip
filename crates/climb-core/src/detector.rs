//! 岩点检测器（ChalkGPT YOLO11s，ONNX，tract 纯 Rust 推理）

use image::RgbaImage;
use std::path::PathBuf;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

const SIZE: usize = 640;
const NMS_IOU: f32 = 0.45;
/// 岩点检测置信度
const HOLD_CONF: f32 = 0.20;

/// 通用 YOLO detect 模型（输出 [1, 4+nc, 8400]）
pub struct YoloDetect {
    model: TypedRunnableModel<TypedModel>,
    num_classes: usize,
}

impl YoloDetect {
    fn load_default(candidates: &[PathBuf]) -> anyhow::Result<YoloDetect> {
        let path = candidates.iter().find(|p| p.is_file()).ok_or_else(|| anyhow::anyhow!("model not found"))?;
        let mut model = tract_onnx::onnx().model_for_path(path)?;
        let fact = InferenceFact::dt_shape(
            f32::datum_type(),
            &[1.to_dim(), 3.to_dim(), (SIZE as i64).to_dim(), (SIZE as i64).to_dim()],
        );
        model.set_input_fact(0, fact)?;
        // 类别数在运行时从输出 shape 推断（ch - 4）
        let model = model.into_typed()?.into_optimized()?.into_runnable()?;
        Ok(YoloDetect { model, num_classes: 1 })
    }

    /// 对图推理，返回所有 (conf, class, [x0,y0,x1,y1])（输入图坐标，px），已做 NMS
    pub fn detect(&self, img: &RgbaImage, conf_min: f32, class_filter: Option<&[usize]>) -> Vec<(f32, usize, [f32; 4])> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w == 0 || h == 0 {
            return Vec::new();
        }
        let scale = (SIZE as f32 / w as f32).min(SIZE as f32 / h as f32);
        let nw = (w as f32 * scale).round() as usize;
        let nh = (h as f32 * scale).round() as usize;
        let dx = (SIZE - nw) / 2;
        let dy = (SIZE - nh) / 2;
        let raw = img.as_raw();
        let mut input = vec![0f32; 3 * SIZE * SIZE];
        for y in 0..SIZE {
            for x in 0..SIZE {
                let (r, g, b) = if x >= dx && x < dx + nw && y >= dy && y < dy + nh {
                    let sx = (((x - dx) as f32 / scale) as usize).min(w - 1);
                    let sy = (((y - dy) as f32 / scale) as usize).min(h - 1);
                    let p = (sy * w + sx) * 4;
                    (raw[p] as f32, raw[p + 1] as f32, raw[p + 2] as f32)
                } else {
                    (114.0, 114.0, 114.0)
                };
                input[0 * SIZE * SIZE + y * SIZE + x] = r / 255.0;
                input[1 * SIZE * SIZE + y * SIZE + x] = g / 255.0;
                input[2 * SIZE * SIZE + y * SIZE + x] = b / 255.0;
            }
        }
        let tensor = match Tensor::from_shape(&[1, 3, SIZE, SIZE], input.as_slice()) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };
        let outputs = match self.model.run(tvec!(tensor.into())) {
            Ok(o) => o,
            Err(_) => return Vec::new(),
        };
        let view = match outputs[0].to_array_view::<f32>() {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let (_b, ch, anchors) = (view.shape()[0], view.shape()[1], view.shape()[2]);
        if ch < 5 || anchors == 0 {
            return Vec::new();
        }
        let nc = (ch - 4).min(self.num_classes.max(ch - 4));
        let mut dets: Vec<(f32, usize, [f32; 4])> = Vec::new();
        for a in 0..anchors {
            // 每类分数取最大
            let mut best_c = 0usize;
            let mut best_s = f32::MIN;
            for c in 0..nc {
                let s = view[[0, 4 + c, a]];
                if s > best_s {
                    best_s = s;
                    best_c = c;
                }
            }
            if best_s < conf_min {
                continue;
            }
            if let Some(cf) = class_filter {
                if !cf.contains(&best_c) {
                    continue;
                }
            }
            let cx = view[[0, 0, a]];
            let cy = view[[0, 1, a]];
            let bw = view[[0, 2, a]];
            let bh = view[[0, 3, a]];
            let x0 = (cx - bw / 2.0 - dx as f32) / scale;
            let y0 = (cy - bh / 2.0 - dy as f32) / scale;
            let x1 = (cx + bw / 2.0 - dx as f32) / scale;
            let y1 = (cy + bh / 2.0 - dy as f32) / scale;
            dets.push((best_s, best_c, [x0, y0, x1, y1]));
        }
        nms_multi(dets)
    }
}

// ---------------- 岩点检测器（ChalkGPT YOLO11s） ----------------

static HOLDS: OnceLock<Result<YoloDetect, String>> = OnceLock::new();

fn holds_candidates() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    for name in ["chalkgpt_holds.onnx", "holds.onnx"] {
        if let Some(p) = super::find_model(name) {
            v.push(p);
        }
    }
    v
}

impl YoloDetect {
    /// 岩点检测全局单例
    pub fn holds() -> Option<&'static YoloDetect> {
        HOLDS
            .get_or_init(|| YoloDetect::load_default(&holds_candidates()).map_err(|e| e.to_string()))
            .as_ref()
            .ok()
    }

    pub fn holds_status() -> &'static str {
        match HOLDS.get_or_init(|| YoloDetect::load_default(&holds_candidates()).map_err(|e| e.to_string())) {
            Ok(_) => "YOLO11s 岩点模型已加载",
            Err(e) => {
                let short: String = e.to_string().chars().take(60).collect();
                Box::leak(format!("岩点模型未启用: {short}").into_boxed_str())
            }
        }
    }

    /// 检测岩点框（整图 + 低置信时 3×3 滑窗二阶段），返回 (conf, class, box) 输入图坐标 px。
    /// class：0=岩点 1=大体积——类别信号对主路有用（大体积需更大取样窗口/触边扩框），不得丢弃
    pub fn detect_holds(&self, img: &RgbaImage) -> Vec<(f32, usize, [f32; 4])> {
        let mut out: Vec<(f32, usize, [f32; 4])> = self.detect(img, HOLD_CONF, None);
        // 二阶段：检出太少时滑窗放大（应对整墙小岩点）
        if out.len() < 8 {
            let (iw, ih) = (img.width() as usize, img.height() as usize);
            for gy in 0..3 {
                for gx in 0..3 {
                    let cw = iw / 2;
                    let ch = ih / 2;
                    if cw < 64 || ch < 64 {
                        continue;
                    }
                    let x0 = ((iw * gx) / 4).min(iw - cw);
                    let y0 = ((ih * gy) / 4).min(ih - ch);
                    let crop = image::imageops::crop_imm(img, x0 as u32, y0 as u32, cw as u32, ch as u32).to_image();
                    for (c, cls, b) in self.detect(&crop, HOLD_CONF + 0.1, None) {
                        out.push((c, cls, [b[0] + x0 as f32, b[1] + y0 as f32, b[2] + x0 as f32, b[3] + y0 as f32]));
                    }
                }
            }
        }
        // 全局 NMS 去重（跨窗重叠）
        nms_multi(out)
    }
}

// ---------------- 人物检测器（COCO YOLO11n，person = class 0） ----------------

/// 人物检测置信度
const PERSON_CONF: f32 = 0.30;

static PERSONS: OnceLock<Result<YoloDetect, String>> = OnceLock::new();

fn persons_candidates() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = Vec::new();
    if let Some(p) = super::find_model("yolo11n_person.onnx") {
        v.push(p);
    }
    v
}

impl YoloDetect {
    /// 人物检测全局单例（可选模型：缺失时人物保色功能静默关闭）
    pub fn persons() -> Option<&'static YoloDetect> {
        PERSONS
            .get_or_init(|| YoloDetect::load_default(&persons_candidates()).map_err(|e| e.to_string()))
            .as_ref()
            .ok()
    }

    pub fn persons_status() -> &'static str {
        match PERSONS.get_or_init(|| YoloDetect::load_default(&persons_candidates()).map_err(|e| e.to_string())) {
            Ok(_) => "YOLO11n 人物模型已加载",
            Err(_) => "人物模型未启用（可选）",
        }
    }

    /// 检测人物框（COCO person 类），返回输入图坐标 px，已做 NMS
    pub fn detect_persons(&self, img: &RgbaImage) -> Vec<(f32, [f32; 4])> {
        self.detect(img, PERSON_CONF, Some(&[0]))
            .into_iter()
            .map(|(c, _, b)| (c, b))
            .collect()
    }
}

static _UNUSED: () = ();

fn nms_multi(mut dets: Vec<(f32, usize, [f32; 4])>) -> Vec<(f32, usize, [f32; 4])> {
    dets.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut keep: Vec<(f32, usize, [f32; 4])> = Vec::new();
    for d in dets {
        if keep.iter().all(|k| iou(d.2, k.2) < NMS_IOU) {
            keep.push(d);
        }
    }
    keep
}

fn iou(a: [f32; 4], b: [f32; 4]) -> f32 {
    let ix0 = a[0].max(b[0]);
    let iy0 = a[1].max(b[1]);
    let ix1 = a[2].min(b[2]);
    let iy1 = a[3].min(b[3]);
    let inter = (ix1 - ix0).max(0.0) * (iy1 - iy0).max(0.0);
    let ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    if ua <= 1e-9 {
        0.0
    } else {
        inter / ua
    }
}

