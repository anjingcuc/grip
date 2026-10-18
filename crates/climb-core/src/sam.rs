//! EdgeSAM（ONNX，tract 纯 Rust 推理）：YOLO 框提示 -> 岩点像素掩码
//!
//! 用法：encode(img) 一次得到 1024-letterbox 图像嵌入（~64x64x256），
//! 之后每个 YOLO 框调 mask_for_box，毫秒级返回分析分辨率掩码。
//! 掩码 logits 在 256×256 网格上，按分析像素中心双线性采样后过阈值，
//! 边缘平滑度优于硬缩放。

use image::RgbaImage;
use std::path::PathBuf;
use std::sync::OnceLock;
use tract_onnx::prelude::*;

const ENC_IN: usize = 1024;
const MASK_GRID: usize = 256;

/// 图像嵌入（encoder 输出的持有者）
pub struct SamEmbedding {
    data: Vec<f32>, // [256,64,64]
    // 分析分辨率 -> 1024 letterbox 的缩放系数
    scale: f32,
}

pub struct EdgeSam {
    encoder: TypedRunnableModel<TypedModel>,
    decoder: TypedRunnableModel<TypedModel>,
}

static SAM: OnceLock<Result<EdgeSam, String>> = OnceLock::new();

fn load_pair() -> anyhow::Result<(TypedRunnableModel<TypedModel>, TypedRunnableModel<TypedModel>)> {
    let enc_path = super::find_model("edge_sam_3x_encoder.onnx").ok_or_else(|| anyhow::anyhow!("未找到 edge_sam_3x_encoder.onnx"))?;
    let dec_path = super::find_model("edge_sam_3x_decoder.onnx").ok_or_else(|| anyhow::anyhow!("未找到 edge_sam_3x_decoder.onnx"))?;
    let square = |d: usize| vec![1.to_dim(), 3.to_dim(), d.to_dim(), d.to_dim()];
    let mut enc = tract_onnx::onnx().model_for_path(&enc_path)?;
    enc.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), &square(ENC_IN)))?;
    let encoder = enc.into_typed()?.into_optimized()?.into_runnable()?;

    let mut dec = tract_onnx::onnx().model_for_path(&dec_path)?;
    dec.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), &[1.to_dim(), 256.to_dim(), 64.to_dim(), 64.to_dim()]))?;
    dec.set_input_fact(1, InferenceFact::dt_shape(f32::datum_type(), &[1.to_dim(), 2.to_dim(), 2.to_dim()]))?;
    dec.set_input_fact(2, InferenceFact::dt_shape(f32::datum_type(), &[1.to_dim(), 2.to_dim()]))?;
    let decoder = dec.into_typed()?.into_optimized()?.into_runnable()?;
    Ok((encoder, decoder))
}

impl EdgeSam {
    pub fn global() -> Option<&'static EdgeSam> {
        SAM.get_or_init(|| match load_pair() {
            Ok((encoder, decoder)) => Ok(EdgeSam { encoder, decoder }),
            Err(e) => Err(e.to_string()),
        })
        .as_ref()
        .ok()
    }

    pub fn status() -> &'static str {
        match SAM.get_or_init(|| match load_pair() {
            Ok((encoder, decoder)) => Ok(EdgeSam { encoder, decoder }),
            Err(e) => Err(e.to_string()),
        }) {
            Ok(_) => "EdgeSAM 已加载",
            Err(e) => {
                let short: String = e.chars().take(60).collect();
                Box::leak(format!("EdgeSAM 未启用: {short}").into_boxed_str())
            }
        }
    }

    /// 编码整图（ResizeLongestSide 1024 + ImageNet 归一化 + 右下零填充）
    pub fn encode(&self, img: &RgbaImage) -> Option<SamEmbedding> {
        let (w, h) = (img.width() as usize, img.height() as usize);
        if w == 0 || h == 0 {
            return None;
        }
        let scale = ENC_IN as f32 / w.max(h) as f32;
        let nw = (w as f32 * scale).round() as usize;
        let nh = (h as f32 * scale).round() as usize;
        let raw = img.as_raw();
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];
        let mut input = vec![0f32; 3 * ENC_IN * ENC_IN];
        // 双线性采样填入 nw×nh 区域，其余为 0（减均值后 0 像素 = (0-mean)/std，与 python 一致）
        for y in 0..nh {
            let sy = ((y as f32 + 0.5) / scale - 0.5).clamp(0.0, h as f32 - 1.0);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let fy = sy - y0 as f32;
            for x in 0..nw {
                let sx = ((x as f32 + 0.5) / scale - 0.5).clamp(0.0, w as f32 - 1.0);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let fx = sx - x0 as f32;
                let p00 = (y0 * w + x0) * 4;
                let p01 = (y0 * w + x1) * 4;
                let p10 = (y1 * w + x0) * 4;
                let p11 = (y1 * w + x1) * 4;
                for c in 0..3 {
                    let v = (raw[p00 + c] as f32 * (1.0 - fx) * (1.0 - fy)
                        + raw[p01 + c] as f32 * fx * (1.0 - fy)
                        + raw[p10 + c] as f32 * (1.0 - fx) * fy
                        + raw[p11 + c] as f32 * fx * fy)
                        / 255.0;
                    input[c * ENC_IN * ENC_IN + y * ENC_IN + x] = (v - mean[c]) / std[c];
                }
            }
        }
        // 零填充区域：按 (0-mean)/std 处理
        for c in 0..3 {
            let base = c * ENC_IN * ENC_IN;
            for y in 0..ENC_IN {
                for x in nw..ENC_IN {
                    if y >= nh || x >= nw {
                        input[base + y * ENC_IN + x] = (0.0 - mean[c]) / std[c];
                    }
                }
            }
        }
        let tensor = Tensor::from_shape(&[1, 3, ENC_IN, ENC_IN], input.as_slice()).ok()?;
        let outputs = self.encoder.run(tvec!(tensor.into())).ok()?;
        let view = outputs[0].to_array_view::<f32>().ok()?;
        let n = 256 * 64 * 64;
        if view.len() < n {
            return None;
        }
        Some(SamEmbedding {
            data: view.iter().take(n).copied().collect(),
            scale,
        })
    }

    /// 框提示解码。返回 (分析分辨率上的 logits 网格采样函数, SAM 置信度)。
    /// 坐标为分析图像素坐标；掩码 logits 在 256 网格，按需双线性采样。
    pub fn mask_for_box(&self, emb: &SamEmbedding, box_xyxy: [f32; 4]) -> Option<(Vec<f32>, f32)> {
        let coords = vec![
            [box_xyxy[0] * emb.scale, box_xyxy[1] * emb.scale],
            [box_xyxy[2] * emb.scale, box_xyxy[3] * emb.scale],
        ];
        let flat: Vec<f32> = coords.iter().flat_map(|p| p.iter().copied()).collect();
        let coords_t = Tensor::from_shape(&[1usize, 2, 2], flat.as_slice()).ok()?;
        let labels_t = Tensor::from_shape(&[1usize, 2], &[2.0f32, 3.0]).ok()?;
        let emb_t = Tensor::from_shape(&[1usize, 256, 64, 64], emb.data.as_slice()).ok()?;
        let outputs = self
            .decoder
            .run(tvec!(emb_t.into(), coords_t.into(), labels_t.into()))
            .ok()?;
        let scores = outputs[0].to_array_view::<f32>().ok()?;
        let masks = outputs[1].to_array_view::<f32>().ok()?;
        if masks.shape().len() != 4 || masks.shape()[2] != MASK_GRID || masks.shape()[3] != MASK_GRID {
            return None;
        }
        let conf = scores.iter().next().copied().unwrap_or(0.0);
        Some((masks.iter().copied().collect(), conf))
    }
}

impl SamEmbedding {
    /// 在分析像素 (x, y) 处双线性采样掩码 logits（>0 为掩码内）
    #[inline]
    pub fn sample_logit(&self, grid: &[f32], x: f32, y: f32) -> f32 {
        // 分析坐标 -> 256 网格坐标
        let gx = x * self.scale / 4.0 - 0.5;
        let gy = y * self.scale / 4.0 - 0.5;
        let x0 = gx.floor().clamp(0.0, (MASK_GRID - 1) as f32) as usize;
        let y0 = gy.floor().clamp(0.0, (MASK_GRID - 1) as f32) as usize;
        let x1 = (x0 + 1).min(MASK_GRID - 1);
        let y1 = (y0 + 1).min(MASK_GRID - 1);
        let fx = (gx - x0 as f32).clamp(0.0, 1.0);
        let fy = (gy - y0 as f32).clamp(0.0, 1.0);
        let g00 = grid[y0 * MASK_GRID + x0];
        let g01 = grid[y0 * MASK_GRID + x1];
        let g10 = grid[y1 * MASK_GRID + x0];
        let g11 = grid[y1 * MASK_GRID + x1];
        g00 * (1.0 - fx) * (1.0 - fy) + g01 * fx * (1.0 - fy) + g10 * (1.0 - fx) * fy + g11 * fx * fy
    }
}
