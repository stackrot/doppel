//! DINOv2 ViT-B/14 forward pass on the GPU, producing L2-normalised
//! GeM-pooled patch descriptors used for near-duplicate matching.
//!
//! Matmuls and attention run in fp16 (RDNA3 WMMA via Vulkan cooperative
//! matrices); the residual stream and LayerNorms stay in fp32, since DINOv2's
//! high-norm tokens swamp small fp16 residual updates.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use burn_tensor::activation::gelu;
use burn_tensor::module::{attention, interpolate, layer_norm, linear};
use burn_tensor::ops::{
	AttentionModuleOptions, InterpolateMode, InterpolateOptions,
};
use burn_tensor::{DType, Device, Tensor, TensorData, f16};
use safetensors::{Dtype, SafeTensors};

pub const DIM: usize = 768;
pub const SIDE: usize = 224;
pub const PATCH: usize = 14;
pub const GRID: usize = SIDE / PATCH;
/// The bottom-right patch is dropped: CLS plus 255 patches makes a
/// 256-token sequence, which runs attention ~3x faster than 257.
pub const TOKENS: usize = GRID * GRID - 1;
pub const PATCH_LEN: usize = 3 * PATCH * PATCH;
pub const BATCH: usize = 32;
const HEADS: usize = 12;
const LAYERS: usize = 12;
const EPS: f64 = 1e-6;

/// fp16 linear layer taking and returning fp16 activations.
struct Affine {
	weight: Tensor<2>,
	bias: Tensor<1>,
}

impl Affine {
	fn apply(&self, x: Tensor<3>) -> Tensor<3> {
		linear(x, self.weight.clone(), Some(self.bias.clone()))
	}
}

/// fp32 LayerNorm whose output feeds an fp16 matmul.
struct Norm {
	gamma: Tensor<1>,
	beta: Tensor<1>,
}

impl Norm {
	fn apply(&self, x: Tensor<3>) -> Tensor<3> {
		layer_norm(x, self.gamma.clone(), Some(self.beta.clone()), EPS)
	}
}

struct Block {
	norm1: Norm,
	qkv: Affine,
	proj: Affine,
	scale1: Tensor<3>,
	norm2: Norm,
	fc1: Affine,
	fc2: Affine,
	scale2: Tensor<3>,
}

impl Block {
	fn forward(&self, x: Tensor<3>) -> Tensor<3> {
		let [b, n, _] = x.dims();
		let head = DIM / HEADS;
		let h = self.norm1.apply(x.clone()).cast(DType::F16);
		let qkv =
			self.qkv.apply(h)
				.reshape([b, n, 3, HEADS, head])
				.permute([2, 0, 3, 1, 4]);
		let mut parts = qkv
			.chunk(3, 0)
			.into_iter()
			.map(|t| t.reshape([b, HEADS, n, head]));
		let (q, k, v) = (
			parts.next().unwrap(),
			parts.next().unwrap(),
			parts.next().unwrap(),
		);
		let opts = AttentionModuleOptions::default();
		let a = attention(q, k, v, None, None, opts)
			.swap_dims(1, 2)
			.reshape([b, n, DIM]);
		let a = self.proj.apply(a).cast(DType::F32);
		let x = x + a * self.scale1.clone();
		let h = self.norm2.apply(x.clone()).cast(DType::F16);
		let h =
			self.fc2.apply(gelu(self.fc1.apply(h)))
				.cast(DType::F32);
		x + h * self.scale2.clone()
	}
}

pub struct Dino {
	device: Device,
	patch: Affine,
	cls: Tensor<3>,
	pos: Tensor<3>,
	blocks: Vec<Block>,
	norm: Norm,
}

struct Weights<'a> {
	st: SafeTensors<'a>,
	device: &'a Device,
}

impl Weights<'_> {
	fn raw(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
		let t = self
			.st
			.tensor(name)
			.with_context(|| format!("weights lack {name}"))?;
		if t.dtype() != Dtype::F32 {
			bail!("{name} is {:?}, expected F32", t.dtype());
		}
		let v =
			t.data().as_chunks::<4>()
				.0
				.iter()
				.map(|b| f32::from_le_bytes(*b))
				.collect();
		Ok((v, t.shape().to_vec()))
	}

	fn full<const D: usize>(
		&self,
		v: Vec<f32>,
		shape: [usize; D],
	) -> Tensor<D> {
		Tensor::from_data(TensorData::new(v, shape), self.device)
	}

	fn half<const D: usize>(
		&self,
		v: Vec<f32>,
		shape: [usize; D],
	) -> Tensor<D> {
		let v: Vec<f16> = v.into_iter().map(f16::from_f32).collect();
		let data = TensorData::new(v, shape);
		Tensor::from_data(data, (self.device, DType::F16))
	}

	fn vector(&self, name: &str) -> Result<Tensor<1>> {
		let (v, _) = self.raw(name)?;
		let n = v.len();
		Ok(self.full(v, [n]))
	}

	/// Layer scale as a broadcastable [1, 1, DIM] tensor.
	fn scale(&self, name: &str) -> Result<Tensor<3>> {
		let (v, _) = self.raw(name)?;
		Ok(self.full(v, [1, 1, DIM]))
	}

	fn norm(&self, prefix: &str) -> Result<Norm> {
		Ok(Norm {
			gamma: self.vector(&format!("{prefix}.weight"))?,
			beta: self.vector(&format!("{prefix}.bias"))?,
		})
	}

	/// PyTorch stores [out, in]; burn's linear wants [in, out]. Several
	/// prefixes are fused side by side along the output dimension.
	fn affine(&self, prefixes: &[&str]) -> Result<Affine> {
		let mut cols = Vec::new();
		let mut bias = Vec::new();
		let mut inputs = 0;
		for p in prefixes {
			let (w, shape) = self.raw(&format!("{p}.weight"))?;
			let (b, _) = self.raw(&format!("{p}.bias"))?;
			inputs = shape[1..].iter().product();
			cols.push((transpose(&w, shape[0], inputs), shape[0]));
			bias.extend(b);
		}
		let outputs: usize = cols.iter().map(|(_, o)| o).sum();
		let mut weight = Vec::with_capacity(inputs * outputs);
		for r in 0..inputs {
			for (w, o) in &cols {
				weight.extend_from_slice(
					&w[r * o..(r + 1) * o],
				);
			}
		}
		Ok(Affine {
			weight: self.half(weight, [inputs, outputs]),
			bias: self.half(bias, [outputs]),
		})
	}

	fn block(&self, i: usize) -> Result<Block> {
		let p = format!("encoder.layer.{i}");
		let att = format!("{p}.attention.attention");
		Ok(Block {
			norm1: self.norm(&format!("{p}.norm1"))?,
			qkv: self.affine(&[
				&format!("{att}.query"),
				&format!("{att}.key"),
				&format!("{att}.value"),
			])?,
			proj: self.affine(&[&format!(
				"{p}.attention.output.dense"
			)])?,
			scale1: self
				.scale(&format!("{p}.layer_scale1.lambda1"))?,
			norm2: self.norm(&format!("{p}.norm2"))?,
			fc1: self.affine(&[&format!("{p}.mlp.fc1")])?,
			fc2: self.affine(&[&format!("{p}.mlp.fc2")])?,
			scale2: self
				.scale(&format!("{p}.layer_scale2.lambda1"))?,
		})
	}

	/// Splits the 37x37 training-resolution position table into the CLS
	/// slot and a bicubically resampled patch grid for our input size.
	fn positions(&self) -> Result<(Vec<f32>, Tensor<3>)> {
		let (v, shape) = self.raw("embeddings.position_embeddings")?;
		let side = ((shape[1] - 1) as f64).sqrt() as usize;
		if side * side + 1 != shape[1] {
			bail!("position table is not square");
		}
		let cls = v[..DIM].to_vec();
		let grid = transpose(&v[DIM..], side * side, DIM);
		let grid = self.full(grid, [1, DIM, side, side]);
		let opts = InterpolateOptions {
			output_size: Some([GRID, GRID]),
			..InterpolateOptions::new(InterpolateMode::Bicubic)
		};
		let grid = interpolate(grid, opts)
			.reshape([1, DIM, GRID * GRID])
			.swap_dims(1, 2)
			.narrow(1, 0, TOKENS);
		Ok((cls, grid))
	}
}

fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
	let mut out = vec![0.0; v.len()];
	for r in 0..rows {
		for c in 0..cols {
			out[c * rows + r] = v[r * cols + c];
		}
	}
	out
}

impl Dino {
	pub fn load(path: &Path, device: &Device) -> Result<Self> {
		let bytes = std::fs::read(path).with_context(|| {
			format!("reading {}", path.display())
		})?;
		let st = SafeTensors::deserialize(&bytes)?;
		let w = Weights { st, device };
		let (mut cls, pos) = w.positions()?;
		let (token, _) = w.raw("embeddings.cls_token")?;
		for (c, t) in cls.iter_mut().zip(token) {
			*c += t;
		}
		let blocks = (0..LAYERS)
			.map(|i| w.block(i))
			.collect::<Result<_>>()?;
		Ok(Self {
			device: device.clone(),
			patch: w.affine(&[
				"embeddings.patch_embeddings.projection",
			])?,
			cls: w.full(cls, [1, 1, DIM]),
			pos,
			blocks,
			norm: w.norm("layernorm")?,
		})
	}

	/// Final normalised tokens, CLS first: [batch, TOKENS + 1, DIM].
	fn forward(&self, patches: Vec<f16>, batch: usize) -> Tensor<3> {
		let data = TensorData::new(patches, [batch, TOKENS, PATCH_LEN]);
		let x = Tensor::<3>::from_data(
			data,
			(&self.device, DType::F16),
		);
		let x = self.patch.apply(x).cast(DType::F32) + self.pos.clone();
		let cls = self.cls.clone().repeat_dim(0, batch);
		let mut x = Tensor::cat(vec![cls, x], 1);
		for b in &self.blocks {
			x = b.forward(x);
		}
		self.norm.apply(x)
	}

	/// Embeds `batch` images given as [batch, TOKENS, PATCH_LEN] patch
	/// rows, returning [batch, DIM] unit vectors: GeM (p = 4) pooling of
	/// the rectified patch tokens. On the sample corpus this separates
	/// edited copies from look-alike photos far better than the CLS token,
	/// which is more semantic.
	pub fn embed(
		&self,
		patches: Vec<f16>,
		batch: usize,
	) -> Result<Vec<f32>> {
		let x = self.forward(patches, batch).narrow(1, 1, TOKENS);
		let gem = x
			.clamp_min(0.0)
			.powi_scalar(4)
			.mean_dim(1)
			.powf_scalar(0.25)
			.reshape([batch, DIM]);
		let len = gem.clone().powi_scalar(2).sum_dim(1).sqrt();
		let out = (gem / len.clamp_min(1e-12)).into_data();
		let out =
			out.as_slice::<f32>().map_err(|e| anyhow!("{e:?}"))?;
		Ok(out.to_vec())
	}
}
