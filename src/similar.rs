//! Brute-force nearest-neighbour search on the GPU: blocked fp16
//! similarity matmuls with per-row top-k, so nothing quadratic ever leaves
//! the card. Descriptors are first whitened against the library's own
//! distribution, which suppresses whatever most files share (one desktop
//! across a thousand screenshots, say) and leaves what tells copies apart.

use anyhow::{Result, anyhow};
use burn_tensor::{DType, Device, Int, Tensor, TensorData, f16};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::model::DIM;

pub struct Hit {
	pub row: usize,
	pub col: usize,
	pub sim: f32,
}

/// Elements of the [block, rows] similarity tile kept on the GPU at once.
const TILE: usize = 1 << 27;
/// Rows sampled to estimate the covariance.
const FIT_ROWS: usize = 20_000;
/// Eigenvalue shrinkage, as a fraction of the mean eigenvalue.
const SHRINK: f64 = 0.1;
/// Weight of the prior, in rows: libraries much smaller than this whiten
/// mostly by the prior, much larger ones mostly by themselves.
const PRIOR_ROWS: f64 = 4000.0;

/// Mean and covariance of 45k COCO photos' descriptors (tools/fit_prior.py).
static PRIOR: &[u8] = include_bytes!("../assets/whiten-prior.bin");

/// PCA whitening: (x - mean) * proj.
struct Whitening {
	mean: Vec<f32>,
	proj: Vec<f32>,
}

fn prior() -> (DVector<f64>, DMatrix<f64>) {
	assert!(PRIOR.starts_with(b"DPWP"));
	let f = |i: usize| {
		let b = &PRIOR[8 + 4 * i..12 + 4 * i];
		f32::from_le_bytes(b.try_into().unwrap()) as f64
	};
	let mean = DVector::from_fn(DIM, |i, _| f(i));
	let cov = DMatrix::from_fn(DIM, DIM, |r, c| f(DIM + r * DIM + c));
	(mean, cov)
}

/// Library statistics blended with the prior, as if the prior were
/// PRIOR_ROWS extra samples.
fn fit(vecs: &[f16]) -> Whitening {
	let m = vecs.len() / DIM;
	let step = m.div_ceil(FIT_ROWS).max(1);
	let rows: Vec<usize> = (0..m).step_by(step).collect();
	let n = rows.len() as f64;
	let x = DMatrix::from_fn(rows.len(), DIM, |r, c| {
		vecs[rows[r] * DIM + c].to_f64()
	});
	let own_mean = x.row_mean().transpose();
	let z = DMatrix::from_fn(rows.len(), DIM, |r, c| {
		x[(r, c)] - own_mean[c]
	});
	let own_cov = z.transpose() * &z / n.max(1.0);
	let (pmean, pcov) = prior();
	let total = n + PRIOR_ROWS;
	let mean = (&own_mean * n + &pmean * PRIOR_ROWS) / total;
	let (da, db) = (&own_mean - &mean, &pmean - &mean);
	let cov = ((own_cov + &da * da.transpose()) * n
		+ (pcov + &db * db.transpose()) * PRIOR_ROWS)
		/ total;
	let eig = SymmetricEigen::new(cov);
	let avg = eig.eigenvalues.mean();
	let mut proj = vec![0f32; DIM * DIM];
	for k in 0..DIM {
		let scale = 1.0
			/ (eig.eigenvalues[k].max(0.0) + SHRINK * avg).sqrt();
		for i in 0..DIM {
			proj[i * DIM + k] =
				(eig.eigenvectors[(i, k)] * scale) as f32;
		}
	}
	Whitening {
		mean: mean.iter().map(|&v| v as f32).collect(),
		proj,
	}
}

/// Whitened, re-normalised fp16 rows on the GPU.
fn prepare(
	device: &Device,
	vecs: &[f16],
	basis: &[f16],
	m: usize,
) -> Tensor<2> {
	let w = fit(basis);
	let x = Tensor::<2>::from_data(
		TensorData::new(vecs.to_vec(), [m, DIM]),
		(device, DType::F16),
	)
	.cast(DType::F32);
	let mean = Tensor::<2>::from_data(
		TensorData::new(w.mean, [1, DIM]),
		device,
	);
	let proj = Tensor::<2>::from_data(
		TensorData::new(w.proj, [DIM, DIM]),
		device,
	);
	let y = (x - mean).matmul(proj);
	let len = y.clone().powi_scalar(2).sum_dim(1).sqrt().clamp_min(1e-12);
	(y / len).cast(DType::F16)
}

/// For every row, its `k` most similar rows belonging to a different owner,
/// keeping those at or above `threshold` (cosine after whitening fitted on
/// `basis`).
pub fn search(
	device: &Device,
	vecs: &[f16],
	basis: &[f16],
	owners: &[u32],
	k: usize,
	threshold: f32,
) -> Result<Vec<Hit>> {
	let m = owners.len();
	if m < 2 {
		return Ok(Vec::new());
	}
	let k = k.min(m);
	let all = prepare(device, vecs, basis, m);
	let all_t = all.clone().transpose();
	let own: Vec<i32> = owners.iter().map(|&o| o as i32).collect();
	let own =
		Tensor::<1, Int>::from_data(TensorData::new(own, [m]), device);
	let block = (TILE / m).clamp(16, 4096);
	let mut hits = Vec::new();
	for r0 in (0..m).step_by(block) {
		let b = block.min(m - r0);
		let sims = all.clone().narrow(0, r0, b).matmul(all_t.clone());
		let rows = own.clone().narrow(0, r0, b).reshape([b, 1]);
		let same = rows
			.expand([b, m])
			.equal(own.clone().reshape([1, m]).expand([b, m]));
		let (vals, idx) =
			sims.mask_fill(same, -1.0).topk_with_indices(k, 1);
		let err = |e| anyhow!("{e:?}");
		let vals = vals
			.into_data()
			.try_into_vec_as::<f32>()
			.map_err(err)?;
		let idx = idx
			.into_data()
			.try_into_vec_as::<i64>()
			.map_err(err)?;
		for (n, (&sim, &col)) in vals.iter().zip(&idx).enumerate() {
			if sim >= threshold {
				let row = r0 + n / k;
				hits.push(Hit {
					row,
					col: col as usize,
					sim,
				});
			}
		}
	}
	Ok(hits)
}
