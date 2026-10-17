//! Image decoding and model preprocessing. JPEGs decode through
//! libjpeg-turbo with DCT-domain downscaling, so a 24 MP photo costs roughly
//! what a 0.4 MP one does; formats the image crate lacks go through ffmpeg.

use std::cell::RefCell;
use std::io::{BufRead, Cursor};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use half::f16;
use image::metadata::Orientation;
use image::{DynamicImage, RgbImage};

use crate::model::{GRID, PATCH, PATCH_LEN, SIDE, TOKENS};

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Short side the crop is taken from, as in DINOv2's preprocessing.
const SHORT: u32 = 256;
/// Luma standard deviation below which a frame is treated as blank.
const FLAT: f32 = 6.0;

/// One model input: TOKENS x PATCH_LEN normalised patch rows.
pub struct Sample {
	pub patches: Vec<f16>,
	pub flat: bool,
}

pub struct Decoded {
	pub sample: Sample,
	pub phash: u64,
	pub width: u32,
	pub height: u32,
}

thread_local! {
	static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
	static JPEG: RefCell<Option<turbojpeg::Decompressor>> =
		const { RefCell::new(None) };
}

fn resize(src: &RgbImage, w: u32, h: u32, opts: &ResizeOptions) -> RgbImage {
	let mut dst = RgbImage::new(w, h);
	RESIZER.with_borrow_mut(|r| r.resize(src, &mut dst, opts).unwrap());
	dst
}

/// Decodes an image file and prepares it for the model and pHash.
pub fn image(path: &Path) -> Result<Decoded> {
	let bytes = std::fs::read(path)?;
	let (rgb, oriented) = load(path, &bytes, SHORT)?;
	let (mut width, mut height) =
		full_size(&bytes).unwrap_or(rgb.dimensions());
	let mut img = DynamicImage::ImageRgb8(rgb);
	if let Some(o) = orientation(&bytes).filter(|_| !oriented) {
		img.apply_orientation(o);
		if transposes(o) {
			(width, height) = (height, width);
		}
	}
	let rgb = img.into_rgb8();
	let r = content(&rgb);
	Ok(Decoded {
		phash: phash(&rgb, &r),
		sample: sample_in(&rgb, &r),
		width,
		height,
	})
}

/// Decodes for display, bounded to roughly `short` pixels on the short side.
pub fn preview(path: &Path, short: u32) -> Result<DynamicImage> {
	let bytes = std::fs::read(path)?;
	let (rgb, oriented) = load(path, &bytes, short)?;
	let mut img = DynamicImage::ImageRgb8(rgb);
	if let Some(o) = orientation(&bytes).filter(|_| !oriented) {
		img.apply_orientation(o);
	}
	Ok(img)
}

fn transposes(o: Orientation) -> bool {
	matches!(
		o,
		Orientation::Rotate90
			| Orientation::Rotate270
			| Orientation::Rotate90FlipH
			| Orientation::Rotate270FlipH
	)
}

/// Returns the decoded image, possibly DCT-downscaled to no less than
/// `short` on its short side, and whether it is already upright. ffmpeg
/// applies HEIF's own rotation, which supersedes any EXIF tag.
fn load(path: &Path, bytes: &[u8], short: u32) -> Result<(RgbImage, bool)> {
	if bytes.starts_with(&[0xff, 0xd8])
		&& let Ok(img) = jpeg(bytes, short)
	{
		return Ok((img, false));
	}
	match image::load_from_memory(bytes) {
		Ok(img) => Ok((img.into_rgb8(), false)),
		Err(_) => Ok((ffmpeg_still(path)?, true)),
	}
}

/// The largest libjpeg-turbo DCT scaling that keeps the short side at or
/// above `short`.
fn jpeg(bytes: &[u8], short: u32) -> Result<RgbImage> {
	JPEG.with_borrow_mut(|slot| {
		if slot.is_none() {
			*slot = Some(turbojpeg::Decompressor::new()?);
		}
		let dec = slot.as_mut().unwrap();
		let header = dec.read_header(bytes)?;
		let min = header.width.min(header.height);
		let denom = [8, 4, 2]
			.into_iter()
			.find(|d| min.div_ceil(*d) >= short as usize)
			.unwrap_or(1);
		let factor = turbojpeg::ScalingFactor::new(1, denom);
		dec.set_scaling_factor(factor)?;
		let h = header.scaled(factor);
		let mut out = turbojpeg::Image {
			pixels: vec![0u8; 3 * h.width * h.height],
			width: h.width,
			pitch: 3 * h.width,
			height: h.height,
			format: turbojpeg::PixelFormat::RGB,
		};
		dec.decompress(bytes, out.as_deref_mut())?;
		RgbImage::from_raw(h.width as u32, h.height as u32, out.pixels)
			.context("bad JPEG buffer")
	})
}

/// Pixel dimensions of the stored image, before any decode-time scaling.
fn full_size(bytes: &[u8]) -> Option<(u32, u32)> {
	if bytes.starts_with(&[0xff, 0xd8]) {
		let h = turbojpeg::read_header(bytes).ok()?;
		return Some((h.width as u32, h.height as u32));
	}
	image::ImageReader::new(Cursor::new(bytes))
		.with_guessed_format()
		.ok()?
		.into_dimensions()
		.ok()
}

fn orientation(bytes: &[u8]) -> Option<Orientation> {
	let exif = exif::Reader::new()
		.read_from_container(&mut Cursor::new(bytes))
		.ok()?;
	let field =
		exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
	let v = field.value.get_uint(0)?;
	Orientation::from_exif(v as u8)
}

/// HEIC, AVIF, JXL and anything else the image crate cannot handle.
fn ffmpeg_still(path: &Path) -> Result<RgbImage> {
	let out = Command::new("ffmpeg")
		.args(["-nostdin", "-v", "error", "-i"])
		.arg(path)
		.args([
			"-frames:v",
			"1",
			"-f",
			"image2pipe",
			"-c:v",
			"ppm",
			"-",
		])
		.stdin(Stdio::null())
		.stderr(Stdio::null())
		.output()
		.context("running ffmpeg")?;
	if !out.status.success() {
		bail!("ffmpeg could not decode");
	}
	read_ppm(&mut Cursor::new(out.stdout))?.context("ffmpeg gave no frame")
}

/// Reads one binary PPM (P6) frame; `None` at a clean end of stream.
pub fn read_ppm(r: &mut impl BufRead) -> Result<Option<RgbImage>> {
	let mut fields = [0u32; 3];
	let mut magic = [0u8; 2];
	match r.read_exact(&mut magic) {
		Ok(()) => {}
		Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
			return Ok(None);
		}
		Err(e) => return Err(e.into()),
	}
	if &magic != b"P6" {
		bail!("not a PPM stream");
	}
	for f in &mut fields {
		*f = ppm_number(r)?;
	}
	let [w, h, max] = fields;
	if max != 255 {
		bail!("unsupported PPM depth {max}");
	}
	let mut buf = vec![0u8; 3 * w as usize * h as usize];
	r.read_exact(&mut buf)?;
	Ok(RgbImage::from_raw(w, h, buf))
}

fn ppm_number(r: &mut impl BufRead) -> Result<u32> {
	let mut n = None::<u32>;
	let mut b = [0u8];
	loop {
		r.read_exact(&mut b)?;
		match b[0] {
			c @ b'0'..=b'9' => {
				let d = (c - b'0') as u32;
				n = Some(n.unwrap_or(0) * 10 + d);
			}
			_ if n.is_some() => return Ok(n.unwrap()),
			_ => {}
		}
	}
}

/// The picture inside any letterbox, pillarbox or mat border.
struct Region {
	x: u32,
	y: u32,
	w: u32,
	h: u32,
}

/// Whether a line of pixels is a uniform near-black or near-white bar.
fn bar<'a>(pixels: impl Iterator<Item = &'a image::Rgb<u8>>) -> bool {
	let (mut lo, mut hi, mut sum, mut n) = (255u8, 0u8, 0u32, 0u32);
	for p in pixels {
		let y = luma(&p.0) as u8;
		(lo, hi) = (lo.min(y), hi.max(y));
		sum += y as u32;
		n += 1;
	}
	let mean = sum / n.max(1);
	hi - lo <= 12 && (mean <= 20 || mean >= 235)
}

/// Trims bars of at least 2% from each side, up to 40% per side, so a
/// letterboxed or bordered copy crops the same picture as its original.
fn content(img: &RgbImage) -> Region {
	let (w, h) = img.dimensions();
	let row = |y: u32| bar((0..w).step_by(2).map(|x| img.get_pixel(x, y)));
	let (mut top, mut bottom) = (0, h);
	while top < h * 2 / 5 && row(top) {
		top += 1;
	}
	while bottom > h * 3 / 5 && row(bottom - 1) {
		bottom -= 1;
	}
	if top < h / 50 {
		top = 0;
	}
	if h - bottom < h / 50 {
		bottom = h;
	}
	let col = |x: u32| {
		bar((top..bottom).step_by(2).map(|y| img.get_pixel(x, y)))
	};
	let (mut left, mut right) = (0, w);
	while left < w * 2 / 5 && col(left) {
		left += 1;
	}
	while right > w * 3 / 5 && col(right - 1) {
		right -= 1;
	}
	if left < w / 50 {
		left = 0;
	}
	if w - right < w / 50 {
		right = w;
	}
	Region {
		x: left,
		y: top,
		w: right - left,
		h: bottom - top,
	}
}

/// Model input for a decoded picture or video frame.
pub fn sample(img: &RgbImage) -> Sample {
	sample_in(img, &content(img))
}

/// Centre square covering SIDE/SHORT of the region's short side, resampled
/// to SIDE x SIDE — the same region DINOv2's resize-then-crop keeps.
fn sample_in(img: &RgbImage, r: &Region) -> Sample {
	let (w, h) = (r.w as f64, r.h as f64);
	let side = w.min(h) * SIDE as f64 / SHORT as f64;
	let (x, y) =
		(r.x as f64 + (w - side) / 2.0, r.y as f64 + (h - side) / 2.0);
	let opts = ResizeOptions::new()
		.resize_alg(ResizeAlg::Convolution(FilterType::CatmullRom))
		.crop(x, y, side, side);
	let crop = resize(img, SIDE as u32, SIDE as u32, &opts);
	patchify(&crop)
}

/// Lays a SIDE x SIDE crop out as patch rows matching the conv weight's
/// (channel, ky, kx) flattening.
fn patchify(crop: &RgbImage) -> Sample {
	let mut patches = Vec::with_capacity(TOKENS * PATCH_LEN);
	let px = crop.as_raw();
	for t in 0..TOKENS {
		let (py, pxo) = (t / GRID * PATCH, t % GRID * PATCH);
		for c in 0..3 {
			for ky in 0..PATCH {
				let row = ((py + ky) * SIDE + pxo) * 3 + c;
				for kx in 0..PATCH {
					let v = px[row + kx * 3] as f32 / 255.0;
					let v = (v - MEAN[c]) / STD[c];
					patches.push(f16::from_f32(v));
				}
			}
		}
	}
	Sample {
		patches,
		flat: luma_std(crop) < FLAT,
	}
}

fn luma(p: &[u8; 3]) -> f32 {
	0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
}

fn luma_std(img: &RgbImage) -> f32 {
	let n = (img.width() * img.height()) as f32;
	let (mut s, mut s2) = (0.0f64, 0.0f64);
	for p in img.as_raw().as_chunks::<3>().0 {
		let y = luma(p) as f64;
		s += y;
		s2 += y * y;
	}
	let mean = s / n as f64;
	((s2 / n as f64 - mean * mean).max(0.0) as f32).sqrt()
}

/// 64-bit DCT perceptual hash (imagehash-style) of the whole picture.
fn phash(img: &RgbImage, r: &Region) -> u64 {
	const N: usize = 32;
	const K: usize = 8;
	let opts = ResizeOptions::new()
		.resize_alg(ResizeAlg::Convolution(FilterType::Box))
		.crop(r.x as f64, r.y as f64, r.w as f64, r.h as f64);
	let small = resize(img, N as u32, N as u32, &opts);
	let gray: Vec<f32> =
		small.as_raw().as_chunks::<3>().0.iter().map(luma).collect();
	let cos = |k: usize, n: usize| {
		(std::f32::consts::PI * (2 * n + 1) as f32 * k as f32
			/ (2 * N) as f32)
			.cos()
	};
	let mut rows = [[0f32; N]; K];
	for (u, row) in rows.iter_mut().enumerate() {
		for (x, out) in row.iter_mut().enumerate() {
			*out = (0..N)
				.map(|y| cos(u, y) * gray[y * N + x])
				.sum();
		}
	}
	let mut coef = [0f32; K * K];
	for u in 0..K {
		for v in 0..K {
			coef[u * K + v] =
				(0..N).map(|x| cos(v, x) * rows[u][x]).sum();
		}
	}
	let mut sorted = coef;
	sorted.sort_by(f32::total_cmp);
	let median = (sorted[K * K / 2 - 1] + sorted[K * K / 2]) / 2.0;
	coef.iter()
		.enumerate()
		.fold(0, |h, (i, &c)| h | ((c > median) as u64) << i)
}
