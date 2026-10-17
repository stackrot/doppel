//! Video probing and frame sampling through ffmpeg. Frames are taken at a
//! fixed interval, so two encodes of the same footage are compared at the
//! same moments whatever their keyframe placement. With VAAPI the whole
//! stream decodes and scales on the GPU's VCN engines and only sampled
//! frames are copied back.

use std::io::BufReader;
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use image::{DynamicImage, RgbImage, imageops};
use serde_json::Value;

use crate::cache::Info;
use crate::decode::{self, Sample};

/// Minimum spacing between sampled frames, in seconds.
const INTERVAL: f64 = 2.0;
const MAX_FRAMES: f64 = 600.0;
/// Short side frames are scaled to before border trimming and cropping.
const SHORT: u32 = 320;

pub struct Probe {
	/// Display geometry, i.e. after rotation.
	pub info: Info,
	/// Clockwise rotation needed for display: 0, 90, 180 or 270.
	turn: u32,
}

pub fn probe(path: &Path) -> Result<Probe> {
	let out = Command::new("ffprobe")
		.args([
			"-v",
			"error",
			"-select_streams",
			"v:0",
			"-show_entries",
		])
		.arg(concat!(
			"stream=codec_name,width,height:",
			"stream_side_data=rotation:format=duration,bit_rate"
		))
		.args(["-of", "json"])
		.arg(path)
		.stdin(Stdio::null())
		.stderr(Stdio::null())
		.output()
		.context("running ffprobe")?;
	let v: Value = serde_json::from_slice(&out.stdout)?;
	let stream = v["streams"].get(0).context("no video stream")?;
	let num = |v: &Value| {
		v.as_f64()
			.or_else(|| v.as_str().and_then(|s| s.parse().ok()))
	};
	let mut width = num(&stream["width"]).context("no width")? as u32;
	let mut height = num(&stream["height"]).context("no height")? as u32;
	let rotation = stream["side_data_list"]
		.as_array()
		.into_iter()
		.flatten()
		.find_map(|d| num(&d["rotation"]))
		.unwrap_or(0.0);
	// ffprobe reports counter-clockwise degrees.
	let turn =
		((-rotation / 90.0).round() as i64).rem_euclid(4) as u32 * 90;
	if turn % 180 == 90 {
		(width, height) = (height, width);
	}
	let info = Info {
		width,
		height,
		duration: num(&v["format"]["duration"]).unwrap_or(0.0),
		codec: stream["codec_name"].as_str().unwrap_or("?").to_owned(),
		bitrate: num(&v["format"]["bit_rate"]).unwrap_or(0.0) as u64,
	};
	Ok(Probe { info, turn })
}

/// Stored (pre-rotation) frame size scaled to SHORT on the short side.
fn scaled(p: &Probe) -> (u32, u32) {
	let (mut w, mut h) = (p.info.width.max(1), p.info.height.max(1));
	if p.turn % 180 == 90 {
		(w, h) = (h, w);
	}
	let even = |v: u64| ((v / 2).max(1) * 2) as u32;
	if w <= h {
		(SHORT, even(h as u64 * SHORT as u64 / w as u64))
	} else {
		(even(w as u64 * SHORT as u64 / h as u64), SHORT)
	}
}

fn upright(img: RgbImage, turn: u32) -> RgbImage {
	match turn {
		90 => imageops::rotate90(&img),
		180 => imageops::rotate180(&img),
		270 => imageops::rotate270(&img),
		_ => img,
	}
}

/// Streams model-ready frames to `sink`; returns how many were produced.
/// Falls back to software decoding when VAAPI cannot handle the stream.
pub fn frames(
	path: &Path,
	probe: &Probe,
	vaapi: bool,
	mut sink: impl FnMut(Sample),
) -> Result<usize> {
	if vaapi && let Ok(n @ 1..) = sampled(path, probe, true, &mut sink) {
		return Ok(n);
	}
	sampled(path, probe, false, &mut sink)
}

fn sampled(
	path: &Path,
	probe: &Probe,
	vaapi: bool,
	sink: &mut impl FnMut(Sample),
) -> Result<usize> {
	let every = INTERVAL.max(probe.info.duration / MAX_FRAMES);
	let (w, h) = scaled(probe);
	let mut cmd = Command::new("ffmpeg");
	cmd.args(["-nostdin", "-v", "error", "-noautorotate"]);
	let filter = if vaapi {
		cmd.args([
			"-hwaccel",
			"vaapi",
			"-hwaccel_output_format",
			"vaapi",
		]);
		format!("fps=1/{every},scale_vaapi=w={w}:h={h}:format=nv12,\
			hwdownload,format=nv12")
	} else {
		cmd.args(["-threads", "4"]);
		format!("fps=1/{every},scale={w}:{h}")
	};
	let mut child = cmd
		.arg("-i")
		.arg(path)
		.args(["-map", "0:v:0", "-an", "-sn", "-dn", "-vf", &filter])
		.args(["-fps_mode", "passthrough", "-f", "image2pipe"])
		.args(["-c:v", "ppm", "-"])
		.stdin(Stdio::null())
		.stderr(Stdio::null())
		.stdout(Stdio::piped())
		.spawn()
		.context("running ffmpeg")?;
	let mut out = BufReader::new(child.stdout.take().unwrap());
	let mut n = 0;
	let res = loop {
		match decode::read_ppm(&mut out) {
			Ok(Some(img)) => {
				sink(decode::sample(&upright(img, probe.turn)));
				n += 1;
			}
			Ok(None) => break Ok(()),
			Err(e) => break Err(e),
		}
	};
	drop(out);
	let status = child.wait()?;
	res?;
	if n == 0 && !status.success() {
		bail!("ffmpeg could not decode");
	}
	Ok(n)
}

/// A representative frame for the review UI.
pub fn still(path: &Path, info: &Info) -> Result<DynamicImage> {
	let at = format!("{:.2}", info.duration * 0.2);
	let out = Command::new("ffmpeg")
		.args(["-nostdin", "-v", "error", "-ss", &at, "-i"])
		.arg(path)
		.args(["-frames:v", "1", "-vf", "scale=-2:720"])
		.args(["-f", "image2pipe", "-c:v", "ppm", "-"])
		.stdin(Stdio::null())
		.stderr(Stdio::null())
		.output()
		.context("running ffmpeg")?;
	let img = decode::read_ppm(&mut std::io::Cursor::new(out.stdout))?
		.context("ffmpeg gave no frame")?;
	Ok(DynamicImage::ImageRgb8(img))
}
