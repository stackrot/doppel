use std::collections::HashSet;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};
use burn_tensor::wgpu::WgpuBackend;
use burn_tensor::{Device, DeviceKind};
use clap::Parser;
use serde::Serialize;
use sha2::{Digest, Sha256};

use doppel::analyse::{self, Options};
use doppel::cache::{Cache, Info, Record};
use doppel::group::{self, Group, Link};
use doppel::model::Dino;
use doppel::scan::{self, Kind};
use doppel::tui;

/// Find duplicate and near-duplicate images and videos.
#[derive(Parser)]
#[command(version)]
struct Args {
	/// Directories to scan.
	#[arg(required = true)]
	paths: Vec<PathBuf>,
	/// How alike two images must be to match, from 0 to 1. Lower values
	/// catch heavier edits but also more look-alike photos.
	#[arg(short, long, default_value_t = 0.70)]
	threshold: f32,
	/// How alike two video frames must be to match, from 0 to 1.
	#[arg(long, default_value_t = 0.60)]
	frame_threshold: f32,
	/// Only find byte-identical files. Skips decoding and needs no GPU.
	#[arg(short, long)]
	exact: bool,
	/// Print the groups as JSON instead of opening the review screen.
	#[arg(long)]
	json: bool,
	/// Decode video on the CPU instead of with VAAPI.
	#[arg(long)]
	no_vaapi: bool,
	/// Number of video decoders (VAAPI unless --no-vaapi).
	#[arg(long, default_value_t = 4)]
	video_jobs: usize,
	/// Number of extra CPU video decoders.
	#[arg(long, default_value_t = 2)]
	cpu_video_jobs: usize,
	/// Number of hashing threads [default: CPU threads].
	#[arg(long)]
	hash_jobs: Option<usize>,
}

const WEIGHTS_URL: &str = concat!(
	"https://huggingface.co/facebook/dinov2-base/resolve/",
	"f9e44c814b77203eaa57a6bdbbd535f21ede1415/model.safetensors"
);
const WEIGHTS_SHA256: &str =
	"d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841";

fn cache_dir() -> Result<PathBuf> {
	Ok(dirs::cache_dir()
		.context("cannot find a cache directory")?
		.join("doppel"))
}

fn sha256(path: &Path) -> Result<String> {
	let mut f = std::fs::File::open(path)?;
	let mut h = Sha256::new();
	let mut buf = vec![0u8; 1 << 20];
	loop {
		let n = f.read(&mut buf)?;
		if n == 0 {
			break;
		}
		h.update(&buf[..n]);
	}
	Ok(format!("{:x}", h.finalize()))
}

/// Fetches the pinned DINOv2 weights on first use.
fn weights() -> Result<PathBuf> {
	let path = cache_dir()?.join("dinov2-base.safetensors");
	if path.exists() {
		return Ok(path);
	}
	std::fs::create_dir_all(path.parent().unwrap())?;
	let part = path.with_extension("part");
	eprintln!("downloading DINOv2 weights (346 MB)");
	let ok = Command::new("curl")
		.args(["-fL", "--progress-bar", "-o"])
		.arg(&part)
		.arg(WEIGHTS_URL)
		.status()
		.context("running curl")?
		.success();
	if !ok {
		bail!("weights download failed");
	}
	if sha256(&part)? != WEIGHTS_SHA256 {
		std::fs::remove_file(&part).ok();
		bail!("downloaded weights failed their checksum");
	}
	std::fs::rename(&part, &path)?;
	Ok(path)
}

#[derive(Serialize)]
struct JsonMember<'a> {
	path: &'a Path,
	size: u64,
	link: Link,
	score: f32,
	keep: bool,
	leader: bool,
	/// Byte-identical copies within the group share a set number.
	#[serde(skip_serializing_if = "Option::is_none")]
	set: Option<usize>,
	#[serde(flatten)]
	info: Option<&'a Info>,
}

#[derive(Serialize)]
struct JsonGroup<'a> {
	kind: Kind,
	link: Link,
	members: Vec<JsonMember<'a>>,
}

fn print_json(
	entries: &[scan::Entry],
	records: &[Record],
	groups: &[Group],
) -> Result<()> {
	let out: Vec<JsonGroup> = groups
		.iter()
		.map(|g| JsonGroup {
			kind: g.kind,
			link: g.link,
			members: g
				.members
				.iter()
				.enumerate()
				.map(|(n, m)| JsonMember {
					path: &entries[m.entry].path,
					size: entries[m.entry].size,
					link: m.link,
					score: m.score,
					keep: n == 0,
					leader: m.leader,
					set: m.set,
					info: records[m.entry].info.as_ref(),
				})
				.collect(),
		})
		.collect();
	serde_json::to_writer_pretty(std::io::stdout().lock(), &out)?;
	println!();
	Ok(())
}

/// The first discrete, else integrated, Vulkan GPU. Software rasterisers
/// such as lavapipe are skipped: they would take hours.
fn gpu() -> Result<Device> {
	for kind in [DeviceKind::DiscreteGpu(0), DeviceKind::IntegratedGpu(0)] {
		let dev = Device::wgpu_options()
			.graphics_api(WgpuBackend::Vulkan)
			.device_kind(kind)
			.init();
		if let Ok(dev) = dev {
			return Ok(dev);
		}
	}
	bail!("no Vulkan GPU found; similarity matching needs one \
		(--exact works without)")
}

fn has_ffmpeg() -> bool {
	["ffmpeg", "ffprobe"].iter().all(|tool| {
		Command::new(tool)
			.arg("-version")
			.stdout(Stdio::null())
			.stderr(Stdio::null())
			.status()
			.is_ok_and(|s| s.success())
	})
}

fn main() -> Result<()> {
	let args = Args::parse();
	let device = if args.exact {
		None
	} else {
		if !has_ffmpeg() {
			bail!("ffmpeg and ffprobe are needed for similarity \
				matching (--exact works without)");
		}
		Some(gpu()?)
	};
	let mut roots = Vec::new();
	for p in &args.paths {
		let p = p
			.canonicalize()
			.with_context(|| format!("{}", p.display()))?;
		roots.push(p);
	}
	let entries = scan::walk(&roots);
	let mut cache = Cache::open(&cache_dir()?.join("cache.sqlite"))?;
	let cached = cache.load()?;
	let mut records: Vec<Record> = entries
		.iter()
		.map(|e| match cached.get(&e.path) {
			Some(r) if r.size == e.size && r.mtime == e.mtime => {
				r.clone()
			}
			_ => Record::fresh(e.size, e.mtime),
		})
		.collect();
	let seen: HashSet<&Path> =
		entries.iter().map(|e| e.path.as_path()).collect();
	cache.prune(&cached, &roots, &seen)?;
	drop(cached);

	let model = match &device {
		Some(dev) => Some(Dino::load(&weights()?, dev)?),
		None => None,
	};
	let opts = Options {
		vaapi: !args.no_vaapi,
		video_jobs: args.video_jobs.max(1),
		cpu_video_jobs: args.cpu_video_jobs,
		hash_jobs: args
			.hash_jobs
			.unwrap_or_else(|| {
				std::thread::available_parallelism()
					.map_or(4, |n| n.get())
			})
			.max(1),
	};
	analyse::run(
		&entries,
		&mut records,
		&mut cache,
		model.as_ref(),
		&opts,
	)?;
	drop(model);
	let failed = records.iter().filter(|r| r.failed).count();
	if failed > 0 {
		let files = tui::count(failed, "file");
		eprintln!("skipped {files} that could not be decoded");
	}
	let t = group::Thresholds {
		image: args.threshold,
		frame: args.frame_threshold,
	};
	let groups = group::build(&entries, &records, device.as_ref(), &t)?;
	if args.json {
		return print_json(&entries, &records, &groups);
	}
	if groups.is_empty() {
		let files = tui::count(entries.len(), "file");
		eprintln!("no duplicates in {files}");
		return Ok(());
	}
	tui::run(&entries, &records, groups)
}
