//! Runs every analysis stage at once so all hardware stays busy: BLAKE3
//! hashing on its own I/O pool, image decoding across every core, several
//! ffmpeg/VAAPI video decoders, and GPU embedding workers, all feeding a
//! single collector that owns the records and the cache.

use std::collections::HashMap;
use std::path::Path;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, bounded};
use half::f16;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;

use crate::cache::{Cache, Info, Record};
use crate::decode;
use crate::model::{BATCH, DIM, Dino, PATCH_LEN, TOKENS};
use crate::scan::{Entry, Kind};
use crate::video;

pub struct Options {
	pub vaapi: bool,
	/// ffmpeg decoders on the VCN engines (when `vaapi`).
	pub video_jobs: usize,
	/// Extra software decoders, so idle cores help once images are done.
	pub cpu_video_jobs: usize,
	pub hash_jobs: usize,
}

const GPU_WORKERS: usize = 2;
const FLUSH: usize = 512;

enum Event {
	Hashed(usize, [u8; 32]),
	Image(usize, Info, u64),
	/// Sent once all of a video's frames have been queued.
	Video(usize, Info, usize),
	Embedded(usize, Vec<f32>),
	Failed(usize),
	Skipped,
}

struct Job {
	owner: usize,
	patches: Vec<f16>,
}

#[derive(Default)]
struct Pending {
	info: Option<Info>,
	phash: Option<u64>,
	expected: Option<usize>,
	embed: Vec<f16>,
}

fn hash(path: &Path, size: u64) -> std::io::Result<[u8; 32]> {
	let mut h = blake3::Hasher::new();
	if size >= 16 << 20 {
		h.update_mmap_rayon(path)?;
	} else {
		h.update_mmap(path)?;
	}
	Ok(*h.finalize().as_bytes())
}

/// Files sharing a size with another file and lacking a cached hash.
fn hash_todo(entries: &[Entry], records: &[Record]) -> Vec<usize> {
	let mut sizes: HashMap<u64, usize> = HashMap::new();
	for e in entries {
		*sizes.entry(e.size).or_default() += 1;
	}
	(0..entries.len())
		.filter(|&i| {
			sizes[&entries[i].size] > 1 && records[i].hash.is_none()
		})
		.collect()
}

fn todo(entries: &[Entry], records: &[Record], kind: Kind) -> Vec<usize> {
	(0..entries.len())
		.filter(|&i| entries[i].kind == kind && !records[i].analysed())
		.collect()
}

fn bar(mp: &MultiProgress, prefix: &'static str, len: usize) -> ProgressBar {
	let style = ProgressStyle::with_template(
		"{prefix:>7} {bar:40.cyan/blue} {pos:>6}/{len:6} \
		{per_sec:>10} eta {eta}",
	)
	.unwrap()
	.progress_chars("━╸ ");
	let pb = mp.add(ProgressBar::new(len as u64));
	pb.set_style(style);
	pb.set_prefix(prefix);
	pb
}

pub fn run(
	entries: &[Entry],
	records: &mut [Record],
	cache: &mut Cache,
	model: Option<&Dino>,
	opts: &Options,
) -> Result<()> {
	let hashes = hash_todo(entries, records);
	let (images, videos) = match model {
		Some(_) => (
			todo(entries, records, Kind::Image),
			todo(entries, records, Kind::Video),
		),
		None => (Vec::new(), Vec::new()),
	};
	let mp = MultiProgress::new();
	let bars = [
		bar(&mp, "hash", hashes.len()),
		bar(&mp, "images", images.len()),
		bar(&mp, "videos", videos.len()),
	];
	let frames = mp.add(ProgressBar::new_spinner());
	frames.set_style(
		ProgressStyle::with_template(
			"{prefix:>7} {pos} embedded, {per_sec}",
		)
		.unwrap(),
	);
	frames.set_prefix("gpu");
	let (ev_tx, ev_rx) = bounded::<Event>(4096);
	let (job_tx, job_rx) = bounded::<Job>(BATCH * 4);
	let pool = rayon::ThreadPoolBuilder::new()
		.num_threads(opts.hash_jobs)
		.build()?;
	thread::scope(|s| {
		let tx = ev_tx.clone();
		s.spawn(move || {
			pool.install(|| {
				hashes.par_iter().for_each(|&i| {
					let e = &entries[i];
					let ev = match hash(&e.path, e.size) {
						Ok(h) => Event::Hashed(i, h),
						Err(_) => Event::Skipped,
					};
					tx.send(ev).ok();
				})
			})
		});
		if let Some(model) = model {
			for _ in 0..GPU_WORKERS {
				let (rx, tx) = (job_rx.clone(), ev_tx.clone());
				s.spawn(move || gpu(model, rx, tx));
			}
			let (tx, jobs) = (ev_tx.clone(), job_tx.clone());
			s.spawn(move || {
				images.par_iter().for_each(|&i| {
					analyse_image(
						&entries[i],
						i,
						&tx,
						&jobs,
					)
				})
			});
			let (queue_tx, queue_rx) =
				bounded::<usize>(videos.len());
			for &i in &videos {
				queue_tx.send(i).unwrap();
			}
			drop(queue_tx);
			let hw = if opts.vaapi { opts.video_jobs } else { 0 };
			let sw = if opts.vaapi { 0 } else { opts.video_jobs };
			for w in 0..hw + sw + opts.cpu_video_jobs {
				let queue = queue_rx.clone();
				let (tx, jobs) =
					(ev_tx.clone(), job_tx.clone());
				s.spawn(move || {
					for i in queue {
						let e = &entries[i];
						analyse_video(
							e,
							i,
							w < hw,
							&tx,
							&jobs,
						);
					}
				});
			}
		}
		drop((ev_tx, job_tx, job_rx));
		collect(entries, records, cache, ev_rx, &bars, &frames)
	})?;
	mp.clear().ok();
	Ok(())
}

fn analyse_image(e: &Entry, i: usize, tx: &Sender<Event>, jobs: &Sender<Job>) {
	let Ok(d) = decode::image(&e.path) else {
		tx.send(Event::Failed(i)).ok();
		return;
	};
	let info = Info {
		width: d.width,
		height: d.height,
		..Default::default()
	};
	tx.send(Event::Image(i, info, d.phash)).ok();
	jobs.send(Job {
		owner: i,
		patches: d.sample.patches,
	})
	.ok();
}

fn analyse_video(
	e: &Entry,
	i: usize,
	vaapi: bool,
	tx: &Sender<Event>,
	jobs: &Sender<Job>,
) {
	let Ok(probe) = video::probe(&e.path) else {
		tx.send(Event::Failed(i)).ok();
		return;
	};
	let mut queued = 0;
	let res = video::frames(&e.path, &probe, vaapi, |s| {
		if !s.flat {
			jobs.send(Job {
				owner: i,
				patches: s.patches,
			})
			.ok();
			queued += 1;
		}
	});
	let ev = match res {
		Ok(_) => Event::Video(i, probe.info, queued),
		Err(_) if queued > 0 => Event::Video(i, probe.info, queued),
		Err(_) => Event::Failed(i),
	};
	tx.send(ev).ok();
}

/// Gathers jobs into fixed-size batches (padding the tail) so the GPU
/// kernels are autotuned once and reused.
fn gpu(model: &Dino, jobs: Receiver<Job>, tx: Sender<Event>) {
	fn push(j: Job, owners: &mut Vec<usize>, buf: &mut Vec<f16>) {
		owners.push(j.owner);
		buf.extend_from_slice(&j.patches);
	}
	let mut owners = Vec::with_capacity(BATCH);
	let mut buf = Vec::with_capacity(BATCH * TOKENS * PATCH_LEN);
	while let Ok(first) = jobs.recv() {
		push(first, &mut owners, &mut buf);
		while owners.len() < BATCH {
			match jobs.recv_timeout(Duration::from_millis(20)) {
				Ok(j) => push(j, &mut owners, &mut buf),
				Err(_) => break,
			}
		}
		buf.resize(BATCH * TOKENS * PATCH_LEN, f16::ZERO);
		let input = std::mem::replace(
			&mut buf,
			Vec::with_capacity(BATCH * TOKENS * PATCH_LEN),
		);
		match model.embed(input, BATCH) {
			Ok(out) => {
				for (k, owner) in owners.drain(..).enumerate() {
					let row = out[k * DIM..(k + 1) * DIM]
						.to_vec();
					tx.send(Event::Embedded(owner, row))
						.ok();
				}
			}
			Err(e) => {
				// Left pending, so nothing is cached and the next
				// run retries.
				eprintln!("GPU batch failed: {e:#}");
				owners.clear();
			}
		}
	}
}

fn collect(
	entries: &[Entry],
	records: &mut [Record],
	cache: &mut Cache,
	events: Receiver<Event>,
	bars: &[ProgressBar; 3],
	frames: &ProgressBar,
) -> Result<()> {
	let mut pending: HashMap<usize, Pending> = HashMap::new();
	let mut dirty = Vec::new();
	let progress = |i: usize| match entries[i].kind {
		Kind::Image => &bars[1],
		Kind::Video => &bars[2],
	};
	for ev in events {
		let idx = match ev {
			Event::Hashed(i, h) => {
				records[i].hash = Some(h);
				dirty.push(i);
				bars[0].inc(1);
				None
			}
			Event::Skipped => {
				bars[0].inc(1);
				None
			}
			Event::Failed(i) => {
				records[i].failed = true;
				dirty.push(i);
				progress(i).inc(1);
				None
			}
			Event::Image(i, info, phash) => {
				let p = pending.entry(i).or_default();
				(p.info, p.phash, p.expected) =
					(Some(info), Some(phash), Some(1));
				Some(i)
			}
			Event::Video(i, info, n) => {
				let p = pending.entry(i).or_default();
				(p.info, p.expected) = (Some(info), Some(n));
				Some(i)
			}
			Event::Embedded(i, row) => {
				frames.inc(1);
				let p = pending.entry(i).or_default();
				p.embed.extend(row
					.into_iter()
					.map(f16::from_f32));
				Some(i)
			}
		};
		if let Some(i) = idx {
			let p = &pending[&i];
			if p.expected == Some(p.embed.len() / DIM) {
				let p = pending.remove(&i).unwrap();
				let r = &mut records[i];
				(r.info, r.phash, r.embed) =
					(p.info, p.phash, Some(p.embed));
				dirty.push(i);
				progress(i).inc(1);
			}
		}
		if dirty.len() >= FLUSH {
			flush(entries, records, cache, &mut dirty)?;
		}
	}
	flush(entries, records, cache, &mut dirty)
}

fn flush(
	entries: &[Entry],
	records: &[Record],
	cache: &mut Cache,
	dirty: &mut Vec<usize>,
) -> Result<()> {
	cache.store(dirty
		.iter()
		.map(|&i| (entries[i].path.as_path(), &records[i])))?;
	dirty.clear();
	Ok(())
}
