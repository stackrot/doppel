//! Filesystem walk: finds image and video files under the given roots.

use std::collections::HashSet;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use rayon::prelude::*;
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
	Image,
	Video,
}

const IMAGES: &[&str] = &[
	"jpg", "jpeg", "jpe", "jfif", "png", "webp", "gif", "bmp", "tif",
	"tiff", "heic", "heif", "avif", "jxl",
];
const VIDEOS: &[&str] = &[
	"mp4", "m4v", "mkv", "webm", "mov", "avi", "wmv", "flv", "mpg", "mpeg",
	"ts", "mts", "m2ts", "3gp", "ogv",
];

pub fn kind_of(path: &Path) -> Option<Kind> {
	let ext = path.extension()?.to_str()?.to_ascii_lowercase();
	if IMAGES.contains(&ext.as_str()) {
		Some(Kind::Image)
	} else if VIDEOS.contains(&ext.as_str()) {
		Some(Kind::Video)
	} else {
		None
	}
}

pub struct Entry {
	pub path: PathBuf,
	pub size: u64,
	/// Nanoseconds since the epoch.
	pub mtime: i64,
	pub kind: Kind,
}

/// Walks every root in parallel. Hidden directories (including Trash) are
/// skipped, symlinks are not followed, and hardlinks to an already-seen
/// inode are dropped since removing them frees nothing.
pub fn walk(roots: &[PathBuf]) -> Vec<Entry> {
	let paths: Vec<PathBuf> = roots
		.iter()
		.flat_map(|root| {
			jwalk::WalkDir::new(root)
				.skip_hidden(true)
				.follow_links(false)
				.into_iter()
				.filter_map(Result::ok)
				.filter(|e| e.file_type().is_file())
				.map(|e| e.path())
				.filter(|p| kind_of(p).is_some())
		})
		.collect();
	let stats: Vec<_> = paths
		.into_par_iter()
		.filter_map(|path| {
			let md = std::fs::symlink_metadata(&path).ok()?;
			Some((path, md))
		})
		.collect();
	let mut inodes = HashSet::new();
	let mut out = Vec::new();
	for (path, md) in stats {
		if md.len() == 0 || !inodes.insert((md.dev(), md.ino())) {
			continue;
		}
		out.push(Entry {
			kind: kind_of(&path).unwrap(),
			size: md.len(),
			mtime: md.mtime() * 1_000_000_000 + md.mtime_nsec(),
			path,
		});
	}
	out
}
