//! Persistent per-file analysis cache in SQLite, keyed by path and
//! invalidated by size or mtime changes.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use anyhow::Result;
use half::f16;
use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

/// Bump when the model or preprocessing changes so stale embeddings are
/// recomputed.
const EMBED_TAG: &str = "dinov2-base/224/255p/gem4/v3";

#[derive(Clone, Debug, Default, Serialize)]
pub struct Info {
	pub width: u32,
	pub height: u32,
	pub duration: f64,
	pub codec: String,
	pub bitrate: u64,
}

#[derive(Clone, Debug, Default)]
pub struct Record {
	pub size: u64,
	pub mtime: i64,
	pub hash: Option<[u8; 32]>,
	pub phash: Option<u64>,
	pub info: Option<Info>,
	/// One DIM-wide row per image, or per usable video frame. `Some` with
	/// no rows means analysed but nothing usable (e.g. an all-black video).
	pub embed: Option<Vec<f16>>,
	pub failed: bool,
}

impl Record {
	pub fn fresh(size: u64, mtime: i64) -> Self {
		Self {
			size,
			mtime,
			..Default::default()
		}
	}

	pub fn analysed(&self) -> bool {
		self.failed || self.embed.is_some()
	}
}

pub struct Cache {
	conn: Connection,
}

fn key(path: &Path) -> &[u8] {
	path.as_os_str().as_bytes()
}

impl Cache {
	pub fn open(path: &Path) -> Result<Self> {
		if let Some(dir) = path.parent() {
			std::fs::create_dir_all(dir)?;
		}
		let conn = Connection::open(path)?;
		conn.execute_batch(
			"PRAGMA journal_mode = WAL;
			PRAGMA synchronous = NORMAL;
			CREATE TABLE IF NOT EXISTS meta (
				key TEXT PRIMARY KEY, value TEXT NOT NULL);
			CREATE TABLE IF NOT EXISTS files (
				path BLOB PRIMARY KEY,
				size INTEGER NOT NULL,
				mtime INTEGER NOT NULL,
				hash BLOB, phash INTEGER,
				width INTEGER, height INTEGER, duration REAL,
				codec TEXT, bitrate INTEGER,
				embed BLOB,
				failed INTEGER NOT NULL DEFAULT 0);",
		)?;
		let tag: Option<String> = conn
			.query_row(
				"SELECT value FROM meta WHERE key = 'embed'",
				[],
				|r| r.get(0),
			)
			.optional()?;
		if tag.as_deref() != Some(EMBED_TAG) {
			conn.execute_batch(
				"UPDATE files SET embed = NULL, failed = 0;",
			)?;
			conn.execute(
				"INSERT OR REPLACE INTO meta VALUES ('embed', ?1)",
				[EMBED_TAG],
			)?;
		}
		Ok(Self { conn })
	}

	pub fn load(&self) -> Result<HashMap<PathBuf, Record>> {
		let mut stmt = self.conn.prepare(
			"SELECT path, size, mtime, hash, phash, width, height,
				duration, codec, bitrate, embed, failed FROM files",
		)?;
		let rows = stmt.query_map([], |r| {
			let path: Vec<u8> = r.get(0)?;
			let hash: Option<Vec<u8>> = r.get(3)?;
			let width: Option<u32> = r.get(5)?;
			let embed: Option<Vec<u8>> = r.get(10)?;
			let info = match width {
				Some(width) => Some(Info {
					width,
					height: r.get(6)?,
					duration: r.get(7)?,
					codec: r.get(8)?,
					bitrate: r.get::<_, i64>(9)? as u64,
				}),
				None => None,
			};
			let rec = Record {
				size: r.get::<_, i64>(1)? as u64,
				mtime: r.get(2)?,
				hash: hash.and_then(|h| h.try_into().ok()),
				phash: r.get::<_, Option<i64>>(4)?
					.map(|p| p as u64),
				info,
				embed: embed.map(|b| {
					b.as_chunks::<2>()
						.0
						.iter()
						.map(|v| f16::from_le_bytes(*v))
						.collect()
				}),
				failed: r.get(11)?,
			};
			Ok((PathBuf::from(OsStr::from_bytes(&path)), rec))
		})?;
		Ok(rows.collect::<rusqlite::Result<_>>()?)
	}

	pub fn store<'a>(
		&mut self,
		items: impl IntoIterator<Item = (&'a Path, &'a Record)>,
	) -> Result<()> {
		let tx = self.conn.transaction()?;
		{
			let mut stmt = tx.prepare_cached(
				"INSERT OR REPLACE INTO files VALUES
				(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
			)?;
			for (path, r) in items {
				let info = r.info.as_ref();
				let embed = r.embed.as_ref().map(|e| {
					e.iter().flat_map(|v| {
						v.to_bits().to_le_bytes()
					})
					.collect::<Vec<u8>>()
				});
				stmt.execute(params![
					key(path),
					r.size as i64,
					r.mtime,
					r.hash.as_ref().map(|h| h.as_slice()),
					r.phash.map(|p| p as i64),
					info.map(|i| i.width),
					info.map(|i| i.height),
					info.map(|i| i.duration),
					info.map(|i| i.codec.as_str()),
					info.map(|i| i.bitrate as i64),
					embed,
					r.failed,
				])?;
			}
		}
		tx.commit()?;
		Ok(())
	}

	/// Drops rows for files under `roots` that the walk no longer found.
	pub fn prune(
		&mut self,
		cached: &HashMap<PathBuf, Record>,
		roots: &[PathBuf],
		seen: &std::collections::HashSet<&Path>,
	) -> Result<()> {
		let tx = self.conn.transaction()?;
		{
			let mut stmt = tx
				.prepare("DELETE FROM files WHERE path = ?1")?;
			for path in cached.keys() {
				let under = roots
					.iter()
					.any(|r| path.starts_with(r));
				if under && !seen.contains(path.as_path()) {
					stmt.execute([key(path)])?;
				}
			}
		}
		tx.commit()?;
		Ok(())
	}
}
