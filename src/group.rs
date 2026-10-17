//! Turns hashes and embeddings into duplicate groups.

use std::cmp::Ordering;
use std::collections::HashMap;

use anyhow::Result;
use burn_tensor::Device;
use half::f16;
use serde::Serialize;

use crate::cache::Record;
use crate::model::DIM;
use crate::scan::{Entry, Kind};
use crate::similar;

/// Ordered strongest first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Link {
	/// Byte-identical.
	Exact,
	/// Same picture or footage re-encoded, resized or recompressed.
	Near,
	/// Edited: cropped, filtered, overlaid, trimmed.
	Similar,
}

impl Link {
	pub fn label(self) -> &'static str {
		match self {
			Link::Exact => "exact",
			Link::Near => "near",
			Link::Similar => "similar",
		}
	}
}

pub struct Member {
	pub entry: usize,
	/// Link to the group's leader (for the leader, its strongest link).
	pub link: Link,
	pub score: f32,
	pub leader: bool,
	/// Byte-identical copies within the group share a set number.
	pub set: Option<usize>,
}

pub struct Group {
	pub kind: Kind,
	/// Loosest link holding the group together.
	pub link: Link,
	/// Best-ranked (suggested keeper) first.
	pub members: Vec<Member>,
}

const NEIGHBOURS: usize = 32;
/// pHash Hamming distance at or below which a match counts as near.
const NEAR_PHASH: u32 = 8;
/// Lowest whitened cosine similarity at which a pHash agreement is trusted.
const PHASH_FLOOR: f32 = 0.45;
/// Fraction of the shorter video's frames that must appear in the other.
const MIN_COVER: f32 = 0.5;
const NEAR_COVER: f32 = 0.8;

type Edge = (usize, usize, Link, f32);

/// Whitened cosine thresholds for counting two images, or two video
/// frames, as the same picture.
pub struct Thresholds {
	pub image: f32,
	pub frame: f32,
}

pub fn build(
	entries: &[Entry],
	records: &[Record],
	device: Option<&Device>,
	t: &Thresholds,
) -> Result<Vec<Group>> {
	let mut edges = exact_edges(records);
	if let Some(dev) = device {
		edges.extend(image_edges(dev, entries, records, t.image)?);
		edges.extend(video_edges(dev, entries, records, t.frame)?);
	}
	Ok(components(entries, records, &edges))
}

fn exact_edges(records: &[Record]) -> Vec<Edge> {
	let mut by_hash: HashMap<[u8; 32], Vec<usize>> = HashMap::new();
	for (i, r) in records.iter().enumerate() {
		if let Some(h) = r.hash {
			by_hash.entry(h).or_default().push(i);
		}
	}
	by_hash.values()
		.flat_map(|v| {
			v.windows(2).map(|w| (w[0], w[1], Link::Exact, 1.0))
		})
		.collect()
}

/// Entries of `kind` with embeddings, plus their stacked rows.
fn embedded(
	entries: &[Entry],
	records: &[Record],
	kind: Kind,
) -> (Vec<usize>, Vec<f16>, Vec<u32>) {
	let mut ids = Vec::new();
	let mut vecs = Vec::new();
	let mut owners = Vec::new();
	for (i, r) in records.iter().enumerate() {
		let Some(e) = r.embed.as_ref().filter(|e| !e.is_empty()) else {
			continue;
		};
		if entries[i].kind != kind {
			continue;
		}
		let n = e.len() / DIM;
		owners.extend(std::iter::repeat_n(ids.len() as u32, n));
		ids.push(i);
		vecs.extend_from_slice(e);
	}
	(ids, vecs, owners)
}

fn image_edges(
	dev: &Device,
	entries: &[Entry],
	records: &[Record],
	threshold: f32,
) -> Result<Vec<Edge>> {
	let (ids, vecs, owners) = embedded(entries, records, Kind::Image);
	let floor = threshold.min(PHASH_FLOOR);
	let hits =
		similar::search(dev, &vecs, &vecs, &owners, NEIGHBOURS, floor)?;
	let mut best: HashMap<(usize, usize), f32> = HashMap::new();
	for h in hits {
		let (a, b) = (ids[h.row], ids[h.col]);
		let key = (a.min(b), a.max(b));
		let s = best.entry(key).or_insert(h.sim);
		*s = s.max(h.sim);
	}
	// Heavy downscales or blur lower the embedding similarity of what is
	// plainly the same picture; an agreeing pHash vouches for those.
	Ok(best.into_iter()
		.filter_map(|((a, b), sim)| {
			let near = match (records[a].phash, records[b].phash) {
				(Some(x), Some(y)) => {
					(x ^ y).count_ones() <= NEAR_PHASH
				}
				_ => false,
			};
			match (near, sim >= threshold) {
				(true, _) => Some((a, b, Link::Near, sim)),
				(false, true) => {
					Some((a, b, Link::Similar, sim))
				}
				(false, false) => None,
			}
		})
		.collect())
}

/// Two videos match when most sampled frames of the shorter one have a
/// close frame in the other, which also catches trimmed clips.
fn video_edges(
	dev: &Device,
	entries: &[Entry],
	records: &[Record],
	threshold: f32,
) -> Result<Vec<Edge>> {
	let (ids, vecs, owners) = embedded(entries, records, Kind::Video);
	if ids.len() < 2 {
		return Ok(Vec::new());
	}
	// Frames from a handful of films are too uniform to whiten against on
	// their own; images, when present, broaden the basis.
	let mut basis = vecs.clone();
	basis.extend(embedded(entries, records, Kind::Image).1);
	let hits = similar::search(
		dev, &vecs, &basis, &owners, NEIGHBOURS, threshold,
	)?;
	let mut frames = vec![0u32; ids.len()];
	for &o in &owners {
		frames[o as usize] += 1;
	}
	// Distinct (frame, other video) matches, counted per video pair.
	let mut seen = std::collections::HashSet::new();
	let mut cover: HashMap<(u32, u32), u32> = HashMap::new();
	for h in &hits {
		let (va, vb) = (owners[h.row], owners[h.col]);
		if seen.insert((h.row, vb)) {
			*cover.entry((va, vb)).or_default() += 1;
		}
	}
	let mut edges = Vec::new();
	for (&(a, b), &n) in &cover {
		if a > b {
			continue;
		}
		let back = cover.get(&(b, a)).copied().unwrap_or(0);
		let (na, nb) = (frames[a as usize], frames[b as usize]);
		let (ca, cb) = (n as f32 / na as f32, back as f32 / nb as f32);
		let short = if na <= nb { ca } else { cb };
		if short < MIN_COVER || na.min(nb) == 1 && na.max(nb) > 2 {
			continue;
		}
		let (ia, ib) = (ids[a as usize], ids[b as usize]);
		let near = ca >= NEAR_COVER
			&& cb >= NEAR_COVER && same_length(
			&records[ia],
			&records[ib],
		);
		let link = if near { Link::Near } else { Link::Similar };
		edges.push((ia, ib, link, short));
	}
	Ok(edges)
}

fn same_length(a: &Record, b: &Record) -> bool {
	let (Some(a), Some(b)) = (&a.info, &b.info) else {
		return false;
	};
	(a.duration - b.duration).abs() <= 1.0f64.max(a.duration * 0.02)
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
	while parent[i] != i {
		parent[i] = parent[parent[i]];
		i = parent[i];
	}
	i
}

/// Strongest first.
fn stronger(a: (Link, f32), b: (Link, f32)) -> bool {
	a.0 < b.0 || a.0 == b.0 && a.1 > b.1
}

/// Leader clustering: byte-identical files collapse into one unit, then
/// the best-connected unclaimed unit claims the unclaimed units it matches
/// directly, and so on. Unlike connected components this cannot chain
/// look-alikes (hundreds of screenshots of one desktop) into one group, and
/// every score is relative to the leader. The suggested keeper is the
/// best-ranked member, which need not be the leader.
fn components(
	entries: &[Entry],
	records: &[Record],
	edges: &[Edge],
) -> Vec<Group> {
	let mut parent: Vec<usize> = (0..entries.len()).collect();
	for &(a, b, link, _) in edges {
		if link == Link::Exact {
			let (ra, rb) =
				(find(&mut parent, a), find(&mut parent, b));
			parent[ra] = rb;
		}
	}
	let mut units: HashMap<usize, Vec<usize>> = HashMap::new();
	let mut adj: HashMap<usize, HashMap<usize, (Link, f32)>> =
		HashMap::new();
	for &(a, b, link, score) in edges {
		for i in [a, b] {
			let unit =
				units.entry(find(&mut parent, i)).or_default();
			if !unit.contains(&i) {
				unit.push(i);
			}
		}
		let (ua, ub) = (find(&mut parent, a), find(&mut parent, b));
		if ua == ub {
			continue;
		}
		for (x, y) in [(ua, ub), (ub, ua)] {
			let e = adj
				.entry(x)
				.or_default()
				.entry(y)
				.or_insert((link, score));
			if stronger((link, score), *e) {
				*e = (link, score);
			}
		}
	}
	let by_rank = |a: &usize, b: &usize| rank(entries, records, *a, *b);
	for unit in units.values_mut() {
		unit.sort_by(by_rank);
	}
	let degree = |u: &usize| adj.get(u).map_or(0, |n| n.len());
	let mut order: Vec<usize> = units.keys().copied().collect();
	order.sort_by(|a, b| {
		degree(b)
			.cmp(&degree(a))
			.then(by_rank(&units[a][0], &units[b][0]))
	});
	let mut claimed = std::collections::HashSet::new();
	let mut groups = Vec::new();
	for u in order {
		if claimed.contains(&u) {
			continue;
		}
		let near: Vec<(usize, (Link, f32))> = adj
			.get(&u)
			.into_iter()
			.flatten()
			.filter(|(v, _)| !claimed.contains(*v))
			.map(|(&v, &e)| (v, e))
			.collect();
		if near.is_empty() && units[&u].len() < 2 {
			continue;
		}
		claimed.insert(u);
		let own =
			if units[&u].len() > 1 {
				(Link::Exact, 1.0)
			} else {
				near.iter()
					.map(|n| n.1)
					.reduce(|a, b| {
						if stronger(b, a) {
							b
						} else {
							a
						}
					})
					.unwrap()
			};
		let mut sets = 0;
		let mut set_of = |unit: &[usize]| {
			(unit.len() > 1).then(|| {
				sets += 1;
				sets - 1
			})
		};
		let set = set_of(&units[&u]);
		let mut members: Vec<Member> = units[&u]
			.iter()
			.enumerate()
			.map(|(k, &i)| {
				let (link, score) = if k == 0 {
					own
				} else {
					(Link::Exact, 1.0)
				};
				Member {
					entry: i,
					link,
					score,
					leader: k == 0,
					set,
				}
			})
			.collect();
		for (v, (link, score)) in near {
			claimed.insert(v);
			let set = set_of(&units[&v]);
			members.extend(units[&v].iter().map(|&i| Member {
				entry: i,
				link,
				score,
				leader: false,
				set,
			}));
		}
		members.sort_by(|a, b| by_rank(&a.entry, &b.entry));
		groups.push(Group {
			kind: entries[members[0].entry].kind,
			link: members.iter().map(|m| m.link).max().unwrap(),
			members,
		});
	}
	groups.sort_by(|a, b| {
		(a.link, a.kind as u8)
			.cmp(&(b.link, b.kind as u8))
			.then(reclaimable(entries, b)
				.cmp(&reclaimable(entries, a)))
	});
	groups
}

/// Bytes freed by keeping only the suggested keeper.
pub fn reclaimable(entries: &[Entry], g: &Group) -> u64 {
	g.members[1..].iter().map(|m| entries[m.entry].size).sum()
}

/// Best first: most pixels, longest, highest bitrate, largest file, oldest,
/// shortest path.
fn rank(entries: &[Entry], records: &[Record], a: usize, b: usize) -> Ordering {
	let key = |i: usize| {
		let info = records[i].info.clone().unwrap_or_default();
		(
			info.width as u64 * info.height as u64,
			(info.duration * 10.0) as u64,
			info.bitrate,
			entries[i].size,
		)
	};
	key(b).cmp(&key(a))
		.then(entries[a].mtime.cmp(&entries[b].mtime))
		.then(entries[a]
			.path
			.as_os_str()
			.len()
			.cmp(&entries[b].path.as_os_str().len()))
}
