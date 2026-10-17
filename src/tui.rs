//! Review UI: browse groups, compare previews side by side (kitty graphics
//! where available), mark files and move them to the freedesktop Trash.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::Result;
use crossbeam_channel::{Receiver, Sender, unbounded};
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use image::DynamicImage;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
	Block, Cell, List, ListItem, ListState, Paragraph, Row, Table,
	TableState,
};
use ratatui::{DefaultTerminal, Frame};
use ratatui_image::StatefulImage;
use ratatui_image::picker::Picker;
use ratatui_image::protocol::StatefulProtocol;

use crate::cache::{Info, Record};
use crate::decode;
use crate::group::{Group, Link, reclaimable};
use crate::scan::{Entry, Kind};
use crate::video;

const PREVIEW_SHORT: u32 = 720;
const PREVIEW_CACHE: usize = 48;

struct Request {
	entry: usize,
	path: PathBuf,
	kind: Kind,
	info: Option<Info>,
}

struct Previews {
	picker: Picker,
	tx: Sender<Request>,
	rx: Receiver<(usize, Option<DynamicImage>)>,
	ready: HashMap<usize, StatefulProtocol>,
	order: VecDeque<usize>,
	asked: HashSet<usize>,
	failed: HashSet<usize>,
}

impl Previews {
	fn new(picker: Picker) -> Self {
		let (tx, jobs) = unbounded::<Request>();
		let (done, rx) = unbounded();
		for _ in 0..2 {
			let (jobs, done) = (jobs.clone(), done.clone());
			thread::spawn(move || {
				for r in jobs {
					let img = load(&r).ok();
					if done.send((r.entry, img)).is_err() {
						break;
					}
				}
			});
		}
		Self {
			picker,
			tx,
			rx,
			ready: HashMap::new(),
			order: VecDeque::new(),
			asked: HashSet::new(),
			failed: HashSet::new(),
		}
	}

	fn want(&mut self, entry: usize, e: &Entry, r: &Record) {
		if !self.asked.insert(entry) {
			return;
		}
		self.tx.send(Request {
			entry,
			path: e.path.clone(),
			kind: e.kind,
			info: r.info.clone(),
		})
		.ok();
	}

	/// Folds finished loads in; returns whether anything changed.
	fn poll(&mut self) -> bool {
		let mut changed = false;
		while let Ok((entry, img)) = self.rx.try_recv() {
			changed = true;
			let Some(img) = img else {
				self.failed.insert(entry);
				continue;
			};
			self.ready.insert(
				entry,
				self.picker.new_resize_protocol(img),
			);
			self.order.push_back(entry);
			if self.order.len() > PREVIEW_CACHE {
				let old = self.order.pop_front().unwrap();
				self.ready.remove(&old);
				self.asked.remove(&old);
			}
		}
		changed
	}
}

fn load(r: &Request) -> Result<DynamicImage> {
	let img = match (r.kind, &r.info) {
		(Kind::Video, Some(info)) => video::still(&r.path, info)?,
		_ => decode::preview(&r.path, PREVIEW_SHORT)?,
	};
	Ok(img.thumbnail(1600, 1600))
}

enum Prompt {
	Group,
	All,
}

struct App<'a> {
	entries: &'a [Entry],
	records: &'a [Record],
	groups: Vec<Group>,
	/// Per group, per member: marked for the Trash.
	marks: Vec<Vec<bool>>,
	groups_state: ListState,
	members_state: TableState,
	previews: Previews,
	prompt: Option<Prompt>,
	status: String,
	freed: u64,
}

/// Exact and near groups: everything but the keeper. Otherwise only the
/// redundant byte-identical copies, keeping the best-ranked of each set.
fn default_marks(g: &Group) -> Vec<bool> {
	if g.link <= Link::Near {
		return (0..g.members.len()).map(|i| i > 0).collect();
	}
	let mut seen = HashSet::new();
	g.members
		.iter()
		.map(|m| m.set.is_some_and(|s| !seen.insert(s)))
		.collect()
}

pub fn run(
	entries: &[Entry],
	records: &[Record],
	groups: Vec<Group>,
) -> Result<()> {
	let mut term = ratatui::init();
	let picker = Picker::from_query_stdio()
		.unwrap_or_else(|_| Picker::halfblocks());
	let mut app = App {
		entries,
		records,
		marks: groups.iter().map(default_marks).collect(),
		groups,
		groups_state: ListState::default().with_selected(Some(0)),
		members_state: TableState::default().with_selected(Some(0)),
		previews: Previews::new(picker),
		prompt: None,
		status: String::new(),
		freed: 0,
	};
	let res = app.event_loop(&mut term);
	ratatui::restore();
	res
}

fn human(bytes: u64) -> String {
	let units = ["B", "KiB", "MiB", "GiB", "TiB"];
	let mut v = bytes as f64;
	let mut u = 0;
	while v >= 1024.0 && u + 1 < units.len() {
		v /= 1024.0;
		u += 1;
	}
	if u == 0 {
		format!("{bytes} B")
	} else {
		format!("{v:.1} {}", units[u])
	}
}

pub fn count(n: usize, noun: &str) -> String {
	if n == 1 {
		format!("1 {noun}")
	} else {
		format!("{n} {noun}s")
	}
}

/// Civil date from nanoseconds since the epoch (Howard Hinnant's algorithm).
fn date(ns: i64) -> String {
	let days = ns.div_euclid(86_400_000_000_000);
	let z = days + 719_468;
	let era = z.div_euclid(146_097);
	let doe = z - era * 146_097;
	let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
	let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
	let mp = (5 * doy + 2) / 153;
	let d = doy - (153 * mp + 2) / 5 + 1;
	let m = if mp < 10 { mp + 3 } else { mp - 9 };
	let y = yoe + era * 400 + (m <= 2) as i64;
	format!("{y:04}-{m:02}-{d:02}")
}

fn link_colour(l: Link) -> Color {
	match l {
		Link::Exact => Color::Green,
		Link::Near => Color::Yellow,
		Link::Similar => Color::Magenta,
	}
}

/// Keeps the tail of `s`, which for paths is the informative end.
fn tail(s: &str, width: usize) -> String {
	let n = s.chars().count();
	if n <= width {
		return s.to_owned();
	}
	let skip = n - width.saturating_sub(1);
	format!("…{}", s.chars().skip(skip).collect::<String>())
}

impl App<'_> {
	fn group(&self) -> usize {
		self.groups_state.selected().unwrap_or(0)
	}

	fn member(&self) -> usize {
		self.members_state.selected().unwrap_or(0)
	}

	fn event_loop(&mut self, term: &mut DefaultTerminal) -> Result<()> {
		loop {
			self.prefetch();
			term.draw(|f| self.draw(f))?;
			let mut dirty = false;
			while !dirty {
				if event::poll(Duration::from_millis(50))? {
					if let Event::Key(k) = event::read()? {
						if k.kind != KeyEventKind::Press
						{
							continue;
						}
						if !self.key(k.code) {
							return Ok(());
						}
					}
					dirty = true;
				}
				dirty |= self.previews.poll();
			}
		}
	}

	fn prefetch(&mut self) {
		let (g, n) = (self.group(), self.groups.len());
		for gi in [g, (g + 1).min(n - 1)] {
			for m in &self.groups[gi].members {
				let e = m.entry;
				self.previews.want(
					e,
					&self.entries[e],
					&self.records[e],
				);
			}
		}
	}

	/// Returns false to quit.
	fn key(&mut self, code: KeyCode) -> bool {
		if let Some(p) = self.prompt.take() {
			if matches!(
				code,
				KeyCode::Char('y') | KeyCode::Char('Y')
			) {
				match p {
					Prompt::Group => {
						self.trash(&[self.group()])
					}
					Prompt::All => {
						let all: Vec<usize> =
							(0..self.groups.len())
								.collect();
						self.trash(&all);
					}
				}
			} else {
				self.status = "cancelled".into();
			}
			return true;
		}
		// Messages last until the next key, then the key help returns.
		self.status.clear();
		let (g, m) = (self.group(), self.member());
		let members = self.groups[g].members.len();
		match code {
			KeyCode::Char('q') | KeyCode::Esc => return false,
			KeyCode::Down | KeyCode::Char('j') => self
				.members_state
				.select(Some((m + 1).min(members - 1))),
			KeyCode::Up | KeyCode::Char('k') => self
				.members_state
				.select(Some(m.saturating_sub(1))),
			KeyCode::Tab | KeyCode::Right | KeyCode::Char('l') => {
				self.select_group(g as isize + 1)
			}
			KeyCode::BackTab
			| KeyCode::Left
			| KeyCode::Char('h') => self.select_group(g as isize - 1),
			KeyCode::PageDown => self.select_group(g as isize + 10),
			KeyCode::PageUp => self.select_group(g as isize - 10),
			KeyCode::Home => self.select_group(0),
			KeyCode::End => {
				self.select_group(self.groups.len() as isize)
			}
			KeyCode::Char(' ') => self.marks[g][m] ^= true,
			KeyCode::Char('a') => {
				self.marks[g] =
					(0..members).map(|i| i > 0).collect()
			}
			KeyCode::Char('u') => {
				self.marks[g] = vec![false; members]
			}
			KeyCode::Char('o') => self.open(g, m),
			KeyCode::Char('t') => self.ask(Prompt::Group),
			KeyCode::Char('T') => self.ask(Prompt::All),
			_ => {}
		}
		true
	}

	fn select_group(&mut self, g: isize) {
		let g = g.clamp(0, self.groups.len() as isize - 1) as usize;
		self.groups_state.select(Some(g));
		self.members_state.select(Some(0));
	}

	fn open(&mut self, g: usize, m: usize) {
		let path = &self.entries[self.groups[g].members[m].entry].path;
		let res = Command::new("xdg-open")
			.arg(path)
			.stdin(Stdio::null())
			.stdout(Stdio::null())
			.stderr(Stdio::null())
			.spawn();
		if let Err(e) = res {
			self.status = format!("xdg-open: {e}");
		}
	}

	fn marked(&self, groups: &[usize]) -> (usize, u64) {
		let mut n = 0;
		let mut bytes = 0;
		for &g in groups {
			for (m, &on) in self.groups[g]
				.members
				.iter()
				.zip(&self.marks[g])
			{
				if on {
					n += 1;
					bytes += self.entries[m.entry].size;
				}
			}
		}
		(n, bytes)
	}

	fn ask(&mut self, p: Prompt) {
		let groups: Vec<usize> = match p {
			Prompt::Group => vec![self.group()],
			Prompt::All => (0..self.groups.len()).collect(),
		};
		let (n, bytes) = self.marked(&groups);
		if n == 0 {
			self.status = "nothing marked".into();
			return;
		}
		self.status = format!(
			"move {} ({}) to Trash? y/n",
			count(n, "file"),
			human(bytes)
		);
		self.prompt = Some(p);
	}

	/// Trashes marked members of `groups`, skipping any group where every
	/// copy is marked so at least one always survives.
	fn trash(&mut self, groups: &[usize]) {
		let (mut moved, mut skipped, mut failed) = (0, 0, 0);
		for &g in groups {
			if self.marks[g].iter().all(|&m| m) {
				skipped += 1;
				continue;
			}
			let mut keep = Vec::new();
			let mut marks = Vec::new();
			let members =
				std::mem::take(&mut self.groups[g].members);
			for (m, on) in members.into_iter().zip(&self.marks[g]) {
				let e = &self.entries[m.entry];
				if *on && trash::delete(&e.path).is_ok() {
					moved += 1;
					self.freed += e.size;
					continue;
				}
				failed += *on as usize;
				marks.push(*on);
				keep.push(m);
			}
			self.groups[g].members = keep;
			self.marks[g] = marks;
		}
		let mut g = 0;
		while g < self.groups.len() {
			if self.groups[g].members.len() < 2 {
				self.groups.remove(g);
				self.marks.remove(g);
			} else {
				g += 1;
			}
		}
		self.status =
			format!("moved {} to Trash", count(moved, "file"));
		if skipped > 0 {
			self.status += &format!(
				", skipped {} with every file marked",
				count(skipped, "group")
			);
		}
		if failed > 0 {
			self.status += &format!(", {failed} failed");
		}
		if self.groups.is_empty() {
			self.groups_state.select(None);
			return;
		}
		self.select_group(self.group() as isize);
	}

	fn draw(&mut self, f: &mut Frame) {
		let [head, body, foot] = Layout::vertical([
			Constraint::Length(1),
			Constraint::Fill(1),
			Constraint::Length(1),
		])
		.areas(f.area());
		let all: Vec<usize> = (0..self.groups.len()).collect();
		let (n, bytes) = self.marked(&all);
		let title = format!(
			" doppel · {} groups · {n} marked ({}) · {} freed",
			self.groups.len(),
			human(bytes),
			human(self.freed)
		);
		f.render_widget(Paragraph::new(title).bold(), head);
		let help = "↑↓ file  ←→ group  space mark  a keep best  u keep all  \
			o open  t trash group  T trash all  q quit";
		let foot_text = if self.status.is_empty() {
			Line::from(help).dim()
		} else {
			Line::from(self.status.as_str()).yellow()
		};
		f.render_widget(Paragraph::new(foot_text), foot);
		if self.groups.is_empty() {
			f.render_widget(
				Paragraph::new("all groups resolved"),
				body,
			);
			return;
		}
		let [list, right] = Layout::horizontal([
			Constraint::Length(42),
			Constraint::Fill(1),
		])
		.areas(body);
		self.draw_groups(f, list);
		let [table, previews] = Layout::vertical([
			Constraint::Percentage(35),
			Constraint::Fill(1),
		])
		.areas(right);
		self.draw_members(f, table);
		self.draw_previews(f, previews);
	}

	fn draw_groups(&mut self, f: &mut Frame, area: Rect) {
		let items: Vec<ListItem> = self
			.groups
			.iter()
			.map(|g| {
				let keeper =
					&self.entries[g.members[0].entry].path;
				let name = keeper
					.file_name()
					.map(|n| {
						n.to_string_lossy().into_owned()
					})
					.unwrap_or_default();
				ListItem::new(Line::from(vec![
					Span::styled(
						format!(
							"{:7} ",
							g.link.label()
						),
						Style::new().fg(link_colour(
							g.link,
						)),
					),
					Span::raw(format!(
						"{:>2}× {:>9}  ",
						g.members.len(),
						human(reclaimable(
							self.entries,
							g
						))
					)),
					Span::raw(name).dim(),
				]))
			})
			.collect();
		let list = List::new(items)
			.block(Block::bordered().title(" groups "))
			.highlight_style(
				Style::new().add_modifier(Modifier::REVERSED),
			);
		f.render_stateful_widget(list, area, &mut self.groups_state);
	}

	fn draw_members(&mut self, f: &mut Frame, area: Rect) {
		let g = self.group();
		let group = &self.groups[g];
		// Fixed columns, eight gaps and two borders.
		let path_width = (area.width as usize).saturating_sub(65);
		let rows =
			group.members
				.iter()
				.zip(&self.marks[g])
				.enumerate()
				.map(|(i, (m, &on))| {
					let e = &self.entries[m.entry];
					let info = self.records[m.entry]
						.info
						.clone()
						.unwrap_or_default();
					let mark =
						if on {
							Span::styled("trash", Style::new().fg(Color::Red).bold())
						} else {
							Span::styled("keep ", Style::new().fg(Color::Green))
						};
					let len = match e.kind {
						Kind::Video => format!(
							"{:.0}s",
							info.duration
						),
						Kind::Image => String::new(),
					};
					let set =
						m.set.map_or(
							String::new(),
							|s| {
								char::from(b'A' + (s % 26) as u8).to_string()
							},
						);
					Row::new(vec![
					Cell::from(mark),
					Cell::from(if i == 0 { "★" } else { "" }),
					Cell::from(Span::raw(set).cyan()),
					Cell::from(Span::styled(
						format!("{} {:.2}", m.link.label(), m.score),
						Style::new().fg(link_colour(m.link)),
					)),
					Cell::from(format!("{}×{}", info.width, info.height)),
					Cell::from(len),
					Cell::from(human(e.size)),
					Cell::from(date(e.mtime)),
					Cell::from(tail(&e.path.to_string_lossy(), path_width)),
				])
				});
		let widths = [
			Constraint::Length(5),
			Constraint::Length(1),
			Constraint::Length(1),
			Constraint::Length(12),
			Constraint::Length(11),
			Constraint::Length(5),
			Constraint::Length(10),
			Constraint::Length(10),
			Constraint::Fill(1),
		];
		let title = format!(
			" {} · {} ",
			group.link.label(),
			match group.kind {
				Kind::Image => "images",
				Kind::Video => "videos",
			}
		);
		let header = Row::new([
			"", "", "≡", "match", "pixels", "len", "size",
			"modified", "path",
		])
		.dim();
		let table = Table::new(rows, widths)
			.header(header)
			.block(Block::bordered().title(title))
			.row_highlight_style(
				Style::new().add_modifier(Modifier::REVERSED),
			);
		f.render_stateful_widget(table, area, &mut self.members_state);
	}

	fn draw_previews(&mut self, f: &mut Frame, area: Rect) {
		let g = self.group();
		let m = self.member();
		let members = &self.groups[g].members;
		let other = if m == 0 { 1 } else { 0 };
		let panes = [
			(m, "selected"),
			(other, if other == 0 { "best" } else { "next" }),
		];
		let [l, r] = Layout::horizontal([Constraint::Fill(1); 2])
			.areas(area);
		for ((idx, label), area) in panes.into_iter().zip([l, r]) {
			let entry = members[idx].entry;
			let name = self.entries[entry]
				.path
				.file_name()
				.map(|n| n.to_string_lossy().into_owned())
				.unwrap_or_default();
			let block = Block::bordered()
				.title(format!(" {label}: {name} "));
			let inner = block.inner(area);
			f.render_widget(block, area);
			if let Some(p) = self.previews.ready.get_mut(&entry) {
				f.render_stateful_widget(
					StatefulImage::default(),
					inner,
					p,
				);
			} else if self.previews.failed.contains(&entry) {
				f.render_widget(
					Paragraph::new("no preview").dim(),
					inner,
				);
			} else {
				f.render_widget(
					Paragraph::new("loading…").dim(),
					inner,
				);
			}
		}
	}
}
