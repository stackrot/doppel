"""Build a doppel test corpus with known ground truth.

Fetch the sources first with tools/fetch_samples.sh ROOT (COCO 2017 photos
and Blender open movies). This writes ROOT/corpus/ and ROOT/manifest.json.
It needs Pillow, pillow-heif, ffmpeg and cjxl.

  python -I tools/make_samples.py ROOT
"""

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

LIBRARY = [
	"phone/DCIM/Camera", "phone/WhatsApp Images", "downloads",
	"edits", "exports/web", "backup/2021/photos", "backup/2023/misc",
	"desktop/screenshots", "shared/album", "old laptop/Pictures",
]

NEAR = ["resize50", "resize25", "upscale", "jpeg_q40", "png", "webp",
	"avif", "heic", "jxl", "exif_rot"]
SIMILAR = ["crop80", "crop60", "gray", "colour", "blur", "text", "border",
	"mirror", "rotate7", "bright", "sticker"]

VNEAR = ["hevc720", "vp9_480", "av1", "h264_360", "mpeg4", "rotated"]
VSIMILAR = ["trim", "watermark", "crop", "letterbox", "mirror",
	"compilation", "speed"]

ROOT = None


def font(size):
	for f in ["/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
			"/usr/share/fonts/noto/NotoSans-Bold.ttf"]:
		if Path(f).exists():
			return ImageFont.truetype(f, size)
	return ImageFont.load_default(size)


def place(rng, ext):
	d = ROOT / "corpus/library" / rng.choice(LIBRARY)
	d.mkdir(parents=True, exist_ok=True)
	return d / f"{rng.getrandbits(48):012x}.{ext}"


def save_jpeg(im, path, q=90, **kw):
	im.convert("RGB").save(path, "JPEG", quality=q, **kw)


def image_variant(src, t, rng, others):
	im = Image.open(src)
	im = ImageOps.exif_transpose(im).convert("RGB")
	w, h = im.size
	ext = {"png": "png", "webp": "webp", "avif": "avif", "heic": "heic",
		"jxl": "jxl"}.get(t, "jpg")
	out = place(rng, ext)
	try:
		render(im, t, out, rng, others)
	except BaseException:
		out.unlink(missing_ok=True)
		raise
	return out


def render(im, t, out, rng, others):
	w, h = im.size
	if t == "resize50":
		im = im.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
	elif t == "resize25":
		im = im.resize((max(64, w // 4), max(64, h // 4)), Image.Resampling.LANCZOS)
	elif t == "upscale":
		im = im.resize((w * 3 // 2, h * 3 // 2), Image.Resampling.BICUBIC)
	elif t == "crop80":
		im = im.crop((w // 10, h // 10, w - w // 10, h - h // 10))
	elif t == "crop60":
		cw, ch = int(w * 0.6), int(h * 0.6)
		x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
		im = im.crop((x, y, x + cw, y + ch))
	elif t == "gray":
		im = ImageOps.grayscale(im).convert("RGB")
	elif t == "colour":
		im = ImageEnhance.Contrast(ImageEnhance.Color(im).enhance(1.8))\
			.enhance(1.3)
	elif t == "blur":
		im = im.filter(ImageFilter.GaussianBlur(3))
	elif t == "bright":
		im = ImageEnhance.Brightness(im).enhance(1.5)
	elif t == "text":
		d = ImageDraw.Draw(im, "RGBA")
		d.rectangle((0, h * 2 // 3, w, h), fill=(0, 0, 0, 140))
		d.text((w // 20, h * 2 // 3 + h // 20), "HOLIDAY 2019 ♥",
			font=font(max(12, h // 9)), fill=(255, 255, 255, 230))
	elif t == "border":
		im = ImageOps.expand(im, border=max(w, h) // 7, fill="white")
	elif t == "mirror":
		im = ImageOps.mirror(im)
	elif t == "rotate7":
		im = im.rotate(7, resample=Image.Resampling.BICUBIC)
	elif t == "sticker":
		st = ImageOps.exif_transpose(Image.open(rng.choice(others)))
		st = st.convert("RGB")
		st.thumbnail((w // 3, h // 3))
		im.paste(st, (w - st.width - w // 30, h // 30))
	if t == "jpeg_q40":
		save_jpeg(im, out, 40)
	elif t == "exif_rot":
		# Stored sideways with an EXIF tag that displays it upright.
		exif = Image.Exif()
		exif[0x0112] = 6
		save_jpeg(im.transpose(Image.Transpose.ROTATE_90), out, 92,
			exif=exif)
	elif t == "png":
		im.save(out, "PNG")
	elif t == "webp":
		im.save(out, "WEBP", quality=75)
	elif t == "avif":
		im.save(out, "AVIF", quality=60)
	elif t == "heic":
		im.save(out, "HEIF", quality=70)
	elif t == "jxl":
		with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
			im.save(tmp.name, "PNG")
			subprocess.run(["cjxl", "-q", "80", "--num_threads", "1",
				"--quiet", tmp.name, str(out)], check=True,
				stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
	else:
		save_jpeg(im, out, rng.choice([82, 88, 95]))


def image_job(args):
	src, seed, n, others = args
	rng = random.Random(seed)
	rel = lambda p: str(p.relative_to(ROOT / "corpus"))
	out = {}
	picks = rng.sample(NEAR + SIMILAR, n)
	for t in picks:
		try:
			p = image_variant(src, t, rng, others)
		except Exception as e:
			print(f"skipped {t} of {src.name}: {e}", flush=True)
			continue
		cls = "near" if t in NEAR else "similar"
		out[rel(p)] = {"source": src.name, "transform": t, "class": cls}
	if rng.random() < 0.3:
		p = place(rng, src.suffix.lstrip("."))
		shutil.copyfile(src, p)
		out[rel(p)] = {"source": src.name, "transform": "copy",
			"class": "exact"}
	if rng.random() < 0.15 and out:
		# Exact copy of a variant, as happens with re-shared edits.
		base = rng.choice(list(out))
		p = place(rng, Path(base).suffix.lstrip("."))
		shutil.copyfile(ROOT / "corpus" / base, p)
		out[rel(p)] = dict(out[base], copy_of=base)
	return out


def extract_coco(dl, corpus):
	origs = []
	for name in ["val2017", "test2017"]:
		dest = corpus / f"originals/coco-{name}"
		if not dest.exists():
			print("extracting", name, flush=True)
			with zipfile.ZipFile(dl / f"coco-{name[:-4]}" / f"{name}.zip") as z:
				z.extractall(dest.parent / "tmp")
			(dest.parent / "tmp" / name).rename(dest)
			(dest.parent / "tmp").rmdir()
		origs += sorted(dest.glob("*.jpg"))
	return origs


def ff(*args):
	subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args],
		check=True)


def duration(path):
	out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
		"format=duration", "-of", "csv=p=0", str(path)],
		capture_output=True, text=True, check=True)
	return float(out.stdout)


def movies(dl, work):
	bbb = work / "bbb.mov"
	if not bbb.exists():
		with zipfile.ZipFile(dl / "bbb/bbb.mov.zip") as z:
			name = z.namelist()[0]
			with z.open(name) as f, open(bbb, "wb") as o:
				shutil.copyfileobj(f, o)
	return {"bbb": bbb, "sintel": dl / "sintel/sintel.mkv",
		"tos": dl / "tos/tos.mov", "ed": dl / "ed/ed.mov",
		"cycles": dl / "cycles/cycles.mov", "reel": dl / "reel/reel.mov"}


def segments(rng, srcs):
	"""Non-overlapping 45-150 s windows over every movie."""
	segs = []
	for name, path in srcs.items():
		t, end = rng.uniform(5, 20), duration(path) - 5
		while t + 45 < end:
			d = min(rng.uniform(45, 150), end - t)
			segs.append((name, path, round(t, 2), round(t + d, 2)))
			t += d + rng.uniform(5, 30)
	return segs


def encode_segment(seg, out):
	name, path, s, e = seg
	ff("-ss", str(s), "-t", str(e - s), "-i", str(path), "-an",
		"-vf", "scale=-2:'trunc(min(1080,ih)/2)*2'", "-c:v", "libx264", "-preset",
		"veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out))


def video_variant(t, seg, src, out, rng, other):
	name, _, s, e = seg
	ranges = [[name, s, e]]
	x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
		"-pix_fmt", "yuv420p"]
	if t == "hevc720":
		ff("-i", src, "-vf", "scale=-2:720", "-c:v", "libx265", "-preset",
			"faster", "-crf", "28", "-x265-params", "log-level=none", out)
	elif t == "vp9_480":
		ff("-i", src, "-vf", "scale=-2:480", "-c:v", "libvpx-vp9", "-crf",
			"36", "-b:v", "0", "-deadline", "realtime", "-cpu-used", "8",
			out)
	elif t == "av1":
		ff("-i", src, "-c:v", "libsvtav1", "-preset", "10", "-crf", "40",
			out)
	elif t == "h264_360":
		ff("-i", src, "-vf", "scale=-2:360", "-c:v", "libx264", "-preset",
			"veryfast", "-crf", "30", out)
	elif t == "mpeg4":
		ff("-i", src, "-vf", "scale=640:-2", "-c:v", "mpeg4", "-q:v", "5",
			out)
	elif t == "rotated":
		# Phone-style: frames stored sideways plus a display matrix.
		with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
			ff("-i", src, "-vf", "transpose=2", *x264, tmp.name)
			ff("-display_rotation:v:0", "-90", "-i", tmp.name, "-c",
				"copy", out)
	elif t == "trim":
		d = e - s
		a = rng.uniform(0.1, 0.4) * d
		b = a + rng.uniform(0.3, 0.5) * d
		ff("-ss", str(a), "-t", str(b - a), "-i", src, *x264, out)
		ranges = [[name, round(s + a, 2), round(s + b, 2)]]
	elif t == "watermark":
		ff("-i", src, "-vf", "drawbox=x=0:y=0:w=iw:h=ih/6:color=black@0.5:"
			"t=fill,drawtext=text='@sharedclips':fontsize=h/10:"
			"fontcolor=white:x=w/30:y=h/30", *x264, out)
	elif t == "crop":
		ff("-i", src, "-vf", "crop=iw*0.75:ih*0.75", *x264, out)
	elif t == "letterbox":
		ff("-i", src, "-vf", "scale=960:-2,pad=960:720:(ow-iw)/2:(oh-ih)/2",
			*x264, out)
	elif t == "mirror":
		ff("-i", src, "-vf", "hflip", *x264, out)
	elif t == "speed":
		ff("-i", src, "-vf", "setpts=0.8*PTS", *x264, out)
	elif t == "compilation":
		oname, osrc, os_, oe = other
		ff("-i", src, "-i", osrc, "-filter_complex",
			"[0:v]scale=1280:720,setsar=1[a];[1:v]scale=1280:720,"
			"setsar=1[b];[a][b]concat=n=2:v=1[v]", "-map", "[v]", *x264,
			out)
		ranges = [[name, s, e], [oname, os_, oe]]
	return ranges


def build_videos(rng, dl, corpus, work):
	srcs = movies(dl, work)
	segs = segments(rng, srcs)
	orig_dir = corpus / "originals/clips"
	orig_dir.mkdir(parents=True, exist_ok=True)
	man = {}
	rel = lambda p: str(p.relative_to(corpus))
	paths = []
	for i, seg in enumerate(segs):
		p = orig_dir / f"clip{i:03d}.mp4"
		paths.append(p)
		man[rel(p)] = {"ranges": [[seg[0], seg[2], seg[3]]],
			"transform": "original", "class": "original"}
	print(f"encoding {len(segs)} clips", flush=True)
	with ThreadPoolExecutor(4) as ex:
		list(ex.map(lambda a: a[1].exists() or encode_segment(a[0], a[1]),
			zip(segs, paths)))
	jobs = []
	for i, seg in enumerate(segs):
		if rng.random() < 0.25:
			continue
		for t in rng.sample(VNEAR + VSIMILAR, rng.randint(2, 4)):
			ext = {"vp9_480": "webm", "av1": "mkv", "mpeg4": "avi"}.get(t,
				"mp4")
			out = place(rng, ext)
			other = segs[rng.randrange(len(segs))]
			other = (other[0], str(paths[segs.index(other)]), other[2],
				other[3])
			jobs.append((t, seg, str(paths[i]), out, random.Random(
				rng.getrandbits(32)), other))
		if rng.random() < 0.3:
			out = place(rng, "mp4")
			shutil.copyfile(paths[i], out)
			man[rel(out)] = {"ranges": [[seg[0], seg[2], seg[3]]],
				"transform": "copy", "class": "exact"}
	print(f"encoding {len(jobs)} video variants", flush=True)

	def run(j):
		t, seg, src, out, r, other = j
		ranges = video_variant(t, seg, src, str(out), r, other)
		cls = "near" if t in VNEAR else "similar"
		return rel(out), {"ranges": ranges, "transform": t, "class": cls}

	with ThreadPoolExecutor(6) as ex:
		man.update(ex.map(run, jobs))
	# Degenerate cases: blank footage and test patterns.
	black = place(rng, "mp4")
	ff("-f", "lavfi", "-i", "color=black:s=1280x720:d=30", "-c:v",
		"libx264", "-pix_fmt", "yuv420p", str(black))
	black2 = place(rng, "mp4")
	shutil.copyfile(black, black2)
	bars = place(rng, "mp4")
	ff("-f", "lavfi", "-i", "testsrc2=s=1280x720:d=60", "-c:v", "libx264",
		"-pix_fmt", "yuv420p", str(bars))
	man[rel(black)] = {"ranges": [["black", 0, 30]], "transform": "original",
		"class": "original"}
	man[rel(black2)] = {"ranges": [["black", 0, 30]], "transform": "copy",
		"class": "exact"}
	man[rel(bars)] = {"ranges": [["bars", 0, 60]], "transform": "original",
		"class": "original"}
	return man


def main():
	global ROOT
	ap = argparse.ArgumentParser()
	ap.add_argument("root", type=Path)
	ap.add_argument("--image-sources", type=int, default=6000)
	ap.add_argument("--variants", type=int, default=4)
	ap.add_argument("--seed", type=int, default=1)
	args = ap.parse_args()
	ROOT = args.root.resolve()
	dl, corpus = ROOT / "downloads", ROOT / "corpus"
	work = ROOT / "work"
	work.mkdir(exist_ok=True)
	rng = random.Random(args.seed)
	origs = extract_coco(dl, corpus)
	man = {"images": {}, "videos": {}}
	for p in origs:
		man["images"][str(p.relative_to(corpus))] = {"source": p.name,
			"transform": "original", "class": "original"}
	picks = rng.sample(origs, args.image_sources)
	jobs = [(p, rng.getrandbits(32), args.variants,
		rng.sample(origs, 8)) for p in picks]
	done = work / "images.json"
	if done.exists():
		man["images"].update(json.loads(done.read_text()))
	else:
		print(f"generating variants of {len(picks)} images", flush=True)
		variants = {}
		with Pool(initializer=init, initargs=(ROOT,)) as pool:
			for out in pool.imap_unordered(image_job, jobs, 16):
				variants.update(out)
		done.write_text(json.dumps(variants))
		man["images"].update(variants)
	man["videos"] = build_videos(rng, dl, corpus, work)
	(ROOT / "manifest.json").write_text(json.dumps(man, indent=1))
	n = len(man["images"]) + len(man["videos"])
	print(f"done: {n} files", flush=True)


def init(root):
	global ROOT
	ROOT = root


if __name__ == "__main__":
	main()
