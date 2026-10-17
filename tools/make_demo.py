"""Build the small library used for the README demo from the sample corpus.

Picks five photo families and two video families from make_samples.py
output and copies them, with 120 unrelated photos and two unrelated clips,
into OUT/Photos and OUT/Videos under phone-style names and dates.

python -I tools/make_demo.py ROOT OUT
"""

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

FAMILIES = {
	"cat": "000000112798.jpg",
	"elephant": "000000149939.jpg",
	"bus": "000000464463.jpg",
	"tennis": "000000351270.jpg",
	"plane": "000000152885.jpg",
}
SUFFIX = {
	"crop80": "crop", "gray": "bw", "blur": "soft", "mirror": "flip",
	"sticker": "collage", "border": "framed", "colour": "vivid",
	"upscale": "large", "rotate7": "tilt", "bright": "bright",
	"resize50": "small", "resize25": "thumb", "jpeg_q40": "web",
}
VIDEOS = {
	"originals/clips/clip010.mp4": ("Videos/Downloads/sintel_scene.mp4", 2023),
	"library/old laptop/Pictures/2f74ac9148dd.mp4":
		("Videos/Edits/sintel_scene_4x3.mp4", 2024),
	"library/edits/e63d7c73dabb.mp4":
		("Videos/Phone/VID_20240301_191233.mp4", 2024),
	"library/downloads/46a106cc36c2.mp4":
		("Videos/Shared/VID-20240302-WA0004.mp4", 2024),
	"originals/clips/clip000.mp4": ("Videos/Downloads/big_buck_bunny.mp4", 2023),
	"library/exports/web/82f47beff2a6.mp4":
		("Videos/Shared/bunny_720p.mp4", 2024),
	"library/edits/487317308327.mp4": ("Videos/Shared/bunny_small.mp4", 2024),
	"library/phone/WhatsApp Images/edc3d9575c3c.mp4":
		("Videos/Edits/bunny_clip.mp4", 2024),
	"originals/clips/clip013.mp4": ("Videos/Downloads/tears_of_steel.mp4", 2023),
	"originals/clips/clip021.mp4": ("Videos/Downloads/elephants_dream.mp4", 2023),
}

rng = random.Random(7)


def camera_name():
	return (f"IMG_2023{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"
		f"_{rng.randint(80000, 215959):06d}")


def main():
	root, out = Path(sys.argv[1]), Path(sys.argv[2])
	corpus = root / "corpus"
	images = json.loads((root / "manifest.json").read_text())["images"]
	by = {}
	for p, v in images.items():
		by.setdefault(v["source"], []).append((p, v["transform"]))

	def put(src, rel, year):
		dst = out / rel
		dst.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(corpus / src, dst)
		t = time.mktime((year, rng.randint(1, 12), rng.randint(1, 28),
			rng.randint(8, 21), rng.randint(0, 59), 0, 0, 0, -1))
		os.utime(dst, (t, t))

	for name, source in FAMILIES.items():
		files = by[source]
		orig = next(p for p, t in files if t == "original")
		img = camera_name()
		put(orig, f"Photos/Camera/{img}.jpg", 2023)
		if name in ("cat", "elephant"):
			put(orig, f"Photos/Backup 2023/{img}.jpg", 2024)
		for p, t in files:
			ext = Path(p).suffix
			if t in ("original", "copy"):
				continue
			if t == "text":
				rel = (f"Photos/WhatsApp/IMG-2024{rng.randint(1, 12):02d}"
					f"{rng.randint(1, 28):02d}-WA{rng.randint(1, 99):04d}{ext}")
			elif t in ("webp", "avif", "jxl", "heic", "png"):
				rel = f"Photos/Downloads/{name}{ext}"
			else:
				rel = f"Photos/Edits/{name}_{SUFFIX.get(t, t)}{ext}"
			put(p, rel, 2024)
	used = set(FAMILIES.values())
	singles = [s for s, ps in by.items() if len(ps) == 1 and s not in used]
	for s in rng.sample(sorted(singles), 120):
		put(by[s][0][0], f"Photos/Camera/{camera_name()}.jpg", 2023)
	for src, (rel, year) in VIDEOS.items():
		put(src, rel, year)


if __name__ == "__main__":
	main()
