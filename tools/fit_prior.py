"""Fit doppel's whitening prior from a populated cache.

Reads image descriptors from doppel's cache for paths containing MATCH (by
default the COCO originals of the sample corpus) and writes their mean and
covariance as assets/whiten-prior.bin: b"DPWP", u32 dim, f32 mean[dim],
f32 cov[dim * dim], all little-endian.

Needs numpy.

python -I tools/fit_prior.py CACHE.sqlite OUT [MATCH]
"""

import sqlite3
import struct
import sys

import numpy as np

DIM = 768
cache, out = sys.argv[1], sys.argv[2]
match = sys.argv[3] if len(sys.argv) > 3 else "/corpus/originals/coco-"
db = sqlite3.connect(cache)
rows = []
for path, blob in db.execute(
		"SELECT path, embed FROM files WHERE embed IS NOT NULL"):
	if match.encode() in path and len(blob) == DIM * 2:
		rows.append(np.frombuffer(blob, "<f2"))
x = np.asarray(rows, np.float64)
mean = x.mean(axis=0)
z = x - mean
cov = z.T @ z / len(x)
with open(out, "wb") as f:
	f.write(b"DPWP" + struct.pack("<I", DIM))
	f.write(mean.astype("<f4").tobytes())
	f.write(cov.astype("<f4").tobytes())
print(f"{len(x)} descriptors -> {out}")
