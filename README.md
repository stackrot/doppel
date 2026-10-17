# doppel

Duplicate and near-duplicate finder for images and videos that keeps every
part of the machine busy at once:

| Stage | Hardware |
| --- | --- |
| BLAKE3 hashing of same-size files (mmap, multi-threaded per file) | NVMe/SATA queues, its own thread pool |
| Image decode (libjpeg-turbo DCT downscaling), border trim, pHash | every CPU core (rayon) |
| Video: full decode, 1 frame per 2 s, scaled on the GPU | both VCN engines (VAAPI), plus software decoders |
| DINOv2 ViT-B/14, fp16 matmuls and attention, fp32 residual stream | GPU via Vulkan cooperative matrices (RDNA3 WMMA) |
| Whitened all-pairs similarity search (blocked fp16 matmul + top-k) | GPU |

The GPU runs through [burn](https://burn.dev) on Vulkan, so there is no ROCm
dependency. A cold run is GPU-bound; a CPU embedding worker was measured at
2% of the GPU's rate and left out. Results go in an SQLite cache
(`~/.cache/doppel/cache.sqlite`) keyed on path, size and mtime, so later runs
only analyse new or changed files.

## Matches

- **exact**: byte-identical.
- **near**: the same picture or footage re-encoded, resized, recompressed,
  format-converted or EXIF-rotated (pHash agrees, or both videos cover each
  other).
- **similar**: edited copies (crops, filters, overlays, borders, mirrored,
  trimmed or sped-up clips, compilations) and burst shots of one moment.

Each image (and each sampled frame) is described by GeM-pooled DINOv2 patch
tokens rather than the CLS token, which is too semantic: two different
photos of zebras score like an edited copy. Descriptors are PCA-whitened
against the library's own distribution, blended with a prior fitted on 45k
COCO photos (`assets/whiten-prior.bin`), so small libraries behave and large
ones adapt (a thousand screenshots of one desktop stop looking alike).
Letterbox, pillarbox and mat borders are trimmed before cropping.

Videos match when most sampled frames of the shorter one have a close frame
in the other, so a trimmed clip groups with its source.

Groups form around the best-connected file, and every member matches that
leader directly, so look-alikes can't chain into giant groups. The suggested
keeper is the best-ranked member: most pixels, then longest, highest
bitrate, largest, oldest, shortest path.

## Usage

```
doppel [-t 0.70] [--frame-threshold 0.60] [--exact] [--json] [--no-vaapi] DIR...
```

The first run downloads the DINOv2 weights (346 MB, pinned revision,
SHA-256 checked). Lower thresholds find more edits and more look-alikes.

Review keys: `↑↓` file, `←→`/Tab group, `space` toggle mark, `a` keep best
and mark the rest, `u` keep all, `o` open, `t` trash marked files in this
group, `T` trash every marked file, `q` quit. Exact and near groups start
marked (best kept); in similar groups only redundant byte-identical copies
(same letter in the `≡` column) start marked. Files go to the freedesktop
Trash, and a group with every copy marked is skipped.

## Sample corpus

`tools/make_samples.py` builds a test corpus with ground truth from COCO
images and Blender open movies (download list in its docstring);
`tools/score.py` scores `doppel --json` output against it, and
`tools/fit_prior.py` refits the whitening prior from a populated cache.

```
python -I tools/make_samples.py ROOT
doppel --json ROOT/corpus > out.json
python -I tools/score.py ROOT out.json
```

On that corpus (72k images, 94 videos, 13 GB; 9700X + RX 7800 XT): 185 s
cold, 4 s warm. Images: 0.95 precision against the leader (most misses are
burst shots), 0.94 recall (near 0.999). Videos: 1.00 precision, 0.96
recall.
