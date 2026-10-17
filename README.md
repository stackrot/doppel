# doppel

doppel finds duplicate images and videos on Linux, including copies that
have been resized, re-encoded, cropped, filtered, watermarked or trimmed. It
runs every stage in parallel so the CPU, GPU, video decoders and disks all
stay busy, then opens a terminal screen where you review each group and send
the copies you don't want to the Trash.

It needs a GPU with a Vulkan driver. There is no CUDA or ROCm dependency,
so AMD cards work out of the box.

## What it finds

| Match | Meaning |
| --- | --- |
| exact | Byte-identical files. |
| near | The same picture or footage saved again: resized, recompressed, converted to another format, or stored rotated with an EXIF tag. |
| similar | Edited copies (crops, colour changes, blur, text or logos on top, added borders, mirrored, trimmed or sped-up clips, compilations) and burst shots of the same moment. |

A trimmed clip groups with the video it came from, and a compilation groups
with each of the clips inside it.

## Requirements

- Linux with a Vulkan GPU driver. Software renderers such as lavapipe are
  ignored because they would take hours.
- ffmpeg and ffprobe.
- A VAAPI driver if you want video decoded on the GPU. Without one, doppel
  decodes on the CPU.
- libjpeg-turbo, SQLite and curl.
- Rust 1.95 or newer to build it.

Only one setup has been tested: an AMD RX 7800 XT with Mesa's RADV driver on
CachyOS. NVIDIA and Intel GPUs should work through Vulkan but have not been
tried.

To see images in the review screen, use a terminal that supports the kitty
graphics protocol or sixel, such as kitty, WezTerm, foot or Konsole. Other
terminals show a coarse preview made of block characters.

## Install

Clone the repository, then:

```
cargo install --path .
```

The first run downloads the DINOv2 model weights (346 MB) into
`~/.cache/doppel/` and checks them against a pinned SHA-256.

## Usage

```
doppel ~/Pictures ~/Videos
```

doppel keeps what it learns about each file in `~/.cache/doppel/cache.sqlite`
and only analyses new or changed files on later runs. It skips hidden
directories (including the Trash), does not follow symlinks, and counts
hardlinks to the same file once.

| Option | Effect |
| --- | --- |
| `-t, --threshold` | How alike two images must be to match, from 0 to 1 (default 0.70). Lower values catch heavier edits but also more look-alike photos. |
| `--frame-threshold` | The same for video frames (default 0.60). |
| `-e, --exact` | Only find byte-identical files. No decoding, no GPU. |
| `--json` | Print the groups as JSON instead of opening the review screen. |
| `--no-vaapi` | Decode video on the CPU. |

`doppel --help` lists the rest.

### Reviewing

| Key | Action |
| --- | --- |
| `↑` `↓` | Move between files |
| `←` `→` or Tab | Move between groups |
| Space | Mark or unmark a file |
| `a` | Keep the best file, mark the rest |
| `u` | Keep every file in the group |
| `o` | Open the file in its default viewer |
| `t` | Move the marked files in this group to the Trash |
| `T` | Move every marked file to the Trash |
| `q` | Quit |

The best file in each group (most pixels, then longest, highest bitrate,
largest, oldest) is listed first with a star. In exact and near groups,
everything else starts marked. In similar groups, only spare copies of
byte-identical files start marked; those share a letter in the `≡` column.
Files go to the freedesktop Trash, so you can restore them, and doppel will
not empty a group completely.

## How it works

| Stage | Runs on |
| --- | --- |
| BLAKE3 hashing of files that share a size | disks, on its own thread pool |
| Image decoding, border trimming, perceptual hash | all CPU cores; JPEGs are downscaled while decoding |
| Video decoding, one frame every 2 seconds (less often past 20 minutes) | the GPU's video engines through VAAPI, plus CPU decoders |
| DINOv2 ViT-B/14 descriptors | GPU, through Vulkan with fp16 matrix units |
| Comparing every file with every other | GPU |

The GPU code uses [burn](https://burn.dev). On large libraries the model is
the slowest stage and the other stages keep it busy. Running the model on the
CPU as well was tried and added only 2%.

Each picture, and each sampled video frame, is described by GeM-pooled
DINOv2 patch features rather than the model's CLS token. The CLS token is
too concerned with what a photo shows, so two different photos of zebras can
score like an edited copy. doppel then whitens the descriptors against your
library's own statistics, blended with a prior computed from 45,000 COCO
photos. In a folder of a thousand screenshots of one desktop, that stops the
shared desktop from making every screenshot match. A perceptual hash
confirms near copies that the model alone rates lower, such as heavy
downscales.

Two videos match when most sampled frames of the shorter one have a close
frame in the other.

Each group forms around the file with the most matches, and every other file
in the group matches that file directly. This stops a chain of look-alikes
from merging into one huge group.

## Accuracy and speed

These figures come from a synthetic test corpus, not real photo libraries:
72,340 images (45,670 COCO photos plus edited copies of 6,000 of them) and
94 videos cut from Blender open movies, 13 GB in total. On a Ryzen 7 9700X
with an RX 7800 XT, the first run took 185 seconds and a repeat run took 4.

| | Precision | Recall | Near-copy recall |
| --- | --- | --- | --- |
| Images | 0.95 | 0.94 | 0.999 |
| Videos | 1.00 | 0.96 | 1.00 |

Precision counts how many files in each group really match the file the
group formed around. Most of the image misses are burst shots of the same
scene, which the corpus labels as different photos. Crops that keep only
36% of the picture are mostly missed (26% found).

To rebuild the corpus and score doppel against it:

```
tools/fetch_samples.sh ROOT
python -I tools/make_samples.py ROOT
doppel --json ROOT/corpus > out.json
python -I tools/score.py ROOT out.json
```

`tools/fit_prior.py` recomputes `assets/whiten-prior.bin` from a cache
filled by running doppel on the corpus.

## Licence

MIT. The DINOv2 weights (Meta, Apache-2.0) are downloaded at first run, not
bundled. `assets/whiten-prior.bin` holds only the mean and covariance of
DINOv2 descriptors computed from COCO 2017 images.
