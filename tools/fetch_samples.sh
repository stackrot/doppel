#!/bin/sh
# Downloads the sample corpus sources (about 12 GB) into ROOT/downloads/,
# laid out the way make_samples.py expects.
set -eu
root=${1:?usage: fetch_samples.sh ROOT}
blender=https://download.blender.org
coco=http://images.cocodataset.org/zips
get() {
	mkdir -p "$root/downloads/$1"
	curl -fL --progress-bar -C - -o "$root/downloads/$1/$2" "$3"
}
get coco-val val2017.zip "$coco/val2017.zip"
get coco-test test2017.zip "$coco/test2017.zip"
get bbb bbb.mov.zip "$blender/peach/bigbuckbunny_movies/big_buck_bunny_1080p_h264.mov.zip"
get sintel sintel.mkv "$blender/durian/movies/Sintel.2010.1080p.mkv"
get tos tos.mov "$blender/demo/movies/ToS/tears_of_steel_720p.mov"
get ed ed.mov "$blender/ED/elephantsdream-720-h264-st-aac.mov"
get cycles cycles.mov "$blender/demo/movies/Cycles_Demoreel_2015.mov"
get reel reel.mov "$blender/demo/movies/Blender_reel_2013.mov"
