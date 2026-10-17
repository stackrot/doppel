"""Record the README demo GIF.

Runs kitty in a headless cage session, so nothing appears on screen, types
a doppel session through kitty's remote control, records it with
wf-recorder and encodes it with gifski. The analysis part is sped up. Needs
cage, kitty, wf-recorder, ffmpeg and gifski. LIBRARY must contain Photos/
and Videos/; marked files from the first group end up in the Trash.

python -I tools/record_demo.py LIBRARY OUT.gif
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SPEEDUP = 4
FPS = 15
WIDTH = 1280


def running(pid):
	"""Whether pid is alive. kitty inherits the recorder from the launch
	script and never reaps it, so a finished recorder stays a zombie."""
	try:
		stat = Path(f"/proc/{pid}/stat").read_text()
	except FileNotFoundError:
		return False
	return stat.rsplit(")", 1)[1].split()[0] != "Z"


def main():
	lib, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
	work = Path(tempfile.mkdtemp(prefix="doppel-demo-"))
	sock = work / "kitty.sock"
	raw = work / "raw.mkv"
	cache = work / "cache"
	(cache / "doppel").mkdir(parents=True)
	weights = Path.home() / ".cache/doppel/dinov2-base.safetensors"
	if weights.exists():
		(cache / "doppel" / weights.name).symlink_to(weights)
	inner = work / "inner.sh"
	inner.write_text(f"""#!/bin/sh
date +%s.%N > '{work}/rec.start'
wf-recorder -y -r {FPS * 2} -f '{raw}' -c libx264 -p crf=10 \\
	-p preset=veryfast >/dev/null 2>&1 &
echo $! > '{work}/rec.pid'
exec kitty --listen-on 'unix:{sock}' -o allow_remote_control=socket-only \\
	-o font_size=11 -o cursor_blink_interval=0 \\
	-o confirm_os_window_close=0 --directory '{lib}' \\
	env XDG_CACHE_HOME='{cache}' PS1='$ ' bash --norc --noprofile
""")
	inner.chmod(0o755)
	env = dict(os.environ, WLR_BACKENDS="headless",
		WLR_HEADLESS_OUTPUTS="1", WLR_LIBINPUT_NO_DEVICES="1")
	env.pop("WAYLAND_DISPLAY", None)
	env.pop("DISPLAY", None)
	cage = subprocess.Popen(["cage", "-d", "--", str(inner)], env=env,
		stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

	def kitty(*args):
		return subprocess.run(["kitty", "@", "--to", f"unix:{sock}", *args],
			capture_output=True, text=True).stdout

	def wait_for(needle, timeout):
		end = time.time() + timeout
		while time.time() < end:
			if sock.exists() and needle in kitty("get-text"):
				return
			time.sleep(0.1)
		raise SystemExit(f"timed out waiting for {needle!r}")

	def key(k, pause):
		kitty("send-key", k)
		time.sleep(pause)

	wait_for("$", 30)
	time.sleep(1.5)
	start = time.time()
	time.sleep(0.8)
	for ch in "doppel Photos Videos":
		kitty("send-text", ch)
		time.sleep(0.07)
	time.sleep(0.4)
	key("enter", 0)
	analysis = time.time()
	wait_for("groups", 300)
	tui = time.time()
	time.sleep(3.5)
	for k, pause in [
		("down", 1.6), ("down", 1.6), ("down", 1.6),
		("right", 2.6), ("down", 1.4), ("down", 1.4), ("down", 1.4),
		("down", 1.6), ("right", 2.4), ("down", 1.4), ("down", 1.6),
		("end", 2.6), ("down", 1.6), ("home", 2.0),
		("t", 2.0), ("y", 2.6), ("q", 1.5),
	]:
		key(k, pause)
	end = time.time()
	pid = int((work / "rec.pid").read_text())
	# The recording is finalised on SIGINT, but in a headless session
	# wf-recorder can then wait forever for another frame.
	os.kill(pid, signal.SIGINT)
	deadline = time.time() + 5
	while running(pid) and time.time() < deadline:
		time.sleep(0.1)
	if running(pid):
		os.kill(pid, signal.SIGKILL)
	cage.terminate()
	cage.wait(timeout=10)

	rec = float((work / "rec.start").read_text())
	a, b, c, d = (start - rec, analysis - rec + 0.5, tui - rec, end - rec)
	cut = work / "cut.mp4"
	subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(raw),
		"-filter_complex",
		f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS[x];"
		f"[0:v]trim={b}:{c},setpts=(PTS-STARTPTS)/{SPEEDUP}[y];"
		f"[0:v]trim={c}:{d},setpts=PTS-STARTPTS[z];"
		f"[x][y][z]concat=n=3:v=1,fps={FPS}[v]",
		"-map", "[v]", "-c:v", "libx264", "-crf", "8", str(cut)],
		check=True)
	subprocess.run(["gifski", "--quiet", "--fps", str(FPS), "--width",
		str(WIDTH), "--quality", "90", "--motion-quality", "100",
		"--lossy-quality", "100", "-o", str(out), str(cut)], check=True)
	print(f"{out} ({out.stat().st_size / 2**20:.1f} MiB), "
		f"analysis {c - b:.0f} s shown at {SPEEDUP}x; work files in {work}")


if __name__ == "__main__":
	main()
