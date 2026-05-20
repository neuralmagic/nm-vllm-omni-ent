#!/usr/bin/env python3
"""Symlink imageio-ffmpeg's bundled binary into /usr/local/bin.

vllm-omni model pipelines (e.g. Qwen3-Omni) shell out to ``ffmpeg`` by name.
Rather than pulling ffmpeg from an external RPM repo (RPM Fusion mirrors are
unreliable on GHA runners and unsupportable for productized images), we reuse
the static binary that imageio-ffmpeg already bundles.
"""

import shutil
import sys
from pathlib import Path

import imageio_ffmpeg

DEST_DIR = Path("/usr/local/bin")

src = Path(imageio_ffmpeg.get_ffmpeg_exe())
if not src.exists():
    print(f"ERROR: imageio-ffmpeg binary not found at {src}", file=sys.stderr)
    sys.exit(1)

for name in ("ffmpeg", "ffprobe"):
    dst = DEST_DIR / name
    # imageio-ffmpeg only ships ffmpeg; skip ffprobe if not present
    if name != "ffmpeg":
        candidate = src.parent / name
        if not candidate.exists():
            print(f"  skip {name} (not bundled by imageio-ffmpeg)")
            continue
        src_bin = candidate
    else:
        src_bin = src

    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src_bin)
    print(f"  {dst} -> {src_bin}")

# Verify it's callable
version = shutil.which("ffmpeg")
if version:
    print(f"OK: ffmpeg available at {version}")
else:
    print("ERROR: ffmpeg not on PATH after shimming", file=sys.stderr)
    sys.exit(1)
