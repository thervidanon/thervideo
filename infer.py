"""Turn an RGB video into a thermal video with TherVideo.

    python infer.py --input rgb.mp4 --output thermal.mp4 \
        --wan_dir checkpoints/Wan2.2-TI2V-5B-Diffusers --checkpoint checkpoints/thervideo_offline.safetensors
"""

import argparse
import os

import numpy as np

from thervideo import TherVideoPipeline
from thervideo.video import read_video, write_video


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="RGB video file or directory of frames")
    ap.add_argument("--output", required=True, help="output .mp4")
    ap.add_argument("--wan_dir", default="checkpoints/Wan2.2-TI2V-5B-Diffusers")
    ap.add_argument("--checkpoint", default="checkpoints/thervideo_offline.safetensors")
    ap.add_argument("--sensor", default="radiometric", choices=["radiometric", "agc8"],
                    help="thermal camera style: radiometric (fixed per-clip scaling) or 8-bit auto-gain")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--side_by_side", action="store_true", help="write RGB and thermal next to each other")
    a = ap.parse_args()

    frames = read_video(a.input, a.max_frames)
    print(f"{len(frames)} frames at 640x512, 10 Hz")
    pipe = TherVideoPipeline.from_pretrained(a.wan_dir, a.checkpoint)
    thermal = pipe(frames, sensor=a.sensor, steps=a.steps, guidance=a.guidance, seed=a.seed)
    out = np.concatenate([frames, np.repeat(thermal[..., None], 3, -1)], 2) if a.side_by_side else thermal
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    write_video(a.output, out)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
