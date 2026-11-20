"""Video input/output helpers: 10 Hz, centre crop to 5:4, 640x512."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

W, H, FPS = 640, 512, 10


def to_model_frame(frame: np.ndarray) -> np.ndarray:
    """Centre-crop an RGB frame to 5:4 and resize it to 640x512."""
    import cv2

    h, w = frame.shape[:2]
    if w * 4 > h * 5:
        nw = int(round(h * 5 / 4)); x0 = (w - nw) // 2; frame = frame[:, x0:x0 + nw]
    else:
        nh = int(round(w * 4 / 5)); y0 = (h - nh) // 2; frame = frame[y0:y0 + nh]
    return cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)


def read_video(path: str, max_frames: Optional[int] = None) -> np.ndarray:
    """Read a video file or a directory of images as uint8 RGB [N, 512, 640, 3], resampled to about 10 Hz."""
    import cv2

    frames = []
    if os.path.isdir(path):
        names = sorted(f for f in os.listdir(path) if f.lower().endswith((".png", ".jpg", ".jpeg")))
        for name in names:
            img = cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR)
            frames.append(to_model_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if max_frames and len(frames) >= max_frames:
                break
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or FPS
        step = max(1, int(round(fps / FPS)))
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if i % step == 0:
                frames.append(to_model_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                if max_frames and len(frames) >= max_frames:
                    break
            i += 1
        cap.release()
    if not frames:
        raise ValueError(f"no frames read from {path}")
    return np.stack(frames)


def write_video(path: str, frames: np.ndarray, fps: int = FPS) -> None:
    """Write uint8 frames ([N, H, W] grey or [N, H, W, 3] RGB) as an H.264 mp4."""
    import imageio

    if frames.ndim == 3:
        frames = np.repeat(frames[..., None], 3, -1)
    imageio.mimwrite(path, list(frames), fps=fps, quality=8, macro_block_size=1)
