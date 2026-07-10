"""
Senxe Cerebellum — Video Utilities
================================
Shared video saving utility.

Provides a utility for creating benchmark demonstration videos:

- ``save_video()``: Encodes a frame list to MP4 with automatic temporal
  downsampling to a target duration.

Uses imageio + ffmpeg for headless-compatible encoding
(no display server required), making it suitable for CI pipelines
and remote server execution.
"""

from __future__ import annotations

import numpy as np
import imageio
import cv2
from tqdm import tqdm
from typing import List, Optional


def save_video(
    frames: List[np.ndarray],
    path: str,
    fps: int = 30,
    target_seconds: int = 20,
) -> None:
    """Save a list of frames as an MP4 video with automatic downsampling.

    If the total frame count exceeds ``fps × target_seconds``, frames are
    uniformly subsampled to fit the target duration while preserving
    temporal coverage across the full recording.

    Args:
        frames: List of (H, W, 3) uint8 RGB frames.
        path: Output file path (e.g., ``"output.mp4"``).
        fps: Video frame rate (default: 30).
        target_seconds: Approximate maximum video duration in seconds.
    """
    if not frames:
        print(f"  [Warning] No frames to save for {path}")
        return

    target = fps * target_seconds
    if len(frames) > target:
        idx = np.linspace(0, len(frames) - 1, target, dtype=int)
        frames = [frames[i] for i in idx]

    print(f"\n  Saving: {path} ({len(frames)} frames, ~{len(frames)/fps:.0f}s)")
    writer = imageio.get_writer(path, fps=fps, quality=8)
    for f in tqdm(frames, desc="  Encode", ncols=90, leave=False):
        writer.append_data(f)
    writer.close()
    print(f"  Saved: {path}")
