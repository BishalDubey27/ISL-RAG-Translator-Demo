"""
Small video helper utilities used by dataset utilities.
"""

import os
from typing import List

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}


def is_video_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in VIDEO_EXTS


def list_videos_in_dir(base_dir: str) -> List[str]:
    """Recursively list video files under `base_dir`.

    Returns absolute paths.
    """
    videos = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if is_video_file(f):
                videos.append(os.path.join(root, f))
    return videos
