"""動画フレーム入出力ユーティリティ(ffmpeg-python ベース)。"""
from __future__ import annotations

from pathlib import Path

import ffmpeg
import numpy as np
from PIL import Image


def probe_video(path: str | Path) -> dict:
    info = ffmpeg.probe(str(path))
    video_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": eval(video_stream["r_frame_rate"]),  # noqa: S307 — "30000/1001" 等の形式
        "duration_sec": float(info["format"]["duration"]),
        "codec": video_stream["codec_name"],
    }


def extract_frame(path: str | Path, timestamp_sec: float = 1.0) -> Image.Image:
    """動画から指定秒位置のフレームを 1 枚抽出して PIL で返す。"""
    meta = probe_video(path)
    width, height = meta["width"], meta["height"]

    out, _ = (
        ffmpeg.input(str(path), ss=timestamp_sec)
        .output("pipe:", vframes=1, format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, capture_stderr=True, quiet=True)
    )
    frame = np.frombuffer(out, dtype=np.uint8).reshape(height, width, 3)
    return Image.fromarray(frame)
