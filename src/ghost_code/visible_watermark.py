"""可視透かし(四隅ローテーション)。

シーン検出してシーン毎に四隅のどこかに 4 文字コードを表示する。
ffmpeg drawtext フィルタで 1 パスで実現。
"""
from __future__ import annotations

import random
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import scenedetect


CORNERS = ["TL", "TR", "BL", "BR"]
CORNER_POS = {
    "TL": ("$margin", "$margin"),
    "TR": ("W-tw-$margin", "$margin"),
    "BL": ("$margin", "H-th-$margin"),
    "BR": ("W-tw-$margin", "H-th-$margin"),
}


@dataclass
class ScenePlacement:
    start_sec: float
    end_sec: float
    corner: str  # "TL" | "TR" | "BL" | "BR"


def detect_scenes(video_path: Path, threshold: float = 27.0) -> list[tuple[float, float]]:
    """PySceneDetect でシーン境界(秒)を検出。閾値が低いほど細かく分割。"""
    scenes = scenedetect.detect(str(video_path), scenedetect.ContentDetector(threshold=threshold))
    if not scenes:
        # シーン無検出時は全体を1シーン扱い
        meta_dur = _probe_duration(video_path)
        return [(0.0, meta_dur)]
    return [(s.get_seconds(), e.get_seconds()) for s, e in scenes]


def _probe_duration(video_path: Path) -> float:
    out = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ], check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


def plan_corner_rotation(
    scenes: list[tuple[float, float]],
    seed: int = 0,
) -> list[ScenePlacement]:
    """各シーンに四隅をランダム割当。連続シーンで同じ四隅を避ける。"""
    rng = random.Random(seed)
    placements: list[ScenePlacement] = []
    last_corner: str | None = None
    for start, end in scenes:
        choices = [c for c in CORNERS if c != last_corner]
        corner = rng.choice(choices)
        placements.append(ScenePlacement(start, end, corner))
        last_corner = corner
    return placements


def build_drawtext_filter(
    placements: list[ScenePlacement],
    code: str,
    fontsize: int = 22,
    margin: int = 24,
    alpha: float = 0.35,
    fontfile: str | None = None,
) -> str:
    """ffmpeg の drawtext フィルタを四隅 × 時間範囲で構築。"""
    by_corner: dict[str, list[tuple[float, float]]] = {c: [] for c in CORNERS}
    for p in placements:
        by_corner[p.corner].append((p.start_sec, p.end_sec))

    parts: list[str] = []
    for corner, ranges in by_corner.items():
        if not ranges:
            continue
        x_expr, y_expr = CORNER_POS[corner]
        x_expr = x_expr.replace("$margin", str(margin))
        y_expr = y_expr.replace("$margin", str(margin))

        enable_expr = "+".join(f"between(t\\,{s:.3f}\\,{e:.3f})" for s, e in ranges)

        font_part = f":fontfile='{fontfile}'" if fontfile else ""
        # ffmpeg drawtext: 文字の背景は付けず、極薄白
        part = (
            f"drawtext=text='{code}'"
            f":x={x_expr}:y={y_expr}"
            f":fontcolor=white@{alpha}"
            f":fontsize={fontsize}"
            f":borderw=1:bordercolor=black@{alpha * 0.6:.2f}"
            f"{font_part}"
            f":enable='{enable_expr}'"
        )
        parts.append(part)
    return ",".join(parts)


def embed_visible_to_video(
    src: Path, dst: Path, code: str,
    seed: int = 0,
    fontsize: int = 22, margin: int = 24, alpha: float = 0.35,
    crf: int = 18,
    scene_threshold: float = 27.0,
) -> list[ScenePlacement]:
    """src → dst に四隅ローテーションで code を可視描画する。"""
    scenes = detect_scenes(src, threshold=scene_threshold)
    placements = plan_corner_rotation(scenes, seed=seed)
    vf = build_drawtext_filter(placements, code, fontsize=fontsize, margin=margin, alpha=alpha)

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return placements
