"""動画への改変攻撃群。流出時に想定される編集・再圧縮を再現する。

各関数は (入力 Path, 出力 Path, **パラメータ) を取って出力を生成する。
ffmpeg のフィルタで統一。
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def reencode(src: Path, dst: Path, crf: int = 28, codec: str = "libx264") -> None:
    """CRF 指定での再エンコード。28 = SNS 再UP 相当、35 = 低画質転載相当。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-c:v", codec, "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def crop_center(src: Path, dst: Path, ratio: float = 0.8) -> None:
    """中央を指定割合でクロップ(ratio=0.8 → 80% の領域を残す)。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"crop=iw*{ratio}:ih*{ratio}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def resize(src: Path, dst: Path, height: int = 720) -> None:
    """高さを指定値に。幅はアスペクト比維持(偶数化)。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def adjust_eq(
    src: Path, dst: Path,
    brightness: float = 0.0, contrast: float = 1.0,
    saturation: float = 1.0, gamma: float = 1.0,
) -> None:
    """明度/コントラスト/彩度/ガンマ補正。"""
    eq = f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}:gamma={gamma}"
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", eq,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def blur(src: Path, dst: Path, sigma: float = 1.0) -> None:
    """Gaussian blur。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"gblur=sigma={sigma}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def add_noise(src: Path, dst: Path, strength: int = 10) -> None:
    """ランダムノイズ付加(0-100、体感 10-30 程度で SNS 加工相当)。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"noise=alls={strength}:allf=t",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def trim(src: Path, dst: Path, duration_sec: float = 5.0, start_sec: float = 1.0) -> None:
    """先頭からの切り抜き。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-ss", str(start_sec), "-t", str(duration_sec),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def change_fps(src: Path, dst: Path, fps: int = 15) -> None:
    """フレームレート変更。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"fps={fps}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def rotate(src: Path, dst: Path, angle_deg: float = 3.0) -> None:
    """わずかに回転(SNS投稿で意図的に撹乱された想定)。"""
    import math
    rad = math.radians(angle_deg)
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", f"rotate={rad}:fillcolor=black",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])


def combined_sns_upload(src: Path, dst: Path) -> None:
    """SNS 再アップの典型的な合成攻撃(クロップ+リサイズ+再エンコード)。"""
    _run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-vf", "crop=iw*0.9:ih*0.9,scale=-2:720",
        "-c:v", "libx264", "-preset", "medium", "-crf", "30",
        "-pix_fmt", "yuv420p", "-c:a", "copy",
        str(dst),
    ])
