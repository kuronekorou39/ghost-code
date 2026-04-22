"""動画への透かし埋め込み・復号。

埋め込み:入力動画の全フレームに同じビット列を TrustMark で埋め込み、H.264 で再エンコード。
復号:等間隔サンプリングで N フレーム取り出し、各フレームで復号、ビット毎に多数決。

Phase 2 では per-frame で同じペイロードを埋め込む(時間冗長化で検出強度UP)。
A/B セグメント方式は Phase 3 で導入。
"""
from __future__ import annotations

import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import ffmpeg
import numpy as np
from PIL import Image
from tqdm import tqdm

from ghost_code.video_io import probe_video
from ghost_code.watermark import Watermarker, compute_psnr


@dataclass
class VideoEmbedResult:
    output_path: Path
    num_frames: int
    avg_psnr_db: float
    elapsed_sec: float


def _ffmpeg_reader(path: Path, width: int, height: int) -> subprocess.Popen:
    """動画を raw RGB ストリームとして読み出す。"""
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _ffmpeg_writer(
    path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int = 18,
    audio_from: Path | None = None,
) -> subprocess.Popen:
    """raw RGB を受け取り H.264 動画として書き出す。"""
    inputs = ["-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", f"{width}x{height}", "-r", str(fps), "-i", "pipe:"]
    if audio_from is not None:
        inputs += ["-i", str(audio_from)]

    maps = []
    if audio_from is not None:
        maps += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy", "-shortest"]

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        *inputs,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        *maps,
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def embed_video(
    input_path: Path,
    output_path: Path,
    bits: str,
    watermarker: Watermarker,
    strength: float = 1.0,
    crf: int = 18,
    max_frames: int | None = None,
    psnr_every: int = 30,
) -> VideoEmbedResult:
    """全フレームに `bits` を埋め込み H.264 で書き出す。"""
    import time

    meta = probe_video(input_path)
    w, h, fps = meta["width"], meta["height"], meta["fps"]
    frame_bytes = w * h * 3

    reader = _ffmpeg_reader(input_path, w, h)
    writer = _ffmpeg_writer(output_path, w, h, fps, crf=crf, audio_from=input_path)

    t0 = time.perf_counter()
    n = 0
    psnr_sum = 0.0
    psnr_n = 0

    total_est = int(meta["duration_sec"] * fps)
    pbar = tqdm(total=(max_frames or total_est), desc=f"embed {output_path.name}", leave=False)

    try:
        while True:
            if max_frames is not None and n >= max_frames:
                break
            raw = reader.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            cover_img = Image.fromarray(frame)
            result = watermarker.embed(cover_img, bits, strength=strength)
            stego_np = np.asarray(result.stego, dtype=np.uint8)
            writer.stdin.write(stego_np.tobytes())

            if n % psnr_every == 0:
                psnr_sum += result.psnr_db
                psnr_n += 1
            n += 1
            pbar.update(1)
    finally:
        pbar.close()
        reader.stdout.close()
        reader.wait(timeout=5)
        if writer.stdin:
            writer.stdin.close()
        writer.wait(timeout=120)

    elapsed = time.perf_counter() - t0
    avg_psnr = psnr_sum / max(psnr_n, 1)
    return VideoEmbedResult(output_path, n, avg_psnr, elapsed)


def extract_sample_frames(
    video_path: Path,
    n_samples: int = 20,
) -> tuple[list[Image.Image], list[float]]:
    """動画から等間隔で N フレーム抽出。戻り値: (フレーム群, 対応秒位置)。

    ffmpeg で PNG に一時書き出し → PIL 読み込み。回転メタデータや SAR 等を
    ffmpeg 側に委ねるため、スマホ縦動画などでも破綻しない。
    """
    import subprocess
    import tempfile

    meta = probe_video(video_path)
    duration = meta["duration_sec"]
    if n_samples == 1:
        timestamps = [duration / 2]
    else:
        margin = duration * 0.05
        timestamps = np.linspace(margin, duration - margin, n_samples).tolist()

    tmp = Path(tempfile.mkdtemp(prefix="ghost_frames_"))
    frames: list[Image.Image] = []
    ts_out: list[float] = []
    for i, ts in enumerate(timestamps):
        out_png = tmp / f"f_{i:04d}.png"
        # -ss を -i の後に置いて正確 seek。回転・色は ffmpeg が PNG に反映。
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(video_path),
            "-ss", str(ts), "-frames:v", "1",
            str(out_png),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            continue
        if not out_png.exists():
            continue
        # load and copy into memory so we can clean up the file
        img = Image.open(out_png).convert("RGB").copy()
        frames.append(img)
        ts_out.append(ts)
    return frames, ts_out


@dataclass
class VideoDecodeResult:
    voted_bits: str
    per_frame_bits: list[str | None]
    detected_frames: int
    num_frames: int


def decode_video_by_vote(
    video_path: Path,
    watermarker: Watermarker,
    n_samples: int = 20,
    reference_video: Path | None = None,
    reference_image: Path | None = None,
    reference_bank_size: int = 12,
) -> VideoDecodeResult:
    """N フレームサンプリング + ビット毎の多数決で最終 bit 列を決定。

    reference_video: 原本動画。与えると 12 フレームの「参照バンク」を作り、
                     各キャプチャフレームに対しバンク全体から最も inliers の多い
                     組み合わせを採用(time offset/ループ再生にも耐える)。
    reference_image: 単一画像を参照として使用(コンテンツ静的な動画向け)。
    """
    import tempfile

    from ghost_code.screen_extract import align_to_reference

    frames, timestamps = extract_sample_frames(video_path, n_samples=n_samples)
    per_frame: list[str | None] = []

    # 参照バンク:原本動画からサンプリングしてファイル化
    ref_bank: list[Path] = []
    if reference_video is not None and reference_video.exists():
        tmp_root = Path(tempfile.mkdtemp(prefix="ghost_refbank_"))
        ref_imgs, _ = extract_sample_frames(reference_video, n_samples=reference_bank_size)
        for i, img in enumerate(ref_imgs):
            p = tmp_root / f"ref_{i:03d}.png"
            img.save(p)
            ref_bank.append(p)
    if reference_image is not None and Path(reference_image).exists():
        ref_bank.append(Path(reference_image))

    frame_tmpdir = Path(tempfile.mkdtemp(prefix="ghost_frm_"))

    for i, f in enumerate(frames):
        # 1. 直接デコード
        decoded: str | None = None
        try:
            bits, detected = watermarker.extract(f)
            if detected:
                decoded = bits
        except Exception:
            pass

        # 2. 位置合わせ後にデコード(参照バンクから最適なものを探索)
        if decoded is None and ref_bank:
            frame_path = frame_tmpdir / f"frame_{i:03d}.png"
            f.save(frame_path)
            best_warped = None
            best_inliers = 0
            for ref_path in ref_bank:
                aligned = align_to_reference(frame_path, ref_path)
                if aligned is None:
                    continue
                warped, _H, inliers = aligned
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_warped = warped

            if best_warped is not None and best_inliers >= 15:
                # 通常 → 回転許容 の2段構え
                for rot in (False, True):
                    try:
                        bits, detected = watermarker.extract(best_warped, rotation=rot)
                        if detected:
                            decoded = bits
                            break
                    except Exception:
                        pass

        per_frame.append(decoded)

    valid = [b for b in per_frame if b is not None]
    if not valid:
        return VideoDecodeResult("", per_frame, 0, len(frames))

    # 全候補のビット長の中央値を採用(Noneや長さ異常を除外)
    from statistics import median
    L = int(median(len(b) for b in valid))
    valid = [b for b in valid if len(b) == L]

    voted = []
    for i in range(L):
        counter = Counter(b[i] for b in valid)
        voted.append(counter.most_common(1)[0][0])
    return VideoDecodeResult("".join(voted), per_frame, len(valid), len(frames))
