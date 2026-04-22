"""任意入力(画像 or 動画)から透かしを復元し、台帳と照合する検出パイプライン。

画像:
  1. 直接デコード
  2. 失敗 or BER 大なら cover.png との ORB 位置合わせ後に再デコード
  3. 台帳(image)から Hamming 距離最小のエントリを返す

動画:
  1. 等間隔で N フレームをサンプリング
  2. 各フレームで直接デコード
  3. ビット毎の多数決で最終 bit 列を決定
  4. 台帳(video)から Hamming 距離最小のエントリを返す
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from ghost_code.registry import TokenEntry, find_best_match, load_registry
from ghost_code.screen_extract import align_to_reference
from ghost_code.video_watermark import decode_video_by_vote
from ghost_code.watermark import Watermarker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_REF = ROOT / "outputs" / "phase1" / "cover.png"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class DetectionResult:
    success: bool
    media_type: Literal["image", "video", "unknown"]
    method: str
    extracted_bits: str | None
    matched_entry: TokenEntry | None
    hamming_distance: int | None
    confidence: float | None
    message: str
    # 動画特有
    frames_detected: int | None = None
    frames_total: int | None = None


_wm_cache: Watermarker | None = None


def get_watermarker() -> Watermarker:
    global _wm_cache
    if _wm_cache is None:
        _wm_cache = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)
    return _wm_cache


def _classify(path: Path) -> Literal["image", "video", "unknown"]:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def _detect_image(
    image_path: Path,
    reference_path: Path,
    max_hamming: int | None,
) -> DetectionResult:
    wm = get_watermarker()
    img = Image.open(image_path).convert("RGB")

    candidates: list[tuple[str, str]] = []
    try:
        bits_a, det_a = wm.extract(img)
        if bits_a:
            candidates.append(("direct", bits_a))
    except Exception:
        pass

    aligned = align_to_reference(image_path, reference_path)
    if aligned is not None:
        warped, _H, inliers = aligned
        if inliers >= 10:
            try:
                bits_b, det_b = wm.extract(warped)
                if bits_b:
                    candidates.append(("aligned", bits_b))
            except Exception:
                pass

    registry = [e for e in load_registry() if e.media_type == "image"]
    return _finalize(candidates, registry, "image", max_hamming)


def _detect_video(
    video_path: Path,
    max_hamming: int | None,
    n_samples: int = 20,
) -> DetectionResult:
    wm = get_watermarker()
    # カメラ撮影対応:原本動画との per-frame 位置合わせを試みる
    registry = [e for e in load_registry() if e.media_type == "video"]
    ref_video: Path | None = None
    if registry:
        candidate = ROOT / registry[0].source_path
        if candidate.exists():
            ref_video = candidate
    dec = decode_video_by_vote(
        video_path, wm,
        n_samples=n_samples,
        reference_video=ref_video,
        reference_image=DEFAULT_IMAGE_REF if not ref_video else None,
    )

    candidates: list[tuple[str, str]] = []
    if dec.voted_bits:
        candidates.append(("video-vote", dec.voted_bits))

    registry = [e for e in load_registry() if e.media_type == "video"]
    result = _finalize(candidates, registry, "video", max_hamming)
    result.frames_detected = dec.detected_frames
    result.frames_total = dec.num_frames
    return result


def _finalize(
    candidates: list[tuple[str, str]],
    registry: list[TokenEntry],
    media_type: Literal["image", "video"],
    max_hamming: int | None,
) -> DetectionResult:
    if not registry:
        return DetectionResult(
            success=False, media_type=media_type, method="none",
            extracted_bits=candidates[0][1] if candidates else None,
            matched_entry=None, hamming_distance=None, confidence=None,
            message=f"{media_type} 台帳が空です。先にトークンを発行してください。",
        )
    if not candidates:
        return DetectionResult(
            success=False, media_type=media_type, method="none",
            extracted_bits=None, matched_entry=None,
            hamming_distance=None, confidence=None,
            message="透かしを復元できませんでした。",
        )

    best: tuple[str, str, TokenEntry, int] | None = None
    for method, bits in candidates:
        match = find_best_match(bits, registry, media_type=media_type)
        if match is None:
            continue
        entry, d = match
        if best is None or d < best[3]:
            best = (method, bits, entry, d)

    if best is None:
        return DetectionResult(
            success=False, media_type=media_type, method="none",
            extracted_bits=candidates[0][1],
            matched_entry=None, hamming_distance=None, confidence=None,
            message="台帳との照合に失敗。",
        )

    method, bits, entry, d = best
    confidence = 1.0 - d / max(len(bits), 1)

    if max_hamming is not None and d > max_hamming:
        return DetectionResult(
            success=False, media_type=media_type, method=method,
            extracted_bits=bits, matched_entry=entry,
            hamming_distance=d, confidence=confidence,
            message=f"最接近エントリ {entry.id} との Hamming 距離 {d} が閾値 {max_hamming} を超過",
        )

    return DetectionResult(
        success=True, media_type=media_type, method=method,
        extracted_bits=bits, matched_entry=entry,
        hamming_distance=d, confidence=confidence,
        message=f"{method} 復元 → {entry.id} ({entry.label}) に一致 (Hamming={d}, 信頼度={confidence:.2%})",
    )


def detect(
    input_path: str | Path,
    reference_path: str | Path = DEFAULT_IMAGE_REF,
    max_hamming: int | None = 10,
    video_samples: int = 60,
) -> DetectionResult:
    """入力ファイル(画像 or 動画)を解析し、最も近いトークンを返す。"""
    path = Path(input_path)
    kind = _classify(path)

    if kind == "image":
        return _detect_image(path, Path(reference_path), max_hamming)
    elif kind == "video":
        return _detect_video(path, max_hamming, n_samples=video_samples)
    else:
        return DetectionResult(
            success=False, media_type="unknown", method="none",
            extracted_bits=None, matched_entry=None,
            hamming_distance=None, confidence=None,
            message=f"未対応の拡張子: {path.suffix}",
        )
