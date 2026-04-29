"""任意入力(画像 or 動画)から透かしを復元し、台帳と照合する検出パイプライン。

二重透かし対応:
  Layer 1(不可視): TrustMark で 40bit 抽出 → HMAC で台帳ユーザー全員と比較
  Layer 2(可視):   OCR で四隅から 4 文字コード抽出 → 台帳と完全一致照合
  両層の結果からコンセンサス判定。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image

from ghost_code.crypto import (
    WatermarkSecret,
    derive_invisible_bits,
    derive_visible_code,
    hamming as crypto_hamming,
    identify_from_invisible_bits,
    identify_from_visible_code,
)
from ghost_code.registry import TokenEntry, find_best_match, load_registry
from ghost_code.screen_extract import align_to_reference
from ghost_code.video_watermark import decode_video_by_vote
from ghost_code.visible_detect import detect_visible_code, VisibleDetection
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
    # 動画の不可視層詳細
    frames_detected: int | None = None
    frames_total: int | None = None
    # 可視層詳細(動画のみ)
    visible_code: str | None = None
    visible_match: TokenEntry | None = None
    visible_confidence: float | None = None
    consensus: str | None = None    # "両層一致" | "不可視のみ" | "可視のみ" | "矛盾" | "全滅"


_wm_cache: Watermarker | None = None
_secret_cache: WatermarkSecret | None = None


def get_watermarker() -> Watermarker:
    global _wm_cache
    if _wm_cache is None:
        _wm_cache = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)
    return _wm_cache


def get_secret() -> WatermarkSecret:
    global _secret_cache
    if _secret_cache is None:
        _secret_cache = WatermarkSecret.load_or_create()
    return _secret_cache


def _classify(path: Path) -> Literal["image", "video", "unknown"]:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


def _identify_invisible(
    extracted_bits: str | None,
    registry: list[TokenEntry],
    media_type: Literal["image", "video"],
    max_hamming: int,
) -> tuple[TokenEntry, int] | None:
    """抜き出されたビット列から media_type の登録ユーザーを特定。"""
    if extracted_bits is None:
        return None
    pool = [e for e in registry if e.media_type == media_type]
    if not pool:
        return None
    # 台帳に保存された bits との Hamming で照合(HMAC 派生済みのため毎回計算不要)
    best: tuple[TokenEntry, int] | None = None
    for e in pool:
        d = crypto_hamming(extracted_bits, e.bits)
        if best is None or d < best[1]:
            best = (e, d)
    if best is None or best[1] > max_hamming:
        return best  # 閾値外でも返す(UI 側で警告表示するため)
    return best


_OCR_CONFUSE = {
    "0": "o", "o": "0",
    "1": "l", "l": "1", "i": "1",
    "5": "s", "s": "5",
    "8": "b", "b": "8",
    "9": "g", "g": "9",
    "z": "2", "2": "z",
}


def _confusable_variants(code: str) -> set[str]:
    """OCR で誤認されがちな文字を入れ替えた候補を生成。"""
    variants = {code}
    for i, c in enumerate(code):
        if c in _OCR_CONFUSE:
            for alt in [_OCR_CONFUSE[c]]:
                variants.add(code[:i] + alt + code[i + 1:])
    return variants


def _identify_visible(
    code: str | None,
    registry: list[TokenEntry],
) -> TokenEntry | None:
    """4 文字コードから登録ユーザーを照合(OCR 誤認の混同字も許容)。"""
    if code is None:
        return None
    code = code.lower()
    candidates = _confusable_variants(code)
    for e in registry:
        if e.media_type != "video":
            continue
        registered = e.visible_code.lower()
        if registered in candidates or _confusable_variants(registered) & candidates:
            return e
    return None


def _detect_image(
    image_path: Path, reference_path: Path, max_hamming: int,
) -> DetectionResult:
    wm = get_watermarker()
    img = Image.open(image_path).convert("RGB")
    registry = load_registry()

    candidates: list[str] = []
    try:
        bits, det = wm.extract(img)
        if det:
            candidates.append(bits)
    except Exception:
        pass

    aligned = align_to_reference(image_path, reference_path)
    if aligned is not None:
        warped, _H, inliers = aligned
        if inliers >= 10:
            try:
                bits, det = wm.extract(warped)
                if det:
                    candidates.append(bits)
            except Exception:
                pass

    best_match: tuple[TokenEntry, int] | None = None
    best_bits: str | None = None
    method = "none"
    for i, bits in enumerate(candidates):
        match = _identify_invisible(bits, registry, "image", max_hamming=999)
        if match is None:
            continue
        if best_match is None or match[1] < best_match[1]:
            best_match = match
            best_bits = bits
            method = "direct" if i == 0 and len(candidates) >= 1 else "aligned"

    if best_bits is None:
        return DetectionResult(
            success=False, media_type="image", method="none",
            extracted_bits=None, matched_entry=None,
            hamming_distance=None, confidence=None,
            message="透かしを復元できませんでした。",
        )

    entry, d = best_match
    confidence = 1.0 - d / max(len(best_bits), 1)
    if d > max_hamming:
        return DetectionResult(
            success=False, media_type="image", method=method,
            extracted_bits=best_bits, matched_entry=entry,
            hamming_distance=d, confidence=confidence,
            message=f"最接近 {entry.id} だが Hamming {d} > 閾値 {max_hamming}",
        )
    return DetectionResult(
        success=True, media_type="image", method=method,
        extracted_bits=best_bits, matched_entry=entry,
        hamming_distance=d, confidence=confidence,
        message=f"{method} 復元 → {entry.id} ({entry.label}) Hamming={d}",
    )


def _detect_video(
    video_path: Path, max_hamming: int, n_samples: int = 60,
) -> DetectionResult:
    wm = get_watermarker()
    registry = load_registry()
    video_pool = [e for e in registry if e.media_type == "video"]

    # ── Layer 1: 不可視透かし ──
    ref_video: Path | None = None
    if video_pool:
        candidate = ROOT / video_pool[0].source_path
        if candidate.exists():
            ref_video = candidate

    dec = decode_video_by_vote(
        video_path, wm,
        n_samples=n_samples,
        reference_video=ref_video,
        reference_image=DEFAULT_IMAGE_REF if not ref_video else None,
    )
    inv_bits = dec.voted_bits if dec.voted_bits else None
    inv_match = _identify_invisible(inv_bits, registry, "video", max_hamming=999)

    # ── Layer 2: 可視透かし ──
    vis_result: VisibleDetection = detect_visible_code(video_path, n_samples=min(n_samples, 30))
    vis_match = _identify_visible(vis_result.code, registry)

    # ── コンセンサス ──
    inv_id = inv_match[0].id if inv_match and inv_match[1] <= max_hamming else None
    vis_id = vis_match.id if vis_match else None

    if inv_id and vis_id:
        consensus = "両層一致" if inv_id == vis_id else "矛盾"
    elif inv_id:
        consensus = "不可視のみ"
    elif vis_id:
        consensus = "可視のみ"
    else:
        consensus = "全滅"

    # 最終判定:不可視層を優先
    if inv_id:
        entry, d = inv_match
        confidence = 1.0 - d / max(len(inv_bits or ""), 1)
        success = True
        method = "video-invisible"
        message = f"{method} → {entry.id} ({entry.label}) Hamming={d}, consensus={consensus}"
    elif vis_id:
        entry = vis_match
        d = 0
        confidence = vis_result.confidence
        success = True
        method = "video-visible"
        message = f"{method} → {entry.id} ({entry.label}) (不可視層は復元不能、可視のみ)"
    else:
        # 両方失敗、近傍があればそれを返す
        entry = inv_match[0] if inv_match else None
        d = inv_match[1] if inv_match else None
        confidence = 1.0 - (d / max(len(inv_bits or ""), 1)) if d is not None else None
        success = False
        method = "none"
        message = "透かしを復元できませんでした。"

    return DetectionResult(
        success=success, media_type="video", method=method,
        extracted_bits=inv_bits,
        matched_entry=entry if success else (inv_match[0] if inv_match else None),
        hamming_distance=d if success else (inv_match[1] if inv_match else None),
        confidence=confidence,
        message=message,
        frames_detected=dec.detected_frames,
        frames_total=dec.num_frames,
        visible_code=vis_result.code,
        visible_match=vis_match,
        visible_confidence=vis_result.confidence,
        consensus=consensus,
    )


def detect(
    input_path: str | Path,
    reference_path: str | Path = DEFAULT_IMAGE_REF,
    max_hamming: int | None = 10,
    video_samples: int = 60,
) -> DetectionResult:
    path = Path(input_path)
    kind = _classify(path)
    if kind == "image":
        return _detect_image(path, Path(reference_path), max_hamming or 10)
    elif kind == "video":
        return _detect_video(path, max_hamming or 10, n_samples=video_samples)
    else:
        return DetectionResult(
            success=False, media_type="unknown", method="none",
            extracted_bits=None, matched_entry=None,
            hamming_distance=None, confidence=None,
            message=f"未対応の拡張子: {path.suffix}",
        )
