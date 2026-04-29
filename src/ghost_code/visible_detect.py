"""可視透かしの検出。

戦略:
  1. 動画から複数フレームをサンプリング
  2. 各フレームの四隅 ROI を切り出し
  3. EasyOCR で英数字を認識
  4. 4 文字コード候補を集計 → 多数決
  5. 完全一致しない場合は編集距離で台帳を検索
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _reader


CODE_RE = re.compile(r"[a-z0-9]{4}")


@dataclass
class VisibleDetection:
    code: str | None
    confidence: float
    frames_checked: int
    candidates: list[tuple[str, int]]  # (code, vote count)


def _crop_corner(img: np.ndarray, corner: str, ratio: float = 0.18) -> np.ndarray:
    h, w = img.shape[:2]
    cw, ch = int(w * ratio), int(h * ratio * 1.5)
    if corner == "TL":
        return img[:ch, :cw]
    if corner == "TR":
        return img[:ch, w - cw:]
    if corner == "BL":
        return img[h - ch:, :cw]
    if corner == "BR":
        return img[h - ch:, w - cw:]
    raise ValueError(corner)


def _enhance(roi: np.ndarray) -> np.ndarray:
    """OCR 精度向上のための前処理。"""
    if roi.ndim == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    # 上半分の白文字を強調(コントラスト強化)
    gray = cv2.equalizeHist(gray)
    # 拡大してから OCR(小さい文字に強い)
    h, w = gray.shape
    scale = max(1, 80 // min(h, w))
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    return gray


def detect_visible_code(
    video_path: Path,
    n_samples: int = 30,
) -> VisibleDetection:
    """動画から 4 文字可視コードを抽出。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VisibleDetection(None, 0.0, 0, [])

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        return VisibleDetection(None, 0.0, 0, [])

    margin = max(int(n_frames * 0.05), 1)
    sample_idx = np.linspace(margin, n_frames - margin, n_samples).astype(int)

    reader = _get_reader()
    candidates: dict[str, int] = {}
    checked = 0

    for idx in sample_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        checked += 1

        for corner in ("TL", "TR", "BL", "BR"):
            roi = _crop_corner(frame, corner)
            roi_pre = _enhance(roi)
            try:
                # detail=0 で文字列のみのリストが返る
                results = reader.readtext(roi_pre, detail=0, paragraph=False, allowlist="abcdefghijklmnopqrstuvwxyz0123456789")
            except Exception:
                continue

            for txt in results:
                txt = txt.strip().lower()
                if not txt:
                    continue
                # ちょうど 4 文字の英数字部分を抽出
                for m in CODE_RE.finditer(txt):
                    code = m.group(0)
                    candidates[code] = candidates.get(code, 0) + 1

    cap.release()

    if not candidates:
        return VisibleDetection(None, 0.0, checked, [])

    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    top_code, top_votes = ranked[0]
    total_votes = sum(candidates.values())
    confidence = top_votes / max(total_votes, 1)
    return VisibleDetection(top_code, confidence, checked, ranked[:10])
