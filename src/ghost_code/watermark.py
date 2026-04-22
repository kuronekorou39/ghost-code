"""TrustMark を用いた静止画への透かし埋め込み・復元の最小ラッパ。

Phase 1 で API 差異の影響を抑え、Phase 2 以降の動画処理に流用する。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from trustmark import TrustMark


@dataclass
class WatermarkResult:
    stego: Image.Image
    psnr_db: float


class Watermarker:
    """100bit ペイロードを画像に埋め込み・復元するラッパ。"""

    def __init__(
        self,
        model_type: str = "Q",
        encoding_type: int = 0,  # 0=BCH_SUPER(40bit 最強), 1=BCH_5(61bit), 2=BCH_4(68bit), 3=BCH_3(75bit)
        use_ecc: bool = True,
        device: str = "",
    ) -> None:
        self._tm = TrustMark(
            use_ECC=use_ecc,
            model_type=model_type,
            encoding_type=encoding_type,
            device=device,
            verbose=False,
        )
        self.payload_len = self._tm.schemaCapacity()

    def embed(self, cover: Image.Image, bits: str, strength: float = 1.0) -> WatermarkResult:
        if len(bits) != self.payload_len:
            raise ValueError(f"bits length {len(bits)} != payload_len {self.payload_len}")
        if any(c not in "01" for c in bits):
            raise ValueError("bits must be a string of '0' and '1'")

        stego = self._tm.encode(cover.convert("RGB"), bits, MODE="binary", WM_STRENGTH=strength)
        psnr = compute_psnr(np.asarray(cover.convert("RGB")), np.asarray(stego))
        return WatermarkResult(stego=stego, psnr_db=psnr)

    def extract(self, stego: Image.Image, rotation: bool = False) -> tuple[str, bool]:
        """戻り値: (復元ビット列, 検出フラグ).

        rotation=True にすると TrustMark が 4 方向の回転も試行(遅いが堅牢)。
        想定 payload_len と異なる長さが返ったら ECC 失敗として None 扱い。
        """
        result = self._tm.decode(stego.convert("RGB"), MODE="binary", ROTATION=rotation)
        secret, detected = result[0], result[1]
        # 想定外の長さ = ECC 内部で別スキーマにフォールバックした状態
        if detected and len(secret) != self.payload_len:
            return secret, False
        return secret, bool(detected)


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def bit_error_rate(a: str, b: str) -> float:
    if len(a) != len(b):
        # 復元失敗時は 0.5 を返す(最大エントロピー扱い)
        return 0.5
    diff = sum(1 for x, y in zip(a, b) if x != y)
    return diff / len(a)


def random_bits(n: int, seed: int | None = None) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(["0", "1"], size=n).tolist())


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
