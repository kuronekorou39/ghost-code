"""デモ用画像トークン発行:cover.png から N ユーザー分の stego 画像と台帳を生成。

HMAC ベースで暗号化派生したビット列を使用(平文の user_id を埋めない)。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ghost_code.crypto import (
    WatermarkSecret,
    find_unique_invisible_bits,
)
from ghost_code.registry import REGISTRY_PATH, TokenEntry, save_registry
from ghost_code.watermark import Watermarker

ROOT = Path(__file__).resolve().parents[2]
COVER = ROOT / "outputs" / "phase1" / "cover.png"
STEGO_DIR = ROOT / "tokens" / "stego"

DEMO_NAMES = [
    "Alice", "Bob", "Charlie", "Daisy", "Ethan", "Fiona", "Grace", "Hiro",
    "Ivy", "Jun", "Kai", "Luna", "Mika", "Noah", "Olivia",
]


def issue_tokens(count: int = 3, strength: float = 1.0) -> list[TokenEntry]:
    STEGO_DIR.mkdir(parents=True, exist_ok=True)
    cover = Image.open(COVER).convert("RGB")
    secret = WatermarkSecret.load_or_create()

    wm = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)
    print(f"payload capacity: {wm.payload_len} bits, issuing {count} tokens...")

    entries: list[TokenEntry] = []
    used_bits: list[str] = []
    t0 = time.perf_counter()

    for i in tqdm(range(count), desc="embed"):
        uid = f"user-{i + 1:03d}"
        label = DEMO_NAMES[i] if i < len(DEMO_NAMES) else uid
        bits, inv_nonce = find_unique_invisible_bits(secret, uid, used_bits, min_distance=12)
        used_bits.append(bits)

        result = wm.embed(cover, bits, strength=strength)
        out_path = STEGO_DIR / f"{uid}.png"
        result.stego.save(out_path)

        entries.append(TokenEntry(
            id=uid, label=label, bits=bits,
            stego_path=str(out_path.relative_to(ROOT)).replace("\\", "/"),
            source_path=str(COVER.relative_to(ROOT)).replace("\\", "/"),
            media_type="image",
            inv_nonce=inv_nonce,
        ))

    save_registry(entries)
    elapsed = time.perf_counter() - t0
    print(f"\n台帳保存: {REGISTRY_PATH}")
    print(f"所要時間: {elapsed:.1f}s ({elapsed / max(count, 1):.2f}s/token)")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    issue_tokens(count=args.count, strength=args.strength)


if __name__ == "__main__":
    main()
