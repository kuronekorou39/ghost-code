"""デモ用トークン発行:cover.png から N ユーザー分の stego 画像と台帳を生成。"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from ghost_code.registry import REGISTRY_PATH, TokenEntry, save_registry
from ghost_code.watermark import Watermarker, random_bits

ROOT = Path(__file__).resolve().parents[2]
COVER = ROOT / "outputs" / "phase1" / "cover.png"
STEGO_DIR = ROOT / "tokens" / "stego"

# 少量のデモ用名前(超過分は "user-NNN" のまま)
DEMO_NAMES = [
    "Alice", "Bob", "Charlie", "Daisy", "Ethan", "Fiona", "Grace", "Hiro",
    "Ivy", "Jun", "Kai", "Luna", "Mika", "Noah", "Olivia",
]


def issue_tokens(count: int = 3, strength: float = 1.0) -> list[TokenEntry]:
    STEGO_DIR.mkdir(parents=True, exist_ok=True)
    cover = Image.open(COVER).convert("RGB")

    wm = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)
    print(f"payload length: {wm.payload_len} bits, issuing {count} tokens...")

    entries: list[TokenEntry] = []
    seen_bits: set[str] = set()
    t0 = time.perf_counter()
    for i in tqdm(range(count), desc="embed"):
        uid = f"user-{i + 1:03d}"
        label = DEMO_NAMES[i] if i < len(DEMO_NAMES) else uid
        # 決定的 seed + 衝突回避
        seed_offset = 0
        while True:
            bits = random_bits(wm.payload_len, seed=1000 + i + seed_offset * 100_000)
            if bits not in seen_bits:
                seen_bits.add(bits)
                break
            seed_offset += 1

        result = wm.embed(cover, bits, strength=strength)
        out_path = STEGO_DIR / f"{uid}.png"
        result.stego.save(out_path)

        entries.append(TokenEntry(
            id=uid,
            label=label,
            bits=bits,
            stego_path=str(out_path.relative_to(ROOT)).replace("\\", "/"),
            source_path=str(COVER.relative_to(ROOT)).replace("\\", "/"),
            media_type="image",
        ))

    save_registry(entries)
    elapsed = time.perf_counter() - t0
    avg = elapsed / max(count, 1)
    print(f"\n台帳保存: {REGISTRY_PATH}")
    print(f"所要時間: {elapsed:.1f}s ({avg:.2f}s/token)")
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3, help="発行するトークン数")
    parser.add_argument("--strength", type=float, default=1.0, help="透かし強度")
    args = parser.parse_args()
    issue_tokens(count=args.count, strength=args.strength)


if __name__ == "__main__":
    main()
