"""動画トークン発行:sample.mp4 から N ユーザー分の stego 動画を生成。"""
from __future__ import annotations

import argparse
from pathlib import Path

from ghost_code.issue_tokens import DEMO_NAMES
from ghost_code.registry import REGISTRY_PATH, TokenEntry, load_registry, save_registry
from ghost_code.video_watermark import embed_video
from ghost_code.watermark import Watermarker, random_bits

ROOT = Path(__file__).resolve().parents[2]
SOURCE_VIDEO = ROOT / "data" / "raw" / "sample.mp4"
STEGO_DIR = ROOT / "tokens" / "stego_video"


def issue_video_tokens(
    count: int = 3,
    strength: float = 1.0,
    crf: int = 18,
    max_frames: int | None = None,
    prefix: str = "vuser",
    replace_existing_videos: bool = True,
) -> list[TokenEntry]:
    STEGO_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"ソース動画がありません: {SOURCE_VIDEO}")

    wm = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)
    print(f"payload length: {wm.payload_len} bits")
    print(f"source: {SOURCE_VIDEO}")
    print(f"issuing {count} video tokens (CRF={crf}, strength={strength})...")

    # 既存 image トークン等を温存しつつ、video エントリは作り直す
    registry = load_registry()
    if replace_existing_videos:
        registry = [e for e in registry if e.media_type != "video"]

    new_entries: list[TokenEntry] = []
    seen_bits: set[str] = {e.bits for e in registry}

    for i in range(count):
        uid = f"{prefix}-{i + 1:03d}"
        label = DEMO_NAMES[i] if i < len(DEMO_NAMES) else uid
        # 動画トークンは image と被らない seed 空間で生成
        seed_offset = 0
        while True:
            bits = random_bits(wm.payload_len, seed=2000 + i + seed_offset * 100_000)
            if bits not in seen_bits:
                seen_bits.add(bits)
                break
            seed_offset += 1

        out_path = STEGO_DIR / f"{uid}.mp4"
        print(f"\n[{i + 1}/{count}] {uid} ({label})  bits={bits}")
        result = embed_video(
            SOURCE_VIDEO, out_path, bits, wm,
            strength=strength, crf=crf, max_frames=max_frames,
        )
        print(f"  → {result.num_frames} frames in {result.elapsed_sec:.1f}s, avg PSNR={result.avg_psnr_db:.2f} dB")

        new_entries.append(TokenEntry(
            id=uid, label=label, bits=bits,
            stego_path=str(out_path.relative_to(ROOT)).replace("\\", "/"),
            source_path=str(SOURCE_VIDEO.relative_to(ROOT)).replace("\\", "/"),
            media_type="video",
        ))

    registry.extend(new_entries)
    save_registry(registry)
    print(f"\n台帳保存: {REGISTRY_PATH}  (video: {len(new_entries)}, total: {len(registry)})")
    return new_entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="デバッグ用:先頭 N フレームのみ処理")
    args = parser.parse_args()
    issue_video_tokens(
        count=args.count, strength=args.strength,
        crf=args.crf, max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
