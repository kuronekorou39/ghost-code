"""動画トークン発行:
  1. HMAC で 40bit 不可視ペイロード派生(衝突回避済み)
  2. 全フレームに不可視透かしを TrustMark で埋める
  3. シーン検出 + 四隅ローテーションで 4 文字可視コードを描画
  4. 台帳に保存
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from ghost_code.crypto import (
    WatermarkSecret,
    find_unique_invisible_bits,
    find_unique_visible_code,
)
from ghost_code.issue_tokens import DEMO_NAMES
from ghost_code.registry import REGISTRY_PATH, TokenEntry, load_registry, save_registry
from ghost_code.video_watermark import embed_video
from ghost_code.visible_watermark import embed_visible_to_video
from ghost_code.watermark import Watermarker

ROOT = Path(__file__).resolve().parents[2]
SOURCE_VIDEO = ROOT / "data" / "raw" / "sample.mp4"
STEGO_DIR = ROOT / "tokens" / "stego_video"
TMP_DIR = ROOT / "tokens" / "_tmp_video"


def issue_video_tokens(
    count: int = 3,
    strength: float = 1.0,
    crf: int = 18,
    visible_alpha: float = 0.35,
    visible_fontsize: int = 22,
    max_frames: int | None = None,
    prefix: str = "vuser",
    replace_existing_videos: bool = True,
) -> list[TokenEntry]:
    STEGO_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"ソース動画なし: {SOURCE_VIDEO}")

    secret = WatermarkSecret.load_or_create()
    wm = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)
    print(f"payload capacity: {wm.payload_len} bits")
    print(f"source: {SOURCE_VIDEO}")
    print(f"issuing {count} video tokens (CRF={crf}, strength={strength})...")

    registry = load_registry()
    if replace_existing_videos:
        registry = [e for e in registry if e.media_type != "video"]
    used_inv = [e.bits for e in registry]
    used_vis = [e.visible_code for e in registry if e.visible_code]

    new_entries: list[TokenEntry] = []
    for i in range(count):
        uid = f"{prefix}-{i + 1:03d}"
        label = DEMO_NAMES[i] if i < len(DEMO_NAMES) else uid

        bits, inv_n = find_unique_invisible_bits(secret, uid, used_inv, min_distance=12)
        code, vis_n = find_unique_visible_code(secret, uid, used_vis)
        used_inv.append(bits)
        used_vis.append(code)

        out_path = STEGO_DIR / f"{uid}.mp4"
        tmp_path = TMP_DIR / f"{uid}_invisible.mp4"

        print(f"\n[{i + 1}/{count}] {uid} ({label})")
        print(f"  invisible bits: {bits}")
        print(f"  visible code:   {code}")

        # 1. 不可視透かし埋め込み
        t0 = time.perf_counter()
        result = embed_video(
            SOURCE_VIDEO, tmp_path, bits, wm,
            strength=strength, crf=crf, max_frames=max_frames,
        )
        t_inv = time.perf_counter() - t0

        # 2. 可視透かし重畳
        t0 = time.perf_counter()
        embed_visible_to_video(
            tmp_path, out_path, code,
            seed=hash(uid) & 0xFFFFFFFF,
            fontsize=visible_fontsize, alpha=visible_alpha,
            crf=crf,
        )
        t_vis = time.perf_counter() - t0
        tmp_path.unlink(missing_ok=True)

        print(f"  → 不可視 {result.num_frames}fr ({t_inv:.1f}s, PSNR={result.avg_psnr_db:.1f}dB) "
              f"+ 可視 ({t_vis:.1f}s)")

        new_entries.append(TokenEntry(
            id=uid, label=label, bits=bits,
            visible_code=code,
            inv_nonce=inv_n, vis_nonce=vis_n,
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
    parser.add_argument("--visible-alpha", type=float, default=0.35)
    parser.add_argument("--visible-fontsize", type=int, default=22)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    issue_video_tokens(
        count=args.count, strength=args.strength,
        crf=args.crf,
        visible_alpha=args.visible_alpha,
        visible_fontsize=args.visible_fontsize,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
