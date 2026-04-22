"""Phase 1 step 1-3: 静止画への埋め込み・デジタル復元の検証。

サンプル動画から 1 フレーム抽出し、TrustMark で 100bit を埋め込み、
そのまま decode して BER と PSNR を測定する。
"""
from __future__ import annotations

from pathlib import Path

from ghost_code.video_io import extract_frame, probe_video
from ghost_code.watermark import Watermarker, bit_error_rate, random_bits

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_VIDEO = ROOT / "data" / "raw" / "sample.mp4"
OUT_DIR = ROOT / "outputs" / "phase1"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] サンプル動画情報: {SAMPLE_VIDEO.name}")
    meta = probe_video(SAMPLE_VIDEO)
    for k, v in meta.items():
        print(f"      {k}: {v}")

    print("[2/5] フレーム抽出 (t=2.0s)")
    cover = extract_frame(SAMPLE_VIDEO, timestamp_sec=2.0)
    print(f"      size: {cover.size}, mode: {cover.mode}")
    cover_path = OUT_DIR / "cover.png"
    cover.save(cover_path)
    print(f"      saved: {cover_path}")

    print("[3/5] Watermarker 初期化 (model=Q, BCH_SUPER)")
    wm = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)
    print(f"      payload capacity: {wm.payload_len} bits")

    payload = random_bits(wm.payload_len, seed=42)
    print(f"      payload: {payload}")

    print("[4/5] 埋め込み (strength=1.0)")
    result = wm.embed(cover, payload, strength=1.0)
    stego_path = OUT_DIR / "stego.png"
    result.stego.save(stego_path)
    print(f"      saved: {stego_path}")
    print(f"      PSNR: {result.psnr_db:.2f} dB")

    print("[5/5] デジタル復元 (歪みなし)")
    extracted, detected = wm.extract(result.stego)
    ber = bit_error_rate(payload, extracted)
    print(f"      detected: {detected}")
    print(f"      extracted: {extracted}")
    print(f"      payload:   {payload}")
    print(f"      BER: {ber:.4f}")

    if ber == 0.0:
        print("\n[OK] デジタル復元 完璧 → 画面撮影テストに進める")
    elif ber < 0.05:
        print("\n[OK] デジタル復元 ほぼ完璧 (ECC で吸収可)")
    else:
        print("\n[NG] デジタル復元時点で BER が高すぎ → 設定見直し")


if __name__ == "__main__":
    main()
