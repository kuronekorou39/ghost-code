"""Phase 1 step 4: スマホ撮影画像から透かし復元。

data/captures/ 以下の全 jpg に対して:
  1. 画面内のstego領域を検出 + homography補正
  2. TrustMark で復号(通常 + DETECTFIRST の2パターン)
  3. 元 payload と BER を比較
"""
from __future__ import annotations

from pathlib import Path

from ghost_code.screen_extract import align_to_reference
from ghost_code.watermark import Watermarker, bit_error_rate, random_bits

ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / "data" / "captures"
OUT = ROOT / "outputs" / "phase1"
REFERENCE = OUT / "cover.png"  # 位置合わせの基準(原本 or stego どちらでも可)


def main() -> None:
    debug_dir = OUT / "capture_debug"
    print("[1/3] Watermarker 初期化 (model=Q, BCH_SUPER, 40bit)")
    wm = Watermarker(model_type="Q", encoding_type=0, use_ecc=True)

    # 埋め込みに使った payload と同じ seed で再生成
    truth = random_bits(wm.payload_len, seed=42)
    print(f"      truth: {truth}")

    captures = sorted(CAPTURES.glob("*.jpg"))
    print(f"\n[2/3] 撮影画像 {len(captures)} 枚を処理")

    results = []
    for cap in captures:
        print(f"\n--- {cap.name} ---")
        out = align_to_reference(cap, REFERENCE, debug_dir=debug_dir)
        if out is None:
            print("  [SKIP] 位置合わせ失敗(特徴点不足)")
            results.append((cap.name, None, None, None))
            continue
        warped, _H, inliers = out
        print(f"  位置合わせ inliers={inliers}, warped saved: {debug_dir / (cap.stem + '_warped.png')}")

        # 通常デコード
        try:
            extracted, detected = wm.extract(warped)
            ber_a = bit_error_rate(truth, extracted)
            print(f"  [A] 通常デコード:    detected={detected}, BER={ber_a:.4f}")
        except Exception as e:
            ber_a = None
            print(f"  [A] 通常デコード失敗: {e}")

        # DETECTFIRST デコード(TrustMark 内部の位置検出を使う)
        try:
            result = wm._tm.decode(warped.convert("RGB"), MODE="binary", DETECTFIRST=True)
            ext_b = result[0]
            det_b = bool(result[1])
            ber_b = bit_error_rate(truth, ext_b)
            print(f"  [B] DETECTFIRST:    detected={det_b}, BER={ber_b:.4f}")
        except Exception as e:
            ber_b = None
            print(f"  [B] DETECTFIRST 失敗: {e}")

        results.append((cap.name, ber_a, ber_b, None))

    # サマリ
    print("\n[3/3] サマリ")
    print(f"{'file':<45} {'BER(通常)':<12} {'BER(DETECTFIRST)':<20}")
    print("-" * 80)
    for name, ba, bb, _ in results:
        ba_s = f"{ba:.4f}" if ba is not None else "N/A"
        bb_s = f"{bb:.4f}" if bb is not None else "N/A"
        print(f"{name:<45} {ba_s:<12} {bb_s:<20}")

    best = min(
        (r for r in results if r[1] is not None or r[2] is not None),
        key=lambda r: min(x for x in (r[1], r[2]) if x is not None),
        default=None,
    )
    if best:
        best_ber = min(x for x in (best[1], best[2]) if x is not None)
        print(f"\nベスト: {best[0]}  BER={best_ber:.4f}")
        if best_ber < 0.1:
            print("→ 目標達成 (BER < 10%)、ECC で救える範囲")
        elif best_ber < 0.25:
            print("→ ギリギリ、ECC 強度 & 撮影条件の両方調整が必要")
        else:
            print("→ 撮影条件 or 表示方法が厳しい。フルスクリーン表示で再撮影推奨")


if __name__ == "__main__":
    main()
