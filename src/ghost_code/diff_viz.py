"""cover と stego の差分ヒートマップを生成して、透かしの所在を可視化する。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "phase1"


def main() -> None:
    cover = np.asarray(Image.open(OUT / "cover.png").convert("RGB"), dtype=np.int16)
    stego = np.asarray(Image.open(OUT / "stego.png").convert("RGB"), dtype=np.int16)

    diff = stego - cover  # 符号付き
    abs_diff = np.abs(diff)

    max_d = int(abs_diff.max())
    mean_d = float(abs_diff.mean())
    std_d = float(diff.std())
    nonzero = float((abs_diff.sum(axis=-1) > 0).mean()) * 100
    print(f"絶対差分 max={max_d}, mean={mean_d:.3f}, std={std_d:.3f}")
    print(f"変化したピクセル割合: {nonzero:.2f}%")
    print(f"  (cover と stego が 1 でも違う画素の割合)")

    # グレースケール合計差分を [0,255] にマップ(増幅)
    heat = abs_diff.sum(axis=-1).astype(np.float32)
    heat = heat / max(heat.max(), 1.0) * 255.0
    Image.fromarray(heat.astype(np.uint8)).save(OUT / "diff_heat.png")

    # ×20 増幅版(肉眼でチェック用)
    amp = np.clip(np.abs(diff).astype(np.float32) * 20.0, 0, 255).astype(np.uint8)
    Image.fromarray(amp).save(OUT / "diff_amp20.png")

    # 並べ比較(cover | stego | heat) 縦並び縮小
    h, w = cover.shape[:2]
    scale = 512 / w
    nh, nw = int(h * scale), int(w * scale)
    c_small = Image.open(OUT / "cover.png").resize((nw, nh))
    s_small = Image.open(OUT / "stego.png").resize((nw, nh))
    heat_rgb = Image.fromarray(heat.astype(np.uint8)).convert("RGB").resize((nw, nh))

    panel = Image.new("RGB", (nw * 3 + 20, nh), "black")
    panel.paste(c_small, (0, 0))
    panel.paste(s_small, (nw + 10, 0))
    panel.paste(heat_rgb, (nw * 2 + 20, 0))
    panel.save(OUT / "compare_panel.png")

    print("\n生成ファイル:")
    print(f"  {OUT / 'diff_heat.png'}    (透かしの分布ヒートマップ)")
    print(f"  {OUT / 'diff_amp20.png'}   (差分を20倍に増幅、肉眼用)")
    print(f"  {OUT / 'compare_panel.png'}(cover/stego/heat の横並び)")


if __name__ == "__main__":
    main()
