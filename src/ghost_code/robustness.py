"""耐性テストハーネス。

stego 動画に各種攻撃を加え、detect() で復元できるか測定。
結果を JSON + Markdown 表で出力。
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from ghost_code import attacks
from ghost_code.detect import detect
from ghost_code.registry import load_registry

ROOT = Path(__file__).resolve().parents[2]

Attack = tuple[str, Callable[[Path, Path], None], str]  # (name, fn, category)


def build_attack_suite() -> list[Attack]:
    """実運用で想定される攻撃を網羅。"""
    return [
        # 再エンコード
        ("reencode_crf23", lambda s, d: attacks.reencode(s, d, crf=23), "reencode"),
        ("reencode_crf28", lambda s, d: attacks.reencode(s, d, crf=28), "reencode"),
        ("reencode_crf32", lambda s, d: attacks.reencode(s, d, crf=32), "reencode"),
        ("reencode_crf36", lambda s, d: attacks.reencode(s, d, crf=36), "reencode"),
        # クロップ
        ("crop_90pct", lambda s, d: attacks.crop_center(s, d, ratio=0.9), "crop"),
        ("crop_75pct", lambda s, d: attacks.crop_center(s, d, ratio=0.75), "crop"),
        ("crop_50pct", lambda s, d: attacks.crop_center(s, d, ratio=0.5), "crop"),
        # リサイズ
        ("resize_720p", lambda s, d: attacks.resize(s, d, height=720), "resize"),
        ("resize_480p", lambda s, d: attacks.resize(s, d, height=480), "resize"),
        ("resize_360p", lambda s, d: attacks.resize(s, d, height=360), "resize"),
        # 色調補正
        ("eq_bright+0.2", lambda s, d: attacks.adjust_eq(s, d, brightness=0.2), "color"),
        ("eq_contrast1.5", lambda s, d: attacks.adjust_eq(s, d, contrast=1.5), "color"),
        ("eq_saturation1.5", lambda s, d: attacks.adjust_eq(s, d, saturation=1.5), "color"),
        ("eq_gamma0.7", lambda s, d: attacks.adjust_eq(s, d, gamma=0.7), "color"),
        # ぼかし / ノイズ
        ("blur_s0.5", lambda s, d: attacks.blur(s, d, sigma=0.5), "blur"),
        ("blur_s1.5", lambda s, d: attacks.blur(s, d, sigma=1.5), "blur"),
        ("noise_10", lambda s, d: attacks.add_noise(s, d, strength=10), "noise"),
        ("noise_30", lambda s, d: attacks.add_noise(s, d, strength=30), "noise"),
        # トリム(短尺切り抜き)
        ("trim_10s", lambda s, d: attacks.trim(s, d, duration_sec=10), "trim"),
        ("trim_5s", lambda s, d: attacks.trim(s, d, duration_sec=5), "trim"),
        ("trim_3s", lambda s, d: attacks.trim(s, d, duration_sec=3), "trim"),
        ("trim_1s", lambda s, d: attacks.trim(s, d, duration_sec=1), "trim"),
        # フレームレート
        ("fps_15", lambda s, d: attacks.change_fps(s, d, fps=15), "fps"),
        # 回転
        ("rotate_3deg", lambda s, d: attacks.rotate(s, d, angle_deg=3), "rotate"),
        # 合成
        ("sns_upload", lambda s, d: attacks.combined_sns_upload(s, d), "combined"),
    ]


@dataclass
class RobustnessRow:
    attack: str
    category: str
    success: bool
    matched_id: str | None
    expected_id: str
    hamming: int | None
    frames_detected: int | None
    frames_total: int | None
    attack_sec: float
    detect_sec: float


def run_suite(
    source_stego: Path,
    expected_id: str,
    out_dir: Path,
    video_samples: int = 80,
    attack_filter: list[str] | None = None,
) -> list[RobustnessRow]:
    out_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir = out_dir / "attacked"
    attacked_dir.mkdir(exist_ok=True)

    suite = build_attack_suite()
    if attack_filter:
        suite = [a for a in suite if a[0] in attack_filter]

    rows: list[RobustnessRow] = []
    for name, fn, category in suite:
        attacked_path = attacked_dir / f"{name}.mp4"
        print(f"\n--- {name} ({category}) ---")

        t0 = time.perf_counter()
        try:
            fn(source_stego, attacked_path)
        except Exception as e:
            print(f"  [ATTACK FAIL] {e}")
            rows.append(RobustnessRow(
                attack=name, category=category, success=False,
                matched_id=None, expected_id=expected_id,
                hamming=None, frames_detected=None, frames_total=None,
                attack_sec=time.perf_counter() - t0, detect_sec=0,
            ))
            continue
        attack_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        result = detect(str(attacked_path), video_samples=video_samples)
        detect_sec = time.perf_counter() - t1

        matched = result.matched_entry.id if result.matched_entry else None
        row = RobustnessRow(
            attack=name, category=category,
            success=result.success and matched == expected_id,
            matched_id=matched, expected_id=expected_id,
            hamming=result.hamming_distance,
            frames_detected=result.frames_detected,
            frames_total=result.frames_total,
            attack_sec=attack_sec, detect_sec=detect_sec,
        )
        rows.append(row)
        ok = "✅" if row.success else "❌"
        print(f"  {ok} match={matched} d={result.hamming_distance} "
              f"frames={result.frames_detected}/{result.frames_total} "
              f"(attack {attack_sec:.1f}s + detect {detect_sec:.1f}s)")

    return rows


def write_report(rows: list[RobustnessRow], out_dir: Path) -> None:
    json_path = out_dir / "robustness.json"
    md_path = out_dir / "robustness.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    # Markdown
    lines = [
        "# 耐性テスト結果",
        "",
        f"全 {len(rows)} 攻撃、成功 {sum(1 for r in rows if r.success)} / "
        f"失敗 {sum(1 for r in rows if not r.success)}",
        "",
        "| 攻撃 | カテゴリ | 結果 | マッチ | Hamming | フレーム検出 | 攻撃(s) | 検出(s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ok = "✅" if r.success else "❌"
        frames = f"{r.frames_detected}/{r.frames_total}" if r.frames_detected is not None else "-"
        matched = r.matched_id or "(なし)"
        hamming = str(r.hamming) if r.hamming is not None else "-"
        lines.append(
            f"| `{r.attack}` | {r.category} | {ok} | {matched} | {hamming} | "
            f"{frames} | {r.attack_sec:.1f} | {r.detect_sec:.1f} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n保存: {json_path}")
    print(f"保存: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="tokens/stego_video/vuser-001.mp4")
    parser.add_argument("--expected", type=str, default="vuser-001")
    parser.add_argument("--out", type=str, default="outputs/robustness")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--only", nargs="*", help="特定の攻撃名のみ実行")
    args = parser.parse_args()

    source = ROOT / args.source
    if not source.exists():
        raise FileNotFoundError(f"stego 動画がありません: {source}")
    # 事前に expected_id が台帳にあるか確認
    reg = load_registry()
    if not any(e.id == args.expected for e in reg):
        print(f"[WARN] expected_id={args.expected} が台帳にありません")

    out_dir = ROOT / args.out
    rows = run_suite(source, args.expected, out_dir, video_samples=args.samples, attack_filter=args.only)
    write_report(rows, out_dir)


if __name__ == "__main__":
    main()
