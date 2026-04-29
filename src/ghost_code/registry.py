"""発行済みトークン台帳。

tokens/registry.json にトークン(bit列)とメタ情報を保持。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "tokens" / "registry.json"


@dataclass
class TokenEntry:
    id: str                                       # 識別子 (例: "user-001" / "vuser-001")
    label: str                                    # 表示名
    bits: str                                     # 埋め込みビット列(不可視透かし、HMAC 派生)
    stego_path: str                               # 生成した stego メディアの相対パス
    source_path: str                              # 元メディアの相対パス
    media_type: Literal["image", "video"] = "image"
    visible_code: str = ""                        # 可視透かし 4 文字コード(動画のみ)
    inv_nonce: int = 0                            # 不可視ビット衝突回避 nonce
    vis_nonce: int = 0                            # 可視コード衝突回避 nonce


def load_registry() -> list[TokenEntry]:
    if not REGISTRY_PATH.exists():
        return []
    with REGISTRY_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    # 旧フォーマット互換: source_image / media_type 欠損を吸収
    out: list[TokenEntry] = []
    for e in raw:
        if "source_path" not in e and "source_image" in e:
            e["source_path"] = e.pop("source_image")
        e.setdefault("media_type", "image")
        e.setdefault("visible_code", "")
        e.setdefault("inv_nonce", 0)
        e.setdefault("vis_nonce", 0)
        out.append(TokenEntry(**e))
    return out


def save_registry(entries: list[TokenEntry]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in entries], f, ensure_ascii=False, indent=2)


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def find_best_match(
    bits: str,
    registry: list[TokenEntry],
    media_type: Literal["image", "video"] | None = None,
) -> tuple[TokenEntry, int] | None:
    """入力 bit 列に最も近い台帳エントリを返す。戻り値: (entry, hamming距離)."""
    pool = registry if media_type is None else [e for e in registry if e.media_type == media_type]
    if not pool:
        return None
    scored = [(e, hamming(bits, e.bits)) for e in pool]
    scored.sort(key=lambda x: x[1])
    return scored[0]
