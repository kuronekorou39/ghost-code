"""HMAC ベースのペイロード派生。

user_id → 40bit を SECRET_KEY を使って決定論的・予測不能に派生する。
攻撃者が TrustMark で抜いても、ビット列から user_id を逆算できない。
検出時は登録ユーザー全員の HMAC 派生値と Hamming 距離で照合(線形探索)。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KEY_PATH = ROOT / ".secrets" / "watermark.key"

INVISIBLE_BITS_LEN = 40
VISIBLE_CODE_LEN = 4
VISIBLE_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"  # 36


class WatermarkSecret:
    """秘密鍵の読み書き。git に絶対入れない。"""

    def __init__(self, key: bytes) -> None:
        if len(key) < 16:
            raise ValueError("key must be >= 16 bytes")
        self._key = key

    @classmethod
    def load_or_create(cls, path: Path = DEFAULT_KEY_PATH) -> "WatermarkSecret":
        if path.exists():
            with path.open("rb") as f:
                return cls(f.read())
        # 新規作成(初回起動時)
        path.parent.mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(32)
        with path.open("wb") as f:
            f.write(key)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass  # Windows
        return cls(key)

    @classmethod
    def from_env(cls, var: str = "GHOST_WM_KEY") -> "WatermarkSecret":
        v = os.environ.get(var)
        if not v:
            raise RuntimeError(f"env var {var} not set")
        return cls(bytes.fromhex(v))

    def hmac_digest(self, msg: str) -> bytes:
        return hmac.new(self._key, msg.encode("utf-8"), hashlib.sha256).digest()


def derive_invisible_bits(secret: WatermarkSecret, user_id: str, nonce: int = 0) -> str:
    """40bit を派生。同じ user_id + nonce なら毎回同じ bit 列。"""
    digest = secret.hmac_digest(f"inv|{user_id}|{nonce}")
    bits_int = int.from_bytes(digest[:5], "big") & ((1 << INVISIBLE_BITS_LEN) - 1)
    return format(bits_int, f"0{INVISIBLE_BITS_LEN}b")


def derive_visible_code(secret: WatermarkSecret, user_id: str, nonce: int = 0) -> str:
    """4 文字 [a-z0-9] を派生。"""
    digest = secret.hmac_digest(f"vis|{user_id}|{nonce}")
    n = int.from_bytes(digest[:8], "big")
    out = []
    for _ in range(VISIBLE_CODE_LEN):
        out.append(VISIBLE_ALPHABET[n % 36])
        n //= 36
    return "".join(out)


def find_unique_invisible_bits(
    secret: WatermarkSecret,
    user_id: str,
    existing_bits: list[str],
    min_distance: int = 12,
    max_attempts: int = 10_000,
) -> tuple[str, int]:
    """既存ユーザーから min_distance bit 以上離れた bit 列を nonce 探索で見つける。"""
    for nonce in range(max_attempts):
        bits = derive_invisible_bits(secret, user_id, nonce=nonce)
        if all(hamming(bits, e) >= min_distance for e in existing_bits):
            return bits, nonce
    raise RuntimeError(
        f"could not find unique invisible bits for {user_id} after {max_attempts} tries"
    )


def find_unique_visible_code(
    secret: WatermarkSecret,
    user_id: str,
    existing_codes: list[str],
    max_attempts: int = 10_000,
) -> tuple[str, int]:
    """既存ユーザーと衝突しない 4 文字コードを nonce 探索で見つける。"""
    existing_set = set(c.lower() for c in existing_codes)
    for nonce in range(max_attempts):
        code = derive_visible_code(secret, user_id, nonce=nonce)
        if code not in existing_set:
            return code, nonce
    raise RuntimeError(f"could not find unique visible code for {user_id}")


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(1 for x, y in zip(a, b) if x != y)


def identify_from_invisible_bits(
    extracted: str,
    user_ids: list[str],
    secret: WatermarkSecret,
    max_distance: int = 10,
) -> tuple[str, int] | None:
    """抜き出されたビット列に最も近い user_id を返す。"""
    if len(extracted) != INVISIBLE_BITS_LEN:
        return None
    best: tuple[str, int] | None = None
    for uid in user_ids:
        expected = derive_invisible_bits(secret, uid)
        d = hamming(extracted, expected)
        if best is None or d < best[1]:
            best = (uid, d)
    if best is None or best[1] > max_distance:
        return None
    return best


def identify_from_visible_code(
    extracted: str,
    user_ids: list[str],
    secret: WatermarkSecret,
) -> str | None:
    """4 文字コードから user_id を逆引き(完全一致)。"""
    if len(extracted) != VISIBLE_CODE_LEN:
        return None
    extracted = extracted.lower()
    for uid in user_ids:
        if derive_visible_code(secret, uid) == extracted:
            return uid
    return None
