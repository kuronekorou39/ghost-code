"""環境セットアップの最小確認スクリプト。"""
from __future__ import annotations

import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"  [{i}] {props.name} (VRAM {vram_gb:.1f} GB, sm_{props.major}{props.minor})")

        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"GPU matmul OK (output sum={y.sum().item():.4f})")


if __name__ == "__main__":
    main()
