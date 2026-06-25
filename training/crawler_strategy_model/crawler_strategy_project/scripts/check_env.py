#!/usr/bin/env python3
# 中文功能说明：Crawler Strategy 训练环境检查脚本，确认关键依赖可导入。

"""Print and validate the training environment."""

from __future__ import annotations

import importlib
import platform
import sys


REQUIRED = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "llamafactory",
]


def main() -> int:
    print(f"python: {sys.version.split()[0]} ({sys.executable})")
    print(f"platform: {platform.platform()}")
    if sys.version_info < (3, 11):
        print("ERROR: LLaMA-Factory 0.9.5 requires Python >= 3.11.")
        return 2

    missing: list[str] = []
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - env diagnostics should show import failures.
            missing.append(name)
            print(f"{name}: MISSING ({exc})")
            continue
        print(f"{name}: {getattr(module, '__version__', 'unknown')}")

    if missing:
        print("ERROR: missing required packages: " + ", ".join(missing))
        return 3

    import torch

    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(idx)
        gb = prop.total_memory / 1024**3
        print(f"cuda:{idx}: {prop.name}, capability={prop.major}.{prop.minor}, memory={gb:.1f}GB")
        if prop.major < 7:
            print("note: this GPU does not support bf16; use fp16 and flash_attn=disabled.")

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return 4

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
