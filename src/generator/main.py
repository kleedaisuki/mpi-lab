"""矩阵样本生成 CLI 入口。 / Matrix sample generation CLI entry point."""

from __future__ import annotations

from .cli import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
