"""矩阵生成器 CLI 测试。 / Matrix generator CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from generator.cli import app

runner = CliRunner()


def test_generate_writes_text_and_manifest(tmp_path: Path) -> None:
    """@brief 验证文本矩阵和 manifest 写出。 / Verify text matrix and manifest output."""
    result = runner.invoke(
        app,
        [
            "generate",
            "--output",
            str(tmp_path / "matrices.txt"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--shape",
            "2x3",
            "--distribution",
            "identity",
            "--distribution",
            "zero",
        ],
    )

    assert result.exit_code == 0, result.output
    text = (tmp_path / "matrices.txt").read_text(encoding="utf-8")
    assert "1" in text
    assert "\n\n" in text

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 2
    assert manifest["matrices"][0]["rows"] == 2
    assert manifest["matrices"][0]["cols"] == 3


def test_suite_writes_npz_bundle(tmp_path: Path) -> None:
    """@brief 验证默认套件可写为 npz。 / Verify the default suite can be written as npz."""
    result = runner.invoke(app, ["suite", "--output", str(tmp_path / "suite.npz"), "--format", "npz"])

    assert result.exit_code == 0, result.output
    bundle = np.load(tmp_path / "suite.npz")
    assert "matrix_0000" in bundle.files
    assert bundle["matrix_0000"].shape == (4, 4)


def test_rejects_bad_shape(tmp_path: Path) -> None:
    """@brief 验证非法形状会失败。 / Verify malformed shapes fail."""
    result = runner.invoke(app, ["generate", "--output", str(tmp_path / "bad.txt"), "--shape", "2-by-3"])

    assert result.exit_code != 0


def test_help_describes_formats_and_examples() -> None:
    """@brief 验证顶层 help 包含格式契约和示例。 / Verify top-level help describes formats and examples."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "MatrixFileReader" in result.output
    assert "Examples" in result.output
    assert "Distributions" in result.output
