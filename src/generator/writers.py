"""矩阵样本输出。 / Matrix sample output writers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .types import FloatMatrix, MatrixSpec, OutputFormat


def write_text(path: Path, matrices: list[FloatMatrix], precision: int) -> None:
    """@brief 写出空行分隔矩阵文本。 / Write blank-line separated matrix text.

    @param path 输出路径。 / Output path.
    @param matrices 矩阵列表。 / Matrix list.
    @param precision 浮点有效数字位数。 / Floating-point significant digits.
    """
    float_format = f"%.{precision}g"
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for matrix_index, matrix in enumerate(matrices):
            if matrix_index > 0:
                output.write("\n\n")
            np.savetxt(output, matrix, fmt=float_format)
        output.write("\n")


def write_csv(path: Path, matrices: list[FloatMatrix], precision: int) -> None:
    """@brief 写出 CSV 矩阵文件。 / Write CSV matrix files.

    @param path 输出路径。 / Output path.
    @param matrices 矩阵列表。 / Matrix list.
    @param precision 浮点有效数字位数。 / Floating-point significant digits.
    """
    if len(matrices) != 1:
        stem = path.with_suffix("")
        suffix = path.suffix or ".csv"
        for index, matrix in enumerate(matrices):
            matrix_path = stem.with_name(f"{stem.name}_{index:04d}").with_suffix(suffix)
            np.savetxt(matrix_path, matrix, delimiter=",", fmt=f"%.{precision}g")
        return
    np.savetxt(path, matrices[0], delimiter=",", fmt=f"%.{precision}g")


def write_matrices(
    path: Path, matrices: list[FloatMatrix], output_format: OutputFormat, precision: int
) -> None:
    """@brief 按指定格式写出矩阵。 / Write matrices in the requested format.

    @param path 输出路径。 / Output path.
    @param matrices 矩阵列表。 / Matrix list.
    @param output_format 输出格式。 / Output format.
    @param precision 浮点有效数字位数。 / Floating-point significant digits.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format is OutputFormat.TEXT:
        write_text(path, matrices, precision)
    elif output_format is OutputFormat.NPY:
        if len(matrices) != 1:
            raise ValueError("npy format supports exactly one matrix; use npz or text for suites")
        with path.open("wb") as output:
            np.save(output, matrices[0])
    elif output_format is OutputFormat.NPZ:
        with path.open("wb") as output:
            np.savez_compressed(
                output,
                **{f"matrix_{index:04d}": matrix for index, matrix in enumerate(matrices)},
            )
    elif output_format is OutputFormat.CSV:
        write_csv(path, matrices, precision)
    else:
        raise ValueError(f"unsupported output format: {output_format}")


def write_manifest(
    path: Path | None,
    specs: list[MatrixSpec],
    output: Path,
    output_format: OutputFormat,
) -> None:
    """@brief 写出 JSON manifest。 / Write a JSON manifest.

    @param path manifest 路径。 / Manifest path.
    @param specs 矩阵规格列表。 / Matrix specification list.
    @param output 矩阵输出路径。 / Matrix output path.
    @param output_format 输出格式。 / Output format.
    """
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "output": str(output),
        "format": output_format.value,
        "count": len(specs),
        "matrices": [asdict(spec) for spec in specs],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
