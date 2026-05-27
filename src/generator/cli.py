"""矩阵样本生成命令行界面。 / Matrix sample generation command-line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .distributions import generate_matrix
from .specs import (
    build_specs,
    default_suite_distributions,
    default_suite_shapes,
    parse_shape,
)
from .types import Distribution, MatrixSpec, OutputFormat
from .writers import write_manifest, write_matrices

HELP_TEXT = """生成 MPI SVD 实验矩阵样本。

默认 text 格式与 C++ MatrixFileReader 兼容：每行是空白分隔数值，空行分隔多块矩阵。
Use generate for hand-picked workloads, and suite for a reproducible coverage set.

Examples:
  matrixgen generate -o experiments/instances/tall.txt -s 1024x128 -d normal
  matrixgen generate -o sparse.txt -s 512x512 -d sparse --density 0.01 --seed 7
  matrixgen suite -o experiments/instances/default-suite.txt --manifest suite.json
"""

app = typer.Typer(
    add_completion=False,
    help=HELP_TEXT,
    epilog="Distributions: uniform, normal, lognormal, integer, rademacher, sparse, "
    "low-rank, ill-conditioned, diagonal, banded, hilbert, identity, cauchy, zero.",
)
console = Console()


def fail_bad_parameter(error: ValueError) -> None:
    """@brief 将领域错误转换为 CLI 参数错误。 / Convert domain errors to CLI parameter errors.

    @param error 领域错误。 / Domain error.
    """
    raise typer.BadParameter(str(error)) from error


def render_summary(specs: list[MatrixSpec], output: Path, output_format: OutputFormat) -> None:
    """@brief 渲染生成摘要表。 / Render a generation summary table.

    @param specs 矩阵规格列表。 / Matrix specification list.
    @param output 输出路径。 / Output path.
    @param output_format 输出格式。 / Output format.
    """
    table = Table(title="Generated matrix samples")
    table.add_column("name")
    table.add_column("shape")
    table.add_column("distribution")
    table.add_column("seed", justify="right")
    for spec in specs:
        table.add_row(spec.name, f"{spec.rows}x{spec.cols}", spec.distribution, str(spec.seed))
    console.print(table)
    console.print(f"[green]wrote[/green] {len(specs)} matrix sample(s) to {output} as {output_format.value}")


@app.command(
    help="""生成自定义矩阵样本。

重复 --shape 和 --distribution 会生成二者的笛卡尔积。默认生成一个 64x64 normal 矩阵。
The output defaults to text so it can be consumed directly by mpilab --input.
""",
)
def generate(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="输出文件路径。 / Output file path.")
    ],
    shape: Annotated[
        list[str] | None,
        typer.Option(
            "--shape",
            "-s",
            help="矩阵形状 ROWSxCOLS，可重复。 / Matrix shape ROWSxCOLS, repeatable.",
        ),
    ] = None,
    distribution: Annotated[
        list[Distribution] | None,
        typer.Option(
            "--distribution",
            "-d",
            help="数值分布，可重复。 / Numeric distribution, repeatable.",
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="输出格式：text、npy、npz、csv。 / Output format."),
    ] = OutputFormat.TEXT,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", help="JSON manifest 输出路径。 / JSON manifest path."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="基础随机种子。 / Base random seed.")] = 42,
    density: Annotated[
        float,
        typer.Option("--density", help="稀疏矩阵非零概率。 / Sparse non-zero probability."),
    ] = 0.05,
    rank: Annotated[
        int | None,
        typer.Option("--rank", help="低秩矩阵目标秩。 / Target rank for low-rank matrices."),
    ] = None,
    condition: Annotated[
        float | None,
        typer.Option("--condition", help="病态矩阵条件数。 / Condition number."),
    ] = None,
    bandwidth: Annotated[
        int | None,
        typer.Option("--bandwidth", help="带状矩阵半带宽。 / Banded half-bandwidth."),
    ] = None,
    low: Annotated[
        float,
        typer.Option("--low", help="均匀/整数分布下界。 / Lower bound for uniform/integer."),
    ] = -1.0,
    high: Annotated[
        float,
        typer.Option("--high", help="均匀/整数分布上界。 / Upper bound for uniform/integer."),
    ] = 1.0,
    mean: Annotated[float, typer.Option("--mean", help="正态/对数正态均值参数。 / Mean parameter.")] = 0.0,
    std: Annotated[
        float,
        typer.Option("--std", help="正态/对数正态标准差参数。 / Standard deviation parameter."),
    ] = 1.0,
    noise: Annotated[
        float,
        typer.Option("--noise", help="低秩矩阵加性噪声标准差。 / Low-rank additive-noise stddev."),
    ] = 0.0,
    scale: Annotated[float, typer.Option("--scale", help="最终数值缩放。 / Final value scale.")] = 1.0,
    offset: Annotated[float, typer.Option("--offset", help="最终数值偏移。 / Final value offset.")] = 0.0,
    precision: Annotated[
        int,
        typer.Option("--precision", help="文本/CSV 有效数字位数。 / Text/CSV significant digits."),
    ] = 17,
) -> None:
    """生成用户指定矩阵样本。 / Generate user-specified matrix samples."""
    try:
        shapes = [parse_shape(item) for item in shape] if shape else [(64, 64)]
        distributions = distribution if distribution else [Distribution.NORMAL]
        specs = build_specs(
            shapes,
            distributions,
            seed,
            density,
            rank,
            condition,
            bandwidth,
            scale,
            offset,
        )
        matrices = [generate_matrix(spec, low, high, mean, std, noise) for spec in specs]
        write_matrices(output, matrices, output_format, precision)
        write_manifest(manifest, specs, output, output_format)
    except ValueError as error:
        fail_bad_parameter(error)
    render_summary(specs, output, output_format)


@app.command(
    help="""生成默认综合实验套件。

套件覆盖小型/方阵/高矩阵/宽矩阵/非 2 次幂 shape，以及 dense、sparse、low-rank、
ill-conditioned、banded、Hilbert、Rademacher 等负载。
""",
)
def suite(
    output: Annotated[
        Path, typer.Option("--output", "-o", help="输出文件路径。 / Output file path.")
    ],
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="输出格式：text、npz、csv。 / Output format."),
    ] = OutputFormat.TEXT,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", help="JSON manifest 输出路径。 / JSON manifest path."),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="基础随机种子。 / Base random seed.")] = 42,
    density: Annotated[
        float,
        typer.Option("--density", help="稀疏矩阵非零概率。 / Sparse non-zero probability."),
    ] = 0.03,
    rank: Annotated[
        int | None,
        typer.Option("--rank", help="低秩矩阵目标秩。 / Target rank for low-rank matrices."),
    ] = None,
    condition: Annotated[float, typer.Option("--condition", help="病态矩阵条件数。 / Condition number.")] = 1.0e10,
    bandwidth: Annotated[int, typer.Option("--bandwidth", help="带状矩阵半带宽。 / Banded half-bandwidth.")] = 3,
    precision: Annotated[
        int,
        typer.Option("--precision", help="文本/CSV 有效数字位数。 / Text/CSV significant digits."),
    ] = 17,
) -> None:
    """生成默认综合实验矩阵套件。 / Generate the default comprehensive experiment matrix suite."""
    try:
        specs = build_specs(
            default_suite_shapes(),
            default_suite_distributions(),
            seed,
            density,
            rank,
            condition,
            bandwidth,
            scale=1.0,
            offset=0.0,
        )
        matrices = [generate_matrix(spec, low=-1.0, high=1.0, mean=0.0, std=1.0, noise=0.0) for spec in specs]
        write_matrices(output, matrices, output_format, precision)
        write_manifest(manifest, specs, output, output_format)
    except ValueError as error:
        fail_bad_parameter(error)
    render_summary(specs, output, output_format)


def main() -> None:
    """@brief 执行 generator 命令行入口。 / Execute the generator command-line entry point."""
    app()
