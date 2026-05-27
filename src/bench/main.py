"""实验流水线命令入口。 / Experiment pipeline command entry point."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .runner import RunOptions, discover_instance_files, load_instance, run_bench

HELP_TEXT = """MPI SVD 实验流水线。

bench discovers JSON instance files under experiments/instances.
Each instance owns one or more jobs.
Jobs are the concurrent scheduling unit.
Experiments inside the same job run sequentially.

这种结构让互不相关的配置可以并发执行。
同一个 job 的 perf.data/stdout/metrics 仍按顺序写入独立 run_dir。

Instance JSON shape:
{
  "schema_version": 1,
  "name": "svd-baseline",
  "jobs": [
    {
      "name": "naive-row-major",
      "build": "relwithdebinfo",
      "arguments": [
        "--input", "experiments/data/tall.txt",
        "--output", "{run_dir}/svd.txt",
        "--metrics", "{run_dir}/metrics.jsonl",
        "--layout", "row-major",
        "--kernel", "naive"
      ],
      "experiments": [
        "native",
        {
          "name": "perf-counters",
          "tool": "perf-stat",
          "repeat": 3,
          "events": ["cycles", "instructions", "cache-misses"]
        }
      ]
    }
  ]
}

Supported tools:
  native
  perf-stat
  perf-record
  callgrind
  cachegrind

Template variables:
  {repo_root}       repository root
  {instances_dir}  instance JSON directory
  {results_dir}    current timestamped result directory
  {instance_file}  current JSON file
  {instance}       slugified instance name
  {job}            slugified job name
  {build}          CMake preset name
  {tool}           profiling tool name
  {experiment}     slugified experiment name
  {repeat}         repeat index
  {run_dir}        per-repeat result directory
  {job_dir}        per-job result directory
"""

RUN_HELP = """发现并运行实验实例。

默认会先对实例里声明的 build preset 执行:
  cmake --preset <build>
  cmake --build --preset <build>

每次运行创建独立结果目录:
  experiments/results/<run_id>/

每个 experiment repeat 都有自己的:
  stdout.txt
  stderr.txt
  tool artifacts

perf-stat writes perf-stat.csv.
perf-record writes perf.data.
Callgrind/Cachegrind write their out files.

Examples:
  bench run
  bench run --jobs 4
  bench run --skip-build --dry-run
  bench run -i experiments/instances -o experiments/results
"""

VALIDATE_HELP = """验证实例 JSON 文件。

validate 只检查 JSON 是否能被 bench schema 接受。
它会展示每个实例的 job 和 experiment 数量。
它不会构建 CMake preset。
它不会执行 mpilab、perf 或 Valgrind。

Examples:
  bench validate
  bench validate -i experiments/instances
"""

app = typer.Typer(
    add_completion=False,
    help=HELP_TEXT,
)
console = Console()


def repo_root_from_cwd() -> Path:
    """@brief 从当前目录推断仓库根目录。 / Infer the repository root from the current directory.

    @return 仓库根目录。 / Repository root.
    """
    return Path.cwd()


def render_report(results_dir: Path, report_count: int, failed_count: int, skipped_count: int) -> None:
    """@brief 渲染运行摘要。 / Render a run summary.

    @param results_dir 结果目录。 / Results directory.
    @param report_count 实验数量。 / Experiment count.
    @param failed_count 失败数量。 / Failure count.
    @param skipped_count 跳过数量。 / Skipped count.
    """
    table = Table(title="Bench run summary")
    table.add_column("results")
    table.add_column("experiments", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("skipped", justify="right")
    table.add_row(str(results_dir), str(report_count), str(failed_count), str(skipped_count))
    console.print(table)
    console.print(f"[green]wrote[/green] {results_dir / 'bench-report.json'}")


def display_path(path: Path, repo_root: Path) -> str:
    """@brief 返回适合显示的路径。 / Return a path suitable for display.

    @param path 原始路径。 / Raw path.
    @param repo_root 仓库根目录。 / Repository root.
    @return 相对仓库路径或绝对路径。 / Repository-relative path or absolute path.
    """
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


@app.command(help=RUN_HELP)
def run(
    instances: Annotated[
        Path,
        typer.Option("--instances", "-i", help="实例 JSON 目录。 / Instance JSON directory."),
    ] = Path("experiments/instances"),
    results: Annotated[
        Path,
        typer.Option("--results", "-o", help="结果输出目录。 / Result output directory."),
    ] = Path("experiments/results"),
    jobs: Annotated[
        int,
        typer.Option("--jobs", "-j", min=1, help="并发 job 数。 / Concurrent job count."),
    ] = max(1, os.cpu_count() or 1),
    build: Annotated[
        bool,
        typer.Option("--build/--skip-build", help="运行前构建 CMake preset。 / Build CMake presets before running."),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="只生成命令和报告,不执行。 / Generate commands and report without executing."),
    ] = False,
) -> None:
    """运行 benchmark 实例。 / Run benchmark instances."""
    repo_root = repo_root_from_cwd()
    instance_files = discover_instance_files(repo_root / instances if not instances.is_absolute() else instances)
    if not instance_files:
        raise typer.BadParameter(f"no JSON instance files found under {instances}")

    options = RunOptions(
        repo_root=repo_root,
        instances_dir=(repo_root / instances).resolve() if not instances.is_absolute() else instances,
        results_dir=(repo_root / results).resolve() if not results.is_absolute() else results,
        max_workers=jobs,
        build=build,
        dry_run=dry_run,
    )
    try:
        report = run_bench(options)
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    failed_count = sum(1 for result in report.results if result.status == "failed")
    skipped_count = sum(1 for result in report.results if result.status == "skipped")
    render_report(Path(report.results_dir), len(report.results), failed_count, skipped_count)
    if failed_count > 0:
        raise typer.Exit(code=1)


@app.command(help=VALIDATE_HELP)
def validate(
    instances: Annotated[
        Path,
        typer.Option("--instances", "-i", help="实例 JSON 目录。 / Instance JSON directory."),
    ] = Path("experiments/instances"),
) -> None:
    """验证 benchmark 实例。 / Validate benchmark instances."""
    repo_root = repo_root_from_cwd()
    instances_dir = (repo_root / instances).resolve() if not instances.is_absolute() else instances
    instance_files = discover_instance_files(instances_dir)
    if not instance_files:
        raise typer.BadParameter(f"no JSON instance files found under {instances}")

    table = Table(title="Bench instances")
    table.add_column("file")
    table.add_column("jobs", justify="right")
    table.add_column("experiments", justify="right")
    for path in instance_files:
        try:
            instance = load_instance(path)
        except ValueError as error:
            raise typer.BadParameter(str(error)) from error
        experiment_count = sum(len(job.experiments) for job in instance.jobs)
        table.add_row(display_path(path, repo_root), str(len(instance.jobs)), str(experiment_count))
    console.print(table)


def main() -> None:
    """执行 bench 命令行入口。 / Execute the bench command-line entry point."""
    app()


if __name__ == "__main__":
    main()
