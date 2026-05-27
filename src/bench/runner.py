"""实验编排执行器。 / Experiment orchestration runner."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .types import BenchJob, ExperimentInstance, ExperimentSpec, ExperimentTool

DEFAULT_PERF_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branches",
    "branch-misses",
]


@dataclass(frozen=True)
class RunOptions:
    """@brief bench 运行选项。 / Bench run options."""

    repo_root: Path
    instances_dir: Path
    results_dir: Path
    max_workers: int
    build: bool
    dry_run: bool


@dataclass(frozen=True)
class ExperimentResult:
    """@brief 单次实验结果。 / Single experiment result."""

    instance: str
    job: str
    experiment: str
    tool: str
    repeat: int
    status: str
    command: list[str]
    started_at: str
    finished_at: str
    duration_seconds: float
    return_code: int | None
    stdout_path: str | None
    stderr_path: str | None
    artifacts: dict[str, str]
    error: str | None = None


@dataclass(frozen=True)
class BenchReport:
    """@brief bench 总报告。 / Aggregate bench report."""

    run_id: str
    started_at: str
    finished_at: str
    duration_seconds: float
    instances_dir: str
    results_dir: str
    dry_run: bool
    max_workers: int
    results: list[ExperimentResult]


def utc_now() -> datetime:
    """@brief 返回当前 UTC 时间。 / Return the current UTC time.

    @return 当前 UTC 时间戳。 / Current UTC timestamp.
    """
    return datetime.now(timezone.utc)


def format_time(value: datetime) -> str:
    """@brief 格式化时间戳。 / Format a timestamp.

    @param value 时间戳。 / Timestamp.
    @return ISO-8601 文本。 / ISO-8601 text.
    """
    return value.isoformat(timespec="seconds")


def slugify(text: str) -> str:
    """@brief 将名称转换为文件系统安全片段。 / Convert a name to a filesystem-safe fragment.

    @param text 原始名称。 / Raw name.
    @return 安全名称。 / Safe name.
    """
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text.strip()).strip("-")
    return slug or "unnamed"


def discover_instance_files(instances_dir: Path) -> list[Path]:
    """@brief 发现实例 JSON 文件。 / Discover instance JSON files.

    @param instances_dir 实例目录。 / Instance directory.
    @return 排序后的 JSON 文件列表。 / Sorted JSON file list.
    """
    if not instances_dir.exists():
        return []
    return sorted(path for path in instances_dir.rglob("*.json") if path.is_file())


def load_instance(path: Path) -> ExperimentInstance:
    """@brief 读取并验证实例文件。 / Load and validate an instance file.

    @param path JSON 文件路径。 / JSON file path.
    @return 实例模型。 / Instance model.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentInstance.model_validate(data)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    except ValidationError as error:
        raise ValueError(f"{path}: invalid instance schema: {error}") from error


def resolve_executable(repo_root: Path, job: BenchJob) -> Path:
    """@brief 解析 job 使用的可执行文件路径。 / Resolve the executable path for a job.

    @param repo_root 仓库根目录。 / Repository root.
    @param job benchmark job。 / Benchmark job.
    @return 可执行文件路径。 / Executable path.
    """
    if job.executable is not None:
        return (repo_root / job.executable).resolve() if not job.executable.is_absolute() else job.executable
    return (repo_root / "build" / job.build / "src" / "mpilab" / "mpilab").resolve()


def expand_text(text: str, context: dict[str, str]) -> str:
    """@brief 展开命令模板文本。 / Expand command-template text.

    @param text 原始文本。 / Raw text.
    @param context 模板上下文。 / Template context.
    @return 展开后的文本。 / Expanded text.
    """
    return text.format_map(context)


def expand_arguments(arguments: list[str], context: dict[str, str]) -> list[str]:
    """@brief 展开参数列表。 / Expand an argument list.

    @param arguments 原始参数。 / Raw arguments.
    @param context 模板上下文。 / Template context.
    @return 展开后的参数。 / Expanded arguments.
    """
    return [expand_text(argument, context) for argument in arguments]


def build_base_command(
    repo_root: Path,
    job: BenchJob,
    experiment: ExperimentSpec,
    context: dict[str, str],
) -> list[str]:
    """@brief 构造 mpilab 基础命令。 / Build the base mpilab command.

    @param repo_root 仓库根目录。 / Repository root.
    @param job benchmark job。 / Benchmark job.
    @param experiment 实验规格。 / Experiment specification.
    @param context 模板上下文。 / Template context.
    @return 基础命令。 / Base command.
    """
    executable = resolve_executable(repo_root, job)
    arguments = expand_arguments(job.arguments, context) + expand_arguments(experiment.arguments, context)
    mpi = job.config.mpi
    if mpi is not None and mpi.enable_flag and "--mpi" not in arguments:
        arguments.append("--mpi")
    command = [str(executable), *arguments]
    if mpi is not None and mpi.processes > 1:
        command = [mpi.launcher, "-np", str(mpi.processes), *mpi.extra_args, *command]
    return command


def build_tool_command(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
) -> tuple[list[str], dict[str, str], str | None]:
    """@brief 构造带剖析工具的命令。 / Build a command wrapped by a profiling tool.

    @param base_command mpilab 基础命令。 / Base mpilab command.
    @param experiment 实验规格。 / Experiment specification.
    @param run_dir 单次运行目录。 / Single-run directory.
    @return 命令、产物和需要检查的工具名。 / Command, artifacts, and required tool name.
    """
    artifacts: dict[str, str] = {}
    if experiment.tool == ExperimentTool.NATIVE:
        return base_command, artifacts, None
    if experiment.tool == ExperimentTool.PERF_STAT:
        stat_path = run_dir / "perf-stat.csv"
        artifacts["perf_stat"] = str(stat_path)
        events = experiment.events or DEFAULT_PERF_EVENTS
        command = [
            "perf",
            "stat",
            "-x",
            ",",
            "-o",
            str(stat_path),
            "-e",
            ",".join(events),
            "--",
            *base_command,
        ]
        return command, artifacts, "perf"
    if experiment.tool == ExperimentTool.PERF_RECORD:
        data_path = run_dir / "perf.data"
        artifacts["perf_data"] = str(data_path)
        command = ["perf", "record", "-g", "-o", str(data_path)]
        if experiment.events:
            command.extend(["-e", ",".join(experiment.events)])
        return [*command, "--", *base_command], artifacts, "perf"
    if experiment.tool == ExperimentTool.CALLGRIND:
        out_path = run_dir / "callgrind.out"
        artifacts["callgrind"] = str(out_path)
        command = ["valgrind", "--tool=callgrind", f"--callgrind-out-file={out_path}", *base_command]
        return command, artifacts, "valgrind"
    if experiment.tool == ExperimentTool.CACHEGRIND:
        out_path = run_dir / "cachegrind.out"
        artifacts["cachegrind"] = str(out_path)
        command = ["valgrind", "--tool=cachegrind", f"--cachegrind-out-file={out_path}", *base_command]
        return command, artifacts, "valgrind"
    return base_command, artifacts, None


def run_process(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    timeout_seconds: float | None,
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[int | None, str | None]:
    """@brief 执行子进程并捕获输出。 / Run a subprocess and capture output.

    @param command 命令。 / Command.
    @param cwd 工作目录。 / Working directory.
    @param env 环境变量。 / Environment variables.
    @param timeout_seconds 超时时间。 / Timeout in seconds.
    @param stdout_path 标准输出路径。 / Stdout path.
    @param stderr_path 标准错误路径。 / Stderr path.
    @return 返回码和错误文本。 / Return code and error text.
    """
    try:
        with (
            stdout_path.open("w", encoding="utf-8") as stdout_file,
            stderr_path.open("w", encoding="utf-8") as stderr_file,
        ):
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                timeout=timeout_seconds,
                check=False,
            )
        return completed.returncode, None
    except subprocess.TimeoutExpired as error:
        return None, f"timed out after {error.timeout} seconds"
    except OSError as error:
        return None, str(error)


def run_one_experiment(
    options: RunOptions,
    instance_path: Path,
    instance: ExperimentInstance,
    job: BenchJob,
    experiment: ExperimentSpec,
    repeat_index: int,
    job_dir: Path,
) -> ExperimentResult:
    """@brief 运行单个实验。 / Run one experiment.

    @param options bench 运行选项。 / Bench run options.
    @param instance_path 实例文件路径。 / Instance file path.
    @param instance 实例模型。 / Instance model.
    @param job benchmark job。 / Benchmark job.
    @param experiment 实验规格。 / Experiment specification.
    @param repeat_index 重复序号。 / Repeat index.
    @param job_dir job 结果目录。 / Job result directory.
    @return 实验结果。 / Experiment result.
    """
    instance_name = instance.name or instance_path.stem
    experiment_name = experiment.name or experiment.tool.value
    run_dir = job_dir / slugify(experiment_name) / f"repeat-{repeat_index:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "repo_root": str(options.repo_root),
        "instances_dir": str(options.instances_dir),
        "results_dir": str(options.results_dir),
        "instance": slugify(instance_name),
        "instance_file": str(instance_path),
        "job": slugify(job.name),
        "build": job.build,
        "tool": experiment.tool.value,
        "experiment": slugify(experiment_name),
        "repeat": str(repeat_index),
        "run_dir": str(run_dir),
        "job_dir": str(job_dir),
    }
    base_command = build_base_command(options.repo_root, job, experiment, context)
    command, artifacts, required_tool = build_tool_command(base_command, experiment, run_dir)
    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    cwd = job.config.cwd or options.repo_root
    if not cwd.is_absolute():
        cwd = options.repo_root / cwd
    env = os.environ.copy()
    env.update(job.config.env)
    env.update(experiment.env)
    timeout_seconds = experiment.timeout_seconds or job.config.timeout_seconds

    started = utc_now()
    start_seconds = time.perf_counter()
    if required_tool is not None and shutil.which(required_tool) is None:
        return_code = None
        error = f"required tool not found: {required_tool}"
        status = "skipped"
    elif options.dry_run:
        return_code = 0
        error = None
        status = "dry-run"
        stdout_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
    else:
        return_code, error = run_process(command, cwd, env, timeout_seconds, stdout_path, stderr_path)
        status = "passed" if return_code == 0 else "failed"
    finished = utc_now()
    duration_seconds = time.perf_counter() - start_seconds

    return ExperimentResult(
        instance=str(instance_path),
        job=job.name,
        experiment=experiment_name,
        tool=experiment.tool.value,
        repeat=repeat_index,
        status=status,
        command=command,
        started_at=format_time(started),
        finished_at=format_time(finished),
        duration_seconds=duration_seconds,
        return_code=return_code,
        stdout_path=str(stdout_path) if stdout_path.exists() else None,
        stderr_path=str(stderr_path) if stderr_path.exists() else None,
        artifacts=artifacts,
        error=error,
    )


def run_job(
    options: RunOptions,
    instance_path: Path,
    instance: ExperimentInstance,
    job: BenchJob,
) -> list[ExperimentResult]:
    """@brief 顺序运行一个 job 内的所有实验。 / Run all experiments in one job sequentially.

    @param options bench 运行选项。 / Bench run options.
    @param instance_path 实例文件路径。 / Instance file path.
    @param instance 实例模型。 / Instance model.
    @param job benchmark job。 / Benchmark job.
    @return 实验结果列表。 / Experiment result list.
    """
    instance_name = instance.name or instance_path.stem
    job_dir = options.results_dir / slugify(instance_name) / slugify(job.name)
    results: list[ExperimentResult] = []
    for experiment in job.experiments:
        for repeat_index in range(1, experiment.repeat + 1):
            result = run_one_experiment(options, instance_path, instance, job, experiment, repeat_index, job_dir)
            results.append(result)
            if result.status == "failed" and not job.config.continue_on_failure:
                return results
    return results


def build_presets(repo_root: Path, build_names: list[str], dry_run: bool) -> None:
    """@brief 配置并构建 CMake preset。 / Configure and build CMake presets.

    @param repo_root 仓库根目录。 / Repository root.
    @param build_names 构建 preset 名称。 / Build preset names.
    @param dry_run 是否只预演。 / Whether this is a dry run.
    """
    for build_name in build_names:
        commands = [
            ["cmake", "--preset", build_name],
            ["cmake", "--build", "--preset", build_name],
        ]
        for command in commands:
            if dry_run:
                continue
            subprocess.run(command, cwd=repo_root, check=True)


def write_report(report: BenchReport, path: Path) -> None:
    """@brief 写出 bench 总报告。 / Write the aggregate bench report.

    @param report 总报告。 / Aggregate report.
    @param path 输出路径。 / Output path.
    """
    payload: dict[str, Any] = asdict(report)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_bench(options: RunOptions) -> BenchReport:
    """@brief 发现实例并并发执行 jobs。 / Discover instances and run jobs concurrently.

    @param options bench 运行选项。 / Bench run options.
    @return bench 总报告。 / Aggregate bench report.
    """
    started = utc_now()
    run_id = started.strftime("%Y%m%dT%H%M%SZ")
    effective_options = replace(options, results_dir=options.results_dir / run_id)
    effective_options.results_dir.mkdir(parents=True, exist_ok=True)
    instance_files = discover_instance_files(options.instances_dir)
    instances = [(path, load_instance(path)) for path in instance_files]
    build_names = sorted({job.build for _, instance in instances for job in instance.jobs})
    if options.build and build_names:
        build_presets(options.repo_root, build_names, options.dry_run)

    futures: list[Future[list[ExperimentResult]]] = []
    results: list[ExperimentResult] = []
    with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
        for instance_path, instance in instances:
            futures.extend(
                executor.submit(run_job, effective_options, instance_path, instance, job)
                for job in instance.jobs
            )
        for future in as_completed(futures):
            results.extend(future.result())

    results.sort(key=lambda item: (item.instance, item.job, item.experiment, item.repeat))
    finished = utc_now()
    report = BenchReport(
        run_id=run_id,
        started_at=format_time(started),
        finished_at=format_time(finished),
        duration_seconds=(finished - started).total_seconds(),
        instances_dir=str(options.instances_dir),
        results_dir=str(effective_options.results_dir),
        dry_run=options.dry_run,
        max_workers=options.max_workers,
        results=results,
    )
    write_report(report, effective_options.results_dir / "bench-report.json")
    return report
