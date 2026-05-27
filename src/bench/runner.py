"""实验编排执行器。 / Experiment orchestration runner."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .types import BenchJob, ExperimentInstance, ExperimentSpec, ExperimentTool, SampleSpec

DEFAULT_PERF_EVENTS = [
    "cycles",
    "instructions",
    "cache-references",
    "cache-misses",
    "branches",
    "branch-misses",
]
PERF_STAT_MIN_COLUMNS = 3

ToolCommand = tuple[list[str], dict[str, str], str | None]
ToolBuilder = Callable[[list[str], ExperimentSpec, Path, dict[str, str]], ToolCommand]


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
class SampleResult:
    """@brief 样本生成结果。 / Matrix sample generation result."""

    instance: str
    sample: str
    output_path: str
    status: str
    command: list[str]
    duration_seconds: float
    return_code: int | None
    error: str | None = None


@dataclass(frozen=True)
class NumericSummary:
    """@brief 数值序列摘要。 / Numeric series summary."""

    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    stdev: float | None


@dataclass(frozen=True)
class ExperimentSummary:
    """@brief repeat 聚合摘要。 / Repeat aggregation summary."""

    instance: str
    job: str
    experiment: str
    tool: str
    repeat_count: int
    status_counts: dict[str, int]
    duration_seconds: NumericSummary
    return_codes: dict[str, int]
    artifacts: dict[str, list[str]]
    perf_stat: dict[str, NumericSummary]


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
    samples: list[SampleResult]
    summaries: list[ExperimentSummary]
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


def template_key(text: str) -> str:
    """@brief 将名称转换为模板变量安全片段。 / Convert a name to a template-safe fragment.

    @param text 原始名称。 / Raw name.
    @return 只含字母数字和下划线的名称。 / Name containing only alphanumerics and underscores.
    """
    key = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip()).strip("_")
    return key or "unnamed"


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


def executable_path_for_build(repo_root: Path, build_name: str) -> Path:
    """@brief 返回构建 preset 对应的 mpilab 路径。 / Return the mpilab path for a build preset.

    @param repo_root 仓库根目录。 / Repository root.
    @param build_name CMake 构建 preset 名称。 / CMake build preset name.
    @return 可执行文件路径。 / Executable path.
    """
    return (repo_root / "build" / build_name / "src" / "mpilab" / "mpilab").resolve()


def sample_output_path(repo_root: Path, sample: SampleSpec) -> Path:
    """@brief 解析样本输出路径。 / Resolve a sample output path.

    @param repo_root 仓库根目录。 / Repository root.
    @param sample 样本声明。 / Sample declaration.
    @return 样本输出路径。 / Sample output path.
    """
    if sample.output is not None:
        return (repo_root / sample.output).resolve() if not sample.output.is_absolute() else sample.output
    return (repo_root / "experiments" / "samples" / f"{slugify(sample.name)}.txt").resolve()


def build_matrixgen_command(repo_root: Path, sample: SampleSpec, context: dict[str, str]) -> tuple[list[str], Path]:
    """@brief 构造 matrixgen 命令。 / Build a matrixgen command.

    @param repo_root 仓库根目录。 / Repository root.
    @param sample 样本声明。 / Sample declaration.
    @param context 模板上下文。 / Template context.
    @return 命令和输出路径。 / Command and output path.
    """
    output_path = sample_output_path(repo_root, sample)
    sample_context = {**context, "sample_name": sample.name, "sample_path": str(output_path)}
    arguments = expand_arguments(sample.arguments, sample_context)
    command = [
        sys.executable,
        "-m",
        "generator.main",
        sample.mode.value,
        "--output",
        str(output_path),
        *arguments,
    ]
    return command, output_path


def sample_template_context(samples: list[SampleSpec], repo_root: Path) -> dict[str, str]:
    """@brief 构造样本模板上下文。 / Build sample template context.

    @param samples 样本声明列表。 / Sample declaration list.
    @param repo_root 仓库根目录。 / Repository root.
    @return 模板上下文。 / Template context.
    """
    context: dict[str, str] = {}
    if samples:
        context["sample_path"] = str(sample_output_path(repo_root, samples[0]))
    for sample in samples:
        context[f"sample_{template_key(sample.name)}"] = str(sample_output_path(repo_root, sample))
    return context


def generate_samples(
    options: RunOptions,
    instances: list[tuple[Path, ExperimentInstance]],
) -> list[SampleResult]:
    """@brief 生成或复用 instance 级样本。 / Generate or reuse instance-level samples.

    @param options bench 运行选项。 / Bench run options.
    @param instances 实例列表。 / Instance list.
    @return 样本结果列表。 / Sample result list.
    """
    results: list[SampleResult] = []
    generated_outputs: set[Path] = set()
    for instance_path, instance in instances:
        instance_name = instance.name or instance_path.stem
        base_context = {
            "repo_root": str(options.repo_root),
            "instances_dir": str(options.instances_dir),
            "results_dir": str(options.results_dir),
            "instance": slugify(instance_name),
            "instance_file": str(instance_path),
        }
        for sample in instance.samples:
            command, output_path = build_matrixgen_command(options.repo_root, sample, base_context)
            started = time.perf_counter()
            status = "generated"
            return_code: int | None = 0
            error: str | None = None
            if sample.reuse_existing and (output_path.exists() or output_path in generated_outputs):
                status = "reused"
            elif options.dry_run:
                status = "dry-run"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                return_code, error = run_process(
                    command,
                    options.repo_root,
                    os.environ.copy(),
                    None,
                    output_path.with_suffix(output_path.suffix + ".matrixgen.stdout.txt"),
                    output_path.with_suffix(output_path.suffix + ".matrixgen.stderr.txt"),
                )
                if return_code != 0:
                    status = "failed"
            if status == "generated":
                generated_outputs.add(output_path)
            results.append(
                SampleResult(
                    instance=str(instance_path),
                    sample=sample.name,
                    output_path=str(output_path),
                    status=status,
                    command=command,
                    duration_seconds=time.perf_counter() - started,
                    return_code=return_code,
                    error=error,
                )
            )
            if status == "failed":
                raise RuntimeError(f"matrixgen failed for sample {sample.name}: {error}")
    return results


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
    executable = executable_path_for_build(repo_root, job.build)
    arguments = expand_arguments(job.arguments, context) + expand_arguments(experiment.arguments, context)
    mpi = job.config.mpi
    if mpi is not None and mpi.enable_flag and "--mpi" not in arguments:
        arguments.append("--mpi")
    command = [str(executable), *arguments]
    if mpi is not None and mpi.processes > 1:
        command = [mpi.launcher, "-np", str(mpi.processes), *mpi.extra_args, *command]
    return command


def expanded_tool_arguments(experiment: ExperimentSpec, context: dict[str, str]) -> list[str]:
    """@brief 展开剖析工具参数。 / Expand profiler tool arguments.

    @param experiment 实验规格。 / Experiment specification.
    @param context 模板上下文。 / Template context.
    @return 展开后的工具参数。 / Expanded tool arguments.
    """
    return expand_arguments(experiment.tool_arguments, context)


def build_native_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造原生命令。 / Build the native command."""
    del experiment, run_dir, context
    return base_command, {}, None


def build_gnu_time_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 GNU time 命令。 / Build a GNU time command."""
    out_path = run_dir / "gnu-time.txt"
    command = ["/usr/bin/time", "-v", "-o", str(out_path), *expanded_tool_arguments(experiment, context), *base_command]
    return command, {"gnu_time": str(out_path)}, "time"


def build_hyperfine_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 hyperfine 命令。 / Build a hyperfine command."""
    json_path = run_dir / "hyperfine.json"
    command_string = shlex.join(base_command)
    command = [
        "hyperfine",
        "--export-json",
        str(json_path),
        *expanded_tool_arguments(experiment, context),
        command_string,
    ]
    return command, {"hyperfine": str(json_path)}, "hyperfine"


def build_perf_stat_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 perf stat 命令。 / Build a perf stat command."""
    stat_path = run_dir / "perf-stat.csv"
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
        *expanded_tool_arguments(experiment, context),
        "--",
        *base_command,
    ]
    return command, {"perf_stat": str(stat_path)}, "perf"


def build_perf_record_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 perf record 命令。 / Build a perf record command."""
    data_path = run_dir / "perf.data"
    command = ["perf", "record", "-g", "-o", str(data_path)]
    if experiment.events:
        command.extend(["-e", ",".join(experiment.events)])
    command = [*command, *expanded_tool_arguments(experiment, context), "--", *base_command]
    return command, {"perf_data": str(data_path)}, "perf"


def build_valgrind_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
    tool_name: str,
    artifact_name: str,
    output_option: str,
) -> ToolCommand:
    """@brief 构造 Valgrind 系列命令。 / Build a Valgrind-family command."""
    out_path = run_dir / f"{artifact_name}.out"
    command = [
        "valgrind",
        f"--tool={tool_name}",
        f"{output_option}={out_path}",
        *expanded_tool_arguments(experiment, context),
        *base_command,
    ]
    return command, {artifact_name.replace("-", "_"): str(out_path)}, "valgrind"


def build_memcheck_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 Memcheck 命令。 / Build a Memcheck command."""
    return build_valgrind_tool(base_command, experiment, run_dir, context, "memcheck", "memcheck", "--log-file")


def build_callgrind_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 Callgrind 命令。 / Build a Callgrind command."""
    return build_valgrind_tool(
        base_command,
        experiment,
        run_dir,
        context,
        "callgrind",
        "callgrind",
        "--callgrind-out-file",
    )


def build_cachegrind_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 Cachegrind 命令。 / Build a Cachegrind command."""
    return build_valgrind_tool(
        base_command,
        experiment,
        run_dir,
        context,
        "cachegrind",
        "cachegrind",
        "--cachegrind-out-file",
    )


def build_massif_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 Massif 命令。 / Build a Massif command."""
    return build_valgrind_tool(base_command, experiment, run_dir, context, "massif", "massif", "--massif-out-file")


def build_dhat_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 DHAT 命令。 / Build a DHAT command."""
    return build_valgrind_tool(base_command, experiment, run_dir, context, "dhat", "dhat", "--log-file")


def build_heaptrack_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 heaptrack 命令。 / Build a heaptrack command."""
    out_path = run_dir / "heaptrack.gz"
    command = ["heaptrack", "-o", str(out_path), *expanded_tool_arguments(experiment, context), *base_command]
    return command, {"heaptrack": str(out_path)}, "heaptrack"


def build_strace_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 strace 命令。 / Build a strace command."""
    out_path = run_dir / "strace.txt"
    command = ["strace", "-f", "-o", str(out_path), *expanded_tool_arguments(experiment, context), *base_command]
    return command, {"strace": str(out_path)}, "strace"


def build_likwid_perfctr_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 LIKWID perfctr 命令。 / Build a LIKWID perfctr command."""
    out_path = run_dir / "likwid-perfctr.txt"
    command = ["likwid-perfctr", "-o", str(out_path), *expanded_tool_arguments(experiment, context), *base_command]
    return command, {"likwid_perfctr": str(out_path)}, "likwid-perfctr"


def build_mpip_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 mpiP 命令。 / Build an mpiP command."""
    out_dir = run_dir / "mpip"
    command = ["mpip", *expanded_tool_arguments(experiment, context), *base_command]
    return command, {"mpip": str(out_dir)}, "mpip"


def build_scorep_tool(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造 Score-P 命令。 / Build a Score-P command."""
    out_dir = run_dir / "scorep"
    command = ["scorep", *expanded_tool_arguments(experiment, context), *base_command]
    return command, {"scorep": str(out_dir)}, "scorep"


TOOL_BUILDERS: dict[ExperimentTool, ToolBuilder] = {
    ExperimentTool.NATIVE: build_native_tool,
    ExperimentTool.GNU_TIME: build_gnu_time_tool,
    ExperimentTool.HYPERFINE: build_hyperfine_tool,
    ExperimentTool.PERF_STAT: build_perf_stat_tool,
    ExperimentTool.PERF_RECORD: build_perf_record_tool,
    ExperimentTool.VALGRIND_MEMCHECK: build_memcheck_tool,
    ExperimentTool.VALGRIND_CALLGRIND: build_callgrind_tool,
    ExperimentTool.VALGRIND_CACHEGRIND: build_cachegrind_tool,
    ExperimentTool.VALGRIND_MASSIF: build_massif_tool,
    ExperimentTool.VALGRIND_DHAT: build_dhat_tool,
    ExperimentTool.HEAPTRACK: build_heaptrack_tool,
    ExperimentTool.STRACE: build_strace_tool,
    ExperimentTool.LIKWID_PERFCTR: build_likwid_perfctr_tool,
    ExperimentTool.MPIP: build_mpip_tool,
    ExperimentTool.SCOREP: build_scorep_tool,
}


def build_tool_command(
    base_command: list[str],
    experiment: ExperimentSpec,
    run_dir: Path,
    context: dict[str, str],
) -> ToolCommand:
    """@brief 构造带剖析工具的命令。 / Build a command wrapped by a profiling tool.

    @param base_command mpilab 基础命令。 / Base mpilab command.
    @param experiment 实验规格。 / Experiment specification.
    @param run_dir 单次运行目录。 / Single-run directory.
    @param context 模板上下文。 / Template context.
    @return 命令、产物和需要检查的工具名。 / Command, artifacts, and required tool name.
    """
    return TOOL_BUILDERS[experiment.tool](base_command, experiment, run_dir, context)


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
    context.update(sample_template_context(instance.samples, options.repo_root))
    base_command = build_base_command(options.repo_root, job, experiment, context)
    command, artifacts, required_tool = build_tool_command(base_command, experiment, run_dir, context)
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
    if options.dry_run:
        return_code = 0
        error = None
        status = "dry-run"
        stdout_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
    elif required_tool is not None and shutil.which(required_tool) is None:
        return_code = None
        error = f"required tool not found: {required_tool}"
        status = "skipped"
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


def ensure_build_presets(repo_root: Path, build_names: list[str], dry_run: bool) -> None:
    """@brief 确保 CMake preset 已构建。 / Ensure CMake presets have been built.

    @param repo_root 仓库根目录。 / Repository root.
    @param build_names 构建 preset 名称。 / Build preset names.
    @param dry_run 是否只预演。 / Whether this is a dry run.
    """
    for build_name in build_names:
        executable = executable_path_for_build(repo_root, build_name)
        if executable.exists():
            continue
        commands = [
            ["cmake", "--preset", build_name],
            ["cmake", "--build", "--preset", build_name],
        ]
        for command in commands:
            if dry_run:
                continue
            subprocess.run(command, cwd=repo_root, check=True)


def summarize_numbers(values: list[float]) -> NumericSummary:
    """@brief 汇总数值序列。 / Summarize a numeric series.

    @param values 数值列表。 / Numeric values.
    @return 数值摘要。 / Numeric summary.
    """
    if not values:
        return NumericSummary(count=0, minimum=None, maximum=None, mean=None, stdev=None)
    mean = sum(values) / len(values)
    if len(values) == 1:
        stdev = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        stdev = math.sqrt(variance)
    return NumericSummary(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=mean,
        stdev=stdev,
    )


def parse_perf_stat(path: Path) -> dict[str, float]:
    """@brief 解析 perf stat CSV 输出。 / Parse perf stat CSV output.

    @param path perf-stat.csv 路径。 / perf-stat.csv path.
    @return event 到数值的映射。 / Mapping from event to numeric value.
    """
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < PERF_STAT_MIN_COLUMNS:
            continue
        raw_value = parts[0].replace(",", "")
        event = parts[2]
        if not raw_value or raw_value == "<not counted>" or not event:
            continue
        try:
            metrics[event] = float(raw_value)
        except ValueError:
            continue
    return metrics


def summarize_perf_stats(results: list[ExperimentResult]) -> dict[str, NumericSummary]:
    """@brief 汇总 perf stat 指标。 / Summarize perf stat metrics.

    @param results 同组实验结果。 / Grouped experiment results.
    @return perf event 摘要。 / Perf event summaries.
    """
    values_by_event: dict[str, list[float]] = {}
    for result in results:
        stat_path = result.artifacts.get("perf_stat")
        if stat_path is None:
            continue
        for event, value in parse_perf_stat(Path(stat_path)).items():
            values_by_event.setdefault(event, []).append(value)
    return {event: summarize_numbers(values) for event, values in sorted(values_by_event.items())}


def summarize_results(results: list[ExperimentResult]) -> list[ExperimentSummary]:
    """@brief 按实验聚合 repeat 结果。 / Aggregate repeat results by experiment.

    @param results 原始实验结果。 / Raw experiment results.
    @return 聚合摘要列表。 / Aggregated summary list.
    """
    grouped: dict[tuple[str, str, str, str], list[ExperimentResult]] = {}
    for result in results:
        key = (result.instance, result.job, result.experiment, result.tool)
        grouped.setdefault(key, []).append(result)

    summaries: list[ExperimentSummary] = []
    for (instance, job, experiment, tool), group in sorted(grouped.items()):
        status_counts: dict[str, int] = {}
        return_codes: dict[str, int] = {}
        artifacts: dict[str, list[str]] = {}
        for result in group:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            return_code_key = "none" if result.return_code is None else str(result.return_code)
            return_codes[return_code_key] = return_codes.get(return_code_key, 0) + 1
            for artifact_name, artifact_path in result.artifacts.items():
                artifacts.setdefault(artifact_name, []).append(artifact_path)
        summaries.append(
            ExperimentSummary(
                instance=instance,
                job=job,
                experiment=experiment,
                tool=tool,
                repeat_count=len(group),
                status_counts=dict(sorted(status_counts.items())),
                duration_seconds=summarize_numbers([result.duration_seconds for result in group]),
                return_codes=dict(sorted(return_codes.items())),
                artifacts=dict(sorted(artifacts.items())),
                perf_stat=summarize_perf_stats(group),
            )
        )
    return summaries


def write_report(report: BenchReport, path: Path) -> None:
    """@brief 写出 bench 总报告。 / Write the aggregate bench report.

    @param report 总报告。 / Aggregate report.
    @param path 输出路径。 / Output path.
    """
    payload: dict[str, Any] = asdict(report)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_summary(summaries: list[ExperimentSummary], path: Path) -> None:
    """@brief 写出聚合摘要。 / Write aggregated summaries.

    @param summaries 聚合摘要。 / Aggregated summaries.
    @param path 输出路径。 / Output path.
    """
    payload = [asdict(summary) for summary in summaries]
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
    sample_results = generate_samples(effective_options, instances)
    build_names = sorted({job.build for _, instance in instances for job in instance.jobs})
    if options.build and build_names:
        ensure_build_presets(options.repo_root, build_names, options.dry_run)

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
    summaries = summarize_results(results)
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
        samples=sample_results,
        summaries=summaries,
        results=results,
    )
    write_report(report, effective_options.results_dir / "bench-report.json")
    write_summary(summaries, effective_options.results_dir / "bench-summary.json")
    return report
