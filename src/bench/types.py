"""实验编排共享类型。 / Shared experiment orchestration types."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ExperimentTool(str, Enum):
    """@brief 实验工具类型。 / Experiment tool type."""

    NATIVE = "native"
    PERF_STAT = "perf-stat"
    PERF_RECORD = "perf-record"
    CALLGRIND = "callgrind"
    CACHEGRIND = "cachegrind"


class MpiConfig(BaseModel):
    """@brief MPI 启动配置。 / MPI launch configuration."""

    model_config = ConfigDict(extra="forbid")

    processes: int = Field(default=1, ge=1)
    launcher: str = "mpirun"
    enable_flag: bool = True
    extra_args: list[str] = Field(default_factory=list)


class JobConfig(BaseModel):
    """@brief 单个 job 的运行配置。 / Runtime configuration for one job."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: float | None = Field(default=None, gt=0.0)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None
    mpi: MpiConfig | None = None
    continue_on_failure: bool = False


class ExperimentSpec(BaseModel):
    """@brief 单次实验规格。 / Single experiment specification."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    tool: ExperimentTool = ExperimentTool.NATIVE
    repeat: int = Field(default=1, ge=1)
    events: list[str] = Field(default_factory=list)
    arguments: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def accept_tool_string(cls, value: Any) -> Any:
        """@brief 允许用字符串简写实验。 / Allow string shorthand for experiments.

        @param value 原始模型输入。 / Raw model input.
        @return 归一化后的模型输入。 / Normalized model input.
        """
        if isinstance(value, str):
            return {"tool": value}
        return value


class BenchJob(BaseModel):
    """@brief 可并发执行的 benchmark job。 / Concurrent benchmark job."""

    model_config = ConfigDict(extra="forbid")

    name: str
    build: str = "relwithdebinfo"
    executable: Path | None = None
    arguments: list[str] = Field(default_factory=list)
    experiments: list[ExperimentSpec] = Field(default_factory=lambda: [ExperimentSpec()])
    config: JobConfig = Field(default_factory=JobConfig)

    @field_validator("name")
    @classmethod
    def require_name(cls, value: str) -> str:
        """@brief 验证 job 名称非空。 / Validate that the job name is non-empty.

        @param value job 名称。 / Job name.
        @return 去除首尾空白后的名称。 / Stripped name.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("job name must not be empty")
        return stripped

    @model_validator(mode="after")
    def require_experiments(self) -> BenchJob:
        """@brief 验证至少包含一个实验。 / Validate at least one experiment is present.

        @return 当前 job。 / Current job.
        """
        if not self.experiments:
            raise ValueError("job must contain at least one experiment")
        return self


class ExperimentInstance(BaseModel):
    """@brief 单个实例 JSON 文件。 / Single instance JSON file."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    name: str | None = None
    jobs: list[BenchJob]

    @model_validator(mode="after")
    def require_jobs(self) -> ExperimentInstance:
        """@brief 验证实例至少包含一个 job。 / Validate that the instance has at least one job.

        @return 当前实例。 / Current instance.
        """
        if not self.jobs:
            raise ValueError("instance must contain at least one job")
        return self
