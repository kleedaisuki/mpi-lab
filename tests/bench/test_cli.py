"""bench CLI 测试。 / Bench CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from bench.main import app

runner = CliRunner()


def write_instance(path: Path) -> None:
    """@brief 写出测试实例。 / Write a test instance.

    @param path 实例 JSON 路径。 / Instance JSON path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "unit-suite",
                "samples": [
                    {
                        "name": "tall normal",
                        "mode": "generate",
                        "arguments": ["--shape", "8x3", "--distribution", "normal"],
                    },
                    {
                        "name": "structured_suite",
                        "mode": "suite",
                    },
                ],
                "jobs": [
                    {
                        "name": "row-major-native",
                        "build": "relwithdebinfo",
                        "arguments": ["--input", "{sample_tall_normal}", "--output", "{run_dir}/out.txt"],
                        "experiments": [
                            {
                                "name": "plain",
                                "tool": "native",
                                "repeat": 2,
                                "arguments": ["--kernel", "naive"],
                            }
                        ],
                    },
                    {
                        "name": "default-build-shortcut",
                        "arguments": ["--input", "{sample_structured_suite}"],
                        "experiments": [
                            {
                                "name": "perf-dry-run",
                                "tool": "perf-stat",
                                "tool_arguments": ["--all-user"],
                                "events": ["cycles"],
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_validate_accepts_instance_json(tmp_path: Path) -> None:
    """@brief 验证实例 JSON 可以被 CLI 校验。 / Verify instance JSON can be validated by the CLI."""
    instances_dir = tmp_path / "instances"
    write_instance(instances_dir / "sample.json")

    result = runner.invoke(app, ["validate", "--instances", str(instances_dir)])

    assert result.exit_code == 0, result.output
    assert "Bench instances" in result.output
    assert "2" in result.output


def test_run_dry_run_writes_report_and_commands(tmp_path: Path) -> None:
    """@brief 验证 dry-run 写出报告和命令。 / Verify dry-run writes report and commands."""
    instances_dir = tmp_path / "instances"
    results_dir = tmp_path / "results"
    write_instance(instances_dir / "sample.json")

    result = runner.invoke(
        app,
        [
            "run",
            "--instances",
            str(instances_dir),
            "--results",
            str(results_dir),
            "--skip-build",
            "--dry-run",
            "--jobs",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    report_paths = list(results_dir.glob("*/bench-report.json"))
    assert len(report_paths) == 1
    summary_path = report_paths[0].with_name("bench-summary.json")
    assert summary_path.exists()
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert len(report["samples"]) == 2
    assert {item["status"] for item in report["samples"]} == {"dry-run"}
    assert len(report["results"]) == 3
    assert len(report["summaries"]) == 2
    assert len(summaries) == 2
    assert any(item["repeat_count"] == 2 for item in summaries)
    assert all("duration_seconds" in item for item in summaries)
    assert {item["status"] for item in report["results"]} == {"dry-run"}
    assert all(item["stdout_path"] for item in report["results"])
    command_text = "\n".join(" ".join(item["command"]) for item in report["results"])
    assert "{run_dir}" not in command_text
    assert "{sample_" not in command_text
    assert "experiments/samples/tall-normal.txt" in command_text
    assert "--all-user" in command_text


def test_help_documents_instance_contract() -> None:
    """@brief 验证 help 描述实例契约。 / Verify help documents the instance contract."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Instance JSON shape" in result.output
    assert "Supported tools" in result.output
    assert "Samples" in result.output
    assert "Template variables" in result.output
    assert "perf-stat" in result.output
    assert "valgrind-callgrind" in result.output
    assert "scorep" in result.output
    assert "heaptrack" in result.output

    run_result = runner.invoke(app, ["run", "--help"])
    assert run_result.exit_code == 0, run_result.output
    assert "bench-summary.json" in run_result.output


def test_rejects_user_supplied_executable(tmp_path: Path) -> None:
    """@brief 验证用户不能配置 executable。 / Verify users cannot configure executable."""
    instances_dir = tmp_path / "instances"
    path = instances_dir / "bad.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "jobs": [
                    {
                        "name": "leaky-build-layout",
                        "build": "relwithdebinfo",
                        "executable": "/bin/echo",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["validate", "--instances", str(instances_dir)])

    assert result.exit_code != 0
    assert "executable" in result.output


def test_reuses_existing_samples_across_instances(tmp_path: Path) -> None:
    """@brief 验证跨 instance 复用样本。 / Verify samples are reused across instances."""
    instances_dir = tmp_path / "instances"
    results_dir = tmp_path / "results"
    sample_path = tmp_path / "shared" / "sample.txt"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text("1 0\n0 1\n", encoding="utf-8")

    for index in range(2):
        path = instances_dir / f"instance-{index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "name": f"suite-{index}",
                    "samples": [
                        {
                            "name": "shared",
                            "output": str(sample_path),
                            "arguments": ["--shape", "2x2", "--distribution", "identity"],
                        }
                    ],
                    "jobs": [
                        {
                            "name": "uses-shared",
                            "arguments": ["--input", "{sample_shared}"],
                            "experiments": ["native"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    result = runner.invoke(
        app,
        [
            "run",
            "--instances",
            str(instances_dir),
            "--results",
            str(results_dir),
            "--skip-build",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    report_paths = list(results_dir.glob("*/bench-report.json"))
    assert len(report_paths) == 1
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    assert len(report["samples"]) == 2
    assert {item["status"] for item in report["samples"]} == {"reused"}
    assert all(item["output_path"] == str(sample_path) for item in report["samples"])
