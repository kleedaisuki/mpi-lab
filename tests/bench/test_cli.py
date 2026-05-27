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
                "jobs": [
                    {
                        "name": "echo-native",
                        "build": "relwithdebinfo",
                        "executable": "/bin/echo",
                        "arguments": ["--input", "{instance_file}", "--output", "{run_dir}/out.txt"],
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
                        "name": "echo-shortcut",
                        "executable": "/bin/echo",
                        "experiments": ["native"],
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
    report = json.loads(report_paths[0].read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert len(report["results"]) == 3
    assert {item["status"] for item in report["results"]} == {"dry-run"}
    assert all(item["stdout_path"] for item in report["results"])
    assert any("{run_dir}" not in " ".join(item["command"]) for item in report["results"])
