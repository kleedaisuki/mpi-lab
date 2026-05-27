"""实验流水线命令入口。 / Experiment pipeline command entry point."""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    add_completion=False,
    help="MPI SVD 实验流水线。 / MPI SVD experiment pipeline.",
)
console = Console()


@app.callback(invoke_without_command=True)
def run() -> None:
    """运行默认 bench 命令。 / Run the default bench command."""
    console.print("bench pipeline is ready")


def main() -> None:
    """执行 bench 命令行入口。 / Execute the bench command-line entry point."""
    app()


if __name__ == "__main__":
    main()
