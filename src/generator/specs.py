"""矩阵样本规格构造。 / Matrix sample specification builders."""

from __future__ import annotations

from .types import Distribution, MatrixSpec


def parse_shape(text: str) -> tuple[int, int]:
    """@brief 解析 ROWSxCOLS 形状。 / Parse a ROWSxCOLS shape.

    @param text 形状文本。 / Shape text.
    @return 行数和列数。 / Row and column counts.
    """
    normalized = text.lower().replace(",", "x").replace("*", "x")
    parts = normalized.split("x")
    if len(parts) != 2:
        raise ValueError("shape must look like ROWSxCOLS, for example 1024x256")
    try:
        rows, cols = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError("shape values must be positive integers") from exc
    validate_shape(rows, cols)
    return rows, cols


def validate_shape(rows: int, cols: int) -> None:
    """@brief 验证矩阵形状。 / Validate matrix shape.

    @param rows 行数。 / Row count.
    @param cols 列数。 / Column count.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")


def bounded_rank(rows: int, cols: int, requested_rank: int | None) -> int:
    """@brief 返回合法低秩矩阵秩。 / Return a valid low-rank matrix rank.

    @param rows 行数。 / Row count.
    @param cols 列数。 / Column count.
    @param requested_rank 用户请求秩。 / User-requested rank.
    @return 截断后的秩。 / Clamped rank.
    """
    limit = min(rows, cols)
    if requested_rank is None:
        return max(1, limit // 8)
    if requested_rank <= 0:
        raise ValueError("rank must be positive")
    return min(requested_rank, limit)


def build_specs(
    shapes: list[tuple[int, int]],
    distributions: list[Distribution],
    seed: int,
    density: float,
    rank: int | None,
    condition: float | None,
    bandwidth: int | None,
    scale: float,
    offset: float,
) -> list[MatrixSpec]:
    """@brief 构造矩阵规格列表。 / Build matrix specifications.

    @param shapes 形状列表。 / Shape list.
    @param distributions 分布列表。 / Distribution list.
    @param seed 基础随机种子。 / Base random seed.
    @param density 稀疏密度。 / Sparse density.
    @param rank 低秩目标秩。 / Low-rank target rank.
    @param condition 条件数。 / Condition number.
    @param bandwidth 带状矩阵半带宽。 / Banded-matrix half-bandwidth.
    @param scale 数值缩放。 / Value scale.
    @param offset 数值偏移。 / Value offset.
    @return 矩阵规格列表。 / Matrix specification list.
    """
    if not 0.0 <= density <= 1.0:
        raise ValueError("density must be in [0, 1]")
    specs: list[MatrixSpec] = []
    for shape_index, (rows, cols) in enumerate(shapes):
        validate_shape(rows, cols)
        for distribution_index, distribution in enumerate(distributions):
            sample_seed = seed + (shape_index * 1009) + distribution_index
            specs.append(
                MatrixSpec(
                    name=f"{distribution.value}_{rows}x{cols}_{shape_index:03d}_{distribution_index:03d}",
                    rows=rows,
                    cols=cols,
                    distribution=distribution.value,
                    seed=sample_seed,
                    density=density,
                    rank=rank,
                    condition=condition,
                    bandwidth=bandwidth,
                    scale=scale,
                    offset=offset,
                )
            )
    return specs


def default_suite_shapes() -> list[tuple[int, int]]:
    """@brief 返回默认实验形状集。 / Return the default experimental shape set.

    @return 覆盖小型、方阵、宽矩阵、高矩阵和非 2 次幂形状的列表。 /
        Shapes covering small, square, wide, tall, and non-power-of-two cases.
    """
    return [
        (4, 4),
        (8, 3),
        (3, 8),
        (31, 31),
        (64, 64),
        (128, 32),
        (32, 128),
        (257, 64),
        (64, 257),
        (512, 512),
    ]


def default_suite_distributions() -> list[Distribution]:
    """@brief 返回默认实验分布集。 / Return the default experimental distribution set.

    @return 覆盖稠密、稀疏、低秩、病态和结构化矩阵的分布列表。 /
        Distributions covering dense, sparse, low-rank, ill-conditioned, and structured matrices.
    """
    return [
        Distribution.UNIFORM,
        Distribution.NORMAL,
        Distribution.SPARSE,
        Distribution.LOW_RANK,
        Distribution.ILL_CONDITIONED,
        Distribution.BANDED,
        Distribution.HILBERT,
        Distribution.RADEMACHER,
    ]
