"""矩阵数值分布生成。 / Matrix numeric distribution generation."""

from __future__ import annotations

import numpy as np

from .specs import bounded_rank
from .types import Distribution, FloatMatrix, MatrixSpec


def random_orthonormal(rng: np.random.Generator, rows: int, cols: int) -> FloatMatrix:
    """@brief 生成随机正交列矩阵。 / Generate a random matrix with orthonormal columns.

    @param rng 随机数生成器。 / Random number generator.
    @param rows 行数。 / Row count.
    @param cols 列数。 / Column count.
    @return 正交列矩阵。 / Matrix with orthonormal columns.
    """
    sample = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(sample, mode="reduced")
    return np.asarray(q, dtype=np.float64)


def generate_matrix(
    spec: MatrixSpec,
    low: float,
    high: float,
    mean: float,
    std: float,
    noise: float,
) -> FloatMatrix:
    """@brief 按规格生成矩阵。 / Generate a matrix from a specification.

    @param spec 矩阵规格。 / Matrix specification.
    @param low 均匀或整数分布下界。 / Lower bound for uniform or integer distributions.
    @param high 均匀或整数分布上界。 / Upper bound for uniform or integer distributions.
    @param mean 正态或对数正态均值参数。 / Mean parameter for normal or log-normal distributions.
    @param std 正态或对数正态标准差参数。 / Standard-deviation parameter for normal or log-normal distributions.
    @param noise 低秩矩阵加性噪声标准差。 / Additive-noise standard deviation for low-rank matrices.
    @return 生成出的矩阵。 / Generated matrix.
    """
    rng = np.random.default_rng(spec.seed)
    shape = (spec.rows, spec.cols)
    distribution = Distribution(spec.distribution)

    if distribution is Distribution.UNIFORM:
        matrix = rng.uniform(low, high, size=shape)
    elif distribution is Distribution.NORMAL:
        matrix = rng.normal(mean, std, size=shape)
    elif distribution is Distribution.LOGNORMAL:
        matrix = rng.lognormal(mean, std, size=shape)
    elif distribution is Distribution.INTEGER:
        matrix = rng.integers(int(low), int(high) + 1, size=shape).astype(np.float64)
    elif distribution is Distribution.RADEMACHER:
        matrix = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=shape)
    elif distribution is Distribution.SPARSE:
        matrix = sparse_matrix(rng, shape, spec.density, mean, std)
    elif distribution is Distribution.LOW_RANK:
        matrix = low_rank_matrix(rng, spec, shape, noise)
    elif distribution is Distribution.ILL_CONDITIONED:
        matrix = ill_conditioned_matrix(rng, spec)
    elif distribution is Distribution.DIAGONAL:
        matrix = diagonal_matrix(rng, shape, low, high)
    elif distribution is Distribution.BANDED:
        matrix = banded_matrix(rng, spec, shape, mean, std)
    elif distribution is Distribution.HILBERT:
        matrix = hilbert_matrix(shape)
    elif distribution is Distribution.IDENTITY:
        matrix = identity_matrix(shape)
    elif distribution is Distribution.CAUCHY:
        matrix = np.clip(rng.standard_cauchy(size=shape), -1.0e6, 1.0e6)
    elif distribution is Distribution.ZERO:
        matrix = np.zeros(shape, dtype=np.float64)
    else:
        raise ValueError(f"unsupported distribution: {distribution}")

    return np.asarray((matrix * spec.scale) + spec.offset, dtype=np.float64)


def sparse_matrix(
    rng: np.random.Generator,
    shape: tuple[int, int],
    density: float,
    mean: float,
    std: float,
) -> FloatMatrix:
    """@brief 生成稀疏随机矩阵。 / Generate a sparse random matrix.

    @param rng 随机数生成器。 / Random number generator.
    @param shape 矩阵形状。 / Matrix shape.
    @param density 非零概率。 / Non-zero probability.
    @param mean 正态分布均值。 / Normal mean.
    @param std 正态分布标准差。 / Normal standard deviation.
    @return 稀疏矩阵。 / Sparse matrix.
    """
    matrix = rng.normal(mean, std, size=shape)
    mask = rng.random(size=shape) < density
    return np.asarray(np.where(mask, matrix, 0.0), dtype=np.float64)


def low_rank_matrix(
    rng: np.random.Generator,
    spec: MatrixSpec,
    shape: tuple[int, int],
    noise: float,
) -> FloatMatrix:
    """@brief 生成低秩矩阵。 / Generate a low-rank matrix.

    @param rng 随机数生成器。 / Random number generator.
    @param spec 矩阵规格。 / Matrix specification.
    @param shape 矩阵形状。 / Matrix shape.
    @param noise 加性噪声标准差。 / Additive-noise standard deviation.
    @return 低秩矩阵。 / Low-rank matrix.
    """
    rank = bounded_rank(spec.rows, spec.cols, spec.rank)
    left = rng.normal(size=(spec.rows, rank))
    right = rng.normal(size=(rank, spec.cols))
    matrix = left @ right
    if noise > 0.0:
        matrix = matrix + rng.normal(0.0, noise, size=shape)
    return np.asarray(matrix, dtype=np.float64)


def ill_conditioned_matrix(rng: np.random.Generator, spec: MatrixSpec) -> FloatMatrix:
    """@brief 生成指定条件数的病态矩阵。 / Generate an ill-conditioned matrix.

    @param rng 随机数生成器。 / Random number generator.
    @param spec 矩阵规格。 / Matrix specification.
    @return 病态矩阵。 / Ill-conditioned matrix.
    """
    rank = min(spec.rows, spec.cols)
    condition = spec.condition if spec.condition is not None else 1.0e8
    if condition < 1.0:
        raise ValueError("condition must be at least 1")
    left = random_orthonormal(rng, spec.rows, rank)
    right = random_orthonormal(rng, spec.cols, rank)
    singular_values = np.geomspace(condition, 1.0, num=rank, dtype=np.float64)
    return np.asarray((left * singular_values) @ right.T, dtype=np.float64)


def diagonal_matrix(
    rng: np.random.Generator,
    shape: tuple[int, int],
    low: float,
    high: float,
) -> FloatMatrix:
    """@brief 生成对角矩阵。 / Generate a diagonal matrix.

    @param rng 随机数生成器。 / Random number generator.
    @param shape 矩阵形状。 / Matrix shape.
    @param low 对角线下界。 / Diagonal lower bound.
    @param high 对角线上界。 / Diagonal upper bound.
    @return 对角矩阵。 / Diagonal matrix.
    """
    matrix = np.zeros(shape, dtype=np.float64)
    diagonal = rng.uniform(low, high, size=min(shape))
    np.fill_diagonal(matrix, diagonal)
    return matrix


def banded_matrix(
    rng: np.random.Generator,
    spec: MatrixSpec,
    shape: tuple[int, int],
    mean: float,
    std: float,
) -> FloatMatrix:
    """@brief 生成带状矩阵。 / Generate a banded matrix.

    @param rng 随机数生成器。 / Random number generator.
    @param spec 矩阵规格。 / Matrix specification.
    @param shape 矩阵形状。 / Matrix shape.
    @param mean 正态分布均值。 / Normal mean.
    @param std 正态分布标准差。 / Normal standard deviation.
    @return 带状矩阵。 / Banded matrix.
    """
    bandwidth = spec.bandwidth if spec.bandwidth is not None else 2
    if bandwidth < 0:
        raise ValueError("bandwidth must be non-negative")
    values = rng.normal(mean, std, size=shape)
    row_index, col_index = np.indices(shape)
    return np.asarray(np.where(np.abs(row_index - col_index) <= bandwidth, values, 0.0), dtype=np.float64)


def hilbert_matrix(shape: tuple[int, int]) -> FloatMatrix:
    """@brief 生成 Hilbert 矩阵。 / Generate a Hilbert matrix.

    @param shape 矩阵形状。 / Matrix shape.
    @return Hilbert 矩阵。 / Hilbert matrix.
    """
    row_index, col_index = np.indices(shape)
    return np.asarray(1.0 / (row_index + col_index + 1.0), dtype=np.float64)


def identity_matrix(shape: tuple[int, int]) -> FloatMatrix:
    """@brief 生成矩形单位矩阵。 / Generate a rectangular identity matrix.

    @param shape 矩阵形状。 / Matrix shape.
    @return 矩形单位矩阵。 / Rectangular identity matrix.
    """
    matrix = np.zeros(shape, dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix
