"""矩阵生成器共享类型。 / Shared matrix-generator types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]


class Distribution(StrEnum):
    """@brief 数值分布类型。 / Numeric distribution type."""

    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    INTEGER = "integer"
    RADEMACHER = "rademacher"
    SPARSE = "sparse"
    LOW_RANK = "low-rank"
    ILL_CONDITIONED = "ill-conditioned"
    DIAGONAL = "diagonal"
    BANDED = "banded"
    HILBERT = "hilbert"
    IDENTITY = "identity"
    CAUCHY = "cauchy"
    ZERO = "zero"


class OutputFormat(StrEnum):
    """@brief 输出格式类型。 / Output format type."""

    TEXT = "text"
    NPY = "npy"
    NPZ = "npz"
    CSV = "csv"


@dataclass(frozen=True)
class MatrixSpec:
    """@brief 单个矩阵样本规格。 / Single matrix sample specification."""

    name: str
    rows: int
    cols: int
    distribution: str
    seed: int
    density: float
    rank: int | None
    condition: float | None
    bandwidth: int | None
    scale: float
    offset: float
