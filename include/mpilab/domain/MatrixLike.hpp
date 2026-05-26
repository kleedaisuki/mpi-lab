#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

/**
 * @file MatrixLike.hpp
 * @brief 矩阵相似概念声明。 Matrix-like concept declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief 矩阵值类型概念：矩阵坐标映射必须返回实数。 Matrix value concept: a matrix coordinate mapping must return a real number.
     *
     * @tparam Value 待检查的返回值类型。 / Return value type to inspect.
     * @note 这里接受可转换为 double 的算术类型，避免把内存布局或具体标量存储策略绑进领域接口。 / This accepts arithmetic types convertible to double so the domain interface does not bind memory layout or concrete scalar storage policy.
     */
    template <typename Value>
    concept MatrixValue = std::is_arithmetic_v<std::remove_cvref_t<Value>> && std::convertible_to<Value, double>;

    /**
     * @brief 矩阵相似概念：矩阵可由二维范围构造，并且是从二维坐标到实数的映射。 Matrix-like concept: a matrix can be constructed from a two-dimensional extent and is a mapping from two-dimensional coordinates to a real number.
     *
     * @tparam Matrix 待检查的矩阵类型。 / Matrix type to inspect.
     * @note 构造参数表示逻辑二维范围，不表示任何内存布局；该概念故意不要求 rows()、cols()、迭代器或连续内存。 / The constructor parameters represent the logical two-dimensional extent, not any memory layout; this concept intentionally does not require rows(), cols(), iterators, or contiguous memory.
     */
    template <typename Matrix>
    concept MatrixLike = std::constructible_from<std::remove_cvref_t<Matrix>, std::size_t, std::size_t>
        && requires(const std::remove_reference_t<Matrix> &matrix, std::size_t x, std::size_t y) {
               { matrix(x, y) } -> MatrixValue;
           };

} // namespace mpilab::domain
