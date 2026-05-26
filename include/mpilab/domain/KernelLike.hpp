#pragma once

#include "mpilab/domain/MatrixLike.hpp"

#include <concepts>
#include <cstddef>
#include <type_traits>

/**
 * @file KernelLike.hpp
 * @brief 核相似概念声明。 Kernel-like concept declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief 核相似概念：核是无状态、可默认构造、可平凡复制的坐标纯函数对象。 Kernel-like concept: a kernel is a stateless, default-constructible, trivially copyable coordinate pure-function object.
     *
     * @tparam Kernel 中文：待检查的核类型。 English: Kernel type to inspect.
     * @tparam Matrix 中文：核操作的矩阵映射类型。 English: Matrix mapping type operated on by the kernel.
     * @note 中文：C++ concept 无法形式化证明无副作用（side effect）或引用透明性（referential transparency）；这里用空类型和 const 调用约束表达可由类型系统检查的“无状态纯函数”部分。 English: A C++ concept cannot formally prove absence of side effects or referential transparency; this encodes the stateless part that the type system can check through an empty type and const invocation.
     */
    template <typename Kernel, typename Matrix>
    concept KernelLike = MatrixLike<Matrix> && std::is_empty_v<std::remove_cvref_t<Kernel>> && std::default_initializable<std::remove_cvref_t<Kernel>> && std::is_trivially_copyable_v<std::remove_cvref_t<Kernel>> && requires(const std::remove_reference_t<Kernel> &kernel, const std::remove_reference_t<Matrix> &matrix, std::size_t x, std::size_t y) {
        { kernel(matrix, x, y) } -> MatrixValue;
    };

} // namespace mpilab::domain
