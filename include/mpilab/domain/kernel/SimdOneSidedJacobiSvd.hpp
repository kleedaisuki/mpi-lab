#pragma once

#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

/**
 * @file SimdOneSidedJacobiSvd.hpp
 * @brief SIMD 单边 Jacobi SVD 算子声明。 SIMD one-sided Jacobi SVD kernel declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief SIMD 单边 Jacobi SVD 算子，按构建体系选择向量化内核。 SIMD one-sided Jacobi SVD kernel that selects vectorized kernels by build architecture.
     *
     * @note x86/x86_64 且启用 SSE2 时使用 SSE2 intrinsics；其他平台回退到标量内核以保持跨平台语义。 / Uses SSE2 intrinsics on x86/x86_64 when SSE2 is enabled; other platforms fall back to scalar kernels for portable semantics.
     */
    struct SimdOneSidedJacobiSvd
    {
        /**
         * @brief 对任意 MatrixLike 矩阵计算 thin SVD。 Compute the thin SVD of any MatrixLike matrix.
         *
         * @tparam Matrix 输入矩阵类型。 / Input matrix type.
         * @param matrix 输入矩阵 A，尺寸为 m-by-n。 / Input matrix A with size m-by-n.
         * @param options 迭代配置。 / Iteration options.
         * @return SVD 结果。 / SVD result.
         * @note 当前 SIMD 版本要求 m >= n。 / The current SIMD version requires m >= n.
         */
        template <MatrixLike Matrix>
        [[nodiscard]] auto operator()(const Matrix& matrix, const OneSidedJacobiSvdOptions& options = {}) const -> OneSidedJacobiSvdResult<std::remove_cvref_t<Matrix>>
        {
            return detail::run_one_sided_jacobi_svd(
                matrix,
                options,
                detail::JacobiSweepStrategy::cyclic,
                detail::JacobiExecutionStrategy::serial,
                detail::JacobiComputeKernel::simd);
        }
    };

} // namespace mpilab::domain
