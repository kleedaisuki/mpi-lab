#pragma once

#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

/**
 * @file AdvancedOneSidedJacobiSvd.hpp
 * @brief 改进串行单边 Jacobi SVD 算子声明。 Advanced serial one-sided Jacobi SVD kernel declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief 改进串行单边 Jacobi SVD 算子，按归一化列相关性动态排序旋转。 Advanced serial one-sided Jacobi SVD kernel that dynamically orders rotations by normalized column correlation.
     *
     * @note 该版本仍是单进程串行实现，但每轮优先处理最不正交的列对，借鉴 dynamic ordering（动态排序）思想。 / This version is still single-process serial code, but each sweep prioritizes the least orthogonal column pairs, following the dynamic-ordering idea.
     */
    struct AdvancedOneSidedJacobiSvd
    {
        /**
         * @brief 对任意 MatrixLike 矩阵计算 thin SVD。 Compute the thin SVD of any MatrixLike matrix.
         *
         * @tparam Matrix 输入矩阵类型。 / Input matrix type.
         * @param matrix 输入矩阵 A，尺寸为 m-by-n。 / Input matrix A with size m-by-n.
         * @param options 迭代配置。 / Iteration options.
         * @return SVD 结果。 / SVD result.
         * @note 当前 advanced 版本要求 m >= n。 / The current advanced version requires m >= n.
         */
        template <MatrixLike Matrix>
        [[nodiscard]] auto operator()(const Matrix& matrix, const OneSidedJacobiSvdOptions& options = {}) const -> OneSidedJacobiSvdResult<std::remove_cvref_t<Matrix>>
        {
            return detail::run_one_sided_jacobi_svd(matrix, options, detail::JacobiSweepStrategy::dynamic_ordered);
        }
    };

} // namespace mpilab::domain
