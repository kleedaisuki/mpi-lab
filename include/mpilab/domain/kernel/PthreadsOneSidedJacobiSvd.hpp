#pragma once

#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

/**
 * @file PthreadsOneSidedJacobiSvd.hpp
 * @brief Pthreads 单边 Jacobi SVD 算子声明。 Pthreads one-sided Jacobi SVD kernel declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief Pthreads 单边 Jacobi SVD 算子，按 phase 并行处理互不相交列对。 Pthreads one-sided Jacobi SVD kernel that processes disjoint column pairs in parallel by phase.
     *
     * @note 该实现复用 infrastructure 的全局惰性初始化线程池。 / This implementation reuses the globally lazy-initialized thread pool from infrastructure.
     */
    struct PthreadsOneSidedJacobiSvd
    {
        /**
         * @brief 对行主序矩阵计算 thin SVD。 Compute the thin SVD of a row-major matrix.
         *
         * @param matrix 输入矩阵 A，尺寸为 m-by-n。 / Input matrix A with size m-by-n.
         * @param options 迭代配置。 / Iteration options.
         * @return SVD 结果。 / SVD result.
         * @note 当前 Pthreads 版本要求 m >= n。 / The current Pthreads version requires m >= n.
         */
        [[nodiscard]] auto operator()(const RowMajorMatrix<double>& matrix, const OneSidedJacobiSvdOptions& options = {}) const -> OneSidedJacobiSvdResult;
    };

} // namespace mpilab::domain
