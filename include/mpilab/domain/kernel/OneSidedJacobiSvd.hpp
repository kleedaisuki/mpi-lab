#pragma once

#include "mpilab/domain/MatrixLike.hpp"
#include "mpilab/domain/matrix/DenseMatrix.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

/**
 * @file OneSidedJacobiSvd.hpp
 * @brief 单边 Jacobi SVD 算子声明。 One-sided Jacobi SVD kernel declarations.
 */

namespace mpilab::domain
{
    /**
     * @brief 单边 Jacobi SVD 配置。 One-sided Jacobi SVD configuration.
     */
    struct OneSidedJacobiSvdOptions
    {
        /**
         * @brief 最大 cyclic sweep 次数。 / Maximum cyclic sweep count.
         */
        std::size_t max_sweeps{100};

        /**
         * @brief 列正交化收敛容差。 / Column-orthogonalization convergence tolerance.
         */
        double tolerance{1.0e-12};
    };

    /**
     * @brief 单边 Jacobi SVD 结果，表示 thin SVD A = U * Sigma * V^T。 One-sided Jacobi SVD result representing the thin SVD A = U * Sigma * V^T.
     */
    template <MatrixLike Matrix>
    struct OneSidedJacobiSvdResult
    {
        /**
         * @brief 左奇异向量矩阵 U，尺寸为 m-by-n。 / Left singular vector matrix U with size m-by-n.
         */
        Matrix u;

        /**
         * @brief 奇异值 Sigma 的对角元素，按非增序排列。 / Diagonal entries of Sigma, sorted in non-increasing order.
         */
        std::vector<double> singular_values;

        /**
         * @brief 右奇异向量矩阵 V，尺寸为 n-by-n。 / Right singular vector matrix V with size n-by-n.
         */
        Matrix v;

        /**
         * @brief 实际执行的 cyclic sweep 次数。 / Actual number of cyclic sweeps executed.
         */
        std::size_t sweeps{0};

        /**
         * @brief 是否在给定容差内收敛。 / Whether the iteration converged within the requested tolerance.
         */
        bool converged{false};
    };

    /**
     * @brief Jacobi 列对。 Jacobi column pair.
     */
    struct JacobiColumnPair
    {
        /**
         * @brief 左列索引。 / Left column index.
         */
        std::size_t left{0};

        /**
         * @brief 右列索引。 / Right column index.
         */
        std::size_t right{0};
    };

    /**
     * @brief Jacobi phase，内部列对两两不共享列。 Jacobi phase whose pairs do not share columns.
     */
    struct JacobiPairPhase
    {
        /**
         * @brief 本 phase 中可并行处理的列对。 / Column pairs that can be processed concurrently in this phase.
         */
        std::vector<JacobiColumnPair> pairs;
    };

    /**
     * @brief 串行 naive 单边 Jacobi SVD 算子。 Serial naive one-sided Jacobi SVD kernel.
     *
     * @note 该算子保持 KernelLike 风格的无状态函数对象（stateless function object），但它是整矩阵分解算子，不是坐标纯函数（coordinate pure function）。 / This kernel keeps the KernelLike-style stateless function-object shape, but it is a whole-matrix factorization operator rather than a coordinate pure function.
     */
    struct OneSidedJacobiSvd
    {
        /**
         * @brief 对任意 MatrixLike 矩阵计算 thin SVD。 Compute the thin SVD of any MatrixLike matrix.
         *
         * @tparam Matrix 输入矩阵类型。 / Input matrix type.
         * @param matrix 输入矩阵 A，尺寸为 m-by-n。 / Input matrix A with size m-by-n.
         * @param options 迭代配置。 / Iteration options.
         * @return SVD 结果。 / SVD result.
         * @note 当前 naive 版本要求 m >= n。 / The current naive version requires m >= n.
         */
        template <MatrixLike Matrix>
        [[nodiscard]] auto operator()(const Matrix &matrix, const OneSidedJacobiSvdOptions &options = {}) const -> OneSidedJacobiSvdResult<std::remove_cvref_t<Matrix>>;
    };

} // namespace mpilab::domain

#include "mpilab/domain/kernel/detail/one_sided_jacobi_svd_detail.hpp"

namespace mpilab::domain
{
    template <MatrixLike Matrix>
    auto OneSidedJacobiSvd::operator()(const Matrix& matrix, const OneSidedJacobiSvdOptions& options) const -> OneSidedJacobiSvdResult<std::remove_cvref_t<Matrix>>
    {
        return detail::run_one_sided_jacobi_svd(matrix, options, detail::JacobiSweepStrategy::cyclic);
    }
} // namespace mpilab::domain
