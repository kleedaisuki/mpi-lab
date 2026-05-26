#pragma once

#include "mpilab/domain/matrix/DenseMatrix.hpp"

#include <cstddef>
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
    struct OneSidedJacobiSvdResult
    {
        /**
         * @brief 左奇异向量矩阵 U，尺寸为 m-by-n。 / Left singular vector matrix U with size m-by-n.
         */
        RowMajorMatrix<double> u;

        /**
         * @brief 奇异值 Sigma 的对角元素，按非增序排列。 / Diagonal entries of Sigma, sorted in non-increasing order.
         */
        std::vector<double> singular_values;

        /**
         * @brief 右奇异向量矩阵 V，尺寸为 n-by-n。 / Right singular vector matrix V with size n-by-n.
         */
        RowMajorMatrix<double> v;

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
     * @brief 串行 naive 单边 Jacobi SVD 算子。 Serial naive one-sided Jacobi SVD kernel.
     *
     * @note 该算子保持 KernelLike 风格的无状态函数对象（stateless function object），但它是整矩阵分解算子，不是坐标纯函数（coordinate pure function）。 / This kernel keeps the KernelLike-style stateless function-object shape, but it is a whole-matrix factorization operator rather than a coordinate pure function.
     */
    struct OneSidedJacobiSvd
    {
        /**
         * @brief 对行主序矩阵计算 thin SVD。 Compute the thin SVD of a row-major matrix.
         *
         * @param matrix 输入矩阵 A，尺寸为 m-by-n。 / Input matrix A with size m-by-n.
         * @param options 迭代配置。 / Iteration options.
         * @return SVD 结果。 / SVD result.
         * @note 当前 naive 版本要求 m >= n。 / The current naive version requires m >= n.
         */
        [[nodiscard]] auto operator()(const RowMajorMatrix<double> &matrix, const OneSidedJacobiSvdOptions &options = {}) const -> OneSidedJacobiSvdResult;
    };

} // namespace mpilab::domain
