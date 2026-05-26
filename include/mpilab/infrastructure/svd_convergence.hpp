#pragma once

#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

/**
 * @file svd_convergence.hpp
 * @brief SVD 收敛性指标声明。 SVD convergence metric declarations.
 */

namespace mpilab::infrastructure
{
    /**
     * @brief SVD 收敛性指标计算选项。 SVD convergence metric calculation options.
     */
    struct SvdConvergenceOptions final
    {
        /**
         * @brief 列正交性容差。 Column-orthogonality tolerance.
         */
        double tolerance = 1.0e-12;

        /**
         * @brief 最大 sweep 次数。 Maximum sweep count.
         */
        std::size_t max_sweeps = 100;

        /**
         * @brief 判定零奇异值的下界。 Floor used to classify zero singular values.
         *
         * @note 奇异值不大于该值的列不会参与左奇异向量正交性检查。 / Columns with singular values no greater than this value are excluded from left singular-vector orthogonality checks.
         */
        double singular_value_floor = 0.0;
    };

    /**
     * @brief SVD 收敛性指标。 SVD convergence metrics.
     */
    struct SvdConvergenceMetrics final
    {
        /**
         * @brief 奇异向量列数量。 Number of singular-vector columns.
         */
        std::size_t column_count = 0;

        /**
         * @brief 参与正交性比较的列对数量。 Number of column pairs included in orthogonality checks.
         */
        std::size_t compared_pair_count = 0;

        /**
         * @brief 因零奇异值跳过的列对数量。 Number of column pairs skipped because of zero singular values.
         */
        std::size_t skipped_pair_count = 0;

        /**
         * @brief 零奇异值数量。 Number of zero singular values.
         */
        std::size_t zero_singular_value_count = 0;

        /**
         * @brief 超过容差的列对数量。 Number of column pairs whose normalized correlation exceeds tolerance.
         */
        std::size_t tolerance_violation_count = 0;

        /**
         * @brief 实际执行 sweep 次数。 Number of executed sweeps.
         */
        std::size_t sweeps = 0;

        /**
         * @brief 最大允许 sweep 次数。 Maximum allowed sweep count.
         */
        std::size_t max_sweeps = 0;

        /**
         * @brief 收敛容差。 Convergence tolerance.
         */
        double tolerance = 0.0;

        /**
         * @brief sweep 预算使用比例。 Fraction of sweep budget used.
         */
        double sweep_budget_fraction = 0.0;

        /**
         * @brief kernel 自报是否收敛。 Whether the kernel reported convergence.
         */
        bool reported_converged = false;

        /**
         * @brief 是否用尽 sweep 预算。 Whether the sweep budget was exhausted.
         */
        bool exhausted_sweep_budget = false;

        /**
         * @brief 后验列正交性是否满足容差。 Whether a posteriori column orthogonality satisfies tolerance.
         */
        bool orthogonality_tolerance_satisfied = true;

        /**
         * @brief 自报状态与后验检查是否同时收敛。 Whether the reported state and a posteriori check both indicate convergence.
         */
        bool converged = false;

        /**
         * @brief 非零左奇异向量列间最大归一化相关性。 Maximum normalized correlation between nonzero left singular-vector columns.
         */
        double max_column_correlation = 0.0;

        /**
         * @brief 非零左奇异向量列间均方根归一化相关性。 RMS normalized correlation between nonzero left singular-vector columns.
         */
        double rms_column_correlation = 0.0;

        /**
         * @brief 左奇异向量 Gram 矩阵非对角 Frobenius 范数。 Off-diagonal Frobenius norm of the left singular-vector Gram matrix.
         */
        double offdiagonal_gram_frobenius_norm = 0.0;

        /**
         * @brief 右奇异向量 Gram 矩阵非对角 Frobenius 范数。 Off-diagonal Frobenius norm of the right singular-vector Gram matrix.
         */
        double right_offdiagonal_gram_frobenius_norm = 0.0;
    };

    /**
     * @brief 从单边 Jacobi SVD 结果计算后验收敛性指标。 Compute a posteriori convergence metrics from a one-sided Jacobi SVD result.
     *
     * @param result SVD 结果。 / SVD result.
     * @param options 收敛性指标选项。 / Convergence metric options.
     * @return SVD 收敛性指标。 / SVD convergence metrics.
     * @note 对单边 Jacobi SVD，非零 U 列之间的归一化相关性反映最终列正交化残差。 / For one-sided Jacobi SVD, normalized correlations between nonzero U columns reflect the final column-orthogonalization residual.
     */
    [[nodiscard]] inline auto evaluate_svd_convergence(
        const domain::OneSidedJacobiSvdResult &result,
        SvdConvergenceOptions options = {}) -> SvdConvergenceMetrics
    {
        const double tolerance = std::max(options.tolerance, std::numeric_limits<double>::epsilon());

        SvdConvergenceMetrics metrics;
        metrics.column_count = result.singular_values.size();
        metrics.sweeps = result.sweeps;
        metrics.max_sweeps = options.max_sweeps;
        metrics.tolerance = tolerance;
        metrics.sweep_budget_fraction = options.max_sweeps == 0 ? 0.0 : static_cast<double>(result.sweeps) / static_cast<double>(options.max_sweeps);
        metrics.reported_converged = result.converged;
        metrics.exhausted_sweep_budget = !result.converged && result.sweeps >= options.max_sweeps;

        for (double singular_value : result.singular_values)
        {
            if (singular_value <= options.singular_value_floor)
            {
                ++metrics.zero_singular_value_count;
            }
        }

        long double sum_squared_correlation = 0.0L;
        long double sum_squared_right_correlation = 0.0L;
        for (std::size_t left = 0; left < metrics.column_count; ++left)
        {
            for (std::size_t right = left + 1; right < metrics.column_count; ++right)
            {
                double right_dot = 0.0;
                for (std::size_t row = 0; row < result.v.height(); ++row)
                {
                    right_dot += result.v(left, row) * result.v(right, row);
                }
                sum_squared_right_correlation += static_cast<long double>(right_dot) * static_cast<long double>(right_dot);

                if (result.singular_values[left] <= options.singular_value_floor || result.singular_values[right] <= options.singular_value_floor)
                {
                    ++metrics.skipped_pair_count;
                    continue;
                }

                double left_dot = 0.0;
                for (std::size_t row = 0; row < result.u.height(); ++row)
                {
                    left_dot += result.u(left, row) * result.u(right, row);
                }

                const double correlation = std::abs(left_dot);
                ++metrics.compared_pair_count;
                metrics.max_column_correlation = std::max(metrics.max_column_correlation, correlation);
                sum_squared_correlation += static_cast<long double>(correlation) * static_cast<long double>(correlation);
                if (correlation > tolerance)
                {
                    ++metrics.tolerance_violation_count;
                }
            }
        }

        if (metrics.compared_pair_count != 0)
        {
            metrics.rms_column_correlation = std::sqrt(static_cast<double>(sum_squared_correlation) / static_cast<double>(metrics.compared_pair_count));
        }
        metrics.offdiagonal_gram_frobenius_norm = std::sqrt(2.0 * static_cast<double>(sum_squared_correlation));
        metrics.right_offdiagonal_gram_frobenius_norm = std::sqrt(2.0 * static_cast<double>(sum_squared_right_correlation));
        metrics.orthogonality_tolerance_satisfied = metrics.tolerance_violation_count == 0;
        metrics.converged = metrics.reported_converged && metrics.orthogonality_tolerance_satisfied;
        return metrics;
    }

    /**
     * @brief 使用 SVD 迭代配置计算后验收敛性指标。 Compute a posteriori convergence metrics with SVD iteration options.
     *
     * @param result SVD 结果。 / SVD result.
     * @param options SVD 迭代配置。 / SVD iteration options.
     * @return SVD 收敛性指标。 / SVD convergence metrics.
     */
    [[nodiscard]] inline auto evaluate_svd_convergence(
        const domain::OneSidedJacobiSvdResult &result,
        const domain::OneSidedJacobiSvdOptions &options) -> SvdConvergenceMetrics
    {
        return evaluate_svd_convergence(result, SvdConvergenceOptions{.tolerance = options.tolerance, .max_sweeps = options.max_sweeps});
    }
} // namespace mpilab::infrastructure
