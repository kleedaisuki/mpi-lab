#pragma once

#include "mpilab/domain/kernel/MpiFriendlyOneSidedJacobiSvd.hpp"
#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

#include <cstddef>

/**
 * @file one_sided_jacobi_svd_detail.hpp
 * @brief 单边 Jacobi SVD 内部实现声明。 Internal one-sided Jacobi SVD implementation declarations.
 */

namespace mpilab::domain::detail
{

    /**
     * @brief Jacobi sweep 策略。 Jacobi sweep strategy.
     */
    enum class JacobiSweepStrategy
    {
        /**
         * @brief 固定 cyclic 列对顺序。 / Fixed cyclic column-pair order.
         */
        cyclic,

        /**
         * @brief 按归一化列相关性动态排序。 / Dynamic ordering by normalized column correlation.
         */
        dynamic_ordered,

        /**
         * @brief Round-robin phase 顺序。 / Round-robin phase order.
         */
        round_robin_phases
    };

    /**
     * @brief 运行单边 Jacobi SVD 共享实现。 Run the shared one-sided Jacobi SVD implementation.
     *
     * @param matrix 输入矩阵。 / Input matrix.
     * @param options 迭代配置。 / Iteration options.
     * @param strategy sweep 策略。 / Sweep strategy.
     * @return SVD 结果。 / SVD result.
     */
    [[nodiscard]] auto run_one_sided_jacobi_svd(const RowMajorMatrix<double>& matrix, const OneSidedJacobiSvdOptions& options, JacobiSweepStrategy strategy) -> OneSidedJacobiSvdResult;

    /**
     * @brief 生成 round-robin phase 调度。 Build a round-robin phase schedule.
     *
     * @param column_count 列数。 / Column count.
     * @return phase 调度。 / Phase schedule.
     */
    [[nodiscard]] auto build_round_robin_phases(std::size_t column_count) -> std::vector<JacobiPairPhase>;

} // namespace mpilab::domain::detail
