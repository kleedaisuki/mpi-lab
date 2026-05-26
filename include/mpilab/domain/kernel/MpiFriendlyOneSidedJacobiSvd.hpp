#pragma once

#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

#include <cstddef>
#include <vector>

/**
 * @file MpiFriendlyOneSidedJacobiSvd.hpp
 * @brief MPI 友好单边 Jacobi SVD 算子声明。 MPI-friendly one-sided Jacobi SVD kernel declarations.
 */

namespace mpilab::domain
{

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
     * @brief Round-robin Jacobi 列对调度器。 Round-robin Jacobi column-pair scheduler.
     */
    struct RoundRobinJacobiPairScheduler
    {
        /**
         * @brief 为给定列数生成一轮 sweep 的 phase 调度。 Generate a phase schedule for one sweep.
         *
         * @param column_count 列数。 / Column count.
         * @return phase 调度。 / Phase schedule.
         */
        [[nodiscard]] auto operator()(std::size_t column_count) const -> std::vector<JacobiPairPhase>;
    };

    /**
     * @brief MPI 友好单边 Jacobi SVD 算子，使用 round-robin phase 调度。 MPI-friendly one-sided Jacobi SVD kernel using round-robin phase scheduling.
     *
     * @note 当前实现仍在单进程内执行 phase，但每个 phase 内列对互不相交，适合后续映射到 MPI rank 或 column block。 / The current implementation still executes phases in one process, but pairs within a phase are disjoint and can later be mapped to MPI ranks or column blocks.
     */
    struct MpiFriendlyOneSidedJacobiSvd
    {
        /**
         * @brief 对行主序矩阵计算 thin SVD。 Compute the thin SVD of a row-major matrix.
         *
         * @param matrix 输入矩阵 A，尺寸为 m-by-n。 / Input matrix A with size m-by-n.
         * @param options 迭代配置。 / Iteration options.
         * @return SVD 结果。 / SVD result.
         * @note 当前 MPI-friendly 版本要求 m >= n。 / The current MPI-friendly version requires m >= n.
         */
        [[nodiscard]] auto operator()(const RowMajorMatrix<double>& matrix, const OneSidedJacobiSvdOptions& options = {}) const -> OneSidedJacobiSvdResult;
    };

} // namespace mpilab::domain
