#pragma once

#include "mpilab/domain/MatrixLike.hpp"
#include "mpilab/domain/matrix/DenseMatrix.hpp"
#include "mpilab/infrastructure/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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
     * @brief Jacobi 执行策略。 Jacobi execution strategy.
     */
    enum class JacobiExecutionStrategy
    {
        /**
         * @brief 当前线程串行执行。 / Execute serially on the current thread.
         */
        serial,

        /**
         * @brief 使用 Pthreads 线程池执行可并行 phase。 / Execute parallel phases with the Pthreads thread pool.
         */
        pthreads
    };

    /**
     * @brief Jacobi 数值内核策略。 Jacobi numeric-kernel strategy.
     */
    enum class JacobiComputeKernel
    {
        /**
         * @brief 标量内核。 / Scalar kernel.
         */
        scalar,

        /**
         * @brief SIMD 内核，按平台能力回退。 / SIMD kernel with platform fallback.
         */
        simd
    };

    [[nodiscard]] inline auto signed_value(double magnitude, double sign_source) -> double
    {
        return sign_source < 0.0 ? -magnitude : magnitude;
    }

    [[nodiscard]] inline auto jacobi_rotation(double left_norm_sq, double right_norm_sq, double cross_dot) -> std::pair<double, double>
    {
        const double tau = (right_norm_sq - left_norm_sq) / (2.0 * cross_dot);
        const double tangent = signed_value(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + (tau * tau)));
        const double cosine = 1.0 / std::sqrt(1.0 + (tangent * tangent));
        const double sine = cosine * tangent;

        return {cosine, sine};
    }

    [[nodiscard]] inline auto build_cyclic_pairs(std::size_t column_count) -> std::vector<JacobiColumnPair>
    {
        std::vector<JacobiColumnPair> pairs;
        pairs.reserve((column_count * (column_count - 1)) / 2);

        for (std::size_t left = 0; left < column_count; ++left)
        {
            for (std::size_t right = left + 1; right < column_count; ++right)
            {
                pairs.push_back({.left = left, .right = right});
            }
        }

        return pairs;
    }

    template <MatrixLike Matrix>
    inline void sort_singular_triplets(OneSidedJacobiSvdResult<Matrix>& result)
    {
        const std::size_t count = result.singular_values.size();
        std::vector<std::size_t> order(count);
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&result](std::size_t left, std::size_t right)
                          { return result.singular_values[left] > result.singular_values[right]; });

        Matrix sorted_u(result.u.width(), result.u.height());
        Matrix sorted_v(result.v.width(), result.v.height());
        std::vector<double> sorted_singular_values(count);

        for (std::size_t sorted_column = 0; sorted_column < count; ++sorted_column)
        {
            const std::size_t source_column = order[sorted_column];
            sorted_singular_values[sorted_column] = result.singular_values[source_column];

            for (std::size_t y = 0; y < result.u.height(); ++y)
            {
                sorted_u.set(sorted_column, y, result.u(source_column, y));
            }

            for (std::size_t y = 0; y < result.v.height(); ++y)
            {
                sorted_v.set(sorted_column, y, result.v(source_column, y));
            }
        }

        result.u = std::move(sorted_u);
        result.v = std::move(sorted_v);
        result.singular_values = std::move(sorted_singular_values);
    }

    template <MatrixLike Matrix>
    void set_identity(Matrix& matrix)
    {
        for (std::size_t y = 0; y < matrix.height(); ++y)
        {
            for (std::size_t x = 0; x < matrix.width(); ++x)
            {
                matrix.set(x, y, x == y ? 1.0 : 0.0);
            }
        }
    }

    template <MatrixLike Matrix>
    [[nodiscard]] auto scalar_column_dot(const Matrix& matrix, std::size_t left, std::size_t right) -> double
    {
        double sum = 0.0;
        for (std::size_t y = 0; y < matrix.height(); ++y)
        {
            sum += matrix(left, y) * matrix(right, y);
        }
        return sum;
    }

    template <MatrixLike Matrix>
    [[nodiscard]] auto simd_column_dot(const Matrix& matrix, std::size_t left, std::size_t right) -> double
    {
        return scalar_column_dot(matrix, left, right);
    }

    template <MatrixLike Matrix>
    [[nodiscard]] auto column_dot(const Matrix& matrix, std::size_t left, std::size_t right, JacobiComputeKernel compute_kernel) -> double
    {
        if (compute_kernel == JacobiComputeKernel::simd)
        {
            return simd_column_dot(matrix, left, right);
        }

        return scalar_column_dot(matrix, left, right);
    }

    template <MatrixLike Matrix>
    [[nodiscard]] auto column_norm(const Matrix& matrix, std::size_t column, JacobiComputeKernel compute_kernel) -> double
    {
        return std::sqrt(column_dot(matrix, column, column, compute_kernel));
    }

    template <MatrixLike Matrix>
    void scalar_rotate_columns(Matrix& matrix, std::size_t left, std::size_t right, double cosine, double sine)
    {
        for (std::size_t y = 0; y < matrix.height(); ++y)
        {
            const double left_value = matrix(left, y);
            const double right_value = matrix(right, y);

            matrix.set(left, y, (cosine * left_value) - (sine * right_value));
            matrix.set(right, y, (sine * left_value) + (cosine * right_value));
        }
    }

    template <MatrixLike Matrix>
    void rotate_columns(Matrix& matrix, std::size_t left, std::size_t right, double cosine, double sine, JacobiComputeKernel)
    {
        scalar_rotate_columns(matrix, left, right, cosine, sine);
    }

    template <MatrixLike Matrix>
    [[nodiscard]] auto rotate_pair_if_needed(Matrix& work, Matrix& right_vectors, JacobiColumnPair pair, double tolerance, JacobiComputeKernel compute_kernel) -> bool
    {
        const double left_norm_sq = column_dot(work, pair.left, pair.left, compute_kernel);
        const double right_norm_sq = column_dot(work, pair.right, pair.right, compute_kernel);
        const double cross_dot = column_dot(work, pair.left, pair.right, compute_kernel);
        const double scale = std::sqrt(left_norm_sq * right_norm_sq);

        if (scale == 0.0 || std::abs(cross_dot) <= tolerance * scale)
        {
            return false;
        }

        const auto [cosine, sine] = jacobi_rotation(left_norm_sq, right_norm_sq, cross_dot);

        rotate_columns(work, pair.left, pair.right, cosine, sine, compute_kernel);
        rotate_columns(right_vectors, pair.left, pair.right, cosine, sine, compute_kernel);
        return true;
    }

    template <MatrixLike Matrix>
    [[nodiscard]] auto build_dynamic_pairs(const Matrix& work, JacobiComputeKernel compute_kernel) -> std::vector<JacobiColumnPair>
    {
        struct ScoredPair
        {
            std::size_t left{0};
            std::size_t right{0};
            double score{0.0};
        };

        const std::size_t column_count = work.width();
        std::vector<ScoredPair> scored_pairs;
        scored_pairs.reserve((column_count * (column_count - 1)) / 2);

        for (std::size_t left = 0; left < column_count; ++left)
        {
            for (std::size_t right = left + 1; right < column_count; ++right)
            {
                const double left_norm_sq = column_dot(work, left, left, compute_kernel);
                const double right_norm_sq = column_dot(work, right, right, compute_kernel);
                const double cross_dot = column_dot(work, left, right, compute_kernel);
                const double scale = std::sqrt(left_norm_sq * right_norm_sq);
                const double score = scale == 0.0 ? 0.0 : std::abs(cross_dot) / scale;

                scored_pairs.push_back({.left = left, .right = right, .score = score});
            }
        }

        std::ranges::sort(scored_pairs, [](const ScoredPair& left, const ScoredPair& right)
                          { return left.score > right.score; });

        std::vector<JacobiColumnPair> pairs;
        pairs.reserve(scored_pairs.size());
        for (const ScoredPair& pair : scored_pairs)
        {
            pairs.push_back({.left = pair.left, .right = pair.right});
        }

        return pairs;
    }

    /**
     * @brief 生成 round-robin phase 调度。 Build a round-robin phase schedule.
     *
     * @param column_count 列数。 / Column count.
     * @return phase 调度。 / Phase schedule.
     */
    [[nodiscard]] auto build_round_robin_phases(std::size_t column_count) -> std::vector<JacobiPairPhase>;

    /**
     * @brief 运行单边 Jacobi SVD 共享模板实现。 Run the shared templated one-sided Jacobi SVD implementation.
     *
     * @tparam Matrix 输入矩阵类型。 / Input matrix type.
     * @param matrix 输入矩阵。 / Input matrix.
     * @param options 迭代配置。 / Iteration options.
     * @param sweep_strategy sweep 策略。 / Sweep strategy.
     * @param execution_strategy 执行策略。 / Execution strategy.
     * @param compute_kernel 数值内核策略。 / Numeric-kernel strategy.
     * @return SVD 结果。 / SVD result.
     */
    template <MatrixLike Matrix>
    [[nodiscard]] auto run_one_sided_jacobi_svd(
        const Matrix& matrix,
        const OneSidedJacobiSvdOptions& options,
        JacobiSweepStrategy sweep_strategy,
        JacobiExecutionStrategy execution_strategy = JacobiExecutionStrategy::serial,
        JacobiComputeKernel compute_kernel = JacobiComputeKernel::scalar) -> OneSidedJacobiSvdResult<std::remove_cvref_t<Matrix>>
    {
        const std::size_t column_count = matrix.width();
        const std::size_t row_count = matrix.height();
        const double tolerance = std::max(options.tolerance, std::numeric_limits<double>::epsilon());

        if (row_count < column_count)
        {
            throw std::invalid_argument("one-sided Jacobi SVD requires matrix.height() >= matrix.width()");
        }

        using WorkMatrix = std::remove_cvref_t<Matrix>;

        WorkMatrix work(column_count, row_count);
        WorkMatrix right_vectors(column_count, column_count);

        for (std::size_t y = 0; y < row_count; ++y)
        {
            for (std::size_t x = 0; x < column_count; ++x)
            {
                work.set(x, y, matrix(x, y));
            }
        }

        set_identity(right_vectors);

        const std::vector<JacobiColumnPair> cyclic_pairs = build_cyclic_pairs(column_count);
        const std::vector<JacobiPairPhase> round_robin_phases = build_round_robin_phases(column_count);
        bool converged = true;
        std::size_t completed_sweeps = 0;

        for (; completed_sweeps < options.max_sweeps; ++completed_sweeps)
        {
            bool rotated = false;

            if (sweep_strategy == JacobiSweepStrategy::round_robin_phases)
            {
                for (const JacobiPairPhase& phase : round_robin_phases)
                {
                    if (execution_strategy == JacobiExecutionStrategy::pthreads && phase.pairs.size() > 1)
                    {
                        std::vector<unsigned char> rotated_flags(phase.pairs.size(), 0U);
                        infrastructure::default_thread_pool().parallel_for(phase.pairs.size(), [&](std::size_t index)
                                                                           { rotated_flags[index] = rotate_pair_if_needed(work, right_vectors, phase.pairs[index], tolerance, compute_kernel) ? 1U : 0U; });
                        rotated = std::ranges::any_of(rotated_flags, [](unsigned char flag)
                                                      { return flag != 0U; }) ||
                                  rotated;
                    }
                    else
                    {
                        for (JacobiColumnPair pair : phase.pairs)
                        {
                            rotated = rotate_pair_if_needed(work, right_vectors, pair, tolerance, compute_kernel) || rotated;
                        }
                    }
                }
            }
            else
            {
                const std::vector<JacobiColumnPair> dynamic_pairs = sweep_strategy == JacobiSweepStrategy::dynamic_ordered ? build_dynamic_pairs(work, compute_kernel) : std::vector<JacobiColumnPair>();
                const std::vector<JacobiColumnPair>& pairs = sweep_strategy == JacobiSweepStrategy::dynamic_ordered ? dynamic_pairs : cyclic_pairs;

                for (JacobiColumnPair pair : pairs)
                {
                    rotated = rotate_pair_if_needed(work, right_vectors, pair, tolerance, compute_kernel) || rotated;
                }
            }

            if (!rotated)
            {
                ++completed_sweeps;
                converged = true;
                break;
            }

            converged = false;
        }

        OneSidedJacobiSvdResult<WorkMatrix> result{
            .u = WorkMatrix(column_count, row_count),
            .singular_values = std::vector<double>(column_count),
            .v = std::move(right_vectors),
            .sweeps = completed_sweeps,
            .converged = converged,
        };

        for (std::size_t column = 0; column < column_count; ++column)
        {
            const double singular_value = column_norm(work, column, compute_kernel);
            result.singular_values[column] = singular_value;

            if (singular_value == 0.0)
            {
                continue;
            }

            for (std::size_t y = 0; y < row_count; ++y)
            {
                result.u.set(column, y, work(column, y) / singular_value);
            }
        }

        sort_singular_triplets(result);
        return result;
    }

} // namespace mpilab::domain::detail
