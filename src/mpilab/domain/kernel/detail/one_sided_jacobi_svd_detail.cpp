#include "one_sided_jacobi_svd_detail.hpp"

#include "mpilab/infrastructure/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__SSE2__) && (defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86))
#include <emmintrin.h>
/**
 * @brief 标记当前构建支持 SSE2 SVD 内核。 Marks that the current build supports the SSE2 SVD kernel.
 */
#define MPILAB_HAS_SSE2_SVD_KERNEL 1
#else
/**
 * @brief 标记当前构建不支持 SSE2 SVD 内核。 Marks that the current build does not support the SSE2 SVD kernel.
 */
#define MPILAB_HAS_SSE2_SVD_KERNEL 0
#endif

namespace mpilab::domain::detail
{
    namespace
    {

        /**
         * @brief 带分数的 Jacobi 列对。 Scored Jacobi column pair.
         */
        struct ScoredJacobiPair
        {
            /**
             * @brief 左列索引。 / Left column index.
             */
            std::size_t left{0};

            /**
             * @brief 右列索引。 / Right column index.
             */
            std::size_t right{0};

            /**
             * @brief 归一化列相关性分数。 / Normalized column-correlation score.
             */
            double score{0.0};
        };

        /**
         * @brief 返回符号拷贝后的幅值。 Return a magnitude with copied sign.
         *
         * @param magnitude 幅值。 / Magnitude.
         * @param sign_source 符号来源。 / Source of the sign.
         * @return 带符号的幅值。 / Signed magnitude.
         */
        [[nodiscard]] auto signed_value(double magnitude, double sign_source) -> double
        {
            return sign_source < 0.0 ? -magnitude : magnitude;
        }

        /**
         * @brief 将矩阵初始化为单位矩阵。 Initialize a matrix as an identity matrix.
         *
         * @param matrix 待初始化矩阵。 / Matrix to initialize.
         */
        void set_identity(RowMajorMatrix<double> &matrix)
        {
            for (std::size_t y = 0; y < matrix.height(); ++y)
            {
                for (std::size_t x = 0; x < matrix.width(); ++x)
                {
                    matrix.set(x, y, x == y ? 1.0 : 0.0);
                }
            }
        }

        /**
         * @brief 计算两个列向量的内积。 Compute the inner product of two columns.
         *
         * @param matrix 输入矩阵。 / Input matrix.
         * @param left 左列索引。 / Left column index.
         * @param right 右列索引。 / Right column index.
         * @return 两列内积。 / Inner product of the two columns.
         */
        [[nodiscard]] auto scalar_column_dot(const RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right) -> double
        {
            double sum = 0.0;
            for (std::size_t y = 0; y < matrix.height(); ++y)
            {
                sum += matrix(left, y) * matrix(right, y);
            }
            return sum;
        }

        /**
         * @brief 使用 SIMD 计算两个列向量的内积。 Compute the inner product of two columns with SIMD.
         *
         * @param matrix 输入矩阵。 / Input matrix.
         * @param left 左列索引。 / Left column index.
         * @param right 右列索引。 / Right column index.
         * @return 两列内积。 / Inner product of the two columns.
         */
        [[nodiscard]] auto simd_column_dot(const RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right) -> double
        {
#if MPILAB_HAS_SSE2_SVD_KERNEL
            __m128d sum = _mm_setzero_pd();
            std::size_t y = 0;

            for (; y + 1 < matrix.height(); y += 2)
            {
                const __m128d left_values = _mm_set_pd(matrix(left, y + 1), matrix(left, y));
                const __m128d right_values = _mm_set_pd(matrix(right, y + 1), matrix(right, y));
                sum = _mm_add_pd(sum, _mm_mul_pd(left_values, right_values));
            }

            alignas(16) double lanes[2]{0.0, 0.0};
            _mm_store_pd(lanes, sum);
            double scalar_sum = lanes[0] + lanes[1];

            for (; y < matrix.height(); ++y)
            {
                scalar_sum += matrix(left, y) * matrix(right, y);
            }

            return scalar_sum;
#else
            return scalar_column_dot(matrix, left, right);
#endif
        }

        /**
         * @brief 计算两个列向量的内积。 Compute the inner product of two columns.
         *
         * @param matrix 输入矩阵。 / Input matrix.
         * @param left 左列索引。 / Left column index.
         * @param right 右列索引。 / Right column index.
         * @param compute_kernel 数值内核策略。 / Numeric-kernel strategy.
         * @return 两列内积。 / Inner product of the two columns.
         */
        [[nodiscard]] auto column_dot(const RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right, JacobiComputeKernel compute_kernel) -> double
        {
            if (compute_kernel == JacobiComputeKernel::simd)
            {
                return simd_column_dot(matrix, left, right);
            }

            return scalar_column_dot(matrix, left, right);
        }

        /**
         * @brief 计算列向量的 Euclidean norm。 Compute the Euclidean norm of a column.
         *
         * @param matrix 输入矩阵。 / Input matrix.
         * @param column 列索引。 / Column index.
         * @return 列范数。 / Column norm.
         */
        [[nodiscard]] auto column_norm(const RowMajorMatrix<double> &matrix, std::size_t column, JacobiComputeKernel compute_kernel) -> double
        {
            return std::sqrt(column_dot(matrix, column, column, compute_kernel));
        }

        /**
         * @brief 对两个列应用右侧 Jacobi rotation。 Apply a right-sided Jacobi rotation to two columns.
         *
         * @param matrix 待更新矩阵。 / Matrix to update.
         * @param left 左列索引。 / Left column index.
         * @param right 右列索引。 / Right column index.
         * @param cosine 旋转 cosine。 / Rotation cosine.
         * @param sine 旋转 sine。 / Rotation sine.
         */
        void scalar_rotate_columns(RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right, double cosine, double sine)
        {
            for (std::size_t y = 0; y < matrix.height(); ++y)
            {
                const double left_value = matrix(left, y);
                const double right_value = matrix(right, y);

                matrix.set(left, y, (cosine * left_value) - (sine * right_value));
                matrix.set(right, y, (sine * left_value) + (cosine * right_value));
            }
        }

        /**
         * @brief 使用 SIMD 对两个列应用右侧 Jacobi rotation。 Apply a right-sided Jacobi rotation to two columns with SIMD.
         *
         * @param matrix 待更新矩阵。 / Matrix to update.
         * @param left 左列索引。 / Left column index.
         * @param right 右列索引。 / Right column index.
         * @param cosine 旋转 cosine。 / Rotation cosine.
         * @param sine 旋转 sine。 / Rotation sine.
         */
        void simd_rotate_columns(RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right, double cosine, double sine)
        {
#if MPILAB_HAS_SSE2_SVD_KERNEL
            const __m128d cosines = _mm_set1_pd(cosine);
            const __m128d sines = _mm_set1_pd(sine);
            std::size_t y = 0;

            for (; y + 1 < matrix.height(); y += 2)
            {
                const __m128d left_values = _mm_set_pd(matrix(left, y + 1), matrix(left, y));
                const __m128d right_values = _mm_set_pd(matrix(right, y + 1), matrix(right, y));
                const __m128d new_left = _mm_sub_pd(_mm_mul_pd(cosines, left_values), _mm_mul_pd(sines, right_values));
                const __m128d new_right = _mm_add_pd(_mm_mul_pd(sines, left_values), _mm_mul_pd(cosines, right_values));
                alignas(16) double left_lanes[2]{0.0, 0.0};
                alignas(16) double right_lanes[2]{0.0, 0.0};

                _mm_store_pd(left_lanes, new_left);
                _mm_store_pd(right_lanes, new_right);
                matrix.set(left, y, left_lanes[0]);
                matrix.set(left, y + 1, left_lanes[1]);
                matrix.set(right, y, right_lanes[0]);
                matrix.set(right, y + 1, right_lanes[1]);
            }

            for (; y < matrix.height(); ++y)
            {
                const double left_value = matrix(left, y);
                const double right_value = matrix(right, y);

                matrix.set(left, y, (cosine * left_value) - (sine * right_value));
                matrix.set(right, y, (sine * left_value) + (cosine * right_value));
            }
#else
            scalar_rotate_columns(matrix, left, right, cosine, sine);
#endif
        }

        /**
         * @brief 对两个列应用右侧 Jacobi rotation。 Apply a right-sided Jacobi rotation to two columns.
         *
         * @param matrix 待更新矩阵。 / Matrix to update.
         * @param left 左列索引。 / Left column index.
         * @param right 右列索引。 / Right column index.
         * @param cosine 旋转 cosine。 / Rotation cosine.
         * @param sine 旋转 sine。 / Rotation sine.
         * @param compute_kernel 数值内核策略。 / Numeric-kernel strategy.
         */
        void rotate_columns(RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right, double cosine, double sine, JacobiComputeKernel compute_kernel)
        {
            if (compute_kernel == JacobiComputeKernel::simd)
            {
                simd_rotate_columns(matrix, left, right, cosine, sine);
                return;
            }

            scalar_rotate_columns(matrix, left, right, cosine, sine);
        }

        /**
         * @brief 对列 Gram 子矩阵计算 Jacobi rotation。 Compute the Jacobi rotation for a two-column Gram submatrix.
         *
         * @param left_norm_sq 左列范数平方。 / Squared norm of the left column.
         * @param right_norm_sq 右列范数平方。 / Squared norm of the right column.
         * @param cross_dot 两列内积。 / Inner product of the two columns.
         * @return cosine 与 sine。 / Cosine and sine.
         */
        [[nodiscard]] auto jacobi_rotation(double left_norm_sq, double right_norm_sq, double cross_dot) -> std::pair<double, double>
        {
            const double tau = (right_norm_sq - left_norm_sq) / (2.0 * cross_dot);
            const double tangent = signed_value(1.0, tau) / (std::abs(tau) + std::sqrt(1.0 + (tau * tau)));
            const double cosine = 1.0 / std::sqrt(1.0 + (tangent * tangent));
            const double sine = cosine * tangent;

            return {cosine, sine};
        }

        /**
         * @brief 尝试旋转一个列对。 Try to rotate one column pair.
         *
         * @param work 当前迭代矩阵。 / Current iterate matrix.
         * @param right_vectors 右奇异向量累积矩阵。 / Accumulated right singular vector matrix.
         * @param pair 列对。 / Column pair.
         * @param tolerance 容差。 / Tolerance.
         * @return 发生旋转时返回 true。 / Returns true when a rotation was applied.
         */
        [[nodiscard]] auto rotate_pair_if_needed(RowMajorMatrix<double> &work, RowMajorMatrix<double> &right_vectors, JacobiColumnPair pair, double tolerance, JacobiComputeKernel compute_kernel) -> bool
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

        /**
         * @brief 生成 cyclic 列对顺序。 Build cyclic column-pair order.
         *
         * @param column_count 列数。 / Column count.
         * @return 列对顺序。 / Column-pair order.
         */
        [[nodiscard]] auto build_cyclic_pairs(std::size_t column_count) -> std::vector<JacobiColumnPair>
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

        /**
         * @brief 生成动态排序列对顺序。 Build dynamically ordered column-pair order.
         *
         * @param work 当前迭代矩阵。 / Current iterate matrix.
         * @return 按分数降序排列的列对。 / Column pairs sorted by descending score.
         */
        [[nodiscard]] auto build_dynamic_pairs(const RowMajorMatrix<double> &work, JacobiComputeKernel compute_kernel) -> std::vector<JacobiColumnPair>
        {
            const std::size_t column_count = work.width();
            std::vector<ScoredJacobiPair> scored_pairs;
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

            std::ranges::sort(scored_pairs, [](const ScoredJacobiPair &left, const ScoredJacobiPair &right)
                              { return left.score > right.score; });

            std::vector<JacobiColumnPair> pairs;
            pairs.reserve(scored_pairs.size());
            for (const ScoredJacobiPair &pair : scored_pairs)
            {
                pairs.push_back({.left = pair.left, .right = pair.right});
            }

            return pairs;
        }

        /**
         * @brief 按奇异值非增序交换 SVD 列。 Swap SVD columns into non-increasing singular-value order.
         *
         * @param result 待排序结果。 / Result to sort.
         */
        void sort_singular_triplets(OneSidedJacobiSvdResult &result)
        {
            const std::size_t count = result.singular_values.size();
            std::vector<std::size_t> order(count);
            std::iota(order.begin(), order.end(), 0);
            std::ranges::sort(order, [&result](std::size_t left, std::size_t right)
                              { return result.singular_values[left] > result.singular_values[right]; });

            RowMajorMatrix<double> sorted_u(result.u.width(), result.u.height());
            RowMajorMatrix<double> sorted_v(result.v.width(), result.v.height());
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

    } // namespace

    auto build_round_robin_phases(std::size_t column_count) -> std::vector<JacobiPairPhase>
    {
        const std::size_t padded_count = column_count % 2 == 0 ? column_count : column_count + 1;
        std::vector<std::size_t> slots(padded_count);
        std::iota(slots.begin(), slots.end(), 0);

        std::vector<JacobiPairPhase> phases;
        if (column_count < 2)
        {
            return phases;
        }

        phases.reserve(padded_count - 1);
        for (std::size_t phase_index = 0; phase_index < padded_count - 1; ++phase_index)
        {
            JacobiPairPhase phase;
            phase.pairs.reserve(padded_count / 2);

            for (std::size_t pair_index = 0; pair_index < padded_count / 2; ++pair_index)
            {
                const std::size_t first = slots[pair_index];
                const std::size_t second = slots[padded_count - 1 - pair_index];

                if (first < column_count && second < column_count)
                {
                    phase.pairs.push_back({.left = std::min(first, second), .right = std::max(first, second)});
                }
            }

            phases.push_back(std::move(phase));

            const std::size_t moved = slots.back();
            for (std::size_t index = padded_count - 1; index > 1; --index)
            {
                slots[index] = slots[index - 1];
            }
            slots[1] = moved;
        }

        return phases;
    }

    auto run_one_sided_jacobi_svd(
        const RowMajorMatrix<double> &matrix,
        const OneSidedJacobiSvdOptions &options,
        JacobiSweepStrategy sweep_strategy,
        JacobiExecutionStrategy execution_strategy,
        JacobiComputeKernel compute_kernel) -> OneSidedJacobiSvdResult
    {
        const std::size_t column_count = matrix.width();
        const std::size_t row_count = matrix.height();
        const double tolerance = std::max(options.tolerance, std::numeric_limits<double>::epsilon());

        if (row_count < column_count)
        {
            throw std::invalid_argument("one-sided Jacobi SVD requires matrix.height() >= matrix.width()");
        }

        RowMajorMatrix<double> work(column_count, row_count);
        RowMajorMatrix<double> right_vectors(column_count, column_count);

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
                for (const JacobiPairPhase &phase : round_robin_phases)
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
                const std::vector<JacobiColumnPair> &pairs = sweep_strategy == JacobiSweepStrategy::dynamic_ordered ? dynamic_pairs : cyclic_pairs;

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

        OneSidedJacobiSvdResult result{
            .u = RowMajorMatrix<double>(column_count, row_count),
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
