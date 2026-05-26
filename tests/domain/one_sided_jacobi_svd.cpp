#include "mpilab/domain/kernel.hpp"
#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/numerical_accuracy.hpp"
#include "mpilab/infrastructure/svd_convergence.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

    /**
     * @brief 默认测试容差。 / Default test tolerance.
     */
    constexpr double tolerance = 1.0e-9;

    /**
     * @brief 判断两个浮点值是否近似相等。 Check whether two floating-point values are approximately equal.
     *
     * @param left 左值。 / Left value.
     * @param right 右值。 / Right value.
     * @param scale 尺度因子。 / Scale factor.
     * @return 近似相等时返回 true。 / Returns true when approximately equal.
     */
    [[nodiscard]] auto almost_equal(double left, double right, double scale = 1.0) -> bool
    {
        return std::abs(left - right) <= tolerance * scale;
    }

    /**
     * @brief 从 SVD 结果重构矩阵。 Reconstruct a matrix from an SVD result.
     *
     * @param result SVD 结果。 / SVD result.
     * @param width 原矩阵宽度。 / Original matrix width.
     * @param height 原矩阵高度。 / Original matrix height.
     * @return 重构矩阵。 / Reconstructed matrix.
     */
    template <mpilab::domain::MatrixLike Matrix>
    [[nodiscard]] auto reconstruct_matrix(const mpilab::domain::OneSidedJacobiSvdResult<Matrix> &result, std::size_t width, std::size_t height) -> Matrix
    {
        Matrix reconstructed(width, height);
        for (std::size_t y = 0; y < height; ++y)
        {
            for (std::size_t x = 0; x < width; ++x)
            {
                double sum = 0.0;
                for (std::size_t column = 0; column < result.singular_values.size(); ++column)
                {
                    sum += result.u(column, y) * result.singular_values[column] * result.v(column, x);
                }
                reconstructed.set(x, y, sum);
            }
        }

        return reconstructed;
    }

    /**
     * @brief 用论文指标断言 SVD 结果有效。 Assert an SVD result with paper-style metrics.
     *
     * @param matrix 原矩阵。 / Original matrix.
     * @param result SVD 结果。 / SVD result.
     * @param options SVD 迭代选项。 / SVD iteration options.
     */
    template <mpilab::domain::MatrixLike Matrix>
    void assert_svd_metrics(
        const Matrix &matrix,
        const mpilab::domain::OneSidedJacobiSvdResult<Matrix> &result,
        const mpilab::domain::OneSidedJacobiSvdOptions &options = {})
    {
        const Matrix reconstructed = reconstruct_matrix(result, matrix.width(), matrix.height());
        const mpilab::infrastructure::NumericalAccuracyMetrics accuracy = mpilab::infrastructure::evaluate_numerical_accuracy(matrix, reconstructed);
        const mpilab::infrastructure::SvdConvergenceMetrics convergence = mpilab::infrastructure::evaluate_svd_convergence(result, options);

        assert(accuracy.element_count == matrix.width() * matrix.height());
        assert(accuracy.nonfinite_pair_count == 0);
        assert(accuracy.root_mean_squared_error <= tolerance * 10.0);
        assert(accuracy.relative_l2_error <= tolerance * 10.0);
        assert(accuracy.relative_linf_error <= tolerance * 10.0);
        assert(convergence.reported_converged);
        assert(convergence.converged);
        assert(convergence.tolerance_violation_count == 0);
        assert(convergence.max_column_correlation <= convergence.tolerance);
        assert(convergence.right_offdiagonal_gram_frobenius_norm <= tolerance * 10.0);
    }

    /**
     * @brief 断言奇异值按非增序排列。 Assert that singular values are sorted in non-increasing order.
     *
     * @param result SVD 结果。 / SVD result.
     */
    template <mpilab::domain::MatrixLike Matrix>
    void assert_singular_values_are_sorted(const mpilab::domain::OneSidedJacobiSvdResult<Matrix> &result)
    {
        for (std::size_t index = 1; index < result.singular_values.size(); ++index)
        {
            assert(result.singular_values[index - 1] + tolerance >= result.singular_values[index]);
        }
    }

    /**
     * @brief 构造一个非对角满列秩测试矩阵。 Build a non-diagonal full-column-rank test matrix.
     *
     * @return 测试矩阵。 / Test matrix.
     */
    [[nodiscard]] auto make_full_rank_matrix() -> mpilab::domain::RowMajorMatrix<double>
    {
        mpilab::domain::RowMajorMatrix<double> matrix(2, 3);
        matrix.set(0, 0, 1.0);
        matrix.set(1, 0, 2.0);
        matrix.set(0, 1, 3.0);
        matrix.set(1, 1, 4.0);
        matrix.set(0, 2, 5.0);
        matrix.set(1, 2, 7.0);
        return matrix;
    }

    static_assert(std::is_empty_v<mpilab::domain::OneSidedJacobiSvd>);
    static_assert(std::is_default_constructible_v<mpilab::domain::OneSidedJacobiSvd>);
    static_assert(std::is_trivially_copyable_v<mpilab::domain::OneSidedJacobiSvd>);
    static_assert(std::is_empty_v<mpilab::domain::AdvancedOneSidedJacobiSvd>);
    static_assert(std::is_default_constructible_v<mpilab::domain::AdvancedOneSidedJacobiSvd>);
    static_assert(std::is_trivially_copyable_v<mpilab::domain::AdvancedOneSidedJacobiSvd>);
    static_assert(std::is_empty_v<mpilab::domain::MpiFriendlyOneSidedJacobiSvd>);
    static_assert(std::is_default_constructible_v<mpilab::domain::MpiFriendlyOneSidedJacobiSvd>);
    static_assert(std::is_trivially_copyable_v<mpilab::domain::MpiFriendlyOneSidedJacobiSvd>);
    static_assert(std::is_empty_v<mpilab::domain::PthreadsOneSidedJacobiSvd>);
    static_assert(std::is_default_constructible_v<mpilab::domain::PthreadsOneSidedJacobiSvd>);
    static_assert(std::is_trivially_copyable_v<mpilab::domain::PthreadsOneSidedJacobiSvd>);
    static_assert(std::is_empty_v<mpilab::domain::SimdOneSidedJacobiSvd>);
    static_assert(std::is_default_constructible_v<mpilab::domain::SimdOneSidedJacobiSvd>);
    static_assert(std::is_trivially_copyable_v<mpilab::domain::SimdOneSidedJacobiSvd>);

} // namespace

/**
 * @brief 单边 Jacobi SVD 语义测试入口。 One-sided Jacobi SVD semantic test entry point.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
int main()
{
    /**
     * @brief 被测无状态 SVD 算子。 / Stateless SVD kernel under test.
     */
    const mpilab::domain::OneSidedJacobiSvd svd;
    const mpilab::domain::AdvancedOneSidedJacobiSvd advanced_svd;
    const mpilab::domain::MpiFriendlyOneSidedJacobiSvd mpi_friendly_svd;
    const mpilab::domain::PthreadsOneSidedJacobiSvd pthreads_svd;
    const mpilab::domain::SimdOneSidedJacobiSvd simd_svd;

    /**
     * @brief 对角矩阵测试。 / Diagonal matrix test.
     */
    mpilab::domain::RowMajorMatrix<double> diagonal(2, 2);
    diagonal.set(0, 0, 3.0);
    diagonal.set(1, 0, 0.0);
    diagonal.set(0, 1, 0.0);
    diagonal.set(1, 1, 2.0);

    const auto diagonal_result = svd(diagonal);
    assert(diagonal_result.converged);
    assert(diagonal_result.singular_values.size() == 2);
    assert(almost_equal(diagonal_result.singular_values[0], 3.0));
    assert(almost_equal(diagonal_result.singular_values[1], 2.0));
    assert_svd_metrics(diagonal, diagonal_result);

    /**
     * @brief 非对角满列秩矩阵测试。 / Non-diagonal full-column-rank matrix test.
     */
    const auto full_rank = make_full_rank_matrix();
    const auto full_rank_result = svd(full_rank);
    const auto advanced_full_rank_result = advanced_svd(full_rank);
    const auto mpi_friendly_full_rank_result = mpi_friendly_svd(full_rank);
    const auto pthreads_full_rank_result = pthreads_svd(full_rank);
    const auto simd_full_rank_result = simd_svd(full_rank);
    assert(full_rank_result.converged);
    assert(advanced_full_rank_result.converged);
    assert(mpi_friendly_full_rank_result.converged);
    assert(pthreads_full_rank_result.converged);
    assert(simd_full_rank_result.converged);
    assert_singular_values_are_sorted(full_rank_result);
    assert_singular_values_are_sorted(advanced_full_rank_result);
    assert_singular_values_are_sorted(mpi_friendly_full_rank_result);
    assert_singular_values_are_sorted(pthreads_full_rank_result);
    assert_singular_values_are_sorted(simd_full_rank_result);
    assert_svd_metrics(full_rank, full_rank_result);
    assert_svd_metrics(full_rank, advanced_full_rank_result);
    assert_svd_metrics(full_rank, mpi_friendly_full_rank_result);
    assert_svd_metrics(full_rank, pthreads_full_rank_result);
    assert_svd_metrics(full_rank, simd_full_rank_result);

    /**
     * @brief 多列矩阵测试，确保 Pthreads phase 并行路径被覆盖。 / Multi-column matrix test covering the Pthreads phase-parallel path.
     */
    mpilab::domain::RowMajorMatrix<double> multi_column(4, 5);
    for (std::size_t y = 0; y < multi_column.height(); ++y)
    {
        for (std::size_t x = 0; x < multi_column.width(); ++x)
        {
            multi_column.set(x, y, static_cast<double>((3 * y) + (2 * x) + ((x + y) % 3)));
        }
    }

    const auto pthreads_multi_result = pthreads_svd(multi_column);
    const auto simd_multi_result = simd_svd(multi_column);
    assert(pthreads_multi_result.converged);
    assert(simd_multi_result.converged);
    assert_singular_values_are_sorted(pthreads_multi_result);
    assert_singular_values_are_sorted(simd_multi_result);
    assert_svd_metrics(multi_column, pthreads_multi_result);
    assert_svd_metrics(multi_column, simd_multi_result);

    /**
     * @brief 非行主序结果类型测试。 / Non-row-major result type test.
     */
    mpilab::domain::ColumnMajorMatrix<double> column_major(2, 3);
    for (std::size_t y = 0; y < full_rank.height(); ++y)
    {
        for (std::size_t x = 0; x < full_rank.width(); ++x)
        {
            column_major.set(x, y, full_rank(x, y));
        }
    }
    const auto column_major_result = advanced_svd(column_major);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(column_major_result.u)>, mpilab::domain::ColumnMajorMatrix<double>>);
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(column_major_result.v)>, mpilab::domain::ColumnMajorMatrix<double>>);
    assert(column_major_result.converged);
    assert_svd_metrics(column_major, column_major_result);

    /**
     * @brief 秩亏矩阵测试。 / Rank-deficient matrix test.
     */
    mpilab::domain::RowMajorMatrix<double> rank_deficient(2, 3);
    rank_deficient.set(0, 0, 1.0);
    rank_deficient.set(1, 0, 2.0);
    rank_deficient.set(0, 1, 2.0);
    rank_deficient.set(1, 1, 4.0);
    rank_deficient.set(0, 2, 3.0);
    rank_deficient.set(1, 2, 6.0);

    const auto rank_deficient_result = svd(rank_deficient);
    assert(rank_deficient_result.converged);
    assert(rank_deficient_result.singular_values[1] <= tolerance);
    assert_svd_metrics(rank_deficient, rank_deficient_result);

    /**
     * @brief Round-robin phase 调度不变量测试。 / Round-robin phase scheduling invariant test.
     */
    const mpilab::domain::RoundRobinJacobiPairScheduler scheduler;
    const std::vector<mpilab::domain::JacobiPairPhase> phases = scheduler(5);
    std::vector<bool> seen_pairs(25, false);
    std::size_t pair_count = 0;
    assert(phases.size() == 5);
    for (const mpilab::domain::JacobiPairPhase &phase : phases)
    {
        std::vector<bool> used_columns(5, false);
        for (mpilab::domain::JacobiColumnPair pair : phase.pairs)
        {
            assert(pair.left < 5);
            assert(pair.right < 5);
            assert(pair.left < pair.right);
            assert(!used_columns[pair.left]);
            assert(!used_columns[pair.right]);
            used_columns[pair.left] = true;
            used_columns[pair.right] = true;
            assert(!seen_pairs[(pair.left * 5) + pair.right]);
            seen_pairs[(pair.left * 5) + pair.right] = true;
            ++pair_count;
        }
    }
    assert(pair_count == 10);

    /**
     * @brief 宽矩阵前置条件测试。 / Wide-matrix precondition test.
     */
    mpilab::domain::RowMajorMatrix<double> wide(3, 2);
    bool rejected_wide_matrix = false;
    try
    {
        static_cast<void>(svd(wide));
    }
    catch (const std::invalid_argument &)
    {
        rejected_wide_matrix = true;
    }

    assert(rejected_wide_matrix);

    rejected_wide_matrix = false;
    try
    {
        static_cast<void>(advanced_svd(wide));
    }
    catch (const std::invalid_argument &)
    {
        rejected_wide_matrix = true;
    }

    assert(rejected_wide_matrix);

    rejected_wide_matrix = false;
    try
    {
        static_cast<void>(mpi_friendly_svd(wide));
    }
    catch (const std::invalid_argument &)
    {
        rejected_wide_matrix = true;
    }

    assert(rejected_wide_matrix);

    rejected_wide_matrix = false;
    try
    {
        static_cast<void>(pthreads_svd(wide));
    }
    catch (const std::invalid_argument &)
    {
        rejected_wide_matrix = true;
    }

    assert(rejected_wide_matrix);

    rejected_wide_matrix = false;
    try
    {
        static_cast<void>(simd_svd(wide));
    }
    catch (const std::invalid_argument &)
    {
        rejected_wide_matrix = true;
    }

    assert(rejected_wide_matrix);

    return 0;
}
