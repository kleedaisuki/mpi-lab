#include "mpilab/domain/kernel.hpp"
#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/svd_convergence.hpp"

#include <cassert>
#include <cmath>

namespace
{
    /**
     * @brief 构造满秩测试矩阵。 Build a full-rank test matrix.
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

    /**
     * @brief 验证真实 SVD 结果的后验收敛性。 Verify a posteriori convergence for a real SVD result.
     */
    void test_converged_svd_result()
    {
        const mpilab::domain::OneSidedJacobiSvd svd;
        const mpilab::domain::OneSidedJacobiSvdOptions options{.max_sweeps = 100, .tolerance = 1.0e-10};
        const auto result = svd(make_full_rank_matrix(), options);
        const auto metrics = mpilab::infrastructure::evaluate_svd_convergence(result, options);

        assert(metrics.column_count == 2);
        assert(metrics.compared_pair_count == 1);
        assert(metrics.skipped_pair_count == 0);
        assert(metrics.zero_singular_value_count == 0);
        assert(metrics.reported_converged);
        assert(metrics.orthogonality_tolerance_satisfied);
        assert(metrics.converged);
        assert(!metrics.exhausted_sweep_budget);
        assert(metrics.max_column_correlation <= metrics.tolerance);
        assert(metrics.offdiagonal_gram_frobenius_norm <= std::sqrt(2.0) * metrics.tolerance);
    }

    /**
     * @brief 验证后验检查能抓住非正交左奇异向量。 Verify that the a posteriori check catches non-orthogonal left singular vectors.
     */
    void test_detects_orthogonality_violation()
    {
        mpilab::domain::OneSidedJacobiSvdResult result{
            .u = mpilab::domain::RowMajorMatrix<double>(2, 2),
            .singular_values = {2.0, 1.0},
            .v = mpilab::domain::RowMajorMatrix<double>(2, 2),
            .sweeps = 3,
            .converged = true,
        };
        result.u.set(0, 0, 1.0);
        result.u.set(0, 1, 0.0);
        result.u.set(1, 0, 1.0);
        result.u.set(1, 1, 0.0);
        result.v.set(0, 0, 1.0);
        result.v.set(0, 1, 0.0);
        result.v.set(1, 0, 0.0);
        result.v.set(1, 1, 1.0);

        const auto metrics = mpilab::infrastructure::evaluate_svd_convergence(
            result,
            mpilab::infrastructure::SvdConvergenceOptions{.tolerance = 1.0e-12, .max_sweeps = 10});

        assert(metrics.reported_converged);
        assert(!metrics.orthogonality_tolerance_satisfied);
        assert(!metrics.converged);
        assert(metrics.tolerance_violation_count == 1);
        assert(metrics.max_column_correlation == 1.0);
    }

    /**
     * @brief 验证零奇异值列会从左列正交性比较中跳过。 Verify that zero-singular-value columns are skipped for left-column orthogonality.
     */
    void test_skips_zero_singular_columns()
    {
        mpilab::domain::OneSidedJacobiSvdResult result{
            .u = mpilab::domain::RowMajorMatrix<double>(2, 2),
            .singular_values = {2.0, 0.0},
            .v = mpilab::domain::RowMajorMatrix<double>(2, 2),
            .sweeps = 10,
            .converged = false,
        };

        const auto metrics = mpilab::infrastructure::evaluate_svd_convergence(
            result,
            mpilab::infrastructure::SvdConvergenceOptions{.tolerance = 1.0e-12, .max_sweeps = 10});

        assert(metrics.zero_singular_value_count == 1);
        assert(metrics.compared_pair_count == 0);
        assert(metrics.skipped_pair_count == 1);
        assert(metrics.exhausted_sweep_budget);
        assert(!metrics.converged);
    }
} // namespace

/**
 * @brief SVD 收敛性指标测试入口。 SVD convergence metric test entry point.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
int main()
{
    test_converged_svd_result();
    test_detects_orthogonality_violation();
    test_skips_zero_singular_columns();
    return 0;
}
