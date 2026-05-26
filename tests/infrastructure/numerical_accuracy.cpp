#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/numerical_accuracy.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace
{
    /**
     * @brief 默认测试容差。 / Default test tolerance.
     */
    constexpr double tolerance = 1.0e-12;

    /**
     * @brief 判断两个数值是否近似相等。 Check whether two values are approximately equal.
     *
     * @param left 左值。 / Left value.
     * @param right 右值。 / Right value.
     * @return 近似相等时返回 true。 / Returns true when the values are approximately equal.
     */
    [[nodiscard]] auto close_to(double left, double right) -> bool
    {
        return std::abs(left - right) <= tolerance;
    }

    /**
     * @brief 验证线性序列指标覆盖论文常用误差。 Verify common paper-style metrics for linear sequences.
     */
    void test_sequence_metrics()
    {
        const std::vector<double> reference{1.0, 2.0, 4.0, 8.0};
        const std::vector<double> measured{1.0, 3.0, 2.0, 12.0};

        const auto metrics = mpilab::infrastructure::evaluate_numerical_accuracy(
            std::span<const double>(reference.data(), reference.size()),
            std::span<const double>(measured.data(), measured.size()));

        assert(metrics.element_count == 4);
        assert(metrics.finite_pair_count == 4);
        assert(metrics.nonfinite_pair_count == 0);
        assert(metrics.exact_match_count == 1);
        assert(close_to(metrics.l1_error_norm, 7.0));
        assert(close_to(metrics.l2_error_norm, std::sqrt(21.0)));
        assert(close_to(metrics.linf_error_norm, 4.0));
        assert(close_to(metrics.reference_l1_norm, 15.0));
        assert(close_to(metrics.reference_l2_norm, std::sqrt(85.0)));
        assert(close_to(metrics.reference_linf_norm, 8.0));
        assert(close_to(metrics.max_absolute_error, 4.0));
        assert(close_to(metrics.mean_absolute_error, 1.75));
        assert(close_to(metrics.mean_squared_error, 5.25));
        assert(close_to(metrics.root_mean_squared_error, std::sqrt(5.25)));
        assert(close_to(metrics.relative_l1_error, 7.0 / 15.0));
        assert(close_to(metrics.relative_l2_error, std::sqrt(21.0) / std::sqrt(85.0)));
        assert(close_to(metrics.relative_linf_error, 0.5));
        assert(close_to(metrics.max_relative_error, 0.5));
        assert(close_to(metrics.mean_relative_error, 0.375));
        assert(close_to(metrics.normalized_root_mean_squared_error, std::sqrt(5.25) / 7.0));
    }

    /**
     * @brief 验证矩阵重载与动态范围选项。 Verify the matrix overload and dynamic-range option.
     */
    void test_matrix_metrics()
    {
        mpilab::domain::RowMajorMatrix<double> reference(2, 2);
        reference.set(0, 0, 0.0);
        reference.set(1, 0, 10.0);
        reference.set(0, 1, 20.0);
        reference.set(1, 1, 30.0);

        mpilab::domain::RowMajorMatrix<double> measured(2, 2);
        measured.set(0, 0, 0.0);
        measured.set(1, 0, 11.0);
        measured.set(0, 1, 18.0);
        measured.set(1, 1, 33.0);

        mpilab::infrastructure::NumericalAccuracyOptions options;
        options.dynamic_range = 30.0;
        const auto metrics = mpilab::infrastructure::evaluate_numerical_accuracy(reference, measured, options);

        assert(metrics.element_count == 4);
        assert(metrics.finite_pair_count == 4);
        assert(close_to(metrics.root_mean_squared_error, std::sqrt(14.0 / 4.0)));
        assert(close_to(metrics.normalized_root_mean_squared_error, std::sqrt(14.0 / 4.0) / 30.0));
        assert(metrics.peak_signal_to_noise_ratio_db > 24.0);
    }

    /**
     * @brief 验证非有限数与零参考值下的相对误差处理。 Verify non-finite and zero-reference relative-error handling.
     */
    void test_nonfinite_and_zero_reference_metrics()
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double infinity = std::numeric_limits<double>::infinity();
        const std::vector<double> reference{0.0, 1.0, 2.0, 3.0};
        const std::vector<double> measured{1.0, nan, infinity, 4.0};

        mpilab::infrastructure::NumericalAccuracyOptions options;
        options.relative_floor = 1.0;
        const auto metrics = mpilab::infrastructure::evaluate_numerical_accuracy(
            std::span<const double>(reference.data(), reference.size()),
            std::span<const double>(measured.data(), measured.size()),
            options);

        assert(metrics.element_count == 4);
        assert(metrics.finite_pair_count == 2);
        assert(metrics.nonfinite_pair_count == 2);
        assert(metrics.nan_count == 1);
        assert(metrics.infinity_count == 1);
        assert(close_to(metrics.l1_error_norm, 2.0));
        assert(close_to(metrics.max_relative_error, 1.0));
        assert(close_to(metrics.mean_relative_error, (1.0 + (1.0 / 3.0)) / 2.0));
    }

    /**
     * @brief 验证非法尺寸会失败。 Verify that invalid sizes fail.
     */
    void test_invalid_sizes()
    {
        const std::vector<double> reference{1.0};
        const std::vector<double> measured{1.0, 2.0};

        bool threw = false;
        try
        {
            static_cast<void>(mpilab::infrastructure::evaluate_numerical_accuracy(
                std::span<const double>(reference.data(), reference.size()),
                std::span<const double>(measured.data(), measured.size())));
        }
        catch (const std::invalid_argument&)
        {
            threw = true;
        }
        assert(threw);
    }
} // namespace

/**
 * @brief 数值正确性指标测试入口。 Numerical accuracy metric test entry point.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
int main()
{
    test_sequence_metrics();
    test_matrix_metrics();
    test_nonfinite_and_zero_reference_metrics();
    test_invalid_sizes();
    return 0;
}
