#pragma once

#include <algorithm>
#include <bit>
#include <concepts>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>

/**
 * @file numerical_accuracy.hpp
 * @brief 数值正确性指标声明。 Numerical accuracy metric declarations.
 */

namespace mpilab::infrastructure
{
    /**
     * @brief 数值正确性指标计算选项。 Numerical accuracy metric calculation options.
     */
    struct NumericalAccuracyOptions final
    {
        /**
         * @brief 逐元素相对误差分母下界。 Element-wise relative-error denominator floor.
         *
         * @note 分母使用 max(abs(reference), relative_floor)。 / The denominator is max(abs(reference), relative_floor).
         */
        double relative_floor = 0.0;

        /**
         * @brief NRMSE 与 PSNR 的动态范围；NaN 表示使用参考值范围。 Dynamic range for NRMSE and PSNR; NaN means using the reference-value range.
         */
        double dynamic_range = std::numeric_limits<double>::quiet_NaN();
    };

    /**
     * @brief HPC 论文常用数值正确性指标。 Numerical accuracy metrics commonly reported in HPC papers.
     */
    struct NumericalAccuracyMetrics final
    {
        /**
         * @brief 输入元素总数。 Total number of input elements.
         */
        std::size_t element_count = 0;

        /**
         * @brief 参考值与被测值均为有限数的元素数。 Number of elements where both reference and measured values are finite.
         */
        std::size_t finite_pair_count = 0;

        /**
         * @brief 至少一侧为非有限数的元素数。 Number of elements where at least one side is non-finite.
         */
        std::size_t nonfinite_pair_count = 0;

        /**
         * @brief 出现 NaN 的元素对数量。 Number of element pairs containing NaN.
         */
        std::size_t nan_count = 0;

        /**
         * @brief 出现 Inf 的元素对数量。 Number of element pairs containing Inf.
         */
        std::size_t infinity_count = 0;

        /**
         * @brief 按 double 转换后完全相等的元素数。 Number of elements exactly equal after conversion to double.
         */
        std::size_t exact_match_count = 0;

        /**
         * @brief 最大 ULP 距离。 Maximum ULP distance.
         */
        std::uint64_t max_ulp_distance = 0;

        /**
         * @brief 误差一范数。 L1 norm of the error.
         */
        double l1_error_norm = 0.0;

        /**
         * @brief 误差二范数。 L2 norm of the error.
         */
        double l2_error_norm = 0.0;

        /**
         * @brief 误差无穷范数。 L-infinity norm of the error.
         */
        double linf_error_norm = 0.0;

        /**
         * @brief 参考值一范数。 L1 norm of the reference.
         */
        double reference_l1_norm = 0.0;

        /**
         * @brief 参考值二范数。 L2 norm of the reference.
         */
        double reference_l2_norm = 0.0;

        /**
         * @brief 参考值无穷范数。 L-infinity norm of the reference.
         */
        double reference_linf_norm = 0.0;

        /**
         * @brief 最大绝对误差。 Maximum absolute error.
         */
        double max_absolute_error = 0.0;

        /**
         * @brief 平均绝对误差。 Mean absolute error.
         */
        double mean_absolute_error = 0.0;

        /**
         * @brief 均方误差。 Mean squared error.
         */
        double mean_squared_error = 0.0;

        /**
         * @brief 均方根误差。 Root mean squared error.
         */
        double root_mean_squared_error = 0.0;

        /**
         * @brief 相对一范数误差。 Relative L1 error.
         */
        double relative_l1_error = 0.0;

        /**
         * @brief 相对二范数误差。 Relative L2 error.
         */
        double relative_l2_error = 0.0;

        /**
         * @brief 相对无穷范数误差。 Relative L-infinity error.
         */
        double relative_linf_error = 0.0;

        /**
         * @brief 最大逐元素相对误差。 Maximum element-wise relative error.
         */
        double max_relative_error = 0.0;

        /**
         * @brief 平均逐元素相对误差。 Mean element-wise relative error.
         */
        double mean_relative_error = 0.0;

        /**
         * @brief 归一化均方根误差。 Normalized root mean squared error.
         */
        double normalized_root_mean_squared_error = 0.0;

        /**
         * @brief 峰值信噪比，单位 dB。 Peak signal-to-noise ratio in dB.
         */
        double peak_signal_to_noise_ratio_db = std::numeric_limits<double>::infinity();
    };

    /**
     * @brief 计算两个线性数值序列之间的正确性指标。 Compute accuracy metrics between two linear numeric sequences.
     *
     * @tparam Reference 参考值标量类型。 / Reference scalar type.
     * @tparam Measured 被测值标量类型。 / Measured scalar type.
     * @param reference 参考序列。 / Reference sequence.
     * @param measured 被测序列。 / Measured sequence.
     * @param options 指标计算选项。 / Metric calculation options.
     * @return 数值正确性指标。 / Numerical accuracy metrics.
     * @throws std::invalid_argument 当序列尺寸不一致或为空时抛出。 / Throws when sequence sizes differ or are empty.
     */
    template <typename Reference, typename Measured>
        requires std::is_arithmetic_v<std::remove_cvref_t<Reference>> && std::is_arithmetic_v<std::remove_cvref_t<Measured>>
    [[nodiscard]] auto evaluate_numerical_accuracy(
        std::span<const Reference> reference,
        std::span<const Measured> measured,
        NumericalAccuracyOptions options = {}) -> NumericalAccuracyMetrics;

    /**
     * @brief 计算两个矩阵之间的正确性指标。 Compute accuracy metrics between two matrices.
     *
     * @tparam ReferenceMatrix 参考矩阵类型。 / Reference matrix type.
     * @tparam MeasuredMatrix 被测矩阵类型。 / Measured matrix type.
     * @param reference 参考矩阵。 / Reference matrix.
     * @param measured 被测矩阵。 / Measured matrix.
     * @param options 指标计算选项。 / Metric calculation options.
     * @return 数值正确性指标。 / Numerical accuracy metrics.
     * @throws std::invalid_argument 当矩阵尺寸不一致或为空时抛出。 / Throws when matrix sizes differ or are empty.
     */
    template <typename ReferenceMatrix, typename MeasuredMatrix>
        requires requires(const ReferenceMatrix &reference_matrix, const MeasuredMatrix &measured_matrix, std::size_t x, std::size_t y) {
            { reference_matrix.width() } -> std::convertible_to<std::size_t>;
            { reference_matrix.height() } -> std::convertible_to<std::size_t>;
            { measured_matrix.width() } -> std::convertible_to<std::size_t>;
            { measured_matrix.height() } -> std::convertible_to<std::size_t>;
            { reference_matrix(x, y) } -> std::convertible_to<double>;
            { measured_matrix(x, y) } -> std::convertible_to<double>;
        }
    [[nodiscard]] auto evaluate_numerical_accuracy(
        const ReferenceMatrix &reference,
        const MeasuredMatrix &measured,
        NumericalAccuracyOptions options = {}) -> NumericalAccuracyMetrics;

    namespace detail
    {
        /**
         * @brief 将 IEEE double 映射到可单调比较的 ULP 序。 Map an IEEE double to monotonic ULP order.
         *
         * @param value 输入值。 / Input value.
         * @return 单调 ULP 序值。 / Monotonic ULP-order value.
         */
        [[nodiscard]] inline auto ordered_double_bits(double value) noexcept -> std::uint64_t
        {
            constexpr std::uint64_t sign_bit = 0x8000000000000000ULL;
            const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
            if ((bits & sign_bit) != 0)
            {
                return (~bits) + 1U;
            }

            return sign_bit | bits;
        }

        /**
         * @brief 计算两个有限 double 的 ULP 距离。 Compute ULP distance between two finite doubles.
         *
         * @param reference 参考值。 / Reference value.
         * @param measured 被测值。 / Measured value.
         * @return ULP 距离。 / ULP distance.
         */
        [[nodiscard]] inline auto ulp_distance(double reference, double measured) noexcept -> std::uint64_t
        {
            const std::uint64_t left = ordered_double_bits(reference);
            const std::uint64_t right = ordered_double_bits(measured);
            return left > right ? left - right : right - left;
        }

        /**
         * @brief 计算非负分母上的比值。 Compute a ratio over a non-negative denominator.
         *
         * @param numerator 分子。 / Numerator.
         * @param denominator 分母。 / Denominator.
         * @return 比值、零或无穷大。 / Ratio, zero, or infinity.
         */
        [[nodiscard]] inline auto ratio_or_limit(double numerator, double denominator) noexcept -> double
        {
            if (denominator > 0.0)
            {
                return numerator / denominator;
            }

            return numerator == 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
        }

        /**
         * @brief 完成累积型指标。 Finalize accumulated metrics.
         *
         * @param metrics 待完成的指标。 / Metrics to finalize.
         * @param sum_squared_error 误差平方和。 / Sum of squared errors.
         * @param sum_squared_reference 参考值平方和。 / Sum of squared reference values.
         * @param sum_relative_error 逐元素相对误差和。 / Sum of element-wise relative errors.
         * @param reference_min 有限参考值最小值。 / Minimum finite reference value.
         * @param reference_max 有限参考值最大值。 / Maximum finite reference value.
         * @param options 指标计算选项。 / Metric calculation options.
         */
        inline void finalize_metrics(
            NumericalAccuracyMetrics &metrics,
            long double sum_squared_error,
            long double sum_squared_reference,
            long double sum_relative_error,
            double reference_min,
            double reference_max,
            NumericalAccuracyOptions options) noexcept
        {
            metrics.l2_error_norm = std::sqrt(static_cast<double>(sum_squared_error));
            metrics.reference_l2_norm = std::sqrt(static_cast<double>(sum_squared_reference));
            metrics.max_absolute_error = metrics.linf_error_norm;
            metrics.relative_l1_error = ratio_or_limit(metrics.l1_error_norm, metrics.reference_l1_norm);
            metrics.relative_l2_error = ratio_or_limit(metrics.l2_error_norm, metrics.reference_l2_norm);
            metrics.relative_linf_error = ratio_or_limit(metrics.linf_error_norm, metrics.reference_linf_norm);

            if (metrics.finite_pair_count == 0)
            {
                metrics.mean_absolute_error = std::numeric_limits<double>::quiet_NaN();
                metrics.mean_squared_error = std::numeric_limits<double>::quiet_NaN();
                metrics.root_mean_squared_error = std::numeric_limits<double>::quiet_NaN();
                metrics.mean_relative_error = std::numeric_limits<double>::quiet_NaN();
                metrics.normalized_root_mean_squared_error = std::numeric_limits<double>::quiet_NaN();
                metrics.peak_signal_to_noise_ratio_db = std::numeric_limits<double>::quiet_NaN();
                return;
            }

            const auto finite_count = static_cast<double>(metrics.finite_pair_count);
            metrics.mean_absolute_error = metrics.l1_error_norm / finite_count;
            metrics.mean_squared_error = static_cast<double>(sum_squared_error) / finite_count;
            metrics.root_mean_squared_error = std::sqrt(metrics.mean_squared_error);
            metrics.mean_relative_error = static_cast<double>(sum_relative_error) / finite_count;

            double range = options.dynamic_range;
            if (std::isnan(range))
            {
                range = reference_max - reference_min;
            }

            metrics.normalized_root_mean_squared_error = ratio_or_limit(metrics.root_mean_squared_error, std::abs(range));
            if (metrics.root_mean_squared_error == 0.0)
            {
                metrics.peak_signal_to_noise_ratio_db = std::numeric_limits<double>::infinity();
            }
            else if (std::abs(range) > 0.0)
            {
                metrics.peak_signal_to_noise_ratio_db = 20.0 * std::log10(std::abs(range) / metrics.root_mean_squared_error);
            }
            else
            {
                metrics.peak_signal_to_noise_ratio_db = std::numeric_limits<double>::quiet_NaN();
            }
        }
    } // namespace detail
} // namespace mpilab::infrastructure

namespace mpilab::infrastructure
{
    template <typename Reference, typename Measured>
        requires std::is_arithmetic_v<std::remove_cvref_t<Reference>> && std::is_arithmetic_v<std::remove_cvref_t<Measured>>
    auto evaluate_numerical_accuracy(
        std::span<const Reference> reference,
        std::span<const Measured> measured,
        NumericalAccuracyOptions options) -> NumericalAccuracyMetrics
    {
        if (reference.size() != measured.size())
        {
            throw std::invalid_argument("numerical accuracy requires equal sequence sizes");
        }
        if (reference.empty())
        {
            throw std::invalid_argument("numerical accuracy requires at least one element");
        }

        NumericalAccuracyMetrics metrics;
        metrics.element_count = reference.size();

        long double sum_squared_error = 0.0L;
        long double sum_squared_reference = 0.0L;
        long double sum_relative_error = 0.0L;
        double reference_min = std::numeric_limits<double>::infinity();
        double reference_max = -std::numeric_limits<double>::infinity();

        for (std::size_t index = 0; index < reference.size(); ++index)
        {
            const double expected = static_cast<double>(reference[index]);
            const double actual = static_cast<double>(measured[index]);
            if (expected == actual)
            {
                ++metrics.exact_match_count;
            }

            if (std::isnan(expected) || std::isnan(actual))
            {
                ++metrics.nan_count;
            }
            if (std::isinf(expected) || std::isinf(actual))
            {
                ++metrics.infinity_count;
            }
            if (!std::isfinite(expected) || !std::isfinite(actual))
            {
                ++metrics.nonfinite_pair_count;
                continue;
            }

            ++metrics.finite_pair_count;
            metrics.max_ulp_distance = std::max(metrics.max_ulp_distance, detail::ulp_distance(expected, actual));

            const double absolute_reference = std::abs(expected);
            const double absolute_error = std::abs(actual - expected);
            const double relative_denominator = std::max(absolute_reference, options.relative_floor);
            const double relative_error = detail::ratio_or_limit(absolute_error, relative_denominator);

            metrics.l1_error_norm += absolute_error;
            sum_squared_error += static_cast<long double>(absolute_error) * static_cast<long double>(absolute_error);
            metrics.linf_error_norm = std::max(metrics.linf_error_norm, absolute_error);
            metrics.reference_l1_norm += absolute_reference;
            sum_squared_reference += static_cast<long double>(expected) * static_cast<long double>(expected);
            metrics.reference_linf_norm = std::max(metrics.reference_linf_norm, absolute_reference);
            metrics.max_relative_error = std::max(metrics.max_relative_error, relative_error);
            sum_relative_error += relative_error;
            reference_min = std::min(reference_min, expected);
            reference_max = std::max(reference_max, expected);
        }

        detail::finalize_metrics(metrics, sum_squared_error, sum_squared_reference, sum_relative_error, reference_min, reference_max, options);
        return metrics;
    }

    template <typename ReferenceMatrix, typename MeasuredMatrix>
        requires requires(const ReferenceMatrix &reference_matrix, const MeasuredMatrix &measured_matrix, std::size_t x, std::size_t y) {
            { reference_matrix.width() } -> std::convertible_to<std::size_t>;
            { reference_matrix.height() } -> std::convertible_to<std::size_t>;
            { measured_matrix.width() } -> std::convertible_to<std::size_t>;
            { measured_matrix.height() } -> std::convertible_to<std::size_t>;
            { reference_matrix(x, y) } -> std::convertible_to<double>;
            { measured_matrix(x, y) } -> std::convertible_to<double>;
        }
    auto evaluate_numerical_accuracy(
        const ReferenceMatrix &reference,
        const MeasuredMatrix &measured,
        NumericalAccuracyOptions options) -> NumericalAccuracyMetrics
    {
        const std::size_t width = static_cast<std::size_t>(reference.width());
        const std::size_t height = static_cast<std::size_t>(reference.height());
        if (width != static_cast<std::size_t>(measured.width()) || height != static_cast<std::size_t>(measured.height()))
        {
            throw std::invalid_argument("numerical accuracy requires equal matrix sizes");
        }
        if (width == 0 || height == 0)
        {
            throw std::invalid_argument("numerical accuracy requires at least one matrix element");
        }

        NumericalAccuracyMetrics metrics;
        metrics.element_count = width * height;

        long double sum_squared_error = 0.0L;
        long double sum_squared_reference = 0.0L;
        long double sum_relative_error = 0.0L;
        double reference_min = std::numeric_limits<double>::infinity();
        double reference_max = -std::numeric_limits<double>::infinity();

        for (std::size_t y = 0; y < height; ++y)
        {
            for (std::size_t x = 0; x < width; ++x)
            {
                const double expected = static_cast<double>(reference(x, y));
                const double actual = static_cast<double>(measured(x, y));
                if (expected == actual)
                {
                    ++metrics.exact_match_count;
                }

                if (std::isnan(expected) || std::isnan(actual))
                {
                    ++metrics.nan_count;
                }
                if (std::isinf(expected) || std::isinf(actual))
                {
                    ++metrics.infinity_count;
                }
                if (!std::isfinite(expected) || !std::isfinite(actual))
                {
                    ++metrics.nonfinite_pair_count;
                    continue;
                }

                ++metrics.finite_pair_count;
                metrics.max_ulp_distance = std::max(metrics.max_ulp_distance, detail::ulp_distance(expected, actual));

                const double absolute_reference = std::abs(expected);
                const double absolute_error = std::abs(actual - expected);
                const double relative_denominator = std::max(absolute_reference, options.relative_floor);
                const double relative_error = detail::ratio_or_limit(absolute_error, relative_denominator);

                metrics.l1_error_norm += absolute_error;
                sum_squared_error += static_cast<long double>(absolute_error) * static_cast<long double>(absolute_error);
                metrics.linf_error_norm = std::max(metrics.linf_error_norm, absolute_error);
                metrics.reference_l1_norm += absolute_reference;
                sum_squared_reference += static_cast<long double>(expected) * static_cast<long double>(expected);
                metrics.reference_linf_norm = std::max(metrics.reference_linf_norm, absolute_reference);
                metrics.max_relative_error = std::max(metrics.max_relative_error, relative_error);
                sum_relative_error += relative_error;
                reference_min = std::min(reference_min, expected);
                reference_max = std::max(reference_max, expected);
            }
        }

        detail::finalize_metrics(metrics, sum_squared_error, sum_squared_reference, sum_relative_error, reference_min, reference_max, options);
        return metrics;
    }
} // namespace mpilab::infrastructure
