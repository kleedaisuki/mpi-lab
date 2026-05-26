#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace mpilab::domain
{
    namespace
    {

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
        [[nodiscard]] auto column_dot(const RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right) -> double
        {
            double sum = 0.0;
            for (std::size_t y = 0; y < matrix.height(); ++y)
            {
                sum += matrix(left, y) * matrix(right, y);
            }
            return sum;
        }

        /**
         * @brief 计算列向量的 Euclidean norm。 Compute the Euclidean norm of a column.
         *
         * @param matrix 输入矩阵。 / Input matrix.
         * @param column 列索引。 / Column index.
         * @return 列范数。 / Column norm.
         */
        [[nodiscard]] auto column_norm(const RowMajorMatrix<double> &matrix, std::size_t column) -> double
        {
            return std::sqrt(column_dot(matrix, column, column));
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
        void rotate_columns(RowMajorMatrix<double> &matrix, std::size_t left, std::size_t right, double cosine, double sine)
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

    auto OneSidedJacobiSvd::operator()(const RowMajorMatrix<double> &matrix, const OneSidedJacobiSvdOptions &options) const -> OneSidedJacobiSvdResult
    {
        const std::size_t column_count = matrix.width();
        const std::size_t row_count = matrix.height();
        const double tolerance = std::max(options.tolerance, std::numeric_limits<double>::epsilon());

        if (row_count < column_count)
        {
            throw std::invalid_argument("OneSidedJacobiSvd requires matrix.height() >= matrix.width()");
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

        bool converged = true;
        std::size_t completed_sweeps = 0;

        for (; completed_sweeps < options.max_sweeps; ++completed_sweeps)
        {
            bool rotated = false;

            for (std::size_t left = 0; left < column_count; ++left)
            {
                for (std::size_t right = left + 1; right < column_count; ++right)
                {
                    const double left_norm_sq = column_dot(work, left, left);
                    const double right_norm_sq = column_dot(work, right, right);
                    const double cross_dot = column_dot(work, left, right);
                    const double scale = std::sqrt(left_norm_sq * right_norm_sq);

                    if (scale == 0.0 || std::abs(cross_dot) <= tolerance * scale)
                    {
                        continue;
                    }

                    const auto [cosine, sine] = jacobi_rotation(left_norm_sq, right_norm_sq, cross_dot);

                    rotate_columns(work, left, right, cosine, sine);
                    rotate_columns(right_vectors, left, right, cosine, sine);
                    rotated = true;
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
            const double singular_value = column_norm(work, column);
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

} // namespace mpilab::domain
