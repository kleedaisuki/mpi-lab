#include "mpilab/application/pipeline.hpp"

#include "mpilab/application/mpi_scheduler.hpp"
#include "mpilab/domain/kernel.hpp"
#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/file_stream.hpp"

#include <cstddef>
#include <vector>

/**
 * @file pipeline.cpp
 * @brief 应用流水线实现。 Application pipeline implementation.
 */

namespace mpilab::application
{
    namespace
    {
        /**
         * @brief 将任意矩阵布局复制为行主序矩阵。 Copy any matrix layout into a row-major matrix.
         *
         * @tparam Matrix 源矩阵类型。 / Source matrix type.
         * @param matrix 源矩阵。 / Source matrix.
         * @return 行主序矩阵副本。 / Row-major matrix copy.
         */
        template <domain::MatrixLike Matrix>
        [[nodiscard]] auto to_row_major(const Matrix &matrix) -> domain::RowMajorMatrix<double>
        {
            domain::RowMajorMatrix<double> row_major(matrix.width(), matrix.height());
            for (std::size_t y = 0; y < matrix.height(); ++y)
            {
                for (std::size_t x = 0; x < matrix.width(); ++x)
                {
                    row_major.set(x, y, static_cast<double>(matrix(x, y)));
                }
            }

            return row_major;
        }

        /**
         * @brief 读取下一块矩阵并正规化为行主序矩阵。 Read the next matrix and normalize it to row-major layout.
         *
         * @tparam Matrix 输入布局矩阵类型。 / Input layout matrix type.
         * @param reader 矩阵文件读取器。 / Matrix file reader.
         * @return 行主序矩阵。 / Row-major matrix.
         */
        template <domain::MatrixLike Matrix>
        [[nodiscard]] auto read_row_major(infrastructure::MatrixFileReader &reader) -> domain::RowMajorMatrix<double>
        {
            return to_row_major(reader.next<Matrix>());
        }

        /**
         * @brief 按配置读取下一块样本矩阵。 Read the next sample matrix according to configuration.
         *
         * @param reader 矩阵文件读取器。 / Matrix file reader.
         * @param layout 输入布局选择。 / Input layout selection.
         * @return 行主序样本矩阵。 / Row-major sample matrix.
         */
        [[nodiscard]] auto read_sample(infrastructure::MatrixFileReader &reader, MatrixLayout layout) -> domain::RowMajorMatrix<double>
        {
            switch (layout)
            {
            case MatrixLayout::row_major:
                return read_row_major<domain::RowMajorMatrix<double>>(reader);
            case MatrixLayout::column_major:
                return read_row_major<domain::ColumnMajorMatrix<double>>(reader);
            case MatrixLayout::strided_row_major:
                return read_row_major<domain::StridedRowMajorMatrix<double>>(reader);
            case MatrixLayout::jagged_row_major:
                return read_row_major<domain::JaggedRowMajorMatrix<double>>(reader);
            case MatrixLayout::blocked_row_major:
                return read_row_major<domain::BlockedRowMajorMatrix<double>>(reader);
            case MatrixLayout::morton:
                return read_row_major<domain::MortonMatrix<double>>(reader);
            }

            return read_row_major<domain::RowMajorMatrix<double>>(reader);
        }

        /**
         * @brief 按配置执行 SVD 算子。 Execute an SVD kernel according to configuration.
         *
         * @param matrix 输入矩阵。 / Input matrix.
         * @param kernel 算子选择。 / Kernel selection.
         * @param options SVD 迭代选项。 / SVD iteration options.
         * @return SVD 结果。 / SVD result.
         */
        [[nodiscard]] auto compute_svd(
            const domain::RowMajorMatrix<double> &matrix,
            SvdKernel kernel,
            const domain::OneSidedJacobiSvdOptions &options) -> domain::OneSidedJacobiSvdResult
        {
            switch (kernel)
            {
            case SvdKernel::naive:
                return domain::OneSidedJacobiSvd{}(matrix, options);
            case SvdKernel::advanced:
                return domain::AdvancedOneSidedJacobiSvd{}(matrix, options);
            case SvdKernel::mpi_friendly:
                return domain::MpiFriendlyOneSidedJacobiSvd{}(matrix, options);
            case SvdKernel::pthreads:
                return domain::PthreadsOneSidedJacobiSvd{}(matrix, options);
            case SvdKernel::simd:
                return domain::SimdOneSidedJacobiSvd{}(matrix, options);
            }

            return domain::OneSidedJacobiSvd{}(matrix, options);
        }

        /**
         * @brief 将奇异值向量转换为单行矩阵。 Convert singular values into a one-row matrix.
         *
         * @param singular_values 奇异值向量。 / Singular value vector.
         * @return 单行奇异值矩阵。 / One-row singular-value matrix.
         */
        [[nodiscard]] auto singular_values_to_matrix(const std::vector<double> &singular_values) -> domain::RowMajorMatrix<double>
        {
            domain::RowMajorMatrix<double> matrix(singular_values.size(), 1);
            for (std::size_t index = 0; index < singular_values.size(); ++index)
            {
                matrix.set(index, 0, singular_values[index]);
            }

            return matrix;
        }

        /**
         * @brief 将 SVD 结果追加到输出矩阵序列。 Append an SVD result to the output matrix sequence.
         *
         * @param result SVD 结果。 / SVD result.
         * @param output 输出矩阵序列。 / Output matrix sequence.
         */
        void append_result_matrices(const domain::OneSidedJacobiSvdResult &result, std::vector<domain::RowMajorMatrix<double>> &output)
        {
            output.push_back(result.u);
            output.push_back(singular_values_to_matrix(result.singular_values));
            output.push_back(result.v);
        }
    } // namespace

    auto run_pipeline(const PipelineConfig &config) -> PipelineReport
    {
        MpiExecutionScope mpi = MpiExecutionScope::create(config.enable_mpi, config.argc, config.argv);

        PipelineReport report;
        report.mpi_enabled = mpi.enabled();
        report.rank = mpi.rank();
        report.size = mpi.size();

        std::vector<domain::RowMajorMatrix<double>> output;

        infrastructure::MatrixFileReader reader(config.input_path);
        while (reader.has_next())
        {
            const domain::RowMajorMatrix<double> sample = read_sample(reader, config.layout);
            const domain::OneSidedJacobiSvdResult result = compute_svd(sample, config.kernel, config.svd_options);
            ++report.samples_read;

            if (mpi.writes_output())
            {
                append_result_matrices(result, output);
                ++report.results_written;
            }
        }

        if (mpi.writes_output())
        {
            infrastructure::write_matrices(config.output_path, output);
        }

        mpi.barrier();
        return report;
    }

} // namespace mpilab::application
