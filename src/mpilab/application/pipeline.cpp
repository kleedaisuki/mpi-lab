#include "mpilab/application/pipeline.hpp"

#include "mpilab/application/mpi_scheduler.hpp"
#include "mpilab/domain/kernel.hpp"
#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/file_stream.hpp"
#include "mpilab/infrastructure/numerical_accuracy.hpp"
#include "mpilab/infrastructure/svd_convergence.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <ios>
#include <limits>
#include <memory>
#include <ostream>
#include <stdexcept>
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

        /**
         * @brief 将矩阵布局转换为稳定文本。 Convert a matrix layout to stable text.
         *
         * @param layout 矩阵布局。 / Matrix layout.
         * @return 矩阵布局文本。 / Matrix-layout text.
         */
        [[nodiscard]] auto to_string(MatrixLayout layout) -> const char *
        {
            switch (layout)
            {
            case MatrixLayout::row_major:
                return "row_major";
            case MatrixLayout::column_major:
                return "column_major";
            case MatrixLayout::strided_row_major:
                return "strided_row_major";
            case MatrixLayout::jagged_row_major:
                return "jagged_row_major";
            case MatrixLayout::blocked_row_major:
                return "blocked_row_major";
            case MatrixLayout::morton:
                return "morton";
            }

            return "row_major";
        }

        /**
         * @brief 将 SVD 算子转换为稳定文本。 Convert an SVD kernel to stable text.
         *
         * @param kernel SVD 算子。 / SVD kernel.
         * @return SVD 算子文本。 / SVD-kernel text.
         */
        [[nodiscard]] auto to_string(SvdKernel kernel) -> const char *
        {
            switch (kernel)
            {
            case SvdKernel::naive:
                return "naive";
            case SvdKernel::advanced:
                return "advanced";
            case SvdKernel::mpi_friendly:
                return "mpi_friendly";
            case SvdKernel::pthreads:
                return "pthreads";
            case SvdKernel::simd:
                return "simd";
            }

            return "naive";
        }

        /**
         * @brief 从 SVD 结果重构矩阵。 Reconstruct a matrix from an SVD result.
         *
         * @param result SVD 结果。 / SVD result.
         * @param width 原矩阵宽度。 / Original matrix width.
         * @param height 原矩阵高度。 / Original matrix height.
         * @return 重构矩阵。 / Reconstructed matrix.
         */
        [[nodiscard]] auto reconstruct_matrix(const domain::OneSidedJacobiSvdResult &result, std::size_t width, std::size_t height) -> domain::RowMajorMatrix<double>
        {
            domain::RowMajorMatrix<double> reconstructed(width, height);
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
         * @brief 写出 JSON 数值。 Write a JSON numeric value.
         *
         * @param output 输出流。 / Output stream.
         * @param name 字段名。 / Field name.
         * @param value 字段值。 / Field value.
         */
        void write_json_number_field(std::ostream &output, const char *name, double value)
        {
            output << '"' << name << "\":";
            if (std::isfinite(value))
            {
                output << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
                return;
            }

            output << "null";
        }

        /**
         * @brief 写出 JSON 布尔字段。 Write a JSON boolean field.
         *
         * @param output 输出流。 / Output stream.
         * @param name 字段名。 / Field name.
         * @param value 字段值。 / Field value.
         */
        void write_json_bool_field(std::ostream &output, const char *name, bool value)
        {
            output << '"' << name << "\":" << (value ? "true" : "false");
        }

        /**
         * @brief 写出 JSON 无符号整数字段。 Write a JSON unsigned-integer field.
         *
         * @param output 输出流。 / Output stream.
         * @param name 字段名。 / Field name.
         * @param value 字段值。 / Field value.
         */
        void write_json_size_field(std::ostream &output, const char *name, std::size_t value)
        {
            output << '"' << name << "\":" << value;
        }

        /**
         * @brief 写出数值正确性指标对象。 Write a numerical accuracy metric object.
         *
         * @param output 输出流。 / Output stream.
         * @param metrics 数值正确性指标。 / Numerical accuracy metrics.
         */
        void write_accuracy_metrics(std::ostream &output, const infrastructure::NumericalAccuracyMetrics &metrics)
        {
            output << '{';
            write_json_size_field(output, "element_count", metrics.element_count);
            output << ',';
            write_json_size_field(output, "finite_pair_count", metrics.finite_pair_count);
            output << ',';
            write_json_size_field(output, "nonfinite_pair_count", metrics.nonfinite_pair_count);
            output << ',';
            write_json_size_field(output, "nan_count", metrics.nan_count);
            output << ',';
            write_json_size_field(output, "infinity_count", metrics.infinity_count);
            output << ',';
            write_json_size_field(output, "exact_match_count", metrics.exact_match_count);
            output << ",\"max_ulp_distance\":" << metrics.max_ulp_distance << ',';
            write_json_number_field(output, "l1_error_norm", metrics.l1_error_norm);
            output << ',';
            write_json_number_field(output, "l2_error_norm", metrics.l2_error_norm);
            output << ',';
            write_json_number_field(output, "linf_error_norm", metrics.linf_error_norm);
            output << ',';
            write_json_number_field(output, "reference_l1_norm", metrics.reference_l1_norm);
            output << ',';
            write_json_number_field(output, "reference_l2_norm", metrics.reference_l2_norm);
            output << ',';
            write_json_number_field(output, "reference_linf_norm", metrics.reference_linf_norm);
            output << ',';
            write_json_number_field(output, "max_absolute_error", metrics.max_absolute_error);
            output << ',';
            write_json_number_field(output, "mean_absolute_error", metrics.mean_absolute_error);
            output << ',';
            write_json_number_field(output, "mean_squared_error", metrics.mean_squared_error);
            output << ',';
            write_json_number_field(output, "root_mean_squared_error", metrics.root_mean_squared_error);
            output << ',';
            write_json_number_field(output, "relative_l1_error", metrics.relative_l1_error);
            output << ',';
            write_json_number_field(output, "relative_l2_error", metrics.relative_l2_error);
            output << ',';
            write_json_number_field(output, "relative_linf_error", metrics.relative_linf_error);
            output << ',';
            write_json_number_field(output, "max_relative_error", metrics.max_relative_error);
            output << ',';
            write_json_number_field(output, "mean_relative_error", metrics.mean_relative_error);
            output << ',';
            write_json_number_field(output, "normalized_root_mean_squared_error", metrics.normalized_root_mean_squared_error);
            output << ',';
            write_json_number_field(output, "peak_signal_to_noise_ratio_db", metrics.peak_signal_to_noise_ratio_db);
            output << '}';
        }

        /**
         * @brief 写出 SVD 收敛性指标对象。 Write an SVD convergence metric object.
         *
         * @param output 输出流。 / Output stream.
         * @param metrics SVD 收敛性指标。 / SVD convergence metrics.
         */
        void write_convergence_metrics(std::ostream &output, const infrastructure::SvdConvergenceMetrics &metrics)
        {
            output << '{';
            write_json_size_field(output, "column_count", metrics.column_count);
            output << ',';
            write_json_size_field(output, "compared_pair_count", metrics.compared_pair_count);
            output << ',';
            write_json_size_field(output, "skipped_pair_count", metrics.skipped_pair_count);
            output << ',';
            write_json_size_field(output, "zero_singular_value_count", metrics.zero_singular_value_count);
            output << ',';
            write_json_size_field(output, "tolerance_violation_count", metrics.tolerance_violation_count);
            output << ',';
            write_json_size_field(output, "sweeps", metrics.sweeps);
            output << ',';
            write_json_size_field(output, "max_sweeps", metrics.max_sweeps);
            output << ',';
            write_json_number_field(output, "tolerance", metrics.tolerance);
            output << ',';
            write_json_number_field(output, "sweep_budget_fraction", metrics.sweep_budget_fraction);
            output << ',';
            write_json_bool_field(output, "reported_converged", metrics.reported_converged);
            output << ',';
            write_json_bool_field(output, "exhausted_sweep_budget", metrics.exhausted_sweep_budget);
            output << ',';
            write_json_bool_field(output, "orthogonality_tolerance_satisfied", metrics.orthogonality_tolerance_satisfied);
            output << ',';
            write_json_bool_field(output, "converged", metrics.converged);
            output << ',';
            write_json_number_field(output, "max_column_correlation", metrics.max_column_correlation);
            output << ',';
            write_json_number_field(output, "rms_column_correlation", metrics.rms_column_correlation);
            output << ',';
            write_json_number_field(output, "offdiagonal_gram_frobenius_norm", metrics.offdiagonal_gram_frobenius_norm);
            output << ',';
            write_json_number_field(output, "right_offdiagonal_gram_frobenius_norm", metrics.right_offdiagonal_gram_frobenius_norm);
            output << '}';
        }

        /**
         * @brief 写出单个样本的 JSONL 指标记录。 Write one JSONL metric record for a sample.
         *
         * @param output 输出流。 / Output stream.
         * @param sample_index 样本索引。 / Sample index.
         * @param sample 输入样本。 / Input sample.
         * @param config 流水线配置。 / Pipeline configuration.
         * @param result SVD 结果。 / SVD result.
         */
        void write_metric_record(
            std::ostream &output,
            std::size_t sample_index,
            const domain::RowMajorMatrix<double> &sample,
            const PipelineConfig &config,
            const domain::OneSidedJacobiSvdResult &result)
        {
            const domain::RowMajorMatrix<double> reconstructed = reconstruct_matrix(result, sample.width(), sample.height());
            const infrastructure::NumericalAccuracyMetrics accuracy = infrastructure::evaluate_numerical_accuracy(sample, reconstructed);
            const infrastructure::SvdConvergenceMetrics convergence = infrastructure::evaluate_svd_convergence(result, config.svd_options);

            output << "{\"sample_index\":" << sample_index << ",\"input\":{\"width\":" << sample.width() << ",\"height\":" << sample.height()
                   << "},\"config\":{\"layout\":\"" << to_string(config.layout) << "\",\"kernel\":\"" << to_string(config.kernel)
                   << "\",\"max_sweeps\":" << config.svd_options.max_sweeps << ',';
            write_json_number_field(output, "tolerance", config.svd_options.tolerance);
            output << "},\"accuracy\":";
            write_accuracy_metrics(output, accuracy);
            output << ",\"convergence\":";
            write_convergence_metrics(output, convergence);
            output << "}\n";
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
        std::unique_ptr<std::ofstream> metrics_output;
        if (mpi.writes_output() && config.metrics_path.has_value())
        {
            metrics_output = std::make_unique<std::ofstream>(config.metrics_path.value(), std::ios::binary | std::ios::trunc);
            if (!*metrics_output)
            {
                throw std::runtime_error("failed to open pipeline metrics JSONL file");
            }
        }

        infrastructure::MatrixFileReader reader(config.input_path);
        while (reader.has_next())
        {
            const domain::RowMajorMatrix<double> sample = read_sample(reader, config.layout);
            const domain::OneSidedJacobiSvdResult result = compute_svd(sample, config.kernel, config.svd_options);
            const std::size_t sample_index = report.samples_read;
            ++report.samples_read;

            if (mpi.writes_output())
            {
                append_result_matrices(result, output);
                ++report.results_written;
                if (metrics_output)
                {
                    write_metric_record(*metrics_output, sample_index, sample, config, result);
                    ++report.metrics_written;
                }
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
