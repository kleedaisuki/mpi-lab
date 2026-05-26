#include "mpilab/application/pipeline.hpp"

#include "mpilab/application/mpi_scheduler.hpp"
#include "mpilab/domain/kernel.hpp"
#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/file_stream.hpp"
#include "mpilab/infrastructure/logger.hpp"
#include "mpilab/infrastructure/numerical_accuracy.hpp"
#include "mpilab/infrastructure/svd_convergence.hpp"

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <ios>
#include <limits>
#include <ostream>
#include <sstream>
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
         * @brief 按配置执行 SVD 算子。 Execute an SVD kernel according to configuration.
         *
         * @param matrix 输入矩阵。 / Input matrix.
         * @param kernel 算子选择。 / Kernel selection.
         * @param options SVD 迭代选项。 / SVD iteration options.
         * @return SVD 结果。 / SVD result.
         */
        /**
         * @brief 按配置执行 SVD 算子。 Execute an SVD kernel according to configuration.
         *
         * @tparam Matrix 输入矩阵类型。 / Input matrix type.
         * @param matrix 输入矩阵。 / Input matrix.
         * @param kernel 算子选择。 / Kernel selection.
         * @param options SVD 迭代选项。 / SVD iteration options.
         * @return SVD 结果。 / SVD result.
         */
        template <domain::MatrixLike Matrix>
        [[nodiscard]] auto compute_svd(
            const Matrix &matrix,
            SvdKernel kernel,
            const domain::OneSidedJacobiSvdOptions &options)
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
        template <domain::MatrixLike Matrix>
        [[nodiscard]] auto singular_values_to_matrix(const std::vector<double> &singular_values) -> Matrix
        {
            Matrix matrix(singular_values.size(), 1);
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
        template <domain::MatrixLike Matrix>
        void append_result_matrices(const domain::OneSidedJacobiSvdResult<Matrix> &result, std::vector<Matrix> &output)
        {
            output.push_back(result.u);
            output.push_back(singular_values_to_matrix<Matrix>(result.singular_values));
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
        template <domain::MatrixLike Matrix>
        [[nodiscard]] auto reconstruct_matrix(const domain::OneSidedJacobiSvdResult<Matrix> &result, std::size_t width, std::size_t height) -> Matrix
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
         * @brief 创建单个样本的 JSONL 指标记录。 Create one JSONL metric record for a sample.
         *
         * @param sample_index 样本索引。 / Sample index.
         * @tparam Matrix 输入样本矩阵类型。 / Input sample matrix type.
         * @param sample 输入样本。 / Input sample.
         * @param config 流水线配置。 / Pipeline configuration.
         * @param result SVD 结果。 / SVD result.
         * @return JSONL 记录文本。 / JSONL record text.
         */
        template <domain::MatrixLike Matrix>
        [[nodiscard]] auto make_metric_record(
            std::size_t sample_index,
            const Matrix &sample,
            const PipelineConfig &config,
            const domain::OneSidedJacobiSvdResult<Matrix> &result) -> std::string
        {
            const Matrix reconstructed = reconstruct_matrix(result, sample.width(), sample.height());
            const infrastructure::NumericalAccuracyMetrics accuracy = infrastructure::evaluate_numerical_accuracy(sample, reconstructed);
            const infrastructure::SvdConvergenceMetrics convergence = infrastructure::evaluate_svd_convergence(result, config.svd_options);

            std::ostringstream output;
            output << "{\"sample_index\":" << sample_index << ",\"input\":{\"width\":" << sample.width() << ",\"height\":" << sample.height()
                   << "},\"config\":{\"layout\":\"" << to_string(config.layout) << "\",\"kernel\":\"" << to_string(config.kernel)
                   << "\",\"max_sweeps\":" << config.svd_options.max_sweeps << ',';
            write_json_number_field(output, "tolerance", config.svd_options.tolerance);
            output << "},\"accuracy\":";
            write_accuracy_metrics(output, accuracy);
            output << ",\"convergence\":";
            write_convergence_metrics(output, convergence);
            output << '}';
            return output.str();
        }

        /**
         * @brief 读取并处理一个指定布局类型的样本。 Read and process one sample with a specific layout type.
         *
         * @tparam Matrix 输入样本矩阵类型。 / Input sample matrix type.
         * @param reader 矩阵文件读取器。 / Matrix file reader.
         * @param config 流水线配置。 / Pipeline configuration.
         * @param metrics_logger JSONL 指标 logger。 / JSONL metrics logger.
         * @param metrics_enabled 是否启用指标输出。 / Whether metric output is enabled.
         * @param report 流水线报告。 / Pipeline report.
         * @param output 输出矩阵序列。 / Output matrix sequence.
         */
        template <domain::MatrixLike Matrix>
        void process_next_sample(
            infrastructure::MatrixFileReader &reader,
            const PipelineConfig &config,
            const infrastructure::Logger &metrics_logger,
            bool metrics_enabled,
            PipelineReport &report,
            std::vector<Matrix> &output)
        {
            const Matrix sample = reader.next<Matrix>();
            const auto result = compute_svd(sample, config.kernel, config.svd_options);
            const std::size_t sample_index = report.samples_read;
            ++report.samples_read;

            append_result_matrices(result, output);
            ++report.results_written;
            if (metrics_enabled)
            {
                metrics_logger.log(infrastructure::LogLevel::jsonl, make_metric_record(sample_index, sample, config, result));
                ++report.metrics_written;
            }
        }

        /**
         * @brief 按指定布局类型运行完整流水线。 Run the full pipeline with a specific layout type.
         *
         * @tparam Matrix 输入和输出矩阵类型。 / Input and output matrix type.
         * @param reader 矩阵文件读取器。 / Matrix file reader.
         * @param config 流水线配置。 / Pipeline configuration.
         * @param mpi MPI 执行作用域。 / MPI execution scope.
         * @param metrics_logger JSONL 指标 logger。 / JSONL metrics logger.
         * @param metrics_enabled 是否启用指标输出。 / Whether metric output is enabled.
         * @param report 流水线报告。 / Pipeline report.
         */
        template <domain::MatrixLike Matrix>
        void run_typed_pipeline(
            infrastructure::MatrixFileReader &reader,
            const PipelineConfig &config,
            const MpiExecutionScope &mpi,
            const infrastructure::Logger &metrics_logger,
            bool metrics_enabled,
            PipelineReport &report)
        {
            std::vector<Matrix> output;
            while (reader.has_next())
            {
                if (mpi.writes_output())
                {
                    process_next_sample<Matrix>(reader, config, metrics_logger, metrics_enabled, report, output);
                }
                else
                {
                    static_cast<void>(reader.next<Matrix>());
                    ++report.samples_read;
                }
            }

            if (mpi.writes_output())
            {
                infrastructure::write_matrices(config.output_path, output);
            }
        }
    } // namespace

    auto run_pipeline(const PipelineConfig &config) -> PipelineReport
    {
        MpiExecutionScope mpi = MpiExecutionScope::create(config.enable_mpi, config.argc, config.argv);

        PipelineReport report;
        report.mpi_enabled = mpi.enabled();
        report.rank = mpi.rank();
        report.size = mpi.size();

        infrastructure::Logger metrics_logger("application.pipeline.metrics", infrastructure::LogLevel::jsonl, false);
        bool metrics_enabled = false;
        if (mpi.writes_output() && config.metrics_path.has_value())
        {
            infrastructure::set_log_file(infrastructure::LogLevel::jsonl, config.metrics_path.value().string(), false);
            metrics_enabled = true;
        }

        infrastructure::MatrixFileReader reader(config.input_path);
        switch (config.layout)
        {
        case MatrixLayout::row_major:
            run_typed_pipeline<domain::RowMajorMatrix<double>>(reader, config, mpi, metrics_logger, metrics_enabled, report);
            break;
        case MatrixLayout::column_major:
            run_typed_pipeline<domain::ColumnMajorMatrix<double>>(reader, config, mpi, metrics_logger, metrics_enabled, report);
            break;
        case MatrixLayout::strided_row_major:
            run_typed_pipeline<domain::StridedRowMajorMatrix<double>>(reader, config, mpi, metrics_logger, metrics_enabled, report);
            break;
        case MatrixLayout::jagged_row_major:
            run_typed_pipeline<domain::JaggedRowMajorMatrix<double>>(reader, config, mpi, metrics_logger, metrics_enabled, report);
            break;
        case MatrixLayout::blocked_row_major:
            run_typed_pipeline<domain::BlockedRowMajorMatrix<double>>(reader, config, mpi, metrics_logger, metrics_enabled, report);
            break;
        case MatrixLayout::morton:
            run_typed_pipeline<domain::MortonMatrix<double>>(reader, config, mpi, metrics_logger, metrics_enabled, report);
            break;
        }

        if (metrics_enabled)
        {
            infrastructure::flush_logs();
            infrastructure::reset_log_output(infrastructure::LogLevel::jsonl);
        }

        mpi.barrier();
        return report;
    }

} // namespace mpilab::application
