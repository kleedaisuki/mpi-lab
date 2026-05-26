#include "mpilab/application/pipeline.hpp"
#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/file_stream.hpp"
#include "mpilab/infrastructure/mpi_runtime.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <ios>
#include <string>

namespace
{
    /**
     * @brief 默认测试容差。 / Default test tolerance.
     */
    constexpr double tolerance = 1.0e-9;

    /**
     * @brief 返回测试临时文件路径。 Return a temporary file path for tests.
     *
     * @param name 文件名。 / File name.
     * @return 临时文件路径。 / Temporary file path.
     */
    [[nodiscard]] auto temp_path(const char *name) -> std::filesystem::path
    {
        return std::filesystem::temp_directory_path() / name;
    }

    /**
     * @brief 将文本写入测试文件。 Write text into a test file.
     *
     * @param path 文件路径。 / File path.
     * @param text 文本内容。 / Text content.
     */
    void write_text(const std::filesystem::path &path, const std::string &text)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << text;
    }

    /**
     * @brief 判断两个浮点值是否近似相等。 Check whether two floating-point values are approximately equal.
     *
     * @param left 左值。 / Left value.
     * @param right 右值。 / Right value.
     * @return 近似相等时返回 true。 / Returns true when approximately equal.
     */
    [[nodiscard]] auto almost_equal(double left, double right) -> bool
    {
        return std::abs(left - right) <= tolerance;
    }

    /**
     * @brief 验证串行 pipeline 读样本、选布局、选算子并写回结果。 Verify that the serial pipeline reads samples, selects layout/kernel, and writes results.
     */
    void test_serial_pipeline_round_trip()
    {
        const std::filesystem::path input_path = temp_path("mpilab_pipeline_input.txt");
        const std::filesystem::path output_path = temp_path("mpilab_pipeline_output.txt");
        const std::filesystem::path metrics_path = temp_path("mpilab_pipeline_metrics.jsonl");
        write_text(input_path, "3 0\n0 2\n\n1 0\n0 1\n");

        const bool mpi_was_initialized = mpilab::infrastructure::mpi_initialized();

        mpilab::application::PipelineConfig config;
        config.input_path = input_path;
        config.output_path = output_path;
        config.metrics_path = metrics_path;
        config.layout = mpilab::application::MatrixLayout::column_major;
        config.kernel = mpilab::application::SvdKernel::advanced;
        config.enable_mpi = false;

        const mpilab::application::PipelineReport report = mpilab::application::run_pipeline(config);
        assert(report.samples_read == 2);
        assert(report.results_written == 2);
        assert(report.metrics_written == 2);
        assert(!report.mpi_enabled);
        assert(report.rank.has_value());
        assert(report.rank.value() == 0);
        assert(mpilab::infrastructure::mpi_initialized() == mpi_was_initialized);

        mpilab::infrastructure::MatrixFileReader reader(output_path);
        const auto first_u = reader.next<mpilab::domain::RowMajorMatrix<double>>();
        const auto first_sigma = reader.next<mpilab::domain::RowMajorMatrix<double>>();
        const auto first_v = reader.next<mpilab::domain::RowMajorMatrix<double>>();
        assert(first_u.width() == 2);
        assert(first_u.height() == 2);
        assert(first_sigma.width() == 2);
        assert(first_sigma.height() == 1);
        assert(first_v.width() == 2);
        assert(first_v.height() == 2);
        assert(almost_equal(first_sigma(0, 0), 3.0));
        assert(almost_equal(first_sigma(1, 0), 2.0));

        static_cast<void>(reader.next<mpilab::domain::RowMajorMatrix<double>>());
        static_cast<void>(reader.next<mpilab::domain::RowMajorMatrix<double>>());
        static_cast<void>(reader.next<mpilab::domain::RowMajorMatrix<double>>());
        assert(!reader.has_next());

        std::ifstream metrics_input(metrics_path, std::ios::binary);
        std::string first_record;
        std::string second_record;
        std::string extra_record;
        assert(static_cast<bool>(std::getline(metrics_input, first_record)));
        assert(static_cast<bool>(std::getline(metrics_input, second_record)));
        assert(!static_cast<bool>(std::getline(metrics_input, extra_record)));
        assert(!first_record.empty());
        assert(first_record.front() == '{');
        assert(first_record.find(R"("sample_index":0)") != std::string::npos);
        assert(first_record.find(R"("layout":"column_major")") != std::string::npos);
        assert(first_record.find(R"("kernel":"advanced")") != std::string::npos);
        assert(first_record.find(R"("accuracy":)") != std::string::npos);
        assert(first_record.find(R"("convergence":)") != std::string::npos);
        assert(first_record.find(R"("root_mean_squared_error")") != std::string::npos);
        assert(first_record.find(R"("max_column_correlation")") != std::string::npos);
        assert(second_record.find(R"("sample_index":1)") != std::string::npos);

        std::filesystem::remove(input_path);
        std::filesystem::remove(output_path);
        std::filesystem::remove(metrics_path);
    }

    /**
     * @brief 验证 MPI 配置路径能通过同一流水线执行。 Verify that the MPI configuration path executes through the same pipeline.
     *
     * @param argc 命令行参数数量指针。 / Command-line argument count pointer.
     * @param argv 命令行参数数组指针。 / Command-line argument vector pointer.
     */
    void test_mpi_enabled_pipeline(int *argc, char ***argv)
    {
        const std::filesystem::path input_path = temp_path("mpilab_pipeline_mpi_input.txt");
        const std::filesystem::path output_path = temp_path("mpilab_pipeline_mpi_output.txt");
        write_text(input_path, "2 0\n0 1\n");

        mpilab::application::PipelineConfig config;
        config.input_path = input_path;
        config.output_path = output_path;
        config.layout = mpilab::application::MatrixLayout::blocked_row_major;
        config.kernel = mpilab::application::SvdKernel::mpi_friendly;
        config.enable_mpi = true;
        config.argc = argc;
        config.argv = argv;

        const mpilab::application::PipelineReport report = mpilab::application::run_pipeline(config);
        assert(report.samples_read == 1);
        assert(report.mpi_enabled);
        assert(report.rank.has_value());
        assert(report.size.has_value());
        if (report.rank.value() == 0)
        {
            assert(report.results_written == 1);
            assert(std::filesystem::exists(output_path));
        }
        else
        {
            assert(report.results_written == 0);
        }

        std::filesystem::remove(input_path);
        std::filesystem::remove(output_path);
    }

} // namespace

/**
 * @brief application pipeline 测试入口。 Application pipeline test entry point.
 *
 * @param argc 命令行参数数量。 / Command-line argument count.
 * @param argv 命令行参数数组。 / Command-line argument vector.
 * @return 成功时返回 0。 / Returns 0 on success.
 */
auto main(int argc, char **argv) -> int
{
    test_serial_pipeline_round_trip();
    test_mpi_enabled_pipeline(&argc, &argv);
    return 0;
}
