#include "mpilab/presentation/cli.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

namespace
{
    /**
     * @brief 解析测试参数。 Parse test arguments.
     *
     * @param arguments 可变命令行参数。 / Mutable command-line arguments.
     * @return 命令行解析结果。 / Command-line parse result.
     */
    [[nodiscard]] auto parse(char *arguments[]) -> mpilab::presentation::CliOptions
    {
        int count = 0;
        while (arguments[count] != nullptr)
        {
            ++count;
        }

        return mpilab::presentation::parse_cli_arguments(count, arguments);
    }

    /**
     * @brief 验证完整配置映射。 Verify complete configuration mapping.
     */
    void test_full_configuration_mapping()
    {
        char program[] = "mpilab";
        char input_option[] = "--input";
        char input_path[] = "input.txt";
        char output_option[] = "--output";
        char output_path[] = "output.txt";
        char metrics_option[] = "--metrics";
        char metrics_path[] = "metrics.jsonl";
        char layout_option[] = "--layout";
        char layout_name[] = "blocked-row-major";
        char kernel_option[] = "--kernel";
        char kernel_name[] = "mpi-friendly";
        char sweeps_option[] = "--max-sweeps";
        char sweeps_value[] = "250";
        char tolerance_option[] = "--tolerance";
        char tolerance_value[] = "1e-9";
        char mpi_option[] = "--mpi";
        char *arguments[] = {
            program,
            input_option,
            input_path,
            output_option,
            output_path,
            metrics_option,
            metrics_path,
            layout_option,
            layout_name,
            kernel_option,
            kernel_name,
            sweeps_option,
            sweeps_value,
            tolerance_option,
            tolerance_value,
            mpi_option,
            nullptr};

        const mpilab::presentation::CliOptions options = parse(arguments);
        assert(options.action == mpilab::presentation::CliAction::run);
        assert(options.config.input_path == input_path);
        assert(options.config.output_path == output_path);
        assert(options.config.metrics_path.has_value());
        assert(options.config.metrics_path.value() == metrics_path);
        assert(options.config.layout == mpilab::application::MatrixLayout::blocked_row_major);
        assert(options.config.kernel == mpilab::application::SvdKernel::mpi_friendly);
        assert(options.config.svd_options.max_sweeps == 250);
        assert(options.config.svd_options.tolerance == 1.0e-9);
        assert(options.config.enable_mpi);
    }

    /**
     * @brief 验证帮助路径不要求输入输出路径。 Verify that help does not require input/output paths.
     */
    void test_help_short_circuit()
    {
        char program[] = "custom-mpilab";
        char help_option[] = "--help";
        char *arguments[] = {program, help_option, nullptr};

        const mpilab::presentation::CliOptions options = parse(arguments);
        assert(options.action == mpilab::presentation::CliAction::help);
        assert(options.program_name == program);

        const std::string text = mpilab::presentation::help_text(options.program_name);
        assert(text.find("custom-mpilab --input PATH --output PATH") != std::string::npos);
        assert(text.find("--kernel NAME") != std::string::npos);
    }

    /**
     * @brief 验证非法选项会失败。 Verify that invalid options fail.
     */
    void test_invalid_option_fails()
    {
        char program[] = "mpilab";
        char bad_option[] = "--wat";
        char *arguments[] = {program, bad_option, nullptr};

        bool threw = false;
        try
        {
            static_cast<void>(parse(arguments));
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }

        assert(threw);
    }

    /**
     * @brief 验证非法容差会失败。 Verify that invalid tolerance values fail.
     */
    void test_invalid_tolerance_fails()
    {
        char program[] = "mpilab";
        char input_option[] = "--input";
        char input_path[] = "input.txt";
        char output_option[] = "--output";
        char output_path[] = "output.txt";
        char tolerance_option[] = "--tolerance";
        char tolerance_value[] = "nan";
        char *arguments[] = {program, input_option, input_path, output_option, output_path, tolerance_option, tolerance_value, nullptr};

        bool threw = false;
        try
        {
            static_cast<void>(parse(arguments));
        }
        catch (const std::invalid_argument &)
        {
            threw = true;
        }

        assert(threw);
    }

} // namespace

/**
 * @brief presentation CLI 测试入口。 Presentation CLI test entry point.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
auto main() -> int
{
    test_full_configuration_mapping();
    test_help_short_circuit();
    test_invalid_option_fails();
    test_invalid_tolerance_fails();
    return 0;
}
