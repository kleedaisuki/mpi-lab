#include "mpilab/presentation/cli.hpp"

#include <charconv>
#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

/**
 * @file cli.cpp
 * @brief 命令行表示层实现。 Command-line presentation layer implementation.
 */

namespace mpilab::presentation
{
    namespace
    {
        /**
         * @brief 判断选项是否需要独立参数。 Return whether an option requires a separate value.
         *
         * @param option 选项文本。 / Option text.
         * @return 需要参数时返回 true。 / True when a value is required.
         */
        [[nodiscard]] auto option_needs_value(std::string_view option) -> bool
        {
            return option == "-i" || option == "--input" || option == "-o" || option == "--output" || option == "-m" || option == "--metrics" ||
                   option == "--layout" || option == "--kernel" || option == "--max-sweeps" || option == "--tolerance";
        }

        /**
         * @brief 读取选项值。 Read an option value.
         *
         * @param argc 命令行参数数量。 / Command-line argument count.
         * @param argv 命令行参数数组。 / Command-line argument vector.
         * @param index 当前索引。 / Current index.
         * @param option 选项文本。 / Option text.
         * @return 选项值。 / Option value.
         */
        [[nodiscard]] auto read_value(int argc, char **argv, int &index, std::string_view option) -> std::string_view
        {
            if ((index + 1) >= argc)
            {
                throw std::invalid_argument("missing value for option " + std::string(option));
            }

            ++index;
            return argv[index];
        }

        /**
         * @brief 解析矩阵布局名称。 Parse a matrix-layout name.
         *
         * @param text 用户输入文本。 / User-provided text.
         * @return 矩阵布局枚举。 / Matrix-layout enum.
         */
        [[nodiscard]] auto parse_layout(std::string_view text) -> application::MatrixLayout
        {
            if (text == "row-major" || text == "row_major")
            {
                return application::MatrixLayout::row_major;
            }
            if (text == "column-major" || text == "column_major")
            {
                return application::MatrixLayout::column_major;
            }
            if (text == "strided-row-major" || text == "strided_row_major")
            {
                return application::MatrixLayout::strided_row_major;
            }
            if (text == "jagged-row-major" || text == "jagged_row_major")
            {
                return application::MatrixLayout::jagged_row_major;
            }
            if (text == "blocked-row-major" || text == "blocked_row_major")
            {
                return application::MatrixLayout::blocked_row_major;
            }
            if (text == "morton")
            {
                return application::MatrixLayout::morton;
            }

            throw std::invalid_argument("invalid layout '" + std::string(text) + "'");
        }

        /**
         * @brief 解析 SVD 算子名称。 Parse an SVD-kernel name.
         *
         * @param text 用户输入文本。 / User-provided text.
         * @return SVD 算子枚举。 / SVD-kernel enum.
         */
        [[nodiscard]] auto parse_kernel(std::string_view text) -> application::SvdKernel
        {
            if (text == "naive")
            {
                return application::SvdKernel::naive;
            }
            if (text == "advanced")
            {
                return application::SvdKernel::advanced;
            }
            if (text == "mpi-friendly" || text == "mpi_friendly")
            {
                return application::SvdKernel::mpi_friendly;
            }
            if (text == "pthreads")
            {
                return application::SvdKernel::pthreads;
            }
            if (text == "simd")
            {
                return application::SvdKernel::simd;
            }

            throw std::invalid_argument("invalid kernel '" + std::string(text) + "'");
        }

        /**
         * @brief 解析 size_t 参数。 Parse a size_t argument.
         *
         * @param text 用户输入文本。 / User-provided text.
         * @param option 选项文本。 / Option text.
         * @return 解析后的整数。 / Parsed integer.
         */
        [[nodiscard]] auto parse_size(std::string_view text, std::string_view option) -> std::size_t
        {
            std::size_t value = 0;
            const char *first = text.data();
            const char *last = text.data() + text.size();
            const std::from_chars_result parsed = std::from_chars(first, last, value);
            if (parsed.ec != std::errc() || parsed.ptr != last)
            {
                throw std::invalid_argument("invalid integer for " + std::string(option) + ": '" + std::string(text) + "'");
            }

            return value;
        }

        /**
         * @brief 解析 double 参数。 Parse a double argument.
         *
         * @param text 用户输入文本。 / User-provided text.
         * @param option 选项文本。 / Option text.
         * @return 解析后的浮点数。 / Parsed floating-point value.
         */
        [[nodiscard]] auto parse_double(std::string_view text, std::string_view option) -> double
        {
            double value = 0.0;
            const char *first = text.data();
            const char *last = text.data() + text.size();
            const std::from_chars_result parsed = std::from_chars(first, last, value);
            if (parsed.ec != std::errc() || parsed.ptr != last || !std::isfinite(value) || value < 0.0 || value > std::numeric_limits<double>::max())
            {
                throw std::invalid_argument("invalid non-negative number for " + std::string(option) + ": '" + std::string(text) + "'");
            }

            return value;
        }

        /**
         * @brief 验证必须提供的路径选项。 Validate required path options.
         *
         * @param options 命令行选项。 / Command-line options.
         */
        void validate_required_paths(const CliOptions &options)
        {
            if (options.config.input_path.empty())
            {
                throw std::invalid_argument("missing required option --input");
            }
            if (options.config.output_path.empty())
            {
                throw std::invalid_argument("missing required option --output");
            }
        }

    } // namespace

    auto help_text(const std::string &program_name) -> std::string
    {
        std::ostringstream text;
        text << "Usage:\n"
             << "  " << program_name << " --input PATH --output PATH [options]\n\n"
             << "Required:\n"
             << "  -i, --input PATH          input matrix text file\n"
             << "  -o, --output PATH         output SVD matrix text file\n\n"
             << "Options:\n"
             << "  -m, --metrics PATH        write per-sample metrics as JSONL\n"
             << "      --layout NAME         row-major, column-major, strided-row-major,\n"
             << "                            jagged-row-major, blocked-row-major, morton\n"
             << "                            default: row-major\n"
             << "      --kernel NAME         naive, advanced, mpi-friendly, pthreads, simd\n"
             << "                            default: naive\n"
             << "      --max-sweeps N        maximum Jacobi sweep count, default: 100\n"
             << "      --tolerance X         Jacobi convergence tolerance, default: 1e-12\n"
             << "      --mpi                 enable MPI execution scope\n"
             << "  -h, --help                show this help message\n\n"
             << "Input format:\n"
             << "  Non-empty lines are matrix rows, values are whitespace separated,\n"
             << "  and blank lines separate multiple matrices.\n";
        return text.str();
    }

    auto parse_cli_arguments(int argc, char **argv) -> CliOptions
    {
        CliOptions options;
        if (argc > 0 && argv != nullptr && argv[0] != nullptr)
        {
            options.program_name = argv[0];
        }

        for (int index = 1; index < argc; ++index)
        {
            const std::string_view argument{argv[index]};
            if (argument == "-h" || argument == "--help")
            {
                options.action = CliAction::help;
                return options;
            }
            if (argument == "--mpi")
            {
                options.config.enable_mpi = true;
                continue;
            }
            if (!option_needs_value(argument))
            {
                throw std::invalid_argument("unknown option '" + std::string(argument) + "'");
            }

            const std::string_view value = read_value(argc, argv, index, argument);
            if (argument == "-i" || argument == "--input")
            {
                options.config.input_path = value;
            }
            else if (argument == "-o" || argument == "--output")
            {
                options.config.output_path = value;
            }
            else if (argument == "-m" || argument == "--metrics")
            {
                options.config.metrics_path = value;
            }
            else if (argument == "--layout")
            {
                options.config.layout = parse_layout(value);
            }
            else if (argument == "--kernel")
            {
                options.config.kernel = parse_kernel(value);
            }
            else if (argument == "--max-sweeps")
            {
                options.config.svd_options.max_sweeps = parse_size(value, argument);
            }
            else if (argument == "--tolerance")
            {
                options.config.svd_options.tolerance = parse_double(value, argument);
            }
        }

        validate_required_paths(options);
        return options;
    }

} // namespace mpilab::presentation
