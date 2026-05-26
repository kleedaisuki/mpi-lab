#pragma once

#include "mpilab/application/pipeline.hpp"

#include <string>

/**
 * @file cli.hpp
 * @brief 命令行表示层声明。 Command-line presentation layer declarations.
 */

namespace mpilab::presentation
{
    /**
     * @brief 命令行动作。 Command-line action.
     */
    enum class CliAction
    {
        /**
         * @brief 运行应用流水线。 / Run the application pipeline.
         */
        run,

        /**
         * @brief 打印帮助信息。 / Print help text.
         */
        help
    };

    /**
     * @brief 命令行解析结果。 Command-line parse result.
     */
    struct CliOptions
    {
        /**
         * @brief 请求的动作。 / Requested action.
         */
        CliAction action{CliAction::run};

        /**
         * @brief 应用流水线配置。 / Application pipeline configuration.
         */
        application::PipelineConfig config{};

        /**
         * @brief 程序名。 / Program name.
         */
        std::string program_name{"mpilab"};
    };

    /**
     * @brief 构造命令行帮助文本。 Build command-line help text.
     *
     * @param program_name 程序名。 / Program name.
     * @return 帮助文本。 / Help text.
     */
    [[nodiscard]] auto help_text(const std::string &program_name) -> std::string;

    /**
     * @brief 解析命令行参数并映射到流水线配置。 Parse command-line arguments and map them to pipeline configuration.
     *
     * @param argc 命令行参数数量。 / Command-line argument count.
     * @param argv 命令行参数数组。 / Command-line argument vector.
     * @return 解析后的命令行选项。 / Parsed command-line options.
     * @throws std::invalid_argument 当参数缺失或非法。 / Throws when arguments are missing or invalid.
     */
    [[nodiscard]] auto parse_cli_arguments(int argc, char **argv) -> CliOptions;

} // namespace mpilab::presentation
