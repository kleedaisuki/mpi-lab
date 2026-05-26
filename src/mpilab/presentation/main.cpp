#include "mpilab/presentation/cli.hpp"

#include "mpilab/infrastructure/logger.hpp"

#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

/**
 * @file main.cpp
 * @brief 命令行程序入口。 Command-line program entry point.
 */

namespace
{
    /**
     * @brief 成功退出码。 Successful exit code.
     */
    constexpr int exit_success = 0;

    /**
     * @brief 执行失败退出码。 Execution failure exit code.
     */
    constexpr int exit_failure = 1;

    /**
     * @brief 命令行用法错误退出码。 Command-line usage error exit code.
     */
    constexpr int exit_usage = 2;

    /**
     * @brief 格式化流水线报告日志。 Format a pipeline report log message.
     *
     * @param report 流水线报告。 / Pipeline report.
     * @return 日志消息。 / Log message.
     */
    [[nodiscard]] auto format_report(const mpilab::application::PipelineReport &report) -> std::string
    {
        std::ostringstream message;
        message << "pipeline completed: samples_read=" << report.samples_read << ", results_written=" << report.results_written
                << ", metrics_written=" << report.metrics_written;
        if (report.rank.has_value())
        {
            message << ", rank=" << report.rank.value();
        }
        if (report.size.has_value())
        {
            message << ", size=" << report.size.value();
        }

        return message.str();
    }
} // namespace

/**
 * @brief mpi-lab 命令行入口。 mpi-lab command-line entry point.
 *
 * @param argc 命令行参数数量。 / Command-line argument count.
 * @param argv 命令行参数数组。 / Command-line argument vector.
 * @return 进程退出码。 / Process exit code.
 */
auto main(int argc, char **argv) -> int
{
    const mpilab::infrastructure::Logger logger("presentation.cli");

    try
    {
        mpilab::presentation::CliOptions options = mpilab::presentation::parse_cli_arguments(argc, argv);
        if (options.action == mpilab::presentation::CliAction::help)
        {
            std::cout << mpilab::presentation::help_text(options.program_name);
            return exit_success;
        }

        options.config.argc = &argc;
        options.config.argv = &argv;
        const mpilab::application::PipelineReport report = mpilab::application::run_pipeline(options.config);
        logger.info(format_report(report));
        mpilab::infrastructure::flush_logs();
        return exit_success;
    }
    catch (const std::invalid_argument &exception)
    {
        const std::string program_name = (argc > 0 && argv != nullptr && argv[0] != nullptr) ? std::string{argv[0]} : std::string{"mpilab"};
        logger.error("command-line error: " + std::string(exception.what()) + "; run '" + program_name + " --help' for usage");
        mpilab::infrastructure::flush_logs();
        return exit_usage;
    }
    catch (const std::exception &exception)
    {
        logger.error("pipeline failed: " + std::string(exception.what()));
        mpilab::infrastructure::flush_logs();
        return exit_failure;
    }
}
