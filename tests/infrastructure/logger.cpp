#include "mpilab/infrastructure/logger.hpp"

#include <cassert>
#include <regex>
#include <sstream>
#include <string>

namespace
{

    /**
     * @brief 将所有等级重定向到同一个测试流。 Redirect all levels to the same test stream.
     *
     * @param output 测试输出流。 / Test output stream.
     */
    void redirect_all_levels(std::ostream& output)
    {
        using mpilab::infrastructure::LogLevel;

        mpilab::infrastructure::set_log_output(LogLevel::trace, output);
        mpilab::infrastructure::set_log_output(LogLevel::debug, output);
        mpilab::infrastructure::set_log_output(LogLevel::info, output);
        mpilab::infrastructure::set_log_output(LogLevel::warning, output);
        mpilab::infrastructure::set_log_output(LogLevel::error, output);
        mpilab::infrastructure::set_log_output(LogLevel::critical, output);
    }

    /**
     * @brief 验证异步 logger 的过滤、格式和输出路由。 Verify filtering, formatting, and routing for the async logger.
     */
    void test_filtering_formatting_and_routing()
    {
        using mpilab::infrastructure::LogLevel;
        using mpilab::infrastructure::Logger;

        std::ostringstream shared_output;
        redirect_all_levels(shared_output);
        mpilab::infrastructure::set_global_log_level(LogLevel::trace);

        const Logger logger("module-a", LogLevel::info);
        logger.debug("debug dropped");
        logger.info("info kept");
        mpilab::infrastructure::flush_logs();

        const std::string shared_text = shared_output.str();
        assert(shared_text.find("debug dropped") == std::string::npos);
        assert(shared_text.find("info kept") != std::string::npos);

        const std::regex line_pattern(
            R"(^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6} \[INFO\] \[module-a\] info kept\n$)");
        assert(std::regex_match(shared_text, line_pattern));

        std::ostringstream info_output;
        std::ostringstream warning_output;
        mpilab::infrastructure::set_log_output(LogLevel::info, info_output);
        mpilab::infrastructure::set_log_output(LogLevel::warning, warning_output);

        logger.info("info routed");
        logger.warning("warning routed");
        mpilab::infrastructure::flush_logs();

        assert(info_output.str().find("info routed") != std::string::npos);
        assert(info_output.str().find("warning routed") == std::string::npos);
        assert(warning_output.str().find("warning routed") != std::string::npos);
        assert(warning_output.str().find("info routed") == std::string::npos);

        mpilab::infrastructure::reset_log_outputs();
    }

    /**
     * @brief 验证全局过滤器会影响所有 logger 实例。 Verify that the global filter affects every logger instance.
     */
    void test_global_filter_applies_to_all_instances()
    {
        using mpilab::infrastructure::LogLevel;
        using mpilab::infrastructure::Logger;

        std::ostringstream output;
        redirect_all_levels(output);
        mpilab::infrastructure::set_global_log_level(LogLevel::error);

        const Logger first("first");
        const Logger second("second");
        first.warning("first warning dropped");
        second.error("second error kept");
        mpilab::infrastructure::flush_logs();

        const std::string text = output.str();
        assert(text.find("first warning dropped") == std::string::npos);
        assert(text.find("second error kept") != std::string::npos);

        mpilab::infrastructure::set_global_log_level(LogLevel::trace);
        mpilab::infrastructure::reset_log_outputs();
    }

} // namespace

/**
 * @brief logger 测试入口。 Logger test entry point.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
int main()
{
    test_filtering_formatting_and_routing();
    test_global_filter_applies_to_all_instances();
    return 0;
}
