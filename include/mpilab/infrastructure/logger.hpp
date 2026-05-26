#pragma once

#include <atomic>
#include <iosfwd>
#include <string>
#include <string_view>

/**
 * @file logger.hpp
 * @brief 异步日志基础设施声明。 Asynchronous logging infrastructure declarations.
 */

namespace mpilab::infrastructure
{

    /**
     * @brief 日志等级。 Log severity level.
     */
    enum class LogLevel
    {
        /**
         * @brief 跟踪级日志。 / Trace-level log.
         */
        trace = 0,

        /**
         * @brief 调试级日志。 / Debug-level log.
         */
        debug = 1,

        /**
         * @brief 信息级日志。 / Info-level log.
         */
        info = 2,

        /**
         * @brief 警告级日志。 / Warning-level log.
         */
        warning = 3,

        /**
         * @brief 错误级日志。 / Error-level log.
         */
        error = 4,

        /**
         * @brief 致命级日志。 / Critical-level log.
         */
        critical = 5,

        /**
         * @brief 关闭日志。 / Disable logging.
         */
        off = 6
    };

    /**
     * @brief 将日志等级转换为稳定文本。 Convert a log level to stable text.
     *
     * @param level 日志等级。 / Log level.
     * @return 日志等级文本。 / Log-level text.
     */
    [[nodiscard]] auto to_string(LogLevel level) -> std::string_view;

    /**
     * @brief 异步 logger 前端实例。 Asynchronous logger front-end instance.
     *
     * @note 多个 Logger 实例共享同一个进程级消息队列、消费者线程和输出路由。 / Multiple Logger instances share the same process-level message queue, consumer thread, and output routing.
     */
    class Logger
    {
    public:
        /**
         * @brief 使用模块名构造 logger。 Construct a logger with a module name.
         *
         * @param name 模块名。 / Module name.
         */
        explicit Logger(std::string name);

        /**
         * @brief 使用模块名和实例过滤等级构造 logger。 Construct a logger with a module name and instance filter level.
         *
         * @param name 模块名。 / Module name.
         * @param level 实例最低输出等级。 / Instance minimum emitted level.
         */
        Logger(std::string name, LogLevel level);

        /**
         * @brief 复制构造 logger 前端。 Copy-construct a logger front end.
         *
         * @param other 另一个 logger。 / Another logger.
         */
        Logger(const Logger& other);

        /**
         * @brief 复制赋值 logger 前端。 Copy-assign a logger front end.
         *
         * @param other 另一个 logger。 / Another logger.
         * @return 当前 logger 引用。 / Reference to this logger.
         */
        auto operator=(const Logger& other) -> Logger&;

        /**
         * @brief 移动构造 logger 前端。 Move-construct a logger front end.
         *
         * @param other 另一个 logger。 / Another logger.
         */
        Logger(Logger&& other) noexcept;

        /**
         * @brief 移动赋值 logger 前端。 Move-assign a logger front end.
         *
         * @param other 另一个 logger。 / Another logger.
         * @return 当前 logger 引用。 / Reference to this logger.
         */
        auto operator=(Logger&& other) noexcept -> Logger&;

        /**
         * @brief 返回模块名。 Return the module name.
         *
         * @return 模块名。 / Module name.
         */
        [[nodiscard]] auto name() const noexcept -> std::string_view;

        /**
         * @brief 设置实例过滤等级。 Set the instance filter level.
         *
         * @param level 实例最低输出等级。 / Instance minimum emitted level.
         */
        void set_level(LogLevel level) noexcept;

        /**
         * @brief 返回实例过滤等级。 Return the instance filter level.
         *
         * @return 实例最低输出等级。 / Instance minimum emitted level.
         */
        [[nodiscard]] auto level() const noexcept -> LogLevel;

        /**
         * @brief 写入指定等级日志。 Write a log message at the given level.
         *
         * @param level 日志等级。 / Log level.
         * @param message 日志消息。 / Log message.
         */
        void log(LogLevel level, std::string_view message) const;

        /**
         * @brief 写入 trace 日志。 Write a trace log message.
         *
         * @param message 日志消息。 / Log message.
         */
        void trace(std::string_view message) const;

        /**
         * @brief 写入 debug 日志。 Write a debug log message.
         *
         * @param message 日志消息。 / Log message.
         */
        void debug(std::string_view message) const;

        /**
         * @brief 写入 info 日志。 Write an info log message.
         *
         * @param message 日志消息。 / Log message.
         */
        void info(std::string_view message) const;

        /**
         * @brief 写入 warning 日志。 Write a warning log message.
         *
         * @param message 日志消息。 / Log message.
         */
        void warning(std::string_view message) const;

        /**
         * @brief 写入 error 日志。 Write an error log message.
         *
         * @param message 日志消息。 / Log message.
         */
        void error(std::string_view message) const;

        /**
         * @brief 写入 critical 日志。 Write a critical log message.
         *
         * @param message 日志消息。 / Log message.
         */
        void critical(std::string_view message) const;

    private:
        /**
         * @brief 模块名。 / Module name.
         */
        std::string name_;

        /**
         * @brief 实例最低输出等级。 / Instance minimum emitted level.
         */
        std::atomic<LogLevel> level_{LogLevel::trace};
    };

    /**
     * @brief 设置全局过滤等级。 Set the global filter level.
     *
     * @param level 全局最低输出等级。 / Global minimum emitted level.
     */
    void set_global_log_level(LogLevel level);

    /**
     * @brief 返回全局过滤等级。 Return the global filter level.
     *
     * @return 全局最低输出等级。 / Global minimum emitted level.
     */
    [[nodiscard]] auto global_log_level() -> LogLevel;

    /**
     * @brief 将某个日志等级重定向到输出流。 Redirect one log level to an output stream.
     *
     * @param level 日志等级。 / Log level.
     * @param output 输出流，调用方负责保证其生命周期覆盖使用期。 / Output stream whose lifetime must cover its use.
     */
    void set_log_output(LogLevel level, std::ostream& output);

    /**
     * @brief 将不低于指定等级的日志重定向到同一个输出流。 Redirect logs at or above a level to the same output stream.
     *
     * @param minimum_level 最低日志等级。 / Minimum log level.
     * @param output 输出流，调用方负责保证其生命周期覆盖使用期。 / Output stream whose lifetime must cover its use.
     */
    void set_log_output_at_or_above(LogLevel minimum_level, std::ostream& output);

    /**
     * @brief 将某个日志等级重定向到文件。 Redirect one log level to a file.
     *
     * @param level 日志等级。 / Log level.
     * @param path 文件路径。 / File path.
     */
    void set_log_file(LogLevel level, const std::string& path);

    /**
     * @brief 将全局输出路由恢复为默认流。 Restore global output routing to default streams.
     */
    void reset_log_outputs();

    /**
     * @brief 阻塞直到异步日志队列被消费并刷新输出流。 Block until the asynchronous log queue is consumed and output streams are flushed.
     */
    void flush_logs();

} // namespace mpilab::infrastructure
