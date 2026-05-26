#include "mpilab/infrastructure/logger.hpp"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <ctime>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

/**
 * @file logger.cpp
 * @brief 异步日志基础设施实现。 Asynchronous logging infrastructure implementation.
 */

namespace mpilab::infrastructure
{
namespace
{

    /**
     * @brief 可路由日志等级数量。 / Number of routable log levels.
     */
    constexpr std::size_t routable_level_count = 6;

    /**
     * @brief 单条日志记录。 Single log record.
     */
    struct LogRecord
    {
        /**
         * @brief 日志等级。 / Log level.
         */
        LogLevel level{LogLevel::info};

        /**
         * @brief 模块名。 / Module name.
         */
        std::string logger_name;

        /**
         * @brief 日志消息。 / Log message.
         */
        std::string message;

        /**
         * @brief 记录创建时间。 / Record creation time.
         */
        std::chrono::system_clock::time_point timestamp{};
    };

    /**
     * @brief 输出路由目标。 Output routing target.
     */
    struct OutputTarget
    {
        /**
         * @brief 非拥有输出流指针。 / Non-owning output stream pointer.
         */
        std::ostream* stream{nullptr};

        /**
         * @brief 可选拥有文件流。 / Optional owning file stream.
         */
        std::shared_ptr<std::ofstream> owned_file{};
    };

    /**
     * @brief 将日志等级转换为数组下标。 Convert a log level to an array index.
     *
     * @param level 日志等级。 / Log level.
     * @return 数组下标。 / Array index.
     */
    [[nodiscard]] auto level_index(LogLevel level) -> std::size_t
    {
        const auto value = static_cast<std::size_t>(level);
        if (value >= routable_level_count) {
            throw std::out_of_range("log level is not routable");
        }

        return value;
    }

    /**
     * @brief 判断消息是否通过阈值过滤。 Test whether a message passes a threshold filter.
     *
     * @param message_level 消息等级。 / Message level.
     * @param filter_level 过滤阈值。 / Filter threshold.
     * @return 通过时为 true。 / True when accepted.
     */
    [[nodiscard]] auto passes_filter(LogLevel message_level, LogLevel filter_level) noexcept -> bool
    {
        return static_cast<int>(message_level) >= static_cast<int>(filter_level)
            && filter_level != LogLevel::off
            && message_level != LogLevel::off;
    }

    /**
     * @brief 线程安全地转换本地时间。 Convert local time in a thread-safe way.
     *
     * @param time 日历时间。 / Calendar time.
     * @return 本地分解时间。 / Broken-down local time.
     */
    [[nodiscard]] auto local_time(std::time_t time) -> std::tm
    {
        std::tm result{};
#if defined(_WIN32)
        localtime_s(&result, &time);
#else
        localtime_r(&time, &result);
#endif
        return result;
    }

    /**
     * @brief 格式化微秒精度时间戳。 Format a timestamp with microsecond precision.
     *
     * @param timestamp 时间点。 / Time point.
     * @return 时间戳文本。 / Timestamp text.
     */
    [[nodiscard]] auto format_timestamp(std::chrono::system_clock::time_point timestamp) -> std::string
    {
        using std::chrono::duration_cast;
        using std::chrono::microseconds;
        using std::chrono::seconds;

        const auto whole_seconds = std::chrono::time_point_cast<seconds>(timestamp);
        const auto fractional = duration_cast<microseconds>(timestamp - whole_seconds).count();
        const std::time_t calendar_time = std::chrono::system_clock::to_time_t(timestamp);
        const std::tm local = local_time(calendar_time);

        std::ostringstream output;
        output << std::put_time(&local, "%Y-%m-%d %H:%M:%S") << '.'
               << std::setw(6) << std::setfill('0') << fractional;
        return output.str();
    }

    /**
     * @brief 格式化日志记录。 Format a log record.
     *
     * @param record 日志记录。 / Log record.
     * @return 完整日志行。 / Complete log line.
     */
    [[nodiscard]] auto format_record(const LogRecord& record) -> std::string
    {
        std::ostringstream output;
        output << format_timestamp(record.timestamp) << " [" << to_string(record.level) << "] ["
               << record.logger_name << "] " << record.message << '\n';
        return output.str();
    }

    /**
     * @brief 进程级异步日志后端。 Process-level asynchronous logging backend.
     */
    class LogBackend
    {
    public:
        /**
         * @brief 构造后端并启动消费者线程。 Construct the backend and start the consumer thread.
         */
        LogBackend()
            : outputs_(default_outputs())
            , worker_([this] {
                consume();
            })
        {
        }

        /**
         * @brief 刷新并销毁后端。 Flush and destroy the backend.
         */
        ~LogBackend()
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            pending_available_.notify_one();

            if (worker_.joinable()) {
                worker_.join();
            }
        }

        /**
         * @brief 禁止复制构造。 Disable copy construction.
         *
         * @param other 另一个后端。 / Another backend.
         */
        LogBackend(const LogBackend& other) = delete;

        /**
         * @brief 禁止复制赋值。 Disable copy assignment.
         *
         * @param other 另一个后端。 / Another backend.
         * @return 当前后端引用。 / Reference to this backend.
         */
        auto operator=(const LogBackend& other) -> LogBackend& = delete;

        /**
         * @brief 入队日志记录。 Enqueue a log record.
         *
         * @param record 日志记录。 / Log record.
         */
        void enqueue(LogRecord record)
        {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                queue_.push_back(std::move(record));
            }
            pending_available_.notify_one();
        }

        /**
         * @brief 设置全局过滤等级。 Set the global filter level.
         *
         * @param level 全局最低输出等级。 / Global minimum emitted level.
         */
        void set_level(LogLevel level)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            level_ = level;
        }

        /**
         * @brief 返回全局过滤等级。 Return the global filter level.
         *
         * @return 全局最低输出等级。 / Global minimum emitted level.
         */
        [[nodiscard]] auto level() -> LogLevel
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return level_;
        }

        /**
         * @brief 设置指定等级输出流。 Set the output stream for a level.
         *
         * @param level 日志等级。 / Log level.
         * @param output 输出流。 / Output stream.
         */
        void set_output(LogLevel level, std::ostream& output)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            drain_locked(lock);
            outputs_.at(level_index(level)) = OutputTarget{&output, nullptr};
        }

        /**
         * @brief 设置指定等级输出文件。 Set the output file for a level.
         *
         * @param level 日志等级。 / Log level.
         * @param path 文件路径。 / File path.
         */
        void set_file(LogLevel level, const std::string& path)
        {
            auto file = std::make_shared<std::ofstream>(path, std::ios::out | std::ios::app);
            if (!file->is_open()) {
                throw std::runtime_error("failed to open log file");
            }

            std::unique_lock<std::mutex> lock(mutex_);
            drain_locked(lock);
            outputs_.at(level_index(level)) = OutputTarget{file.get(), std::move(file)};
        }

        /**
         * @brief 恢复默认输出路由。 Restore default output routing.
         */
        void reset_outputs()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            drain_locked(lock);
            outputs_ = default_outputs();
        }

        /**
         * @brief 等待队列清空并刷新输出流。 Wait until the queue is empty and flush output streams.
         */
        void flush()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            drained_.wait(lock, [this] {
                return queue_.empty() && active_writes_ == 0;
            });
            flush_outputs_locked();
        }

    private:
        /**
         * @brief 创建默认输出路由。 Create the default output routing.
         *
         * @return 默认输出路由表。 / Default output routing table.
         */
        [[nodiscard]] static auto default_outputs() -> std::array<OutputTarget, routable_level_count>
        {
            return {
                OutputTarget{&std::clog, nullptr},
                OutputTarget{&std::clog, nullptr},
                OutputTarget{&std::clog, nullptr},
                OutputTarget{&std::cerr, nullptr},
                OutputTarget{&std::cerr, nullptr},
                OutputTarget{&std::cerr, nullptr},
            };
        }

        /**
         * @brief 消费者线程主循环。 Main loop for the consumer thread.
         */
        void consume()
        {
            while (true) {
                LogRecord record;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    pending_available_.wait(lock, [this] {
                        return stopping_ || !queue_.empty();
                    });

                    if (stopping_ && queue_.empty()) {
                        flush_outputs_locked();
                        drained_.notify_all();
                        return;
                    }

                    record = std::move(queue_.front());
                    queue_.pop_front();
                    ++active_writes_;
                }

                write(record);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    --active_writes_;
                    if (queue_.empty() && active_writes_ == 0) {
                        drained_.notify_all();
                    }
                }
            }
        }

        /**
         * @brief 写入一条记录。 Write one record.
         *
         * @param record 日志记录。 / Log record.
         */
        void write(const LogRecord& record)
        {
            const std::string line = format_record(record);

            std::lock_guard<std::mutex> lock(mutex_);
            OutputTarget& target = outputs_.at(level_index(record.level));
            if (target.stream != nullptr) {
                (*target.stream) << line;
            }
        }

        /**
         * @brief 在持锁状态刷新所有输出流。 Flush all output streams while holding the lock.
         */
        void flush_outputs_locked()
        {
            for (OutputTarget& target : outputs_) {
                if (target.stream != nullptr) {
                    target.stream->flush();
                }
            }
        }

        /**
         * @brief 在持锁对象上等待队列耗尽并刷新输出流。 Wait for queue drain on an owned lock and flush output streams.
         *
         * @param lock 已持有的后端锁。 / Already-owned backend lock.
         */
        void drain_locked(std::unique_lock<std::mutex>& lock)
        {
            drained_.wait(lock, [this] {
                return queue_.empty() && active_writes_ == 0;
            });
            flush_outputs_locked();
        }

        /**
         * @brief 队列互斥锁。 / Queue mutex.
         */
        std::mutex mutex_;

        /**
         * @brief 待消费日志队列。 / Pending log queue.
         */
        std::deque<LogRecord> queue_;

        /**
         * @brief 日志可用条件变量。 / Log-available condition variable.
         */
        std::condition_variable pending_available_;

        /**
         * @brief 队列已清空条件变量。 / Queue-drained condition variable.
         */
        std::condition_variable drained_;

        /**
         * @brief 按等级分派的输出路由。 / Output routing table by level.
         */
        std::array<OutputTarget, routable_level_count> outputs_;

        /**
         * @brief 消费者线程。 / Consumer thread.
         */
        std::thread worker_;

        /**
         * @brief 全局最低输出等级。 / Global minimum emitted level.
         */
        LogLevel level_{LogLevel::trace};

        /**
         * @brief 正在写入的记录数量。 / Number of records currently being written.
         */
        std::size_t active_writes_{0};

        /**
         * @brief 是否请求停止。 / Whether stop was requested.
         */
        bool stopping_{false};
    };

    /**
     * @brief 返回进程级日志后端。 Return the process-level logging backend.
     *
     * @return 日志后端引用。 / Reference to the logging backend.
     */
    [[nodiscard]] auto backend() -> LogBackend&
    {
        /**
         * @brief 惰性初始化的全局日志后端。 / Lazily initialized global logging backend.
         */
        static LogBackend instance;
        return instance;
    }

} // namespace

auto to_string(LogLevel level) -> std::string_view
{
    switch (level) {
    case LogLevel::trace:
        return "TRACE";
    case LogLevel::debug:
        return "DEBUG";
    case LogLevel::info:
        return "INFO";
    case LogLevel::warning:
        return "WARN";
    case LogLevel::error:
        return "ERROR";
    case LogLevel::critical:
        return "CRITICAL";
    case LogLevel::off:
        return "OFF";
    }

    return "UNKNOWN";
}

Logger::Logger(std::string name)
    : name_(std::move(name))
{
}

Logger::Logger(std::string name, LogLevel level)
    : name_(std::move(name))
    , level_(level)
{
}

Logger::Logger(const Logger& other)
    : name_(other.name_)
    , level_(other.level_.load())
{
}

auto Logger::operator=(const Logger& other) -> Logger&
{
    if (this == &other) {
        return *this;
    }

    name_ = other.name_;
    level_.store(other.level_.load());
    return *this;
}

Logger::Logger(Logger&& other) noexcept
    : name_(std::move(other.name_))
    , level_(other.level_.load())
{
}

auto Logger::operator=(Logger&& other) noexcept -> Logger&
{
    if (this == &other) {
        return *this;
    }

    name_ = std::move(other.name_);
    level_.store(other.level_.load());
    return *this;
}

auto Logger::name() const noexcept -> std::string_view
{
    return name_;
}

void Logger::set_level(LogLevel level) noexcept
{
    level_.store(level);
}

auto Logger::level() const noexcept -> LogLevel
{
    return level_.load();
}

void Logger::log(LogLevel level, std::string_view message) const
{
    if (!passes_filter(level, level_.load()) || !passes_filter(level, global_log_level())) {
        return;
    }

    backend().enqueue(LogRecord{
        level,
        name_,
        std::string(message),
        std::chrono::system_clock::now(),
    });
}

void Logger::trace(std::string_view message) const
{
    log(LogLevel::trace, message);
}

void Logger::debug(std::string_view message) const
{
    log(LogLevel::debug, message);
}

void Logger::info(std::string_view message) const
{
    log(LogLevel::info, message);
}

void Logger::warning(std::string_view message) const
{
    log(LogLevel::warning, message);
}

void Logger::error(std::string_view message) const
{
    log(LogLevel::error, message);
}

void Logger::critical(std::string_view message) const
{
    log(LogLevel::critical, message);
}

void set_global_log_level(LogLevel level)
{
    backend().set_level(level);
}

auto global_log_level() -> LogLevel
{
    return backend().level();
}

void set_log_output(LogLevel level, std::ostream& output)
{
    backend().set_output(level, output);
}

void set_log_output_at_or_above(LogLevel minimum_level, std::ostream& output)
{
    if (minimum_level == LogLevel::off) {
        return;
    }

    for (std::size_t index = level_index(minimum_level); index < routable_level_count; ++index) {
        backend().set_output(static_cast<LogLevel>(index), output);
    }
}

void set_log_file(LogLevel level, const std::string& path)
{
    backend().set_file(level, path);
}

void reset_log_outputs()
{
    backend().reset_outputs();
}

void flush_logs()
{
    backend().flush();
}

} // namespace mpilab::infrastructure
