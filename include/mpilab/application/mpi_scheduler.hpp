#pragma once

#include "mpilab/infrastructure/mpi_runtime.hpp"

#include <optional>

/**
 * @file mpi_scheduler.hpp
 * @brief 应用层 MPI 调度辅助声明。 Application-level MPI scheduling helper declarations.
 */

namespace mpilab::application
{

    /**
     * @brief MPI 执行作用域。 MPI execution scope.
     *
     * @note 当配置未启用 MPI 时保持串行空作用域；启用时惰性创建 MPI 运行时守卫（MpiRuntime）。 / Keeps an empty serial scope when MPI is disabled by configuration; lazily creates an MPI runtime guard (MpiRuntime) when enabled.
     */
    class MpiExecutionScope final
    {
    public:
        /**
         * @brief 构造串行执行作用域。 Construct a serial execution scope.
         */
        MpiExecutionScope() noexcept = default;

        /**
         * @brief 按配置创建 MPI 执行作用域。 Create an MPI execution scope from configuration.
         *
         * @param enabled 是否启用 MPI。 / Whether MPI is enabled.
         * @param argc 命令行参数数量指针，可为空。 / Optional command-line argument count pointer.
         * @param argv 命令行参数数组指针，可为空。 / Optional command-line argument vector pointer.
         * @return MPI 执行作用域。 / MPI execution scope.
         * @throws std::runtime_error 当 MPI 初始化失败。 / Throws when MPI initialization fails.
         */
        [[nodiscard]] static auto create(bool enabled, int *argc = nullptr, char ***argv = nullptr) -> MpiExecutionScope;

        /**
         * @brief 返回配置是否请求 MPI。 Return whether configuration requested MPI.
         *
         * @return 请求 MPI 时返回 true。 / Returns true when MPI was requested.
         */
        [[nodiscard]] auto enabled() const noexcept -> bool;

        /**
         * @brief 返回当前 rank。 Return the current rank.
         *
         * @return 当前 rank；查询失败时为空。 / Current rank; empty when the query fails.
         */
        [[nodiscard]] auto rank() const noexcept -> std::optional<int>;

        /**
         * @brief 返回进程数量。 Return the process count.
         *
         * @return 进程数量；查询失败时为空。 / Process count; empty when the query fails.
         */
        [[nodiscard]] auto size() const noexcept -> std::optional<int>;

        /**
         * @brief 返回当前进程是否负责写输出。 Return whether the current process should write output.
         *
         * @return 串行模式或 rank 0 时返回 true。 / Returns true in serial mode or on rank 0.
         */
        [[nodiscard]] auto writes_output() const noexcept -> bool;

        /**
         * @brief 在启用 MPI 时执行 barrier。 Run a barrier when MPI is enabled.
         *
         * @throws std::runtime_error 当 MPI barrier 失败。 / Throws when the MPI barrier fails.
         */
        void barrier() const;

    private:
        /**
         * @brief 配置是否请求 MPI。 / Whether configuration requested MPI.
         */
        bool enabled_{false};

        /**
         * @brief MPI 运行时守卫。 / MPI runtime guard.
         */
        std::optional<infrastructure::MpiRuntime> runtime_;
    };

} // namespace mpilab::application
