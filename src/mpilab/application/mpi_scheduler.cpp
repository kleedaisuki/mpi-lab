#include "mpilab/application/mpi_scheduler.hpp"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

/**
 * @file mpi_scheduler.cpp
 * @brief 应用层 MPI 调度辅助实现。 Application-level MPI scheduling helper implementation.
 */

namespace mpilab::application
{
    namespace
    {
        /**
         * @brief 构造 MPI 操作失败信息。 Build an MPI operation failure message.
         *
         * @param error MPI 错误值。 / MPI error value.
         * @return 错误信息。 / Error message.
         */
        [[nodiscard]] auto mpi_error_message(const infrastructure::MpiError &error) -> std::string
        {
            return "MPI operation failed: " + std::string(error.operation) + " returned " + std::to_string(error.code);
        }
    } // namespace

    auto MpiExecutionScope::create(bool enabled, int *argc, char ***argv) -> MpiExecutionScope
    {
        MpiExecutionScope scope;
        scope.enabled_ = enabled;

        if (!enabled)
        {
            return scope;
        }

        infrastructure::MpiRuntimeResult result = infrastructure::MpiRuntime::create(argc, argv);
        if (!result.has_value())
        {
            throw std::runtime_error(mpi_error_message(result.error.value()));
        }

        scope.runtime_ = std::move(result.runtime.value());
        return scope;
    }

    auto MpiExecutionScope::enabled() const noexcept -> bool
    {
        return enabled_;
    }

    auto MpiExecutionScope::rank() const noexcept -> std::optional<int>
    {
        if (runtime_.has_value())
        {
            return runtime_->rank();
        }

        return infrastructure::world_rank();
    }

    auto MpiExecutionScope::size() const noexcept -> std::optional<int>
    {
        if (runtime_.has_value())
        {
            return runtime_->size();
        }

        return infrastructure::world_size();
    }

    auto MpiExecutionScope::writes_output() const noexcept -> bool
    {
        const std::optional<int> current_rank = rank();
        return !enabled_ || !current_rank.has_value() || current_rank.value() == 0;
    }

    void MpiExecutionScope::barrier() const
    {
        if (!enabled_)
        {
            return;
        }

        const std::optional<infrastructure::MpiError> error = runtime_.has_value() ? runtime_->barrier() : infrastructure::world_barrier();
        if (error.has_value())
        {
            throw std::runtime_error(mpi_error_message(error.value()));
        }
    }

} // namespace mpilab::application
