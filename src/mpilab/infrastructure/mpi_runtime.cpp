#include "mpilab/infrastructure/mpi_runtime.hpp"

#include <optional>
#include <utility>

#if MPILAB_HAS_MPI
#include <mpi.h>
#endif

/**
 * @file mpi_runtime.cpp
 * @brief MPI 运行时基础设施实现。 MPI runtime infrastructure implementation.
 */

namespace mpilab::infrastructure
{
    namespace
    {

#if MPILAB_HAS_MPI
        /**
         * @brief 将 MPI 调用结果转换为可选错误。 Convert an MPI call result to an optional error.
         *
         * @param result MPI 调用返回码。 / MPI call return code.
         * @param operation 操作名称。 / Operation name.
         * @return 成功时返回空，失败时返回错误。 / Empty on success, error on failure.
         */
        [[nodiscard]] auto mpi_error_from_result(int result, const char *operation) noexcept -> std::optional<MpiError>
        {
            if (result == MPI_SUCCESS)
            {
                return std::nullopt;
            }

            return MpiError{result, operation};
        }
#endif

        /**
         * @brief 在拥有生命周期时终结 MPI。 Finalize MPI when the guard owns the lifetime.
         *
         * @param owns_mpi_lifetime 是否拥有 MPI 生命周期。 / Whether the guard owns the MPI lifetime.
         */
        void finalize_owned_mpi(bool owns_mpi_lifetime) noexcept
        {
#if MPILAB_HAS_MPI
            if (!owns_mpi_lifetime || !mpi_initialized() || mpi_finalized())
            {
                return;
            }

            static_cast<void>(MPI_Finalize());
#else
            static_cast<void>(owns_mpi_lifetime);
#endif
        }

    } // namespace

    auto mpi_support_enabled() noexcept -> bool
    {
        return MPILAB_HAS_MPI != 0;
    }

    auto mpi_initialized() noexcept -> bool
    {
#if MPILAB_HAS_MPI
        int initialized = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS)
        {
            return false;
        }

        return initialized != 0;
#else
        return false;
#endif
    }

    auto mpi_finalized() noexcept -> bool
    {
#if MPILAB_HAS_MPI
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS)
        {
            return false;
        }

        return finalized != 0;
#else
        return false;
#endif
    }

    auto MpiRuntime::create(int *argc, char ***argv) noexcept -> MpiRuntimeResult
    {
        MpiRuntime runtime;

#if MPILAB_HAS_MPI
        if (mpi_finalized())
        {
            return MpiRuntimeResult{std::nullopt, MpiError{MPI_ERR_OTHER, "MPI has already been finalized"}};
        }

        if (mpi_initialized())
        {
            return MpiRuntimeResult{std::move(runtime), std::nullopt};
        }

        if (std::optional<MpiError> error = mpi_error_from_result(MPI_Init(argc, argv), "MPI_Init"))
        {
            return MpiRuntimeResult{std::nullopt, error};
        }

        runtime.owns_mpi_lifetime_ = true;
#else
        static_cast<void>(argc);
        static_cast<void>(argv);
#endif

        return MpiRuntimeResult{std::move(runtime), std::nullopt};
    }

    MpiRuntime::~MpiRuntime() noexcept
    {
        finalize_owned_mpi(owns_mpi_lifetime_);
    }

    MpiRuntime::MpiRuntime(MpiRuntime &&other) noexcept
        : owns_mpi_lifetime_(std::exchange(other.owns_mpi_lifetime_, false))
    {
    }

    auto MpiRuntime::operator=(MpiRuntime &&other) noexcept -> MpiRuntime &
    {
        if (this == &other)
        {
            return *this;
        }

        finalize_owned_mpi(owns_mpi_lifetime_);
        owns_mpi_lifetime_ = std::exchange(other.owns_mpi_lifetime_, false);
        return *this;
    }

    auto MpiRuntime::owns_mpi_lifetime() const noexcept -> bool
    {
        return owns_mpi_lifetime_;
    }

    auto MpiRuntime::initialized() const noexcept -> bool
    {
        return mpi_initialized();
    }

    auto MpiRuntime::finalized() const noexcept -> bool
    {
        return mpi_finalized();
    }

    auto MpiRuntime::rank() const noexcept -> std::optional<int>
    {
        return world_rank();
    }

    auto MpiRuntime::size() const noexcept -> std::optional<int>
    {
        return world_size();
    }

    auto MpiRuntime::is_root() const noexcept -> bool
    {
        const std::optional<int> current_rank = rank();
        return current_rank.has_value() && current_rank.value() == 0;
    }

    auto MpiRuntime::barrier() const noexcept -> std::optional<MpiError>
    {
        return world_barrier();
    }

    auto MpiRuntimeResult::has_value() const noexcept -> bool
    {
        return runtime.has_value();
    }

    auto world_rank() noexcept -> std::optional<int>
    {
#if MPILAB_HAS_MPI
        if (!mpi_initialized() || mpi_finalized())
        {
            return 0;
        }

        int rank = 0;
        if (mpi_error_from_result(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank"))
        {
            return std::nullopt;
        }

        return rank;
#else
        return 0;
#endif
    }

    auto world_size() noexcept -> std::optional<int>
    {
#if MPILAB_HAS_MPI
        if (!mpi_initialized() || mpi_finalized())
        {
            return 1;
        }

        int size = 1;
        if (mpi_error_from_result(MPI_Comm_size(MPI_COMM_WORLD, &size), "MPI_Comm_size"))
        {
            return std::nullopt;
        }

        return size;
#else
        return 1;
#endif
    }

    auto world_barrier() noexcept -> std::optional<MpiError>
    {
#if MPILAB_HAS_MPI
        if (!mpi_initialized() || mpi_finalized())
        {
            return std::nullopt;
        }

        return mpi_error_from_result(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
#else
        return std::nullopt;
#endif
    }

} // namespace mpilab::infrastructure
