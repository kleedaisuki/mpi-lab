#pragma once

/**
 * @file mpi_runtime.hpp
 * @brief MPI 运行时基础设施声明。 MPI runtime infrastructure declarations.
 */

#ifndef MPILAB_HAS_MPI
#define MPILAB_HAS_MPI 0
#endif

#include <optional>
#include <string_view>

namespace mpilab::infrastructure
{

    /**
     * @brief MPI 错误值。 MPI error value.
     */
    struct MpiError
    {
        /**
         * @brief MPI 返回码。 / MPI return code.
         */
        int code{0};

        /**
         * @brief 失败操作名称。 / Failed operation name.
         */
        std::string_view operation;
    };

    /**
     * @brief MPI 运行时创建结果。 MPI runtime creation result.
     */
    struct MpiRuntimeResult;

    /**
     * @brief 返回构建是否启用了 MPI 支持。 Return whether the build enables MPI support.
     *
     * @return 启用 MPI 支持时返回 true。 / Returns true when MPI support is enabled.
     */
    [[nodiscard]] auto mpi_support_enabled() noexcept -> bool;

    /**
     * @brief 返回 MPI 是否已经初始化。 Return whether MPI has been initialized.
     *
     * @return MPI 已初始化时返回 true。 / Returns true when MPI has been initialized.
     * @note 无 MPI 构建始终返回 false。 / Non-MPI builds always return false.
     */
    [[nodiscard]] auto mpi_initialized() noexcept -> bool;

    /**
     * @brief 返回 MPI 是否已经终结。 Return whether MPI has been finalized.
     *
     * @return MPI 已终结时返回 true。 / Returns true when MPI has been finalized.
     * @note 无 MPI 构建始终返回 false。 / Non-MPI builds always return false.
     */
    [[nodiscard]] auto mpi_finalized() noexcept -> bool;

    /**
     * @brief RAII 风格的 MPI 运行时守卫。 RAII-style MPI runtime guard.
     *
     * @note 如果 MPI 已由调用方初始化，本对象不会在析构时终结 MPI。 / If MPI was already initialized by the caller, this object will not finalize MPI during destruction.
     */
    class MpiRuntime
    {
    public:
        /**
         * @brief 构造串行回退运行时。 Construct a serial fallback runtime.
         */
        MpiRuntime() noexcept = default;

        /**
         * @brief 销毁运行时并在拥有 MPI 生命周期时终结 MPI。 Destroy the runtime and finalize MPI when it owns the MPI lifetime.
         */
        ~MpiRuntime() noexcept;

        /**
         * @brief 创建运行时并在需要时初始化 MPI。 Create the runtime and initialize MPI when needed.
         *
         * @param argc 命令行参数数量指针，可为空。 / Optional command-line argument count pointer.
         * @param argv 命令行参数数组指针，可为空。 / Optional command-line argument vector pointer.
         * @return 运行时创建结果。 / Runtime creation result.
         */
        [[nodiscard]] static auto create(int *argc = nullptr, char ***argv = nullptr) noexcept -> MpiRuntimeResult;

        /**
         * @brief 移动构造 MPI 运行时守卫。 Move-construct an MPI runtime guard.
         *
         * @param other 另一个 MPI 运行时守卫。 / Another MPI runtime guard.
         */
        MpiRuntime(MpiRuntime &&other) noexcept;

        /**
         * @brief 移动赋值 MPI 运行时守卫。 Move-assign an MPI runtime guard.
         *
         * @param other 另一个 MPI 运行时守卫。 / Another MPI runtime guard.
         * @return 当前 MPI 运行时守卫引用。 / Reference to this MPI runtime guard.
         */
        auto operator=(MpiRuntime &&other) noexcept -> MpiRuntime &;

        /**
         * @brief 禁止复制构造。 Disable copy construction.
         *
         * @param other 另一个 MPI 运行时守卫。 / Another MPI runtime guard.
         */
        MpiRuntime(const MpiRuntime &other) = delete;

        /**
         * @brief 禁止复制赋值。 Disable copy assignment.
         *
         * @param other 另一个 MPI 运行时守卫。 / Another MPI runtime guard.
         * @return 当前 MPI 运行时守卫引用。 / Reference to this MPI runtime guard.
         */
        auto operator=(const MpiRuntime &other) -> MpiRuntime & = delete;

        /**
         * @brief 返回当前进程是否拥有 MPI 生命周期。 Return whether this object owns the MPI lifetime.
         *
         * @return 拥有 MPI 生命周期时返回 true。 / Returns true when this object owns the MPI lifetime.
         */
        [[nodiscard]] auto owns_mpi_lifetime() const noexcept -> bool;

        /**
         * @brief 返回 MPI 是否已经初始化。 Return whether MPI has been initialized.
         *
         * @return MPI 已初始化时返回 true。 / Returns true when MPI has been initialized.
         */
        [[nodiscard]] auto initialized() const noexcept -> bool;

        /**
         * @brief 返回 MPI 是否已经终结。 Return whether MPI has been finalized.
         *
         * @return MPI 已终结时返回 true。 / Returns true when MPI has been finalized.
         */
        [[nodiscard]] auto finalized() const noexcept -> bool;

        /**
         * @brief 返回 MPI_COMM_WORLD 中的 rank。 Return the rank in MPI_COMM_WORLD.
         *
         * @return 当前 rank；失败时返回空。 / Current rank; empty on failure.
         */
        [[nodiscard]] auto rank() const noexcept -> std::optional<int>;

        /**
         * @brief 返回 MPI_COMM_WORLD 中的进程数量。 Return the process count in MPI_COMM_WORLD.
         *
         * @return 进程数量；失败时返回空。 / Process count; empty on failure.
         */
        [[nodiscard]] auto size() const noexcept -> std::optional<int>;

        /**
         * @brief 返回当前 rank 是否为根 rank。 Return whether the current rank is the root rank.
         *
         * @return rank 为 0 时返回 true。 / Returns true when rank is 0.
         */
        [[nodiscard]] auto is_root() const noexcept -> bool;

        /**
         * @brief 在 MPI_COMM_WORLD 上执行 barrier。 Run a barrier on MPI_COMM_WORLD.
         *
         * @return 成功时返回空，失败时返回错误。 / Empty on success, error on failure.
         * @note 无 MPI 构建中这是成功的空操作。 / This is a successful no-op in non-MPI builds.
         */
        [[nodiscard]] auto barrier() const noexcept -> std::optional<MpiError>;

    private:
        /**
         * @brief 是否由本对象初始化 MPI。 / Whether this object initialized MPI.
         */
        bool owns_mpi_lifetime_{false};
    };

    /**
     * @brief MPI 运行时创建结果。 MPI runtime creation result.
     */
    struct MpiRuntimeResult
    {
        /**
         * @brief 创建成功时的运行时守卫。 / Runtime guard when creation succeeds.
         */
        std::optional<MpiRuntime> runtime;

        /**
         * @brief 创建失败时的错误。 / Error when creation fails.
         */
        std::optional<MpiError> error;

        /**
         * @brief 返回创建是否成功。 Return whether creation succeeded.
         *
         * @return 创建成功时返回 true。 / Returns true when creation succeeded.
         */
        [[nodiscard]] auto has_value() const noexcept -> bool;
    };

    /**
     * @brief 返回 MPI_COMM_WORLD 中的 rank。 Return the rank in MPI_COMM_WORLD.
     *
     * @return 当前 rank；失败时返回空。 / Current rank; empty on failure.
     */
    [[nodiscard]] auto world_rank() noexcept -> std::optional<int>;

    /**
     * @brief 返回 MPI_COMM_WORLD 中的进程数量。 Return the process count in MPI_COMM_WORLD.
     *
     * @return 进程数量；失败时返回空。 / Process count; empty on failure.
     */
    [[nodiscard]] auto world_size() noexcept -> std::optional<int>;

    /**
     * @brief 在 MPI_COMM_WORLD 上执行 barrier。 Run a barrier on MPI_COMM_WORLD.
     *
     * @return 成功时返回空，失败时返回错误。 / Empty on success, error on failure.
     * @note 无 MPI 构建中这是成功的空操作。 / This is a successful no-op in non-MPI builds.
     */
    [[nodiscard]] auto world_barrier() noexcept -> std::optional<MpiError>;

} // namespace mpilab::infrastructure
