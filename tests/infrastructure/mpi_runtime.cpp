#include "mpilab/infrastructure/mpi_runtime.hpp"

#include <cassert>
#include <optional>

namespace
{

    /**
     * @brief 验证无显式初始化时的串行查询契约。 Verify the serial query contract without explicit initialization.
     */
    void test_serial_query_contract()
    {
        const std::optional<int> rank = mpilab::infrastructure::world_rank();
        const std::optional<int> size = mpilab::infrastructure::world_size();

        assert(rank.has_value());
        assert(size.has_value());
        assert(rank.value() == 0);
        assert(size.value() >= 1);
        assert(!mpilab::infrastructure::world_barrier().has_value());
    }

    /**
     * @brief 验证 MPI 运行时守卫的 rank 和 size 查询。 Verify rank and size queries on the MPI runtime guard.
     *
     * @param argc 命令行参数数量指针。 / Command-line argument count pointer.
     * @param argv 命令行参数数组指针。 / Command-line argument vector pointer.
     */
    void test_runtime_guard(int* argc, char*** argv)
    {
        mpilab::infrastructure::MpiRuntimeResult result = mpilab::infrastructure::MpiRuntime::create(argc, argv);

        assert(result.has_value());
        assert(result.runtime.has_value());
        assert(!result.error.has_value());

        mpilab::infrastructure::MpiRuntime& runtime = result.runtime.value();
        const std::optional<int> rank = runtime.rank();
        const std::optional<int> size = runtime.size();

        assert(rank.has_value());
        assert(size.has_value());
        assert(rank.value() >= 0);
        assert(size.value() >= 1);
        assert(runtime.is_root() == (rank.value() == 0));
        assert(!runtime.barrier().has_value());

        if (mpilab::infrastructure::mpi_support_enabled()) {
            assert(runtime.initialized());
        } else {
            assert(!runtime.initialized());
            assert(!runtime.owns_mpi_lifetime());
        }
    }

} // namespace

/**
 * @brief MPI 运行时测试入口。 MPI runtime test entry point.
 *
 * @param argc 命令行参数数量。 / Command-line argument count.
 * @param argv 命令行参数数组。 / Command-line argument vector.
 * @return 成功时返回 0。 / Returns 0 on success.
 */
auto main(int argc, char** argv) -> int
{
    test_serial_query_contract();
    test_runtime_guard(&argc, &argv);
    return 0;
}
