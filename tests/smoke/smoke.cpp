#include "mpilab/application/mpi_scheduler.hpp"
#include "mpilab/application/pipeline.hpp"
#include "mpilab/domain/KernelLike.hpp"
#include "mpilab/domain/MatrixLike.hpp"
#include "mpilab/domain/kernel.hpp"
#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/file_stream.hpp"
#include "mpilab/infrastructure/logger.hpp"
#include "mpilab/infrastructure/mpi_runtime.hpp"

/**
 * @brief 冒烟测试入口，验证公共头文件可被下游目标包含。 Smoke test entry point that verifies public headers are consumable by downstream targets.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
int main()
{
    return 0;
}
