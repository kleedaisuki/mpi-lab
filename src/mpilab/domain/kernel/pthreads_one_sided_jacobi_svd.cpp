#include "mpilab/domain/kernel/PthreadsOneSidedJacobiSvd.hpp"

#include "detail/one_sided_jacobi_svd_detail.hpp"

/**
 * @file pthreads_one_sided_jacobi_svd.cpp
 * @brief Pthreads 单边 Jacobi SVD 算子实现。 Pthreads one-sided Jacobi SVD kernel implementation.
 */

namespace mpilab::domain
{

auto PthreadsOneSidedJacobiSvd::operator()(const RowMajorMatrix<double>& matrix, const OneSidedJacobiSvdOptions& options) const -> OneSidedJacobiSvdResult
{
    return detail::run_one_sided_jacobi_svd(
        matrix,
        options,
        detail::JacobiSweepStrategy::round_robin_phases,
        detail::JacobiExecutionStrategy::pthreads,
        detail::JacobiComputeKernel::scalar);
}

} // namespace mpilab::domain
