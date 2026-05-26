#include "mpilab/domain/kernel/SimdOneSidedJacobiSvd.hpp"

#include "detail/one_sided_jacobi_svd_detail.hpp"

/**
 * @file simd_one_sided_jacobi_svd.cpp
 * @brief SIMD 单边 Jacobi SVD 算子实现。 SIMD one-sided Jacobi SVD kernel implementation.
 */

namespace mpilab::domain
{

auto SimdOneSidedJacobiSvd::operator()(const RowMajorMatrix<double>& matrix, const OneSidedJacobiSvdOptions& options) const -> OneSidedJacobiSvdResult
{
    return detail::run_one_sided_jacobi_svd(
        matrix,
        options,
        detail::JacobiSweepStrategy::cyclic,
        detail::JacobiExecutionStrategy::serial,
        detail::JacobiComputeKernel::simd);
}

} // namespace mpilab::domain
