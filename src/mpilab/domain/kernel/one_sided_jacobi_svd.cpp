#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

#include "detail/one_sided_jacobi_svd_detail.hpp"

/**
 * @file one_sided_jacobi_svd.cpp
 * @brief Naive 单边 Jacobi SVD 算子实现。 Naive one-sided Jacobi SVD kernel implementation.
 */

namespace mpilab::domain
{

auto OneSidedJacobiSvd::operator()(const RowMajorMatrix<double>& matrix, const OneSidedJacobiSvdOptions& options) const -> OneSidedJacobiSvdResult
{
    return detail::run_one_sided_jacobi_svd(matrix, options, detail::JacobiSweepStrategy::cyclic);
}

} // namespace mpilab::domain
