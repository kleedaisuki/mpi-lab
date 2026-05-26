#include "mpilab/domain/kernel/AdvancedOneSidedJacobiSvd.hpp"

#include "detail/one_sided_jacobi_svd_detail.hpp"

/**
 * @file advanced_one_sided_jacobi_svd.cpp
 * @brief 改进串行单边 Jacobi SVD 算子实现。 Advanced serial one-sided Jacobi SVD kernel implementation.
 */

namespace mpilab::domain
{

auto AdvancedOneSidedJacobiSvd::operator()(const RowMajorMatrix<double>& matrix, const OneSidedJacobiSvdOptions& options) const -> OneSidedJacobiSvdResult
{
    return detail::run_one_sided_jacobi_svd(matrix, options, detail::JacobiSweepStrategy::dynamic_ordered);
}

} // namespace mpilab::domain
