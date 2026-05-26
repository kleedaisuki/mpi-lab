#include "mpilab/domain/kernel/MpiFriendlyOneSidedJacobiSvd.hpp"

#include "detail/one_sided_jacobi_svd_detail.hpp"

/**
 * @file mpi_friendly_one_sided_jacobi_svd.cpp
 * @brief MPI 友好单边 Jacobi SVD 算子实现。 MPI-friendly one-sided Jacobi SVD kernel implementation.
 */

namespace mpilab::domain
{

    auto RoundRobinJacobiPairScheduler::operator()(std::size_t column_count) const -> std::vector<JacobiPairPhase>
    {
        return detail::build_round_robin_phases(column_count);
    }

    auto MpiFriendlyOneSidedJacobiSvd::operator()(const RowMajorMatrix<double> &matrix, const OneSidedJacobiSvdOptions &options) const -> OneSidedJacobiSvdResult
    {
        return detail::run_one_sided_jacobi_svd(matrix, options, detail::JacobiSweepStrategy::round_robin_phases);
    }

} // namespace mpilab::domain
