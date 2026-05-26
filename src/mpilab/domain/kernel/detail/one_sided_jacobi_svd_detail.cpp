#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"
#include "mpilab/domain/kernel/detail/one_sided_jacobi_svd_detail.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

/**
 * @file one_sided_jacobi_svd_detail.cpp
 * @brief 单边 Jacobi SVD 非模板调度辅助实现。 Non-template scheduling helpers for one-sided Jacobi SVD.
 */

namespace mpilab::domain::detail
{

    auto build_round_robin_phases(std::size_t column_count) -> std::vector<JacobiPairPhase>
    {
        const std::size_t padded_count = column_count % 2 == 0 ? column_count : column_count + 1;
        std::vector<std::size_t> slots(padded_count);
        std::iota(slots.begin(), slots.end(), 0);

        std::vector<JacobiPairPhase> phases;
        if (column_count < 2)
        {
            return phases;
        }

        phases.reserve(padded_count - 1);
        for (std::size_t phase_index = 0; phase_index < padded_count - 1; ++phase_index)
        {
            JacobiPairPhase phase;
            phase.pairs.reserve(padded_count / 2);

            for (std::size_t pair_index = 0; pair_index < padded_count / 2; ++pair_index)
            {
                const std::size_t first = slots[pair_index];
                const std::size_t second = slots[padded_count - 1 - pair_index];

                if (first < column_count && second < column_count)
                {
                    phase.pairs.push_back({.left = std::min(first, second), .right = std::max(first, second)});
                }
            }

            phases.push_back(std::move(phase));

            const std::size_t moved = slots.back();
            for (std::size_t index = padded_count - 1; index > 1; --index)
            {
                slots[index] = slots[index - 1];
            }
            slots[1] = moved;
        }

        return phases;
    }

} // namespace mpilab::domain::detail
