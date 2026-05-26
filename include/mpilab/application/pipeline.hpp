#pragma once

#include "mpilab/domain/kernel/OneSidedJacobiSvd.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>

/**
 * @file pipeline.hpp
 * @brief 应用流水线声明。 Application pipeline declarations.
 */

namespace mpilab::application
{

    /**
     * @brief 输入矩阵布局选择。 Input matrix layout selection.
     */
    enum class MatrixLayout
    {
        /**
         * @brief 行主序布局。 / Row-major layout.
         */
        row_major,

        /**
         * @brief 列主序布局。 / Column-major layout.
         */
        column_major,

        /**
         * @brief 带跨距行主序布局。 / Strided row-major layout.
         */
        strided_row_major,

        /**
         * @brief 逐行分配行主序布局。 / Jagged row-major layout.
         */
        jagged_row_major,

        /**
         * @brief 分块行主序布局。 / Blocked row-major layout.
         */
        blocked_row_major,

        /**
         * @brief Morton Z 阶布局。 / Morton Z-order layout.
         */
        morton
    };

    /**
     * @brief SVD 算子选择。 SVD kernel selection.
     */
    enum class SvdKernel
    {
        /**
         * @brief 朴素串行单边 Jacobi SVD。 / Naive serial one-sided Jacobi SVD.
         */
        naive,

        /**
         * @brief 动态排序单边 Jacobi SVD。 / Dynamically ordered one-sided Jacobi SVD.
         */
        advanced,

        /**
         * @brief MPI 友好 phase 调度单边 Jacobi SVD。 / MPI-friendly phase-scheduled one-sided Jacobi SVD.
         */
        mpi_friendly,

        /**
         * @brief Pthreads phase 并行单边 Jacobi SVD。 / Pthreads phase-parallel one-sided Jacobi SVD.
         */
        pthreads,

        /**
         * @brief SIMD 单边 Jacobi SVD。 / SIMD one-sided Jacobi SVD.
         */
        simd
    };

    /**
     * @brief 应用流水线配置。 Application pipeline configuration.
     */
    struct PipelineConfig
    {
        /**
         * @brief 输入样本文件路径。 / Input sample file path.
         */
        std::filesystem::path input_path;

        /**
         * @brief 输出结果文件路径。 / Output result file path.
         */
        std::filesystem::path output_path;

        /**
         * @brief 输入矩阵布局。 / Input matrix layout.
         */
        MatrixLayout layout{MatrixLayout::row_major};

        /**
         * @brief SVD 算子。 / SVD kernel.
         */
        SvdKernel kernel{SvdKernel::naive};

        /**
         * @brief SVD 迭代选项。 / SVD iteration options.
         */
        domain::OneSidedJacobiSvdOptions svd_options{};

        /**
         * @brief 是否启用 MPI 执行作用域。 / Whether to enable the MPI execution scope.
         */
        bool enable_mpi{false};

        /**
         * @brief 命令行参数数量指针，可为空。 / Optional command-line argument count pointer.
         */
        int *argc{nullptr};

        /**
         * @brief 命令行参数数组指针，可为空。 / Optional command-line argument vector pointer.
         */
        char ***argv{nullptr};
    };

    /**
     * @brief 应用流水线执行报告。 Application pipeline execution report.
     */
    struct PipelineReport
    {
        /**
         * @brief 已读取样本数量。 / Number of samples read.
         */
        std::size_t samples_read{0};

        /**
         * @brief 已写出 SVD 结果数量。 / Number of SVD results written.
         */
        std::size_t results_written{0};

        /**
         * @brief 配置是否启用 MPI。 / Whether configuration enabled MPI.
         */
        bool mpi_enabled{false};

        /**
         * @brief 当前进程 rank。 / Current process rank.
         */
        std::optional<int> rank;

        /**
         * @brief MPI 进程数量。 / MPI process count.
         */
        std::optional<int> size;
    };

    /**
     * @brief 运行应用流水线。 Run the application pipeline.
     *
     * @param config 流水线配置。 / Pipeline configuration.
     * @return 流水线执行报告。 / Pipeline execution report.
     * @throws std::runtime_error 当 MPI 初始化或同步失败。 / Throws when MPI initialization or synchronization fails.
     * @throws infrastructure::MatrixFileFormatError 当矩阵文本格式非法。 / Throws when matrix text format is invalid.
     * @note 每个输入样本写出三块矩阵：U、单行 Sigma、V。 / Each input sample writes three matrices: U, one-row Sigma, and V.
     */
    [[nodiscard]] auto run_pipeline(const PipelineConfig &config) -> PipelineReport;

} // namespace mpilab::application
