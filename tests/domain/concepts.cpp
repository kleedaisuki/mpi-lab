#include "mpilab/domain/KernelLike.hpp"
#include "mpilab/domain/MatrixLike.hpp"
#include "mpilab/domain/matrix.hpp"

#include <cstddef>

namespace
{

    /**
     * @brief 测试矩阵映射：按坐标返回实数。 Test matrix mapping: returns a real number by coordinate.
     */
    struct CoordinateMatrix
    {
        /**
         * @brief 构造测试矩阵映射。 Construct a test matrix mapping.
         *
         * @param width 中文：逻辑横向范围。 English: Logical horizontal extent.
         * @param height 中文：逻辑纵向范围。 English: Logical vertical extent.
         */
        CoordinateMatrix(std::size_t width, std::size_t height)
        {
            static_cast<void>(width);
            static_cast<void>(height);
        }

        /**
         * @brief 按坐标读取矩阵值。 Read a matrix value by coordinate.
         *
         * @param x 中文：横向坐标。 English: Horizontal coordinate.
         * @param y 中文：纵向坐标。 English: Vertical coordinate.
         * @return 中文：坐标对应的矩阵值。 English: Matrix value at the coordinate.
         */
        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> double
        {
            return static_cast<double>(x) + static_cast<double>(y);
        }
    };

    /**
     * @brief 测试核：无状态地读取矩阵坐标值。 Test kernel: statelessly reads a matrix coordinate value.
     */
    struct ReadKernel
    {
        /**
         * @brief 执行核函数。 Execute the kernel function.
         *
         * @tparam Matrix 中文：矩阵映射类型。 English: Matrix mapping type.
         * @param matrix 中文：矩阵映射。 English: Matrix mapping.
         * @param x 中文：横向坐标。 English: Horizontal coordinate.
         * @param y 中文：纵向坐标。 English: Vertical coordinate.
         * @return 中文：核函数结果。 English: Kernel result.
         */
        template <mpilab::domain::MatrixLike Matrix>
        [[nodiscard]] auto operator()(const Matrix &matrix, std::size_t x, std::size_t y) const -> double
        {
            return matrix(x, y);
        }
    };

    /**
     * @brief 测试坏矩阵：返回非实数值。 Test bad matrix: returns a non-real value.
     */
    struct NonRealMatrix
    {
        /**
         * @brief 构造测试坏矩阵。 Construct a test bad matrix.
         *
         * @param width 中文：逻辑横向范围。 English: Logical horizontal extent.
         * @param height 中文：逻辑纵向范围。 English: Logical vertical extent.
         */
        NonRealMatrix(std::size_t width, std::size_t height)
        {
            static_cast<void>(width);
            static_cast<void>(height);
        }

        /**
         * @brief 按坐标返回非实数值。 Return a non-real value by coordinate.
         *
         * @param x 中文：横向坐标。 English: Horizontal coordinate.
         * @param y 中文：纵向坐标。 English: Vertical coordinate.
         * @return 中文：非实数值。 English: Non-real value.
         */
        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> const char *
        {
            static_cast<void>(x);
            static_cast<void>(y);
            return "";
        }
    };

    /**
     * @brief 测试有状态核：应被 KernelLike 拒绝。 Test stateful kernel: should be rejected by KernelLike.
     */
    struct StatefulKernel
    {
        /**
         * @brief 中文：核内部状态。 English: Kernel internal state.
         */
        int offset{0};

        /**
         * @brief 执行带状态核函数。 Execute a stateful kernel function.
         *
         * @param matrix 中文：矩阵映射。 English: Matrix mapping.
         * @param x 中文：横向坐标。 English: Horizontal coordinate.
         * @param y 中文：纵向坐标。 English: Vertical coordinate.
         * @return 中文：核函数结果。 English: Kernel result.
         */
        [[nodiscard]] auto operator()(const CoordinateMatrix &matrix, std::size_t x, std::size_t y) const -> double
        {
            return matrix(x, y) + static_cast<double>(offset);
        }
    };

    /**
     * @brief 测试不可统一构造矩阵：应被 MatrixLike 拒绝。 Test non-uniformly constructible matrix: should be rejected by MatrixLike.
     */
    struct NonUniformMatrix
    {
        /**
         * @brief 默认构造测试矩阵。 Default-construct a test matrix.
         */
        NonUniformMatrix() = default;

        /**
         * @brief 按坐标读取矩阵值。 Read a matrix value by coordinate.
         *
         * @param x 中文：横向坐标。 English: Horizontal coordinate.
         * @param y 中文：纵向坐标。 English: Vertical coordinate.
         * @return 中文：坐标对应的矩阵值。 English: Matrix value at the coordinate.
         */
        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> double
        {
            return static_cast<double>(x) + static_cast<double>(y);
        }
    };

    static_assert(mpilab::domain::MatrixLike<CoordinateMatrix>);
    static_assert(!mpilab::domain::MatrixLike<NonRealMatrix>);
    static_assert(!mpilab::domain::MatrixLike<NonUniformMatrix>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::RowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::ColumnMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::StridedRowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::JaggedRowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::BlockedRowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::MortonMatrix<>>);
static_assert(mpilab::domain::KernelLike<ReadKernel, CoordinateMatrix>);
static_assert(!mpilab::domain::KernelLike<StatefulKernel, CoordinateMatrix>);

} // namespace

/**
 * @brief 领域概念测试入口。 Domain concept test entry point.
 *
 * @return 中文：成功时返回 0。 English: Returns 0 on success.
 */
int main()
{
    return 0;
}
