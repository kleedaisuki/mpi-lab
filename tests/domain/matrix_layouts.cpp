#include "mpilab/domain/MatrixLike.hpp"
#include "mpilab/domain/matrix.hpp"

#include <cassert>
#include <cstddef>
#include <ranges>
#include <vector>

namespace
{

/**
 * @brief 以稳定公式填充矩阵，便于比较不同布局的逻辑等价性。 Fill a matrix with a stable formula so logical equivalence across layouts can be compared.
 *
 * @tparam Matrix 矩阵类型。 / Matrix type.
 * @param matrix 待填充矩阵。 / Matrix to populate.
 */
template <typename Matrix>
void fill_matrix(Matrix& matrix)
{
    /**
     * @brief 逻辑高度缓存。 / Cached logical height.
     */
    const std::size_t height = matrix.height();

    /**
     * @brief 逻辑宽度缓存。 / Cached logical width.
     */
    const std::size_t width = matrix.width();

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            matrix.set(x, y, static_cast<double>(100 * y + x));
        }
    }
}

/**
 * @brief 断言矩阵逻辑值与填充公式一致。 Assert that matrix logical values match the fill formula.
 *
 * @tparam Matrix 矩阵类型。 / Matrix type.
 * @param matrix 待验证矩阵。 / Matrix to validate.
 */
template <typename Matrix>
void assert_logical_values(const Matrix& matrix)
{
    /**
     * @brief 逻辑高度缓存。 / Cached logical height.
     */
    const std::size_t height = matrix.height();

    /**
     * @brief 逻辑宽度缓存。 / Cached logical width.
     */
    const std::size_t width = matrix.width();

    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            assert(matrix(x, y) == static_cast<double>(100 * y + x));
        }
    }
}

static_assert(mpilab::domain::MatrixLike<mpilab::domain::RowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::ColumnMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::StridedRowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::JaggedRowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::BlockedRowMajorMatrix<>>);
static_assert(mpilab::domain::MatrixLike<mpilab::domain::MortonMatrix<>>);

} // namespace

/**
 * @brief 矩阵布局测试入口。 Matrix layout test entry point.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
int main()
{
    /**
     * @brief 行主序测试矩阵。 / Row-major test matrix.
     */
    mpilab::domain::RowMajorMatrix<> row_major(3, 2);

    /**
     * @brief 列主序测试矩阵。 / Column-major test matrix.
     */
    mpilab::domain::ColumnMajorMatrix<> column_major(3, 2);

    /**
     * @brief 带跨距测试矩阵。 / Strided test matrix.
     */
    mpilab::domain::StridedRowMajorMatrix<> strided(3, 2, 5);

    /**
     * @brief 逐行分配测试矩阵。 / Jagged per-row test matrix.
     */
    mpilab::domain::JaggedRowMajorMatrix<> jagged(3, 2);

    /**
     * @brief 分块测试矩阵。 / Blocked-layout test matrix.
     */
    mpilab::domain::BlockedRowMajorMatrix<> blocked(4, 4, 2, 2);

    /**
     * @brief Morton 测试矩阵。 / Morton-layout test matrix.
     */
    mpilab::domain::MortonMatrix<> morton(4, 4);

    fill_matrix(row_major);
    fill_matrix(column_major);
    fill_matrix(strided);
    fill_matrix(jagged);
    fill_matrix(blocked);
    fill_matrix(morton);

    assert_logical_values(row_major);
    assert_logical_values(column_major);
    assert_logical_values(strided);
    assert_logical_values(jagged);
    assert_logical_values(blocked);
    assert_logical_values(morton);

    /**
     * @brief 期望的行主序底层存储。 / Expected row-major backing storage.
     */
    const std::vector<double> expected_row_major{0.0, 1.0, 2.0, 100.0, 101.0, 102.0};

    /**
     * @brief 期望的列主序底层存储。 / Expected column-major backing storage.
     */
    const std::vector<double> expected_column_major{0.0, 100.0, 1.0, 101.0, 2.0, 102.0};

    /**
     * @brief 期望的带跨距底层存储。 / Expected strided backing storage.
     */
    const std::vector<double> expected_strided{0.0, 1.0, 2.0, 0.0, 0.0, 100.0, 101.0, 102.0, 0.0, 0.0};

    /**
     * @brief 期望的逐行分配第一行存储。 / Expected first-row storage for the jagged layout.
     */
    const std::vector<double> expected_jagged_row0{0.0, 1.0, 2.0};

    /**
     * @brief 期望的逐行分配第二行存储。 / Expected second-row storage for the jagged layout.
     */
    const std::vector<double> expected_jagged_row1{100.0, 101.0, 102.0};

    /**
     * @brief 期望的分块底层存储。 / Expected blocked backing storage.
     */
    const std::vector<double> expected_blocked{
        0.0, 1.0, 100.0, 101.0,
        2.0, 3.0, 102.0, 103.0,
        200.0, 201.0, 300.0, 301.0,
        202.0, 203.0, 302.0, 303.0};

    /**
     * @brief 期望的 Morton 底层存储。 / Expected Morton backing storage.
     */
    const std::vector<double> expected_morton{
        0.0, 1.0, 100.0, 101.0,
        2.0, 3.0, 102.0, 103.0,
        200.0, 201.0, 300.0, 301.0,
        202.0, 203.0, 302.0, 303.0};

    assert(std::ranges::equal(row_major.storage(), expected_row_major));
    assert(std::ranges::equal(column_major.storage(), expected_column_major));
    assert(std::ranges::equal(strided.storage(), expected_strided));
    assert(std::ranges::equal(jagged.row_storage(0), expected_jagged_row0));
    assert(std::ranges::equal(jagged.row_storage(1), expected_jagged_row1));
    assert(std::ranges::equal(blocked.storage(), expected_blocked));
    assert(std::ranges::equal(morton.storage(), expected_morton));

    assert(row_major.linear_index(1, 1) == 4);
    assert(column_major.linear_index(1, 1) == 3);
    assert(strided.linear_index(1, 1) == 6);
    assert(blocked.linear_index(2, 0) == 4);
    assert(blocked.linear_index(0, 2) == 8);
    assert(morton.linear_index(2, 0) == 4);
    assert(morton.linear_index(0, 2) == 8);
    assert(strided.stride() == 5);
    assert(strided.storage_size() == 10);
    assert(jagged.row_count() == 2);
    assert(jagged.row_width() == 3);
    assert(blocked.block_width() == 2);
    assert(blocked.block_height() == 2);
    assert(morton.side() == 4);

    /**
     * @brief 非法 stride 自动正规化后的测试矩阵。 / Test matrix after automatic normalization of an invalid stride.
     */
    mpilab::domain::StridedRowMajorMatrix<> normalized(3, 2, 1);

    /**
     * @brief 非法块大小自动正规化后的测试矩阵。 / Test matrix after automatic normalization of an invalid block size.
     */
    mpilab::domain::BlockedRowMajorMatrix<> normalized_blocked(3, 2, 0, 0);

    assert(normalized.stride() == 3);
    assert(normalized.storage_size() == 6);
    assert(normalized_blocked.block_width() == 1);
    assert(normalized_blocked.block_height() == 1);

    return 0;
}
