#include "mpilab/domain/matrix/detail/MatrixMemory.hpp"

/**
 * @file layout_utils.cpp
 * @brief 矩阵布局辅助函数定义。 Matrix layout helper function definitions.
 */

namespace mpilab::domain::detail
{

    /**
     * @brief 计算行主序线性索引。 Compute a row-major linear index.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return 线性索引。 / Linear index.
     */
    auto row_major_linear_index(std::size_t width, std::size_t x, std::size_t y) noexcept -> std::size_t
    {
        return y * width + x;
    }

    /**
     * @brief 计算列主序线性索引。 Compute a column-major linear index.
     *
     * @param height 逻辑高度。 / Logical height.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return 线性索引。 / Linear index.
     */
    auto column_major_linear_index(std::size_t height, std::size_t x, std::size_t y) noexcept -> std::size_t
    {
        return x * height + y;
    }

    /**
     * @brief 正规化跨距。 Normalize a stride.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param stride 请求跨距。 / Requested stride.
     * @return 合法跨距。 / Valid stride.
     */
    auto normalize_stride(std::size_t width, std::size_t stride) noexcept -> std::size_t
    {
        return stride < width ? width : stride;
    }

    /**
     * @brief 计算带跨距行主序线性索引。 Compute a strided row-major linear index.
     *
     * @param stride 物理行跨度。 / Physical row stride.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return 线性索引。 / Linear index.
     */
    auto strided_row_major_linear_index(std::size_t stride, std::size_t x, std::size_t y) noexcept -> std::size_t
    {
        return y * stride + x;
    }

    /**
     * @brief 正规化块维度。 Normalize a tile extent.
     *
     * @param block_extent 请求块维度。 / Requested tile extent.
     * @return 合法块维度。 / Valid tile extent.
     */
    auto normalize_block_extent(std::size_t block_extent) noexcept -> std::size_t
    {
        return block_extent == 0 ? 1 : block_extent;
    }

    /**
     * @brief 计算分块布局所需的物理容量。 Compute the physical storage extent required by a blocked layout.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param height 逻辑高度。 / Logical height.
     * @param block_width 块宽度。 / Tile width.
     * @param block_height 块高度。 / Tile height.
     * @return 物理槽位数量。 / Number of physical slots.
     */
    auto blocked_storage_extent(
        std::size_t width,
        std::size_t height,
        std::size_t block_width,
        std::size_t block_height) noexcept -> std::size_t
    {
        const std::size_t block_columns = (width + block_width - 1) / block_width;
        const std::size_t block_rows = (height + block_height - 1) / block_height;
        return block_columns * block_rows * block_width * block_height;
    }

    /**
     * @brief 计算分块布局线性索引。 Compute a blocked-layout linear index.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param block_width 块宽度。 / Tile width.
     * @param block_height 块高度。 / Tile height.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return 线性索引。 / Linear index.
     */
    auto blocked_linear_index(
        std::size_t width,
        std::size_t block_width,
        std::size_t block_height,
        std::size_t x,
        std::size_t y) noexcept -> std::size_t
    {
        const std::size_t block_column = x / block_width;
        const std::size_t block_row = y / block_height;
        const std::size_t local_x = x % block_width;
        const std::size_t local_y = y % block_height;
        const std::size_t block_columns = (width + block_width - 1) / block_width;
        const std::size_t block_index = block_row * block_columns + block_column;
        const std::size_t block_offset = local_y * block_width + local_x;
        const std::size_t block_area = block_width * block_height;
        return block_index * block_area + block_offset;
    }

    /**
     * @brief 计算覆盖逻辑矩阵的最小 2 的幂边长。 Compute the minimal power-of-two side covering a logical matrix.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param height 逻辑高度。 / Logical height.
     * @return 覆盖边长。 / Covering side length.
     */
    auto morton_covering_power_of_two(std::size_t width, std::size_t height) noexcept -> std::size_t
    {
        std::size_t side = 1;
        const std::size_t required = width > height ? width : height;

        while (side < required) {
            side *= 2;
        }

        return side;
    }

    /**
     * @brief 计算 Morton 线性索引。 Compute a Morton linear index.
     *
     * @param side 内部方阵边长。 / Internal square side length.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return Morton 线性索引。 / Morton linear index.
     */
    auto morton_linear_index(std::size_t side, std::size_t x, std::size_t y) noexcept -> std::size_t
    {
        std::size_t morton = 0;

        for (std::size_t bit = 0; (static_cast<std::size_t>(1) << bit) < side; ++bit) {
            morton |= ((x >> bit) & static_cast<std::size_t>(1)) << (2 * bit);
            morton |= ((y >> bit) & static_cast<std::size_t>(1)) << (2 * bit + 1);
        }

        return morton;
    }

} // namespace mpilab::domain::detail
