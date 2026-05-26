#pragma once

#include <concepts>
#include <cstddef>
#include <memory>

/**
 * @file MatrixMemory.hpp
 * @brief 矩阵内存与布局辅助声明。 Matrix memory and layout helper declarations.
 */

namespace mpilab::domain::detail
{

    /**
     * @brief 标准兼容分配器概念：要求类型满足 allocator_traits 可用，并为目标标量提供标准分配接口。 Standard-compatible allocator concept: requires allocator_traits support and standard allocation interfaces for the target scalar.
     *
     * @tparam Scalar 目标标量类型。 / Target scalar type.
     * @tparam Allocator 待检查的分配器类型。 / Allocator type to inspect.
     */
    template <typename Scalar, typename Allocator>
    concept StandardAllocator = std::copy_constructible<Allocator>
        && requires {
               typename std::allocator_traits<Allocator>::value_type;
               typename std::allocator_traits<Allocator>::pointer;
           }
        && std::same_as<typename std::allocator_traits<Allocator>::value_type, Scalar>
        && requires(
            Allocator allocator,
            typename std::allocator_traits<Allocator>::pointer pointer,
            Scalar* raw_pointer,
            std::size_t size,
            const Scalar& value) {
               { std::allocator_traits<Allocator>::allocate(allocator, size) }
                   -> std::same_as<typename std::allocator_traits<Allocator>::pointer>;
               std::allocator_traits<Allocator>::deallocate(allocator, pointer, size);
               std::allocator_traits<Allocator>::construct(allocator, raw_pointer, value);
               std::allocator_traits<Allocator>::destroy(allocator, raw_pointer);
               { std::to_address(pointer) } -> std::same_as<Scalar*>;
           };

    /**
     * @brief 计算行主序线性索引。 Compute a row-major linear index.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return 线性索引。 / Linear index.
     */
    [[nodiscard]] auto row_major_linear_index(std::size_t width, std::size_t x, std::size_t y) noexcept -> std::size_t;

    /**
     * @brief 计算列主序线性索引。 Compute a column-major linear index.
     *
     * @param height 逻辑高度。 / Logical height.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return 线性索引。 / Linear index.
     */
    [[nodiscard]] auto column_major_linear_index(std::size_t height, std::size_t x, std::size_t y) noexcept -> std::size_t;

    /**
     * @brief 正规化跨距。 Normalize a stride.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param stride 请求跨距。 / Requested stride.
     * @return 合法跨距。 / Valid stride.
     */
    [[nodiscard]] auto normalize_stride(std::size_t width, std::size_t stride) noexcept -> std::size_t;

    /**
     * @brief 计算带跨距行主序线性索引。 Compute a strided row-major linear index.
     *
     * @param stride 物理行跨度。 / Physical row stride.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return 线性索引。 / Linear index.
     */
    [[nodiscard]] auto strided_row_major_linear_index(std::size_t stride, std::size_t x, std::size_t y) noexcept
        -> std::size_t;

    /**
     * @brief 正规化块维度。 Normalize a tile extent.
     *
     * @param block_extent 请求块维度。 / Requested tile extent.
     * @return 合法块维度。 / Valid tile extent.
     */
    [[nodiscard]] auto normalize_block_extent(std::size_t block_extent) noexcept -> std::size_t;

    /**
     * @brief 计算分块布局所需的物理容量。 Compute the physical storage extent required by a blocked layout.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param height 逻辑高度。 / Logical height.
     * @param block_width 块宽度。 / Tile width.
     * @param block_height 块高度。 / Tile height.
     * @return 物理槽位数量。 / Number of physical slots.
     */
    [[nodiscard]] auto blocked_storage_extent(
        std::size_t width,
        std::size_t height,
        std::size_t block_width,
        std::size_t block_height) noexcept -> std::size_t;

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
    [[nodiscard]] auto blocked_linear_index(
        std::size_t width,
        std::size_t block_width,
        std::size_t block_height,
        std::size_t x,
        std::size_t y) noexcept -> std::size_t;

    /**
     * @brief 计算覆盖逻辑矩阵的最小 2 的幂边长。 Compute the minimal power-of-two side covering a logical matrix.
     *
     * @param width 逻辑宽度。 / Logical width.
     * @param height 逻辑高度。 / Logical height.
     * @return 覆盖边长。 / Covering side length.
     */
    [[nodiscard]] auto morton_covering_power_of_two(std::size_t width, std::size_t height) noexcept -> std::size_t;

    /**
     * @brief 计算 Morton 线性索引。 Compute a Morton linear index.
     *
     * @param side 内部方阵边长。 / Internal square side length.
     * @param x 横向坐标。 / Horizontal coordinate.
     * @param y 纵向坐标。 / Vertical coordinate.
     * @return Morton 线性索引。 / Morton linear index.
     */
    [[nodiscard]] auto morton_linear_index(std::size_t side, std::size_t x, std::size_t y) noexcept -> std::size_t;

} // namespace mpilab::domain::detail
