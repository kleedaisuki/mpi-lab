#include "mpilab/domain/matrix/detail/MatrixMemory.hpp"

namespace mpilab::domain::detail
{

    auto row_major_linear_index(std::size_t width, std::size_t x, std::size_t y) noexcept -> std::size_t
    {
        return y * width + x;
    }

    auto column_major_linear_index(std::size_t height, std::size_t x, std::size_t y) noexcept -> std::size_t
    {
        return x * height + y;
    }

    auto normalize_stride(std::size_t width, std::size_t stride) noexcept -> std::size_t
    {
        return stride < width ? width : stride;
    }

    auto strided_row_major_linear_index(std::size_t stride, std::size_t x, std::size_t y) noexcept -> std::size_t
    {
        return y * stride + x;
    }

    auto normalize_block_extent(std::size_t block_extent) noexcept -> std::size_t
    {
        return block_extent == 0 ? 1 : block_extent;
    }

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

    auto morton_covering_power_of_two(std::size_t width, std::size_t height) noexcept -> std::size_t
    {
        std::size_t side = 1;
        const std::size_t required = width > height ? width : height;

        while (side < required) {
            side *= 2;
        }

        return side;
    }

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
