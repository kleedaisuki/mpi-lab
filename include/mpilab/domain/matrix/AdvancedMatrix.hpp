#pragma once

#include "mpilab/domain/matrix/detail/MatrixMemory.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

/**
 * @file AdvancedMatrix.hpp
 * @brief 高级矩阵布局声明。 Advanced matrix layout declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief 分块行主序矩阵：按固定大小小块顺序存储。 Blocked row-major matrix: stores values in fixed-size tiles.
     *
     * @tparam Scalar 中文：标量类型。 English: Scalar type.
     * @tparam Allocator 中文：分配器类型。 English: Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class BlockedRowMajorMatrix
    {
    public:
        using value_type = Scalar;
        using allocator_type = Allocator;
        using pointer = typename std::allocator_traits<allocator_type>::pointer;

        BlockedRowMajorMatrix(std::size_t width, std::size_t height)
            : BlockedRowMajorMatrix(width, height, 2, 2)
        {
        }

        BlockedRowMajorMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : BlockedRowMajorMatrix(width, height, 2, 2, allocator)
        {
        }

        BlockedRowMajorMatrix(std::size_t width, std::size_t height, std::size_t block_width, std::size_t block_height)
            : BlockedRowMajorMatrix(width, height, block_width, block_height, allocator_type())
        {
        }

        BlockedRowMajorMatrix(
            std::size_t width,
            std::size_t height,
            std::size_t block_width,
            std::size_t block_height,
            const allocator_type& allocator)
            : width_(width),
              height_(height),
              block_width_(detail::normalize_block_extent(block_width)),
              block_height_(detail::normalize_block_extent(block_height)),
              allocator_(allocator)
        {
            allocate_values(storage_size());
        }

        BlockedRowMajorMatrix(const BlockedRowMajorMatrix& other)
            : width_(other.width_),
              height_(other.height_),
              block_width_(other.block_width_),
              block_height_(other.block_height_),
              allocator_(copy_allocator(other.allocator_))
        {
            allocate_values(other.storage_size());
            copy_from(other);
        }

        BlockedRowMajorMatrix(BlockedRowMajorMatrix&& other) noexcept
            : width_(other.width_),
              height_(other.height_),
              block_width_(other.block_width_),
              block_height_(other.block_height_),
              values_(other.values_),
              allocator_(std::move(other.allocator_))
        {
            other.width_ = 0;
            other.height_ = 0;
            other.block_width_ = 1;
            other.block_height_ = 1;
            other.values_ = pointer();
        }

        ~BlockedRowMajorMatrix()
        {
            release_values();
        }

        auto operator=(BlockedRowMajorMatrix other) noexcept -> BlockedRowMajorMatrix&
        {
            swap(other);
            return *this;
        }

        void swap(BlockedRowMajorMatrix& other) noexcept
        {
            using std::swap;
            swap(width_, other.width_);
            swap(height_, other.height_);
            swap(block_width_, other.block_width_);
            swap(block_height_, other.block_height_);
            swap(values_, other.values_);
            swap(allocator_, other.allocator_);
        }

        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> const value_type&
        {
            return std::to_address(values_)[detail::blocked_linear_index(width_, block_width_, block_height_, x, y)];
        }

        void set(std::size_t x, std::size_t y, const value_type& value)
        {
            std::to_address(values_)[detail::blocked_linear_index(width_, block_width_, block_height_, x, y)] = value;
        }

        [[nodiscard]] auto width() const -> std::size_t { return width_; }
        [[nodiscard]] auto height() const -> std::size_t { return height_; }
        [[nodiscard]] auto block_width() const -> std::size_t { return block_width_; }
        [[nodiscard]] auto block_height() const -> std::size_t { return block_height_; }

        [[nodiscard]] auto storage_size() const -> std::size_t
        {
            return detail::blocked_storage_extent(width_, height_, block_width_, block_height_);
        }

        [[nodiscard]] auto linear_index(std::size_t x, std::size_t y) const -> std::size_t
        {
            return detail::blocked_linear_index(width_, block_width_, block_height_, x, y);
        }

        [[nodiscard]] auto storage() const -> std::span<const value_type>
        {
            return std::span<const value_type>(std::to_address(values_), storage_size());
        }

        [[nodiscard]] auto get_allocator() const -> const allocator_type&
        {
            return allocator_;
        }

    private:
        [[nodiscard]] static auto copy_allocator(const allocator_type& allocator) -> allocator_type
        {
            return std::allocator_traits<allocator_type>::select_on_container_copy_construction(allocator);
        }

        void allocate_values(std::size_t size)
        {
            if (size == 0) {
                return;
            }

            values_ = std::allocator_traits<allocator_type>::allocate(allocator_, size);
            std::size_t constructed = 0;
            try {
                for (; constructed < size; ++constructed) {
                    std::allocator_traits<allocator_type>::construct(allocator_, std::to_address(values_) + constructed);
                }
            } catch (...) {
                for (std::size_t index = 0; index < constructed; ++index) {
                    std::allocator_traits<allocator_type>::destroy(allocator_, std::to_address(values_) + index);
                }
                std::allocator_traits<allocator_type>::deallocate(allocator_, values_, size);
                values_ = pointer();
                throw;
            }
        }

        void copy_from(const BlockedRowMajorMatrix& other)
        {
            for (std::size_t index = 0; index < storage_size(); ++index) {
                std::to_address(values_)[index] = std::to_address(other.values_)[index];
            }
        }

        void release_values() noexcept
        {
            const std::size_t size = storage_size();
            if (size == 0 || std::to_address(values_) == nullptr) {
                return;
            }

            for (std::size_t index = 0; index < size; ++index) {
                std::allocator_traits<allocator_type>::destroy(allocator_, std::to_address(values_) + index);
            }

            std::allocator_traits<allocator_type>::deallocate(allocator_, values_, size);
            values_ = pointer();
        }

        std::size_t width_{0};
        std::size_t height_{0};
        std::size_t block_width_{1};
        std::size_t block_height_{1};
        pointer values_{};
        allocator_type allocator_{};
    };

    /**
     * @brief Morton 矩阵：使用 Z-order（Z 序, Z-order）近似保存二维局部性。 Morton matrix: preserves two-dimensional locality approximately via Z-order.
     *
     * @tparam Scalar 中文：标量类型。 English: Scalar type.
     * @tparam Allocator 中文：分配器类型。 English: Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class MortonMatrix
    {
    public:
        using value_type = Scalar;
        using allocator_type = Allocator;
        using pointer = typename std::allocator_traits<allocator_type>::pointer;

        MortonMatrix(std::size_t width, std::size_t height)
            : MortonMatrix(width, height, allocator_type())
        {
        }

        MortonMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : width_(width), height_(height), side_(detail::morton_covering_power_of_two(width, height)), allocator_(allocator)
        {
            allocate_values(storage_size());
        }

        MortonMatrix(const MortonMatrix& other)
            : width_(other.width_), height_(other.height_), side_(other.side_), allocator_(copy_allocator(other.allocator_))
        {
            allocate_values(other.storage_size());
            copy_from(other);
        }

        MortonMatrix(MortonMatrix&& other) noexcept
            : width_(other.width_),
              height_(other.height_),
              side_(other.side_),
              values_(other.values_),
              allocator_(std::move(other.allocator_))
        {
            other.width_ = 0;
            other.height_ = 0;
            other.side_ = 1;
            other.values_ = pointer();
        }

        ~MortonMatrix()
        {
            release_values();
        }

        auto operator=(MortonMatrix other) noexcept -> MortonMatrix&
        {
            swap(other);
            return *this;
        }

        void swap(MortonMatrix& other) noexcept
        {
            using std::swap;
            swap(width_, other.width_);
            swap(height_, other.height_);
            swap(side_, other.side_);
            swap(values_, other.values_);
            swap(allocator_, other.allocator_);
        }

        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> const value_type&
        {
            return std::to_address(values_)[detail::morton_linear_index(side_, x, y)];
        }

        void set(std::size_t x, std::size_t y, const value_type& value)
        {
            std::to_address(values_)[detail::morton_linear_index(side_, x, y)] = value;
        }

        [[nodiscard]] auto width() const -> std::size_t { return width_; }
        [[nodiscard]] auto height() const -> std::size_t { return height_; }
        [[nodiscard]] auto side() const -> std::size_t { return side_; }
        [[nodiscard]] auto storage_size() const -> std::size_t { return side_ * side_; }

        [[nodiscard]] auto linear_index(std::size_t x, std::size_t y) const -> std::size_t
        {
            return detail::morton_linear_index(side_, x, y);
        }

        [[nodiscard]] auto storage() const -> std::span<const value_type>
        {
            return std::span<const value_type>(std::to_address(values_), storage_size());
        }

        [[nodiscard]] auto get_allocator() const -> const allocator_type&
        {
            return allocator_;
        }

    private:
        [[nodiscard]] static auto copy_allocator(const allocator_type& allocator) -> allocator_type
        {
            return std::allocator_traits<allocator_type>::select_on_container_copy_construction(allocator);
        }

        void allocate_values(std::size_t size)
        {
            if (size == 0) {
                return;
            }

            values_ = std::allocator_traits<allocator_type>::allocate(allocator_, size);
            std::size_t constructed = 0;
            try {
                for (; constructed < size; ++constructed) {
                    std::allocator_traits<allocator_type>::construct(allocator_, std::to_address(values_) + constructed);
                }
            } catch (...) {
                for (std::size_t index = 0; index < constructed; ++index) {
                    std::allocator_traits<allocator_type>::destroy(allocator_, std::to_address(values_) + index);
                }
                std::allocator_traits<allocator_type>::deallocate(allocator_, values_, size);
                values_ = pointer();
                throw;
            }
        }

        void copy_from(const MortonMatrix& other)
        {
            for (std::size_t index = 0; index < storage_size(); ++index) {
                std::to_address(values_)[index] = std::to_address(other.values_)[index];
            }
        }

        void release_values() noexcept
        {
            const std::size_t size = storage_size();
            if (size == 0 || std::to_address(values_) == nullptr) {
                return;
            }

            for (std::size_t index = 0; index < size; ++index) {
                std::allocator_traits<allocator_type>::destroy(allocator_, std::to_address(values_) + index);
            }

            std::allocator_traits<allocator_type>::deallocate(allocator_, values_, size);
            values_ = pointer();
        }

        std::size_t width_{0};
        std::size_t height_{0};
        std::size_t side_{1};
        pointer values_{};
        allocator_type allocator_{};
    };

} // namespace mpilab::domain
