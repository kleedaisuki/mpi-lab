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
     * @tparam Scalar 标量类型。 / Scalar type.
     * @tparam Allocator 分配器类型。 / Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class BlockedRowMajorMatrix
    {
    public:
        /**
         * @brief 标量类型别名。 Scalar type alias.
         */
        using value_type = Scalar;

        /**
         * @brief 分配器类型别名。 Allocator type alias.
         */
        using allocator_type = Allocator;

        /**
         * @brief 指针类型别名。 Pointer type alias.
         */
        using pointer = typename std::allocator_traits<allocator_type>::pointer;

        /**
         * @brief 使用默认块大小构造分块矩阵。 Construct a blocked matrix with the default tile extent.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         */
        BlockedRowMajorMatrix(std::size_t width, std::size_t height)
            : BlockedRowMajorMatrix(width, height, 2, 2)
        {
        }

        /**
         * @brief 使用默认块大小和分配器构造分块矩阵。 Construct a blocked matrix with the default tile extent and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param allocator 分配器实例。 / Allocator instance.
         */
        BlockedRowMajorMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : BlockedRowMajorMatrix(width, height, 2, 2, allocator)
        {
        }

        /**
         * @brief 使用显式块大小构造分块矩阵。 Construct a blocked matrix with an explicit tile extent.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param block_width 块宽度。 / Tile width.
         * @param block_height 块高度。 / Tile height.
         */
        BlockedRowMajorMatrix(std::size_t width, std::size_t height, std::size_t block_width, std::size_t block_height)
            : BlockedRowMajorMatrix(width, height, block_width, block_height, allocator_type())
        {
        }

        /**
         * @brief 使用显式块大小和分配器构造分块矩阵。 Construct a blocked matrix with an explicit tile extent and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param block_width 块宽度。 / Tile width.
         * @param block_height 块高度。 / Tile height.
         * @param allocator 分配器实例。 / Allocator instance.
         */
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

        /**
         * @brief 拷贝构造分块矩阵。 Copy-construct a blocked matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
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

        /**
         * @brief 移动构造分块矩阵。 Move-construct a blocked matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
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

        /**
         * @brief 析构分块矩阵。 Destroy the blocked matrix.
         */
        ~BlockedRowMajorMatrix()
        {
            release_values();
        }

        /**
         * @brief 以值语义赋值分块矩阵。 Assign the blocked matrix with value semantics.
         *
         * @param other 源矩阵。 / Source matrix.
         * @return 当前矩阵引用。 / Reference to this matrix.
         */
        auto operator=(BlockedRowMajorMatrix other) noexcept -> BlockedRowMajorMatrix&
        {
            swap(other);
            return *this;
        }

        /**
         * @brief 交换两个分块矩阵。 Swap two blocked matrices.
         *
         * @param other 另一个矩阵。 / Another matrix.
         */
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

        /**
         * @brief 读取指定坐标的矩阵值。 Read the matrix value at the given coordinate.
         *
         * @param x 横向坐标。 / Horizontal coordinate.
         * @param y 纵向坐标。 / Vertical coordinate.
         * @return 坐标对应的标量值。 / Scalar value at the coordinate.
         */
        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> const value_type&
        {
            return std::to_address(values_)[detail::blocked_linear_index(width_, block_width_, block_height_, x, y)];
        }

        /**
         * @brief 写入指定坐标的矩阵值。 Write a matrix value at the given coordinate.
         *
         * @param x 横向坐标。 / Horizontal coordinate.
         * @param y 纵向坐标。 / Vertical coordinate.
         * @param value 待写入的标量值。 / Scalar value to store.
         */
        void set(std::size_t x, std::size_t y, const value_type& value)
        {
            std::to_address(values_)[detail::blocked_linear_index(width_, block_width_, block_height_, x, y)] = value;
        }

        /**
         * @brief 返回逻辑宽度。 Return the logical width.
         *
         * @return 矩阵宽度。 / Matrix width.
         */
        [[nodiscard]] auto width() const -> std::size_t { return width_; }

        /**
         * @brief 返回逻辑高度。 Return the logical height.
         *
         * @return 矩阵高度。 / Matrix height.
         */
        [[nodiscard]] auto height() const -> std::size_t { return height_; }

        /**
         * @brief 返回正规化后的块宽度。 Return the normalized tile width.
         *
         * @return 块宽度。 / Tile width.
         */
        [[nodiscard]] auto block_width() const -> std::size_t { return block_width_; }

        /**
         * @brief 返回正规化后的块高度。 Return the normalized tile height.
         *
         * @return 块高度。 / Tile height.
         */
        [[nodiscard]] auto block_height() const -> std::size_t { return block_height_; }

        /**
         * @brief 返回线性存储槽位数量。 Return the number of linear storage slots.
         *
         * @return 底层存储大小。 / Underlying storage size.
         */
        [[nodiscard]] auto storage_size() const -> std::size_t
        {
            return detail::blocked_storage_extent(width_, height_, block_width_, block_height_);
        }

        /**
         * @brief 返回坐标对应的线性索引。 Return the linear index for the coordinate.
         *
         * @param x 横向坐标。 / Horizontal coordinate.
         * @param y 纵向坐标。 / Vertical coordinate.
         * @return 线性存储索引。 / Linear storage index.
         */
        [[nodiscard]] auto linear_index(std::size_t x, std::size_t y) const -> std::size_t
        {
            return detail::blocked_linear_index(width_, block_width_, block_height_, x, y);
        }

        /**
         * @brief 返回底层连续存储。 Return the underlying contiguous storage.
         *
         * @return 底层存储视图。 / View of the underlying storage.
         */
        [[nodiscard]] auto storage() const -> std::span<const value_type>
        {
            return std::span<const value_type>(std::to_address(values_), storage_size());
        }

        /**
         * @brief 返回分配器实例。 Return the allocator instance.
         *
         * @return 分配器实例引用。 / Reference to the allocator instance.
         */
        [[nodiscard]] auto get_allocator() const -> const allocator_type&
        {
            return allocator_;
        }

    private:
        /**
         * @brief 选择拷贝构造使用的分配器。 Select the allocator used during copy construction.
         *
         * @param allocator 源分配器。 / Source allocator.
         * @return 目标分配器。 / Target allocator.
         */
        [[nodiscard]] static auto copy_allocator(const allocator_type& allocator) -> allocator_type
        {
            return std::allocator_traits<allocator_type>::select_on_container_copy_construction(allocator);
        }

        /**
         * @brief 分配并值初始化全部元素。 Allocate and value-initialize all elements.
         *
         * @param size 元素数量。 / Element count.
         */
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

        /**
         * @brief 从另一个矩阵复制元素。 Copy elements from another matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        void copy_from(const BlockedRowMajorMatrix& other)
        {
            for (std::size_t index = 0; index < storage_size(); ++index) {
                std::to_address(values_)[index] = std::to_address(other.values_)[index];
            }
        }

        /**
         * @brief 释放底层存储。 Release the underlying storage.
         */
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

        /**
         * @brief 矩阵逻辑宽度。 / Matrix logical width.
         */
        std::size_t width_{0};

        /**
         * @brief 矩阵逻辑高度。 / Matrix logical height.
         */
        std::size_t height_{0};

        /**
         * @brief 块逻辑宽度。 / Tile logical width.
         */
        std::size_t block_width_{1};

        /**
         * @brief 块逻辑高度。 / Tile logical height.
         */
        std::size_t block_height_{1};

        /**
         * @brief 连续存储起始指针。 / Contiguous storage pointer.
         */
        pointer values_{};

        /**
         * @brief 分配器实例。 / Allocator instance.
         */
        allocator_type allocator_{};
    };

    /**
     * @brief Morton 矩阵：使用 Z-order（Z 序, Z-order）近似保存二维局部性。 Morton matrix: preserves two-dimensional locality approximately via Z-order.
     *
     * @tparam Scalar 标量类型。 / Scalar type.
     * @tparam Allocator 分配器类型。 / Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class MortonMatrix
    {
    public:
        /**
         * @brief 标量类型别名。 Scalar type alias.
         */
        using value_type = Scalar;

        /**
         * @brief 分配器类型别名。 Allocator type alias.
         */
        using allocator_type = Allocator;

        /**
         * @brief 指针类型别名。 Pointer type alias.
         */
        using pointer = typename std::allocator_traits<allocator_type>::pointer;

        /**
         * @brief 使用给定尺寸构造 Morton 矩阵。 Construct a Morton matrix with the given extent.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         */
        MortonMatrix(std::size_t width, std::size_t height)
            : MortonMatrix(width, height, allocator_type())
        {
        }

        /**
         * @brief 使用给定尺寸和分配器构造 Morton 矩阵。 Construct a Morton matrix with the given extent and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param allocator 分配器实例。 / Allocator instance.
         */
        MortonMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : width_(width), height_(height), side_(detail::morton_covering_power_of_two(width, height)), allocator_(allocator)
        {
            allocate_values(storage_size());
        }

        /**
         * @brief 拷贝构造 Morton 矩阵。 Copy-construct a Morton matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        MortonMatrix(const MortonMatrix& other)
            : width_(other.width_), height_(other.height_), side_(other.side_), allocator_(copy_allocator(other.allocator_))
        {
            allocate_values(other.storage_size());
            copy_from(other);
        }

        /**
         * @brief 移动构造 Morton 矩阵。 Move-construct a Morton matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
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

        /**
         * @brief 析构 Morton 矩阵。 Destroy the Morton matrix.
         */
        ~MortonMatrix()
        {
            release_values();
        }

        /**
         * @brief 以值语义赋值 Morton 矩阵。 Assign the Morton matrix with value semantics.
         *
         * @param other 源矩阵。 / Source matrix.
         * @return 当前矩阵引用。 / Reference to this matrix.
         */
        auto operator=(MortonMatrix other) noexcept -> MortonMatrix&
        {
            swap(other);
            return *this;
        }

        /**
         * @brief 交换两个 Morton 矩阵。 Swap two Morton matrices.
         *
         * @param other 另一个矩阵。 / Another matrix.
         */
        void swap(MortonMatrix& other) noexcept
        {
            using std::swap;
            swap(width_, other.width_);
            swap(height_, other.height_);
            swap(side_, other.side_);
            swap(values_, other.values_);
            swap(allocator_, other.allocator_);
        }

        /**
         * @brief 读取指定坐标的矩阵值。 Read the matrix value at the given coordinate.
         *
         * @param x 横向坐标。 / Horizontal coordinate.
         * @param y 纵向坐标。 / Vertical coordinate.
         * @return 坐标对应的标量值。 / Scalar value at the coordinate.
         */
        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> const value_type&
        {
            return std::to_address(values_)[detail::morton_linear_index(side_, x, y)];
        }

        /**
         * @brief 写入指定坐标的矩阵值。 Write a matrix value at the given coordinate.
         *
         * @param x 横向坐标。 / Horizontal coordinate.
         * @param y 纵向坐标。 / Vertical coordinate.
         * @param value 待写入的标量值。 / Scalar value to store.
         */
        void set(std::size_t x, std::size_t y, const value_type& value)
        {
            std::to_address(values_)[detail::morton_linear_index(side_, x, y)] = value;
        }

        /**
         * @brief 返回逻辑宽度。 Return the logical width.
         *
         * @return 矩阵宽度。 / Matrix width.
         */
        [[nodiscard]] auto width() const -> std::size_t { return width_; }

        /**
         * @brief 返回逻辑高度。 Return the logical height.
         *
         * @return 矩阵高度。 / Matrix height.
         */
        [[nodiscard]] auto height() const -> std::size_t { return height_; }

        /**
         * @brief 返回覆盖逻辑矩阵的内部方阵边长。 Return the internal square side covering the logical matrix.
         *
         * @return 内部方阵边长。 / Internal square side length.
         */
        [[nodiscard]] auto side() const -> std::size_t { return side_; }

        /**
         * @brief 返回线性存储槽位数量。 Return the number of linear storage slots.
         *
         * @return 底层存储大小。 / Underlying storage size.
         */
        [[nodiscard]] auto storage_size() const -> std::size_t { return side_ * side_; }

        /**
         * @brief 返回坐标对应的 Morton 线性索引。 Return the Morton linear index for the coordinate.
         *
         * @param x 横向坐标。 / Horizontal coordinate.
         * @param y 纵向坐标。 / Vertical coordinate.
         * @return Morton 线性索引。 / Morton linear index.
         */
        [[nodiscard]] auto linear_index(std::size_t x, std::size_t y) const -> std::size_t
        {
            return detail::morton_linear_index(side_, x, y);
        }

        /**
         * @brief 返回底层连续存储。 Return the underlying contiguous storage.
         *
         * @return 底层存储视图。 / View of the underlying storage.
         */
        [[nodiscard]] auto storage() const -> std::span<const value_type>
        {
            return std::span<const value_type>(std::to_address(values_), storage_size());
        }

        /**
         * @brief 返回分配器实例。 Return the allocator instance.
         *
         * @return 分配器实例引用。 / Reference to the allocator instance.
         */
        [[nodiscard]] auto get_allocator() const -> const allocator_type&
        {
            return allocator_;
        }

    private:
        /**
         * @brief 选择拷贝构造使用的分配器。 Select the allocator used during copy construction.
         *
         * @param allocator 源分配器。 / Source allocator.
         * @return 目标分配器。 / Target allocator.
         */
        [[nodiscard]] static auto copy_allocator(const allocator_type& allocator) -> allocator_type
        {
            return std::allocator_traits<allocator_type>::select_on_container_copy_construction(allocator);
        }

        /**
         * @brief 分配并值初始化全部元素。 Allocate and value-initialize all elements.
         *
         * @param size 元素数量。 / Element count.
         */
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

        /**
         * @brief 从另一个矩阵复制元素。 Copy elements from another matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        void copy_from(const MortonMatrix& other)
        {
            for (std::size_t index = 0; index < storage_size(); ++index) {
                std::to_address(values_)[index] = std::to_address(other.values_)[index];
            }
        }

        /**
         * @brief 释放底层存储。 Release the underlying storage.
         */
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

        /**
         * @brief 矩阵逻辑宽度。 / Matrix logical width.
         */
        std::size_t width_{0};

        /**
         * @brief 矩阵逻辑高度。 / Matrix logical height.
         */
        std::size_t height_{0};

        /**
         * @brief 内部 Morton 方阵边长。 / Internal Morton square side length.
         */
        std::size_t side_{1};

        /**
         * @brief 连续存储起始指针。 / Contiguous storage pointer.
         */
        pointer values_{};

        /**
         * @brief 分配器实例。 / Allocator instance.
         */
        allocator_type allocator_{};
    };

} // namespace mpilab::domain
