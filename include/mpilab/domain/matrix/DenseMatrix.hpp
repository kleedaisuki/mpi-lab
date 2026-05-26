#pragma once

#include "mpilab/domain/matrix/detail/MatrixMemory.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

/**
 * @file DenseMatrix.hpp
 * @brief 稠密矩阵布局声明。 Dense matrix layout declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief 行主序矩阵：按行连续存储二维标量。 Row-major matrix: stores two-dimensional scalars contiguously by rows.
     *
     * @tparam Scalar 标量类型。 / Scalar type.
     * @tparam Allocator 分配器类型。 / Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class RowMajorMatrix
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
         * @brief 使用给定尺寸构造行主序矩阵。 Construct a row-major matrix with the given extent.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         */
        RowMajorMatrix(std::size_t width, std::size_t height)
            : RowMajorMatrix(width, height, allocator_type())
        {
        }

        /**
         * @brief 使用给定尺寸和分配器构造行主序矩阵。 Construct a row-major matrix with the given extent and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param allocator 分配器实例。 / Allocator instance.
         */
        RowMajorMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : width_(width), height_(height), allocator_(allocator)
        {
            allocate_values(width_ * height_);
        }

        /**
         * @brief 拷贝构造行主序矩阵。 Copy-construct a row-major matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        RowMajorMatrix(const RowMajorMatrix& other)
            : width_(other.width_), height_(other.height_), allocator_(copy_allocator(other.allocator_))
        {
            allocate_values(other.storage_size());
            copy_from(other);
        }

        /**
         * @brief 移动构造行主序矩阵。 Move-construct a row-major matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        RowMajorMatrix(RowMajorMatrix&& other) noexcept
            : width_(other.width_), height_(other.height_), values_(other.values_), allocator_(std::move(other.allocator_))
        {
            other.width_ = 0;
            other.height_ = 0;
            other.values_ = pointer();
        }

        /**
         * @brief 析构行主序矩阵。 Destroy the row-major matrix.
         */
        ~RowMajorMatrix()
        {
            release_values();
        }

        /**
         * @brief 以值语义赋值行主序矩阵。 Assign the row-major matrix with value semantics.
         *
         * @param other 源矩阵。 / Source matrix.
         * @return 当前矩阵引用。 / Reference to this matrix.
         */
        auto operator=(RowMajorMatrix other) noexcept -> RowMajorMatrix&
        {
            swap(other);
            return *this;
        }

        /**
         * @brief 交换两个行主序矩阵。 Swap two row-major matrices.
         *
         * @param other 另一个矩阵。 / Another matrix.
         */
        void swap(RowMajorMatrix& other) noexcept
        {
            using std::swap;

            swap(width_, other.width_);
            swap(height_, other.height_);
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
            return std::to_address(values_)[detail::row_major_linear_index(width_, x, y)];
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
            std::to_address(values_)[detail::row_major_linear_index(width_, x, y)] = value;
        }

        /**
         * @brief 返回逻辑宽度。 Return the logical width.
         *
         * @return 矩阵宽度。 / Matrix width.
         */
        [[nodiscard]] auto width() const -> std::size_t
        {
            return width_;
        }

        /**
         * @brief 返回逻辑高度。 Return the logical height.
         *
         * @return 矩阵高度。 / Matrix height.
         */
        [[nodiscard]] auto height() const -> std::size_t
        {
            return height_;
        }

        /**
         * @brief 返回线性存储槽位数量。 Return the number of linear storage slots.
         *
         * @return 底层存储大小。 / Underlying storage size.
         */
        [[nodiscard]] auto storage_size() const -> std::size_t
        {
            return width_ * height_;
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
            return detail::row_major_linear_index(width_, x, y);
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
        void copy_from(const RowMajorMatrix& other)
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
         * @brief 连续存储起始指针。 / Contiguous storage pointer.
         */
        pointer values_{};

        /**
         * @brief 分配器实例。 / Allocator instance.
         */
        allocator_type allocator_{};
    };

    /**
     * @brief 列主序矩阵：按列连续存储二维标量。 Column-major matrix: stores two-dimensional scalars contiguously by columns.
     *
     * @tparam Scalar 标量类型。 / Scalar type.
     * @tparam Allocator 分配器类型。 / Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class ColumnMajorMatrix
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
         * @brief 使用给定尺寸构造列主序矩阵。 Construct a column-major matrix with the given extent.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         */
        ColumnMajorMatrix(std::size_t width, std::size_t height)
            : ColumnMajorMatrix(width, height, allocator_type())
        {
        }

        /**
         * @brief 使用给定尺寸和分配器构造列主序矩阵。 Construct a column-major matrix with the given extent and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param allocator 分配器实例。 / Allocator instance.
         */
        ColumnMajorMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : width_(width), height_(height), allocator_(allocator)
        {
            allocate_values(width_ * height_);
        }

        /**
         * @brief 拷贝构造列主序矩阵。 Copy-construct a column-major matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        ColumnMajorMatrix(const ColumnMajorMatrix& other)
            : width_(other.width_), height_(other.height_), allocator_(copy_allocator(other.allocator_))
        {
            allocate_values(other.storage_size());
            copy_from(other);
        }

        /**
         * @brief 移动构造列主序矩阵。 Move-construct a column-major matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        ColumnMajorMatrix(ColumnMajorMatrix&& other) noexcept
            : width_(other.width_), height_(other.height_), values_(other.values_), allocator_(std::move(other.allocator_))
        {
            other.width_ = 0;
            other.height_ = 0;
            other.values_ = pointer();
        }

        /**
         * @brief 析构列主序矩阵。 Destroy the column-major matrix.
         */
        ~ColumnMajorMatrix()
        {
            release_values();
        }

        /**
         * @brief 以值语义赋值列主序矩阵。 Assign the column-major matrix with value semantics.
         *
         * @param other 源矩阵。 / Source matrix.
         * @return 当前矩阵引用。 / Reference to this matrix.
         */
        auto operator=(ColumnMajorMatrix other) noexcept -> ColumnMajorMatrix&
        {
            swap(other);
            return *this;
        }

        /**
         * @brief 交换两个列主序矩阵。 Swap two column-major matrices.
         *
         * @param other 另一个矩阵。 / Another matrix.
         */
        void swap(ColumnMajorMatrix& other) noexcept
        {
            using std::swap;

            swap(width_, other.width_);
            swap(height_, other.height_);
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
            return std::to_address(values_)[detail::column_major_linear_index(height_, x, y)];
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
            std::to_address(values_)[detail::column_major_linear_index(height_, x, y)] = value;
        }

        /**
         * @brief 返回逻辑宽度。 Return the logical width.
         *
         * @return 矩阵宽度。 / Matrix width.
         */
        [[nodiscard]] auto width() const -> std::size_t
        {
            return width_;
        }

        /**
         * @brief 返回逻辑高度。 Return the logical height.
         *
         * @return 矩阵高度。 / Matrix height.
         */
        [[nodiscard]] auto height() const -> std::size_t
        {
            return height_;
        }

        /**
         * @brief 返回线性存储槽位数量。 Return the number of linear storage slots.
         *
         * @return 底层存储大小。 / Underlying storage size.
         */
        [[nodiscard]] auto storage_size() const -> std::size_t
        {
            return width_ * height_;
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
            return detail::column_major_linear_index(height_, x, y);
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
        void copy_from(const ColumnMajorMatrix& other)
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
         * @brief 连续存储起始指针。 / Contiguous storage pointer.
         */
        pointer values_{};

        /**
         * @brief 分配器实例。 / Allocator instance.
         */
        allocator_type allocator_{};
    };

} // namespace mpilab::domain
