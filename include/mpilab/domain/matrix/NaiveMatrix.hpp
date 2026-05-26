#pragma once

#include "mpilab/domain/matrix/detail/MatrixMemory.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

/**
 * @file NaiveMatrix.hpp
 * @brief 朴素矩阵布局声明。 Naive matrix layout declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief 逐行分配矩阵：每一行单独拥有一段存储。 Per-row allocated matrix: each row owns a separate storage segment.
     *
     * @tparam Scalar 标量类型。 / Scalar type.
     * @tparam Allocator 分配器类型。 / Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class JaggedRowMajorMatrix
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
         * @brief 标量指针类型别名。 Scalar pointer type alias.
         */
        using value_pointer = typename std::allocator_traits<allocator_type>::pointer;

    private:
        /**
         * @brief 行指针分配器类型别名。 Row-pointer allocator type alias.
         */
        using row_pointer_allocator_type =
            typename std::allocator_traits<allocator_type>::template rebind_alloc<value_pointer>;

        /**
         * @brief 行指针类型别名。 Row pointer type alias.
         */
        using row_pointer = typename std::allocator_traits<row_pointer_allocator_type>::pointer;

    public:
        /**
         * @brief 使用给定尺寸构造逐行分配矩阵。 Construct a jagged row-major matrix with the given extent.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         */
        JaggedRowMajorMatrix(std::size_t width, std::size_t height)
            : JaggedRowMajorMatrix(width, height, allocator_type())
        {
        }

        /**
         * @brief 使用给定尺寸和分配器构造逐行分配矩阵。 Construct a jagged row-major matrix with the given extent and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param allocator 分配器实例。 / Allocator instance.
         */
        JaggedRowMajorMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : width_(width), height_(height), allocator_(allocator)
        {
            allocate_rows();
        }

        /**
         * @brief 拷贝构造逐行分配矩阵。 Copy-construct a jagged row-major matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        JaggedRowMajorMatrix(const JaggedRowMajorMatrix& other)
            : width_(other.width_), height_(other.height_), allocator_(copy_allocator(other.allocator_))
        {
            allocate_rows();
            copy_from(other);
        }

        /**
         * @brief 移动构造逐行分配矩阵。 Move-construct a jagged row-major matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        JaggedRowMajorMatrix(JaggedRowMajorMatrix&& other) noexcept
            : width_(other.width_), height_(other.height_), rows_(other.rows_), allocator_(std::move(other.allocator_))
        {
            other.width_ = 0;
            other.height_ = 0;
            other.rows_ = row_pointer();
        }

        /**
         * @brief 析构逐行分配矩阵。 Destroy the jagged row-major matrix.
         */
        ~JaggedRowMajorMatrix()
        {
            release_rows();
        }

        /**
         * @brief 以值语义赋值逐行分配矩阵。 Assign the jagged row-major matrix with value semantics.
         *
         * @param other 源矩阵。 / Source matrix.
         * @return 当前矩阵引用。 / Reference to this matrix.
         */
        auto operator=(JaggedRowMajorMatrix other) noexcept -> JaggedRowMajorMatrix&
        {
            swap(other);
            return *this;
        }

        /**
         * @brief 交换两个逐行分配矩阵。 Swap two jagged row-major matrices.
         *
         * @param other 另一个矩阵。 / Another matrix.
         */
        void swap(JaggedRowMajorMatrix& other) noexcept
        {
            using std::swap;

            swap(width_, other.width_);
            swap(height_, other.height_);
            swap(rows_, other.rows_);
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
            return std::to_address(std::to_address(rows_)[y])[x];
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
            std::to_address(std::to_address(rows_)[y])[x] = value;
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
         * @brief 返回逻辑行数。 Return the logical row count.
         *
         * @return 矩阵行数。 / Matrix row count.
         */
        [[nodiscard]] auto row_count() const -> std::size_t
        {
            return height_;
        }

        /**
         * @brief 返回指定行的连续存储视图。 Return a contiguous storage view for the given row.
         *
         * @param y 纵向坐标。 / Vertical coordinate.
         * @return 行存储视图。 / Row storage view.
         */
        [[nodiscard]] auto row_storage(std::size_t y) const -> std::span<const value_type>
        {
            return std::span<const value_type>(std::to_address(std::to_address(rows_)[y]), width_);
        }

        /**
         * @brief 返回每行逻辑宽度。 Return the logical width of each row.
         *
         * @return 行宽度。 / Row width.
         */
        [[nodiscard]] auto row_width() const -> std::size_t
        {
            return width_;
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
         * @brief 分配并值初始化单行元素。 Allocate and value-initialize one row.
         *
         * @param row 待写入的行指针引用。 / Row pointer reference to populate.
         */
        void allocate_row(value_pointer& row)
        {
            if (width_ == 0) {
                row = value_pointer();
                return;
            }

            row = std::allocator_traits<allocator_type>::allocate(allocator_, width_);
            std::size_t constructed = 0;

            try {
                for (; constructed < width_; ++constructed) {
                    std::allocator_traits<allocator_type>::construct(allocator_, std::to_address(row) + constructed);
                }
            } catch (...) {
                for (std::size_t index = 0; index < constructed; ++index) {
                    std::allocator_traits<allocator_type>::destroy(allocator_, std::to_address(row) + index);
                }
                std::allocator_traits<allocator_type>::deallocate(allocator_, row, width_);
                row = value_pointer();
                throw;
            }
        }

        /**
         * @brief 分配全部行句柄和行存储。 Allocate all row handles and row storage segments.
         */
        void allocate_rows()
        {
            if (height_ == 0) {
                return;
            }

            row_pointer_allocator_type row_allocator(allocator_);
            rows_ = std::allocator_traits<row_pointer_allocator_type>::allocate(row_allocator, height_);
            std::size_t constructed_row_handles = 0;

            try {
                for (; constructed_row_handles < height_; ++constructed_row_handles) {
                    std::allocator_traits<row_pointer_allocator_type>::construct(
                        row_allocator,
                        std::to_address(rows_) + constructed_row_handles,
                        value_pointer());
                }

                for (std::size_t row = 0; row < height_; ++row) {
                    allocate_row(std::to_address(rows_)[row]);
                }
            } catch (...) {
                for (std::size_t row = 0; row < constructed_row_handles; ++row) {
                    std::allocator_traits<row_pointer_allocator_type>::destroy(row_allocator, std::to_address(rows_) + row);
                }
                std::allocator_traits<row_pointer_allocator_type>::deallocate(row_allocator, rows_, height_);
                rows_ = row_pointer();
                release_rows();
                throw;
            }
        }

        /**
         * @brief 从另一个矩阵复制元素。 Copy elements from another matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        void copy_from(const JaggedRowMajorMatrix& other)
        {
            for (std::size_t y = 0; y < height_; ++y) {
                for (std::size_t x = 0; x < width_; ++x) {
                    std::to_address(std::to_address(rows_)[y])[x] = std::to_address(std::to_address(other.rows_)[y])[x];
                }
            }
        }

        /**
         * @brief 释放全部行存储和行句柄。 Release all row storage segments and row handles.
         */
        void release_rows() noexcept
        {
            if (std::to_address(rows_) == nullptr) {
                return;
            }

            row_pointer_allocator_type row_allocator(allocator_);

            for (std::size_t row = 0; row < height_; ++row) {
                value_pointer current_row = std::to_address(rows_)[row];
                if (std::to_address(current_row) == nullptr) {
                    std::allocator_traits<row_pointer_allocator_type>::destroy(row_allocator, std::to_address(rows_) + row);
                    continue;
                }

                for (std::size_t index = 0; index < width_; ++index) {
                    std::allocator_traits<allocator_type>::destroy(allocator_, std::to_address(current_row) + index);
                }

                if (width_ != 0) {
                    std::allocator_traits<allocator_type>::deallocate(allocator_, current_row, width_);
                }

                std::allocator_traits<row_pointer_allocator_type>::destroy(row_allocator, std::to_address(rows_) + row);
            }

            std::allocator_traits<row_pointer_allocator_type>::deallocate(row_allocator, rows_, height_);
            rows_ = row_pointer();
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
         * @brief 行指针数组起始指针。 / Pointer to the row-pointer array.
         */
        row_pointer rows_{};

        /**
         * @brief 标量分配器实例。 / Scalar allocator instance.
         */
        allocator_type allocator_{};
    };

} // namespace mpilab::domain
