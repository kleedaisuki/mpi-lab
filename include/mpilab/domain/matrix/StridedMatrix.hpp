#pragma once

#include "mpilab/domain/matrix/detail/MatrixMemory.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <utility>

/**
 * @file StridedMatrix.hpp
 * @brief 带跨距矩阵布局声明。 Strided matrix layout declarations.
 */

namespace mpilab::domain
{

    /**
     * @brief 行主序带跨距矩阵：每行可带尾部填充。 Strided row-major matrix: each row may include tail padding.
     *
     * @tparam Scalar 标量类型。 / Scalar type.
     * @tparam Allocator 分配器类型。 / Allocator type.
     */
    template <typename Scalar = double, typename Allocator = std::allocator<Scalar>>
        requires detail::StandardAllocator<Scalar, Allocator>
    class StridedRowMajorMatrix
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
         * @brief 使用紧凑行跨度构造带跨距矩阵。 Construct a strided matrix using a compact row pitch.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         */
        StridedRowMajorMatrix(std::size_t width, std::size_t height)
            : StridedRowMajorMatrix(width, height, width)
        {
        }

        /**
         * @brief 使用紧凑行跨度和分配器构造带跨距矩阵。 Construct a strided matrix with a compact row pitch and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param allocator 分配器实例。 / Allocator instance.
         */
        StridedRowMajorMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : StridedRowMajorMatrix(width, height, width, allocator)
        {
        }

        /**
         * @brief 使用显式行跨度构造带跨距矩阵。 Construct a strided matrix using an explicit row pitch.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param stride 每行实际槽位数。 / Physical slots per row.
         */
        StridedRowMajorMatrix(std::size_t width, std::size_t height, std::size_t stride)
            : StridedRowMajorMatrix(width, height, stride, allocator_type())
        {
        }

        /**
         * @brief 使用显式行跨度和分配器构造带跨距矩阵。 Construct a strided matrix using an explicit row pitch and allocator.
         *
         * @param width 逻辑宽度。 / Logical width.
         * @param height 逻辑高度。 / Logical height.
         * @param stride 每行实际槽位数。 / Physical slots per row.
         * @param allocator 分配器实例。 / Allocator instance.
         */
        StridedRowMajorMatrix(std::size_t width, std::size_t height, std::size_t stride, const allocator_type& allocator)
            : width_(width), height_(height), stride_(detail::normalize_stride(width, stride)), allocator_(allocator)
        {
            allocate_values(storage_size());
        }

        /**
         * @brief 拷贝构造带跨距矩阵。 Copy-construct a strided matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        StridedRowMajorMatrix(const StridedRowMajorMatrix& other)
            : width_(other.width_),
              height_(other.height_),
              stride_(other.stride_),
              allocator_(copy_allocator(other.allocator_))
        {
            allocate_values(other.storage_size());
            copy_from(other);
        }

        /**
         * @brief 移动构造带跨距矩阵。 Move-construct a strided matrix.
         *
         * @param other 源矩阵。 / Source matrix.
         */
        StridedRowMajorMatrix(StridedRowMajorMatrix&& other) noexcept
            : width_(other.width_),
              height_(other.height_),
              stride_(other.stride_),
              values_(other.values_),
              allocator_(std::move(other.allocator_))
        {
            other.width_ = 0;
            other.height_ = 0;
            other.stride_ = 0;
            other.values_ = pointer();
        }

        /**
         * @brief 析构带跨距矩阵。 Destroy the strided matrix.
         */
        ~StridedRowMajorMatrix()
        {
            release_values();
        }

        /**
         * @brief 以值语义赋值带跨距矩阵。 Assign the strided matrix with value semantics.
         *
         * @param other 源矩阵。 / Source matrix.
         * @return 当前矩阵引用。 / Reference to this matrix.
         */
        auto operator=(StridedRowMajorMatrix other) noexcept -> StridedRowMajorMatrix&
        {
            swap(other);
            return *this;
        }

        /**
         * @brief 交换两个带跨距矩阵。 Swap two strided matrices.
         *
         * @param other 另一个矩阵。 / Another matrix.
         */
        void swap(StridedRowMajorMatrix& other) noexcept
        {
            using std::swap;

            swap(width_, other.width_);
            swap(height_, other.height_);
            swap(stride_, other.stride_);
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
            return std::to_address(values_)[detail::strided_row_major_linear_index(stride_, x, y)];
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
            std::to_address(values_)[detail::strided_row_major_linear_index(stride_, x, y)] = value;
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
         * @brief 返回物理行跨度。 Return the physical row pitch.
         *
         * @return 每行物理槽位数。 / Physical slot count per row.
         */
        [[nodiscard]] auto stride() const -> std::size_t
        {
            return stride_;
        }

        /**
         * @brief 返回线性存储槽位数量。 Return the number of linear storage slots.
         *
         * @return 底层存储大小。 / Underlying storage size.
         */
        [[nodiscard]] auto storage_size() const -> std::size_t
        {
            return stride_ * height_;
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
            return detail::strided_row_major_linear_index(stride_, x, y);
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

        void copy_from(const StridedRowMajorMatrix& other)
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

        /**
         * @brief 矩阵逻辑宽度。 / Matrix logical width.
         */
        std::size_t width_{0};

        /**
         * @brief 矩阵逻辑高度。 / Matrix logical height.
         */
        std::size_t height_{0};

        /**
         * @brief 矩阵物理行跨度。 / Matrix physical row pitch.
         */
        std::size_t stride_{0};

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
