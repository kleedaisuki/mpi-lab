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
     * @tparam Scalar 中文：标量类型。 English: Scalar type.
     * @tparam Allocator 中文：分配器类型。 English: Allocator type.
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
        JaggedRowMajorMatrix(std::size_t width, std::size_t height)
            : JaggedRowMajorMatrix(width, height, allocator_type())
        {
        }

        JaggedRowMajorMatrix(std::size_t width, std::size_t height, const allocator_type& allocator)
            : width_(width), height_(height), allocator_(allocator)
        {
            allocate_rows();
        }

        JaggedRowMajorMatrix(const JaggedRowMajorMatrix& other)
            : width_(other.width_), height_(other.height_), allocator_(copy_allocator(other.allocator_))
        {
            allocate_rows();
            copy_from(other);
        }

        JaggedRowMajorMatrix(JaggedRowMajorMatrix&& other) noexcept
            : width_(other.width_), height_(other.height_), rows_(other.rows_), allocator_(std::move(other.allocator_))
        {
            other.width_ = 0;
            other.height_ = 0;
            other.rows_ = row_pointer();
        }

        ~JaggedRowMajorMatrix()
        {
            release_rows();
        }

        auto operator=(JaggedRowMajorMatrix other) noexcept -> JaggedRowMajorMatrix&
        {
            swap(other);
            return *this;
        }

        void swap(JaggedRowMajorMatrix& other) noexcept
        {
            using std::swap;

            swap(width_, other.width_);
            swap(height_, other.height_);
            swap(rows_, other.rows_);
            swap(allocator_, other.allocator_);
        }

        [[nodiscard]] auto operator()(std::size_t x, std::size_t y) const -> const value_type&
        {
            return std::to_address(std::to_address(rows_)[y])[x];
        }

        void set(std::size_t x, std::size_t y, const value_type& value)
        {
            std::to_address(std::to_address(rows_)[y])[x] = value;
        }

        [[nodiscard]] auto width() const -> std::size_t
        {
            return width_;
        }

        [[nodiscard]] auto height() const -> std::size_t
        {
            return height_;
        }

        [[nodiscard]] auto row_count() const -> std::size_t
        {
            return height_;
        }

        [[nodiscard]] auto row_storage(std::size_t y) const -> std::span<const value_type>
        {
            return std::span<const value_type>(std::to_address(std::to_address(rows_)[y]), width_);
        }

        [[nodiscard]] auto row_width() const -> std::size_t
        {
            return width_;
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

        void copy_from(const JaggedRowMajorMatrix& other)
        {
            for (std::size_t y = 0; y < height_; ++y) {
                for (std::size_t x = 0; x < width_; ++x) {
                    std::to_address(std::to_address(rows_)[y])[x] = std::to_address(std::to_address(other.rows_)[y])[x];
                }
            }
        }

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

        std::size_t width_{0};
        std::size_t height_{0};
        row_pointer rows_{};
        allocator_type allocator_{};
    };

} // namespace mpilab::domain
