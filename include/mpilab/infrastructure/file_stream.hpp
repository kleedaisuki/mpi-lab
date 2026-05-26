#pragma once

#include "mpilab/domain/MatrixLike.hpp"

#include <charconv>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <stdexcept>
#include <system_error>
#include <vector>

/**
 * @file file_stream.hpp
 * @brief 基于内存映射文本文件的矩阵流声明。 Matrix stream declarations over memory-mapped text files.
 */

namespace mpilab::infrastructure
{
    namespace detail
    {
        /**
         * @brief 已解析矩阵文本数据。 Parsed matrix text data.
         */
        struct MatrixTextData final
        {
            /**
             * @brief 矩阵宽度。 Matrix width.
             */
            std::size_t width = 0;

            /**
             * @brief 矩阵高度。 Matrix height.
             */
            std::size_t height = 0;

            /**
             * @brief 行主序线性值。 Row-major linear values.
             */
            std::vector<double> values;
        };

        /**
         * @brief 写入已序列化矩阵文本。 Write serialized matrix text.
         *
         * @param path 输出文件路径。 / Output file path.
         * @param text 已校验文本。 / Validated text.
         */
        void write_matrix_text(const std::filesystem::path &path, std::string_view text);
    } // namespace detail

    /**
     * @brief 矩阵文本文件格式错误。 Matrix text file format error.
     */
    class MatrixFileFormatError final : public std::runtime_error
    {
    public:
        /**
         * @brief 使用错误信息构造格式错误。 Construct a format error with a message.
         *
         * @param message 错误信息。 / Error message.
         */
        explicit MatrixFileFormatError(const char *message);

        /**
         * @brief 使用错误信息构造格式错误。 Construct a format error with a message.
         *
         * @param message 错误信息。 / Error message.
         */
        explicit MatrixFileFormatError(const std::string &message);
    };

    /**
     * @brief 内存映射矩阵输入流。 Memory-mapped matrix input stream.
     *
     * @note 文本格式中非空行表示矩阵行，连续空行分隔矩阵；每行元素以空白分隔。 / In the text format, non-empty lines are matrix rows, consecutive empty lines separate matrices, and row elements are whitespace-separated.
     */
    class MatrixFileReader final
    {
    public:
        /**
         * @brief 从路径打开矩阵文本文件。 Open a matrix text file from a path.
         *
         * @param path 文件路径。 / File path.
         */
        explicit MatrixFileReader(const std::filesystem::path &path);

        /**
         * @brief 移动构造矩阵输入流。 Move-construct a matrix input stream.
         *
         * @param other 源输入流。 / Source input stream.
         */
        MatrixFileReader(MatrixFileReader &&other) noexcept;

        /**
         * @brief 移动赋值矩阵输入流。 Move-assign a matrix input stream.
         *
         * @param other 源输入流。 / Source input stream.
         * @return 当前输入流引用。 / Reference to this input stream.
         */
        auto operator=(MatrixFileReader &&other) noexcept -> MatrixFileReader &;

        /**
         * @brief 析构矩阵输入流并释放映射。 Destroy the matrix input stream and release the mapping.
         */
        ~MatrixFileReader();

        MatrixFileReader(const MatrixFileReader &) = delete;
        auto operator=(const MatrixFileReader &) -> MatrixFileReader & = delete;

        /**
         * @brief 判断是否还有下一块矩阵。 Return whether another matrix is available.
         *
         * @return 若存在下一块矩阵则为 true。 / True if another matrix is available.
         */
        [[nodiscard]] auto has_next() const -> bool;

        /**
         * @brief 读取下一块矩阵。 Read the next matrix.
         *
         * @tparam Matrix 目标矩阵类型。 / Target matrix type.
         * @return 下一块矩阵。 / Next matrix.
         * @throws MatrixFileFormatError 当格式非法或已经到达末尾时抛出。 / Throws when the format is invalid or the stream is exhausted.
         */
        template <domain::MatrixLike Matrix>
        auto next() -> Matrix
        {
            detail::MatrixTextData data = next_data();
            Matrix matrix(data.width, data.height);
            for (std::size_t y = 0; y < data.height; ++y)
            {
                for (std::size_t x = 0; x < data.width; ++x)
                {
                    matrix.set(x, y, data.values[(y * data.width) + x]);
                }
            }

            return matrix;
        }

    private:
        /**
         * @brief 私有实现类型。 Private implementation type.
         */
        struct Impl;

        /**
         * @brief 私有实现指针。 Private implementation pointer.
         */
        std::unique_ptr<Impl> impl_;

        /**
         * @brief 读取下一块矩阵的中性文本数据。 Read neutral text data for the next matrix.
         *
         * @return 已解析矩阵文本数据。 / Parsed matrix text data.
         */
        auto next_data() -> detail::MatrixTextData;
    };

    /**
     * @brief 将矩阵序列写为内存映射文本文件。 Write a matrix sequence as a memory-mapped text file.
     *
     * @tparam Matrix 矩阵类型。 / Matrix type.
     * @param path 输出文件路径。 / Output file path.
     * @param matrices 待写出的矩阵序列。 / Matrix sequence to write.
     * @throws MatrixFileFormatError 当矩阵不能表示为该文本格式时抛出。 / Throws when a matrix cannot be represented by this text format.
     */
    template <domain::MatrixLike Matrix>
    void write_matrices(const std::filesystem::path &path, std::span<const Matrix> matrices);

    /**
     * @brief 将矩阵向量写为内存映射文本文件。 Write a matrix vector as a memory-mapped text file.
     *
     * @tparam Matrix 矩阵类型。 / Matrix type.
     * @param path 输出文件路径。 / Output file path.
     * @param matrices 待写出的矩阵向量。 / Matrix vector to write.
     * @throws MatrixFileFormatError 当矩阵不能表示为该文本格式时抛出。 / Throws when a matrix cannot be represented by this text format.
     */
    template <domain::MatrixLike Matrix>
    void write_matrices(const std::filesystem::path &path, const std::vector<Matrix> &matrices);

} // namespace mpilab::infrastructure

namespace mpilab::infrastructure
{
    template <domain::MatrixLike Matrix>
    void write_matrices(const std::filesystem::path &path, std::span<const Matrix> matrices)
    {
        std::string output;
        for (std::size_t matrix_index = 0; matrix_index < matrices.size(); ++matrix_index)
        {
            const Matrix &matrix = matrices[matrix_index];
            if (matrix.width() == 0 || matrix.height() == 0)
            {
                throw MatrixFileFormatError("matrix text format cannot represent empty matrices");
            }

            for (std::size_t y = 0; y < matrix.height(); ++y)
            {
                for (std::size_t x = 0; x < matrix.width(); ++x)
                {
                    if (x != 0)
                    {
                        output.push_back(' ');
                    }

                    char buffer[64] = {};
                    const std::to_chars_result written = std::to_chars(
                        buffer,
                        buffer + sizeof(buffer),
                        static_cast<double>(matrix(x, y)),
                        std::chars_format::general,
                        std::numeric_limits<double>::max_digits10);
                    if (written.ec != std::errc())
                    {
                        throw MatrixFileFormatError("failed to format floating-point value");
                    }
                    output.append(buffer, written.ptr);
                }
                output.push_back('\n');
            }

            if (matrix_index + 1 != matrices.size())
            {
                output.push_back('\n');
            }
        }

        detail::write_matrix_text(path, output);
    }

    template <domain::MatrixLike Matrix>
    void write_matrices(const std::filesystem::path &path, const std::vector<Matrix> &matrices)
    {
        write_matrices(path, std::span<const Matrix>(matrices.data(), matrices.size()));
    }
} // namespace mpilab::infrastructure
