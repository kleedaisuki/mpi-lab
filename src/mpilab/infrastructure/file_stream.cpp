#include "mpilab/infrastructure/file_stream.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace mpilab::infrastructure
{
    namespace
    {
        /**
         * @brief 判断字符是否为格式空白。 Return whether a character is format whitespace.
         *
         * @param value 待检查字符。 / Character to inspect.
         * @return 若为空白则为 true。 / True if the character is whitespace.
         */
        [[nodiscard]] auto is_space(char value) -> bool
        {
            return value == ' ' || value == '\t' || value == '\r' || value == '\n' || value == '\f' || value == '\v';
        }

        /**
         * @brief 判断行是否为空白行。 Return whether a line is blank.
         *
         * @param line 行视图。 / Line view.
         * @return 若只包含空白则为 true。 / True if it contains only whitespace.
         */
        [[nodiscard]] auto is_blank_line(std::string_view line) -> bool
        {
            return std::ranges::all_of(line, is_space);
        }

        /**
         * @brief 去除行尾回车。 Remove a trailing carriage return.
         *
         * @param line 行视图。 / Line view.
         * @return 去除回车后的行视图。 / Line view without the carriage return.
         */
        [[nodiscard]] auto strip_cr(std::string_view line) -> std::string_view
        {
            if (!line.empty() && line.back() == '\r')
            {
                line.remove_suffix(1);
            }

            return line;
        }

        /**
         * @brief 构造带行号的格式错误。 Build a format error with a line number.
         *
         * @param line_number 行号。 / Line number.
         * @param message 错误信息。 / Error message.
         * @return 格式错误。 / Format error.
         */
        [[nodiscard]] auto line_error(std::size_t line_number, std::string_view message) -> MatrixFileFormatError
        {
            std::ostringstream output;
            output << "matrix file format error at line " << line_number << ": " << message;
            return MatrixFileFormatError(output.str());
        }

        /**
         * @brief 解析一行矩阵元素。 Parse matrix elements from one row.
         *
         * @param line 行视图。 / Line view.
         * @param line_number 行号。 / Line number.
         * @return 行元素。 / Row elements.
         */
        [[nodiscard]] auto parse_row(std::string_view line, std::size_t line_number) -> std::vector<double>
        {
            std::vector<double> row;
            const char *cursor = line.data();
            const char *const end = line.data() + line.size();

            while (cursor != end)
            {
                while (cursor != end && is_space(*cursor))
                {
                    ++cursor;
                }
                if (cursor == end)
                {
                    break;
                }

                double value = 0.0;
                const std::from_chars_result parsed = std::from_chars(cursor, end, value);
                if (parsed.ec != std::errc())
                {
                    throw line_error(line_number, "invalid floating-point value");
                }
                if (parsed.ptr == cursor)
                {
                    throw line_error(line_number, "empty floating-point value");
                }

                cursor = parsed.ptr;
                if (cursor != end && !is_space(*cursor))
                {
                    throw line_error(line_number, "matrix values must be separated by whitespace");
                }
                row.push_back(value);
            }

            if (row.empty())
            {
                throw line_error(line_number, "matrix row must contain at least one value");
            }

            return row;
        }

        /**
         * @brief 将行缓冲区移动为中性矩阵文本数据。 Move row buffers into neutral matrix text data.
         *
         * @param rows 行缓冲区。 / Row buffers.
         * @return 已解析矩阵文本数据。 / Parsed matrix text data.
         */
        [[nodiscard]] auto rows_to_data(const std::vector<std::vector<double>> &rows) -> detail::MatrixTextData
        {
            if (rows.empty())
            {
                throw MatrixFileFormatError("matrix must contain at least one row");
            }

            detail::MatrixTextData data;
            data.width = rows.front().size();
            data.height = rows.size();
            data.values.reserve(data.width * data.height);
            for (const std::vector<double> &row : rows)
            {
                if (row.size() != data.width)
                {
                    throw MatrixFileFormatError("matrix rows must have equal widths");
                }
                data.values.insert(data.values.end(), row.begin(), row.end());
            }

            return data;
        }

        /**
         * @brief 内存映射只读文件。 Read-only memory-mapped file.
         */
        class ReadOnlyMappedFile final
        {
        public:
            /**
             * @brief 打开并映射文件。 Open and map a file.
             *
             * @param path 文件路径。 / File path.
             */
            explicit ReadOnlyMappedFile(const std::filesystem::path &path)
            {
                open(path);
            }

            /**
             * @brief 移动构造映射文件。 Move-construct a mapped file.
             *
             * @param other 源映射。 / Source mapping.
             */
            ReadOnlyMappedFile(ReadOnlyMappedFile &&other) noexcept
            {
                move_from(other);
            }

            /**
             * @brief 移动赋值映射文件。 Move-assign a mapped file.
             *
             * @param other 源映射。 / Source mapping.
             * @return 当前映射引用。 / Reference to this mapping.
             */
            auto operator=(ReadOnlyMappedFile &&other) noexcept -> ReadOnlyMappedFile &
            {
                if (this != &other)
                {
                    close();
                    move_from(other);
                }

                return *this;
            }

            /**
             * @brief 析构并释放映射。 Destroy and release the mapping.
             */
            ~ReadOnlyMappedFile()
            {
                close();
            }

            ReadOnlyMappedFile(const ReadOnlyMappedFile &) = delete;
            auto operator=(const ReadOnlyMappedFile &) -> ReadOnlyMappedFile & = delete;

            /**
             * @brief 返回映射数据。 Return mapped data.
             *
             * @return 数据指针。 / Data pointer.
             */
            [[nodiscard]] auto data() const -> const char *
            {
                return data_;
            }

            /**
             * @brief 返回映射大小。 Return mapped size.
             *
             * @return 字节数。 / Byte count.
             */
            [[nodiscard]] auto size() const -> std::size_t
            {
                return size_;
            }

        private:
            /**
             * @brief 打开并映射文件的跨平台入口。 Cross-platform entry for opening and mapping a file.
             *
             * @param path 文件路径。 / File path.
             */
            void open(const std::filesystem::path &path)
            {
                const std::uintmax_t file_size = std::filesystem::file_size(path);
#if UINTMAX_MAX > SIZE_MAX
                if (file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max()))
                {
                    throw std::length_error("matrix file is too large to map into this process");
                }
#endif
                size_ = static_cast<std::size_t>(file_size);
                if (size_ == 0)
                {
                    data_ = nullptr;
                    return;
                }

#if defined(_WIN32)
                file_ = CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if (file_ == INVALID_HANDLE_VALUE)
                {
                    throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "CreateFileW failed");
                }

                mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
                if (mapping_ == nullptr)
                {
                    const DWORD error = GetLastError();
                    static_cast<void>(CloseHandle(file_));
                    file_ = INVALID_HANDLE_VALUE;
                    throw std::system_error(static_cast<int>(error), std::system_category(), "CreateFileMappingW failed");
                }

                data_ = static_cast<const char *>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, size_));
                if (data_ == nullptr)
                {
                    const DWORD error = GetLastError();
                    static_cast<void>(CloseHandle(mapping_));
                    static_cast<void>(CloseHandle(file_));
                    mapping_ = nullptr;
                    file_ = INVALID_HANDLE_VALUE;
                    throw std::system_error(static_cast<int>(error), std::system_category(), "MapViewOfFile failed");
                }
#else
                fd_ = ::open(path.c_str(), O_RDONLY);
                if (fd_ == -1)
                {
                    throw std::system_error(errno, std::generic_category(), "open failed");
                }

                void *mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
                if (mapped == MAP_FAILED)
                {
                    const int error = errno;
                    static_cast<void>(::close(fd_));
                    fd_ = -1;
                    throw std::system_error(error, std::generic_category(), "mmap failed");
                }
                data_ = static_cast<const char *>(mapped);
#endif
            }

            /**
             * @brief 释放平台资源。 Release platform resources.
             */
            void close() noexcept
            {
#if defined(_WIN32)
                if (data_ != nullptr)
                {
                    static_cast<void>(UnmapViewOfFile(data_));
                }
                if (mapping_ != nullptr)
                {
                    static_cast<void>(CloseHandle(mapping_));
                }
                if (file_ != INVALID_HANDLE_VALUE)
                {
                    static_cast<void>(CloseHandle(file_));
                }
                file_ = INVALID_HANDLE_VALUE;
                mapping_ = nullptr;
#else
                if (data_ != nullptr)
                {
                    static_cast<void>(::munmap(const_cast<char *>(data_), size_));
                }
                if (fd_ != -1)
                {
                    static_cast<void>(::close(fd_));
                }
                fd_ = -1;
#endif
                data_ = nullptr;
                size_ = 0;
            }

            /**
             * @brief 从另一个映射移动资源。 Move resources from another mapping.
             *
             * @param other 源映射。 / Source mapping.
             */
            void move_from(ReadOnlyMappedFile &other) noexcept
            {
                data_ = other.data_;
                size_ = other.size_;
#if defined(_WIN32)
                file_ = other.file_;
                mapping_ = other.mapping_;
                other.file_ = INVALID_HANDLE_VALUE;
                other.mapping_ = nullptr;
#else
                fd_ = other.fd_;
                other.fd_ = -1;
#endif
                other.data_ = nullptr;
                other.size_ = 0;
            }

            /**
             * @brief 映射数据指针。 Mapped data pointer.
             */
            const char *data_ = nullptr;

            /**
             * @brief 映射字节数。 Mapped byte count.
             */
            std::size_t size_ = 0;

#if defined(_WIN32)
            /**
             * @brief Windows 文件句柄。 Windows file handle.
             */
            HANDLE file_ = INVALID_HANDLE_VALUE;

            /**
             * @brief Windows 文件映射句柄。 Windows file-mapping handle.
             */
            HANDLE mapping_ = nullptr;
#else
            /**
             * @brief POSIX 文件描述符。 POSIX file descriptor.
             */
            int fd_ = -1;
#endif
        };

        /**
         * @brief 将文本写入内存映射文件。 Write text into a memory-mapped file.
         *
         * @param path 输出路径。 / Output path.
         * @param text 文本内容。 / Text content.
         */
        void write_mapped_text(const std::filesystem::path &path, std::string_view text)
        {
            if (text.empty())
            {
                std::ofstream output(path, std::ios::binary | std::ios::trunc);
                if (!output)
                {
                    throw std::system_error(errno, std::generic_category(), "failed to create empty matrix file");
                }
                return;
            }

#if defined(_WIN32)
            HANDLE file = CreateFileW(path.wstring().c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (file == INVALID_HANDLE_VALUE)
            {
                throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "CreateFileW failed");
            }

            const auto close_file = [&file]() noexcept
            {
                if (file != INVALID_HANDLE_VALUE)
                {
                    static_cast<void>(CloseHandle(file));
                    file = INVALID_HANDLE_VALUE;
                }
            };

            LARGE_INTEGER file_size = {};
            file_size.QuadPart = static_cast<LONGLONG>(text.size());
            if (SetFilePointerEx(file, file_size, nullptr, FILE_BEGIN) == 0 || SetEndOfFile(file) == 0)
            {
                const DWORD error = GetLastError();
                close_file();
                throw std::system_error(static_cast<int>(error), std::system_category(), "failed to resize output file");
            }

            HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
            if (mapping == nullptr)
            {
                const DWORD error = GetLastError();
                close_file();
                throw std::system_error(static_cast<int>(error), std::system_category(), "CreateFileMappingW failed");
            }

            void *mapped = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, text.size());
            if (mapped == nullptr)
            {
                const DWORD error = GetLastError();
                static_cast<void>(CloseHandle(mapping));
                close_file();
                throw std::system_error(static_cast<int>(error), std::system_category(), "MapViewOfFile failed");
            }

            std::memcpy(mapped, text.data(), text.size());
            static_cast<void>(FlushViewOfFile(mapped, text.size()));
            static_cast<void>(UnmapViewOfFile(mapped));
            static_cast<void>(CloseHandle(mapping));
            close_file();
#else
            const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0666);
            if (fd == -1)
            {
                throw std::system_error(errno, std::generic_category(), "open failed");
            }

            const auto close_fd = [fd]() noexcept
            {
                static_cast<void>(::close(fd));
            };

            if (::ftruncate(fd, static_cast<off_t>(text.size())) == -1)
            {
                const int error = errno;
                close_fd();
                throw std::system_error(error, std::generic_category(), "ftruncate failed");
            }

            void *mapped = ::mmap(nullptr, text.size(), PROT_WRITE, MAP_SHARED, fd, 0);
            if (mapped == MAP_FAILED)
            {
                const int error = errno;
                close_fd();
                throw std::system_error(error, std::generic_category(), "mmap failed");
            }

            std::memcpy(mapped, text.data(), text.size());
            static_cast<void>(::msync(mapped, text.size(), MS_SYNC));
            static_cast<void>(::munmap(mapped, text.size()));
            close_fd();
#endif
        }
    } // namespace

    /**
     * @brief MatrixFileReader 的私有实现。 Private implementation for MatrixFileReader.
     */
    struct MatrixFileReader::Impl final
    {
        /**
         * @brief 构造私有实现。 Construct the private implementation.
         *
         * @param path 文件路径。 / File path.
         */
        explicit Impl(const std::filesystem::path &path) : file(path), bytes(file.data(), file.size())
        {
            skip_blank_lines();
        }

        /**
         * @brief 跳过空白矩阵分隔行。 Skip blank matrix separator lines.
         */
        void skip_blank_lines()
        {
            while (position < bytes.size())
            {
                const std::size_t line_start = position;
                const std::size_t line_end = bytes.find('\n', line_start);
                const std::size_t end = line_end == std::string_view::npos ? bytes.size() : line_end;
                if (!is_blank_line(strip_cr(bytes.substr(line_start, end - line_start))))
                {
                    return;
                }

                position = line_end == std::string_view::npos ? bytes.size() : line_end + 1;
                ++line_number;
            }
        }

        /**
         * @brief 底层只读映射文件。 Underlying read-only mapped file.
         */
        ReadOnlyMappedFile file;

        /**
         * @brief 文件字节视图。 File byte view.
         */
        std::string_view bytes;

        /**
         * @brief 当前解析位置。 Current parse position.
         */
        std::size_t position = 0;

        /**
         * @brief 当前行号。 Current line number.
         */
        std::size_t line_number = 1;
    };

    MatrixFileFormatError::MatrixFileFormatError(const char *message) : std::runtime_error(message)
    {
    }

    MatrixFileFormatError::MatrixFileFormatError(const std::string &message) : std::runtime_error(message)
    {
    }

    MatrixFileReader::MatrixFileReader(const std::filesystem::path &path) : impl_(std::make_unique<Impl>(path))
    {
    }

    MatrixFileReader::MatrixFileReader(MatrixFileReader &&other) noexcept = default;

    auto MatrixFileReader::operator=(MatrixFileReader &&other) noexcept -> MatrixFileReader & = default;

    MatrixFileReader::~MatrixFileReader() = default;

    auto MatrixFileReader::has_next() const -> bool
    {
        return impl_ != nullptr && impl_->position < impl_->bytes.size();
    }

    auto MatrixFileReader::next_data() -> detail::MatrixTextData
    {
        if (!has_next())
        {
            throw MatrixFileFormatError("no matrix remains in stream");
        }

        std::vector<std::vector<double>> rows;
        std::size_t expected_width = 0;

        while (impl_->position < impl_->bytes.size())
        {
            const std::size_t line_start = impl_->position;
            const std::size_t line_end = impl_->bytes.find('\n', line_start);
            const std::size_t end = line_end == std::string_view::npos ? impl_->bytes.size() : line_end;
            const std::string_view line = strip_cr(impl_->bytes.substr(line_start, end - line_start));

            impl_->position = line_end == std::string_view::npos ? impl_->bytes.size() : line_end + 1;
            const std::size_t current_line = impl_->line_number;
            ++impl_->line_number;

            if (is_blank_line(line))
            {
                if (!rows.empty())
                {
                    impl_->skip_blank_lines();
                    break;
                }
                continue;
            }

            std::vector<double> row = parse_row(line, current_line);
            if (rows.empty())
            {
                expected_width = row.size();
            }
            else if (row.size() != expected_width)
            {
                throw line_error(current_line, "matrix rows must have equal widths");
            }
            rows.push_back(std::move(row));
        }

        if (rows.empty())
        {
            throw MatrixFileFormatError("matrix must contain at least one row");
        }

        return rows_to_data(rows);
    }

    void detail::write_matrix_text(const std::filesystem::path &path, std::string_view text)
    {
        write_mapped_text(path, text);
    }

} // namespace mpilab::infrastructure
