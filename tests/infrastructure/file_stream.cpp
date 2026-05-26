#include "mpilab/domain/matrix.hpp"
#include "mpilab/infrastructure/file_stream.hpp"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    /**
     * @brief 返回测试临时文件路径。 Return a temporary file path for tests.
     *
     * @param name 文件名。 / File name.
     * @return 临时文件路径。 / Temporary file path.
     */
    [[nodiscard]] auto temp_path(const char *name) -> std::filesystem::path
    {
        return std::filesystem::temp_directory_path() / name;
    }

    /**
     * @brief 将文本写入测试文件。 Write text into a test file.
     *
     * @param path 文件路径。 / File path.
     * @param text 文本内容。 / Text content.
     */
    void write_text(const std::filesystem::path &path, const std::string &text)
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output << text;
    }

    /**
     * @brief 读取整个文本文件。 Read an entire text file.
     *
     * @param path 文件路径。 / File path.
     * @return 文件内容。 / File content.
     */
    [[nodiscard]] auto read_text(const std::filesystem::path &path) -> std::string
    {
        std::ifstream input(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    }

    /**
     * @brief 验证多矩阵 mmap 输入流。 Verify a multi-matrix mmap input stream.
     */
    void test_reads_multiple_matrix_types()
    {
        const std::filesystem::path path = temp_path("mpilab_file_stream_read.txt");
        write_text(path, " 1 2 3\r\n4 5 6\r\n\r\n\r\n7.5 8.5\n9.5 10.5\n");

        mpilab::infrastructure::MatrixFileReader reader(path);
        assert(reader.has_next());

        const auto first = reader.next<mpilab::domain::RowMajorMatrix<double>>();
        assert(first.width() == 3);
        assert(first.height() == 2);
        assert(first(0, 0) == 1.0);
        assert(first(2, 1) == 6.0);

        assert(reader.has_next());
        const auto second = reader.next<mpilab::domain::ColumnMajorMatrix<double>>();
        assert(second.width() == 2);
        assert(second.height() == 2);
        assert(second(0, 0) == 7.5);
        assert(second(1, 1) == 10.5);

        assert(!reader.has_next());
        std::filesystem::remove(path);
    }

    /**
     * @brief 验证非法矩阵文本会被拒绝。 Verify that malformed matrix text is rejected.
     */
    void test_rejects_bad_format()
    {
        const std::filesystem::path path = temp_path("mpilab_file_stream_bad.txt");
        write_text(path, "1 2\n3\n");

        mpilab::infrastructure::MatrixFileReader reader(path);
        bool threw = false;
        try
        {
            static_cast<void>(reader.next<mpilab::domain::RowMajorMatrix<double>>());
        }
        catch (const mpilab::infrastructure::MatrixFileFormatError &)
        {
            threw = true;
        }
        assert(threw);
        std::filesystem::remove(path);
    }

    /**
     * @brief 验证矩阵写出和回读。 Verify matrix writing and round-tripping.
     */
    void test_writes_and_round_trips()
    {
        const std::filesystem::path path = temp_path("mpilab_file_stream_write.txt");

        std::vector<mpilab::domain::RowMajorMatrix<double>> matrices;
        matrices.emplace_back(2, 2);
        matrices.back().set(0, 0, 1.25);
        matrices.back().set(1, 0, 2.5);
        matrices.back().set(0, 1, 3.75);
        matrices.back().set(1, 1, 4.0);
        matrices.emplace_back(1, 1);
        matrices.back().set(0, 0, -8.0);

        mpilab::infrastructure::write_matrices(path, matrices);
        assert(read_text(path) == "1.25 2.5\n3.75 4\n\n-8\n");

        mpilab::infrastructure::MatrixFileReader reader(path);
        const auto first = reader.next<mpilab::domain::JaggedRowMajorMatrix<double>>();
        const auto second = reader.next<mpilab::domain::RowMajorMatrix<double>>();
        assert(first(1, 0) == 2.5);
        assert(second(0, 0) == -8.0);
        assert(!reader.has_next());

        std::filesystem::remove(path);
    }

    /**
     * @brief 验证不可表示的空矩阵会被拒绝。 Verify that unrepresentable empty matrices are rejected.
     */
    void test_rejects_empty_matrix_on_write()
    {
        const std::filesystem::path path = temp_path("mpilab_file_stream_empty.txt");
        std::vector<mpilab::domain::RowMajorMatrix<double>> matrices;
        matrices.emplace_back(0, 2);

        bool threw = false;
        try
        {
            mpilab::infrastructure::write_matrices(path, matrices);
        }
        catch (const mpilab::infrastructure::MatrixFileFormatError &)
        {
            threw = true;
        }
        assert(threw);
    }

} // namespace

/**
 * @brief file_stream 测试入口。 File-stream test entry point.
 *
 * @return 成功时返回 0。 / Returns 0 on success.
 */
int main()
{
    test_reads_multiple_matrix_types();
    test_rejects_bad_format();
    test_writes_and_round_trips();
    test_rejects_empty_matrix_on_write();
    return 0;
}
