#pragma once

#include <cstddef>
#include <functional>

/**
 * @file thread_pool.hpp
 * @brief Pthreads 线程池声明。 Pthreads thread-pool declarations.
 */

namespace mpilab::infrastructure
{

    /**
     * @brief 基于 Pthreads 的固定大小线程池。 Fixed-size thread pool based on Pthreads.
     */
    class ThreadPool
    {
    public:
        /**
         * @brief 使用 worker 数量构造线程池。 Construct a thread pool with the worker count.
         *
         * @param worker_count worker 线程数量。 / Number of worker threads.
         */
        explicit ThreadPool(std::size_t worker_count);

        /**
         * @brief 销毁线程池并等待 worker 退出。 Destroy the pool and join workers.
         */
        ~ThreadPool();

        /**
         * @brief 禁止复制构造。 Disable copy construction.
         *
         * @param other 另一个线程池。 / Another thread pool.
         */
        ThreadPool(const ThreadPool& other) = delete;

        /**
         * @brief 禁止复制赋值。 Disable copy assignment.
         *
         * @param other 另一个线程池。 / Another thread pool.
         * @return 当前线程池引用。 / Reference to this thread pool.
         */
        auto operator=(const ThreadPool& other) -> ThreadPool& = delete;

        /**
         * @brief 并行执行半开区间 [0, count) 上的任务。 Execute work over the half-open range [0, count) in parallel.
         *
         * @param count 任务元素数量。 / Number of work items.
         * @param task 单个元素任务。 / Per-item task.
         */
        void parallel_for(std::size_t count, const std::function<void(std::size_t)>& task);

        /**
         * @brief 返回 worker 数量。 Return the worker count.
         *
         * @return worker 线程数量。 / Number of worker threads.
         */
        [[nodiscard]] auto worker_count() const -> std::size_t;

    private:
        /**
         * @brief Pthreads worker 入口函数。 Pthreads worker entry point.
         *
         * @param argument 线程池实现指针。 / Thread-pool implementation pointer.
         * @return pthread 入口返回值。 / pthread entry return value.
         */
        static auto worker_entry(void* argument) -> void*;

        /**
         * @brief 私有实现类型。 Private implementation type.
         */
        struct Impl;

        /**
         * @brief 私有实现指针。 / Private implementation pointer.
         */
        Impl* impl_{nullptr};
    };

    /**
     * @brief 返回全局惰性初始化线程池。 Return the globally lazy-initialized thread pool.
     *
     * @return 全局线程池引用。 / Reference to the global thread pool.
     */
    [[nodiscard]] auto default_thread_pool() -> ThreadPool&;

} // namespace mpilab::infrastructure
