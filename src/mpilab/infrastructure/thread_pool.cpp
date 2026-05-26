#include "mpilab/infrastructure/thread_pool.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <pthread.h>
#include <stdexcept>
#include <unistd.h>
#include <utility>
#include <vector>

/**
 * @file thread_pool.cpp
 * @brief Pthreads 线程池实现。 Pthreads thread-pool implementation.
 */

namespace mpilab::infrastructure
{
namespace
{

    /**
     * @brief 读取在线处理器数量。 Read the online processor count.
     *
     * @return 在线处理器数量。 / Online processor count.
     */
    [[nodiscard]] auto online_processor_count() -> std::size_t
    {
        const long count = sysconf(_SC_NPROCESSORS_ONLN);
        if (count <= 0) {
            return 1;
        }

        return static_cast<std::size_t>(count);
    }

} // namespace

struct ThreadPool::Impl
{
    /**
     * @brief worker 线程句柄。 / Worker thread handles.
     */
    std::vector<pthread_t> workers;

    /**
     * @brief 待执行任务队列。 / Pending task queue.
     */
    std::deque<std::function<void()>> tasks;

    /**
     * @brief 队列互斥锁。 / Queue mutex.
     */
    std::mutex mutex;

    /**
     * @brief 任务可用条件变量。 / Task-available condition variable.
     */
    std::condition_variable task_available;

    /**
     * @brief 所有任务完成条件变量。 / All-tasks-finished condition variable.
     */
    std::condition_variable tasks_finished;

    /**
     * @brief 正在运行的任务数量。 / Number of active tasks.
     */
    std::size_t active_tasks{0};

    /**
     * @brief 是否请求停止。 / Whether stop was requested.
     */
    bool stopping{false};
};

auto ThreadPool::worker_entry(void* argument) -> void*
{
    auto* impl = static_cast<ThreadPool::Impl*>(argument);

    while (true) {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(impl->mutex);
            impl->task_available.wait(lock, [&impl] {
                return impl->stopping || !impl->tasks.empty();
            });

            if (impl->stopping && impl->tasks.empty()) {
                return nullptr;
            }

            task = std::move(impl->tasks.front());
            impl->tasks.pop_front();
            ++impl->active_tasks;
        }

        task();

        {
            std::lock_guard<std::mutex> lock(impl->mutex);
            --impl->active_tasks;
            if (impl->tasks.empty() && impl->active_tasks == 0) {
                impl->tasks_finished.notify_all();
            }
        }
    }
}

ThreadPool::ThreadPool(std::size_t worker_count)
    : impl_(new Impl())
{
    const std::size_t normalized_worker_count = std::max<std::size_t>(1, worker_count);
    impl_->workers.resize(normalized_worker_count);

    for (pthread_t& worker : impl_->workers) {
        const int rc = pthread_create(&worker, nullptr, ThreadPool::worker_entry, impl_);
        if (rc != 0) {
            {
                std::lock_guard<std::mutex> lock(impl_->mutex);
                impl_->stopping = true;
            }
            impl_->task_available.notify_all();
            throw std::runtime_error("failed to create pthread worker");
        }
    }
}

ThreadPool::~ThreadPool()
{
    if (impl_ == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        impl_->stopping = true;
    }
    impl_->task_available.notify_all();

    for (pthread_t worker : impl_->workers) {
        pthread_join(worker, nullptr);
    }

    delete impl_;
    impl_ = nullptr;
}

void ThreadPool::parallel_for(std::size_t count, const std::function<void(std::size_t)>& task)
{
    if (count == 0) {
        return;
    }

    const std::size_t chunk_count = std::min(count, worker_count());
    const std::size_t chunk_size = (count + chunk_count - 1) / chunk_count;
    std::exception_ptr first_exception;
    std::mutex exception_mutex;

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        for (std::size_t chunk = 0; chunk < chunk_count; ++chunk) {
            const std::size_t begin = chunk * chunk_size;
            const std::size_t end = std::min(count, begin + chunk_size);
            if (begin >= end) {
                continue;
            }

            impl_->tasks.push_back([begin, end, &task, &first_exception, &exception_mutex] {
                try {
                    for (std::size_t index = begin; index < end; ++index) {
                        task(index);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> exception_lock(exception_mutex);
                    if (first_exception == nullptr) {
                        first_exception = std::current_exception();
                    }
                }
            });
        }
    }

    impl_->task_available.notify_all();

    {
        std::unique_lock<std::mutex> lock(impl_->mutex);
        impl_->tasks_finished.wait(lock, [this] {
            return impl_->tasks.empty() && impl_->active_tasks == 0;
        });
    }

    if (first_exception != nullptr) {
        std::rethrow_exception(first_exception);
    }
}

auto ThreadPool::worker_count() const -> std::size_t
{
    return impl_->workers.size();
}

auto default_thread_pool() -> ThreadPool&
{
    /**
     * @brief 全局惰性初始化线程池。 / Globally lazy-initialized thread pool.
     */
    static ThreadPool pool(std::max<std::size_t>(1, online_processor_count()));
    return pool;
}

} // namespace mpilab::infrastructure
