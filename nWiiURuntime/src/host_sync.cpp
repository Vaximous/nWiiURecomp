#include "runtime/host_sync.h"

#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <thread>

namespace nwii::runtime::host_sync {
namespace {
struct Request {
    const std::function<void()>* work{};
    std::exception_ptr error;
    bool done{};
};

struct State {
    std::recursive_mutex hle;
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<Request*> requests;
    uint64_t kicks{};
    std::atomic<int> waiters{};
    std::atomic<bool> has_requests{};
    std::thread::id main_id{};
};

State& state() {
    static State* instance = new State();
    return *instance;
}

thread_local bool g_marshalled = false;
} // namespace

bool parallel() {
    static const bool enabled = [] {
        const char* value = std::getenv("NWIIU_PARALLEL");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

void mark_main_thread() { state().main_id = std::this_thread::get_id(); }

bool on_main_thread() {
    return std::this_thread::get_id() == state().main_id;
}

std::recursive_mutex& hle_mutex() { return state().hle; }

HleLock::HleLock() {
    if (!parallel()) {
        return;
    }
    auto& hle = state().hle;
    if (!on_main_thread()) {
        hle.lock();
        held_ = true;
        return;
    }
    while (!hle.try_lock()) {
        service_main();
        std::this_thread::yield();
    }
    held_ = true;
}

HleLock::~HleLock() {
    if (held_) {
        state().hle.unlock();
    }
}

void run_on_main(const std::function<void()>& work) {
    if (!parallel() || on_main_thread()) {
        work();
        return;
    }
    auto& s = state();
    Request request;
    request.work = &work;
    {
        std::unique_lock lock(s.mutex);
        s.requests.push_back(&request);
        s.has_requests.store(true, std::memory_order_release);
        s.cv.notify_all();
        s.cv.wait(lock, [&] { return request.done; });
    }
    g_marshalled = true;
    if (request.error) {
        std::rethrow_exception(request.error);
    }
}

bool take_marshalled() {
    const bool value = g_marshalled;
    g_marshalled = false;
    return value;
}

void service_main() {
    auto& s = state();
    if (!s.has_requests.load(std::memory_order_acquire)) {
        return;
    }
    while (true) {
        Request* request = nullptr;
        {
            std::lock_guard lock(s.mutex);
            if (s.requests.empty()) {
                s.has_requests.store(false, std::memory_order_release);
                return;
            }
            request = s.requests.front();
            s.requests.pop_front();
        }
        try {
            (*request->work)();
        } catch (...) {
            request->error = std::current_exception();
        }
        {
            std::lock_guard lock(s.mutex);
            request->done = true;
        }
        s.cv.notify_all();
    }
}

void kick() {
    auto& s = state();
    if (s.waiters.load(std::memory_order_acquire) == 0) {
        return;
    }
    {
        std::lock_guard lock(s.mutex);
        ++s.kicks;
    }
    s.cv.notify_all();
}

void idle_wait(std::chrono::microseconds timeout) {
    auto& s = state();
    const bool main = on_main_thread();
    if (main) {
        service_main();
    }
    s.waiters.fetch_add(1, std::memory_order_acq_rel);
    {
        std::unique_lock lock(s.mutex);
        const uint64_t seen = s.kicks;
        s.cv.wait_for(lock, timeout, [&] {
            return s.kicks != seen || (main && !s.requests.empty());
        });
    }
    s.waiters.fetch_sub(1, std::memory_order_acq_rel);
    if (main) {
        service_main();
    }
}

namespace {
std::function<void()>& shutdown_hook() {
    static auto* hook = new std::function<void()>();
    return *hook;
}
} // namespace

void set_shutdown(std::function<void()> stop) {
    shutdown_hook() = std::move(stop);
}

void shutdown() {
    auto stop = std::move(shutdown_hook());
    shutdown_hook() = nullptr;
    if (stop) {
        stop();
    }
}

} // namespace nwii::runtime::host_sync
