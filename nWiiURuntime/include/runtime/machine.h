#pragma once

#include "runtime/cafe_coreinit.h"
#include "runtime/cafe_olv.h"
#include "runtime/cafe_filesystem.h"
#include "runtime/cafe_runtime.h"
#include "runtime/executor.h"
#include "runtime/native_hooks.h"

#include <array>
#include <atomic>
#include <chrono>
#include <exception>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <string>
#include <thread>
#include <vector>

namespace nwii::runtime {
inline constexpr uint64_t kSchedulerQuantum = 10000;

class Machine {
public:
    // `hle_hooks` maps a guest address to the name of a native replacement
    // (see native_hooks.h). It defaults to the WWHD set so that call sites
    // predating game profiles behave unchanged; an unknown name throws here
    // rather than quietly never firing.
    explicit Machine(
        ExecutionImage& image, std::filesystem::path title_root = {},
        std::optional<std::filesystem::path> save_root = std::nullopt,
        std::optional<std::filesystem::path> shared_font = std::nullopt,
        const std::map<uint32_t, std::string>& hle_hooks =
            default_hle_hooks());
    Machine(const Machine&) = delete;
    Machine& operator=(const Machine&) = delete;
    Machine(Machine&&) = delete;
    Machine& operator=(Machine&&) = delete;
    ~Machine();

    // NWIIU_PARALLEL: stops and joins the core workers. Must run while the
    // renderer (which they hand GX2 events to) is still alive.
    void stop_workers();

    CPUContext& main_context();
    uint32_t main_thread_address() const;
    uint32_t add_thread(CPUContext context, int32_t priority);
    void add_suspended_thread(uint32_t address, CPUContext context,
                              int32_t priority, uint8_t affinity);
    bool has_thread(uint32_t address) const;
    void set_thread_affinity(uint32_t thread_address, uint8_t affinity);
    void resume_thread(uint32_t thread_address);
    CPUContext& context(uint32_t thread_address);
    const CPUContext& context(uint32_t thread_address) const;

    ExecutionStop run(uint64_t global_instruction_budget,
                      uint64_t quantum);

    CafeCoreinit& coreinit() { return coreinit_; }
    const CafeCoreinit& coreinit() const { return coreinit_; }
    CafeRuntime& cafe_runtime() { return cafe_runtime_; }
    const CafeRuntime& cafe_runtime() const { return cafe_runtime_; }
    Executor& executor() { return executor_; }

    uint32_t current_thread_address() const;
    uint32_t current_core_id() const;
    void pin_current_core();
    void unpin_current_core();
    int32_t current_thread_priority() const;
    uint64_t current_time_ticks() const;
    void queue_callback(uint32_t function, uint32_t argument,
                        bool external = false);
    // Callbacks with up to four arguments (r3..r6), e.g. FS async results.
    void queue_callback_args(uint32_t function,
                             const std::array<uint32_t, 4>& arguments);
    void set_thread_priority(uint32_t thread_address, int32_t priority);
    // Returns true when the current thread suspended itself.
    bool suspend_thread(uint32_t thread_address);
    bool thread_exited(uint32_t thread_address) const;
    bool return_from_callback();
    void clear_active_callback() {
        pending_callbacks_.clear();
        scheduled_threads_.fill(0);
        scheduler_slot_ = 0;
        callback_saved_context_.reset();
        active_callback_.reset();
        callback_thread_address_ = 0;
    }
    void block_current(uint32_t wait_object);
    void exit_current(int32_t value);
    void block_thread(uint32_t thread_address, uint32_t wait_object);
    void wake_thread(uint32_t thread_address);
    void wake_waiters(uint32_t wait_object);
    void sleep_current(uint64_t delta_ticks);
    void dump_threads(std::ostream& output) const;

private:
    enum class ThreadState { suspended, ready, running, waiting, moribund };

    struct Thread {
        CPUContext cpu;
        uint32_t address;
        int32_t priority;
        ThreadState state;
        uint8_t affinity;
        uint32_t wait_object{};
        uint64_t enqueue_sequence{};
        uint64_t wake_tick{};
        bool sleeping{};
        uint8_t pinned_core{0xFF};
        // Parallel scheduling: a host worker is still inside this thread's
        // slice (it may already be ready again after a cross-core wake).
        bool on_cpu{};
        // The thread issues GX2 calls; only the main worker runs it.
        bool gpu_affine{};
        int64_t ready_since_ns{};
    };
    enum class StepResult { ran, idle, stopped };
    struct PendingCallback {
        uint32_t function;
        uint32_t argument;
        uint32_t owner_thread;
        std::array<uint32_t, 3> extra{};
    };


    Thread& thread(uint32_t address);
    const Thread& thread(uint32_t address) const;
    Thread* select_ready(uint32_t excluded_thread = 0);
    void enqueue(Thread& thread);
    void begin_callback(Thread& thread);
    void wake_elapsed_sleepers();
    bool wake_earliest_sleeper();
    ExecutionStop scheduler_stop(StopCategory category, const char* reason,
                                 uint64_t instruction_count) const;

    // Per-host-worker scheduling state (slot 0: the main/serial thread).
    Thread*& cur();
    Thread* cur() const;
    uint32_t& cur_core();
    uint32_t cur_core() const;
    uint64_t& cur_slice_start();
    uint64_t cur_slice_start() const;

    ExecutionStop run_parallel(uint64_t budget, uint64_t quantum);
    StepResult parallel_step(uint64_t quantum, uint64_t& executed,
                             std::chrono::microseconds& idle_for);
    Thread* select_parallel(bool main_worker, uint32_t& core);
    void worker_main(int slot);
    uint64_t wall_ticks() const;
    void sync_wall_ticks();

    ExecutionImage& image_;
    CafeCoreinit coreinit_;
    CafeFilesystem filesystem_;
    CafeOlv olv_;
    CafeRuntime cafe_runtime_;
    Executor executor_;
    std::deque<Thread> threads_;
    std::array<Thread*, 3> current_slots_{};
    std::array<uint32_t, 3> core_slots_{};
    std::array<uint64_t, 3> slice_start_slots_{};
    uint32_t last_active_thread_{};
    uint32_t cross_core_wake_{};
    std::array<uint32_t, 3> scheduled_threads_{};
    std::array<uint32_t, 3> pinned_threads_{};
    uint8_t scheduler_slot_{};
    uint64_t enqueue_sequence_{};
    std::deque<PendingCallback> pending_callbacks_;
    std::optional<CPUContext> callback_saved_context_;
    std::optional<PendingCallback> active_callback_;
    uint32_t callback_thread_address_{};
    uint32_t callback_return_address_{};

    // NWIIU_PARALLEL state; everything not atomic is under the HLE lock.
    std::vector<std::thread> workers_;
    std::atomic<bool> stopping_{};
    std::atomic<int> live_workers_{};
    std::optional<ExecutionStop> terminal_;
    std::exception_ptr worker_error_;
    std::array<bool, 3> core_busy_{};
    uint64_t parallel_quantum_{kSchedulerQuantum};
    std::chrono::steady_clock::time_point wall_start_{};
    uint64_t tick_base_{};
    uint64_t tick_skip_{};
    bool wall_clock_{};
};
} // namespace nwii::runtime
