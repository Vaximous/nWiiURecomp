#include "runtime/machine.h"
#include "runtime/host_sync.h"

#include <algorithm>
#include <tuple>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace nwii::runtime {
namespace {
// Which per-worker slot the calling host thread uses: 0 for the main thread
// (also the only one outside NWIIU_PARALLEL), 1..2 for the core workers.
thread_local int g_worker_slot = 0;

int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool env_flag(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

constexpr uint64_t kWallTimerHz = 62'156'250;
} // namespace

Machine::Thread*& Machine::cur() { return current_slots_[g_worker_slot]; }
Machine::Thread* Machine::cur() const { return current_slots_[g_worker_slot]; }
uint32_t& Machine::cur_core() { return core_slots_[g_worker_slot]; }
uint32_t Machine::cur_core() const { return core_slots_[g_worker_slot]; }
uint64_t& Machine::cur_slice_start() {
    return slice_start_slots_[g_worker_slot];
}
uint64_t Machine::cur_slice_start() const {
    return slice_start_slots_[g_worker_slot];
}

Machine::Machine(ExecutionImage& image, std::filesystem::path title_root,
                 std::optional<std::filesystem::path> save_root,
                 std::optional<std::filesystem::path> shared_font,
                 const std::map<uint32_t, std::string>& hle_hooks)
    : image_(image), coreinit_(image, std::move(shared_font)),
      filesystem_(image, std::move(title_root), std::move(save_root)),
      olv_(image), cafe_runtime_(image), executor_(image) {
    host_sync::mark_main_thread();
    host_sync::set_shutdown([this] { stop_workers(); });
    // NWIIU_FLAT_MEMORY (implied by NWIIU_PARALLEL): back guest memory with
    // one reserved host range. Required for parallel cores, and faster.
    if (host_sync::parallel() || env_flag("NWIIU_FLAT_MEMORY")) {
        if (!image_.memory.enable_host_flat() && host_sync::parallel()) {
            throw std::runtime_error(
                "NWIIU_PARALLEL needs a flat host mapping of guest memory");
        }
    }
    coreinit_.attach_machine(*this);
    cafe_runtime_.attach_machine(*this);
    coreinit_.register_handlers(cafe_runtime_);
    filesystem_.register_handlers(cafe_runtime_);
    olv_.register_handlers(cafe_runtime_);
    cafe_runtime_.register_extra_handlers();
    cafe_runtime_.register_imports(executor_);
    for (const auto& [address, name] : hle_hooks) {
        const NativeThunk thunk = find_native_hook(name);
        if (thunk == nullptr) {
            throw std::runtime_error("unknown HLE hook '" + name + "'");
        }
        executor_.register_patch(address, thunk);
    }

    CPUContext main;
    initialize_cpu(image_, main);
    threads_.push_back(
        {main, coreinit_.main_thread_address(), 16, ThreadState::ready, 7, 0,
         enqueue_sequence_++});
    coreinit_.initialize_thread(coreinit_.main_thread_address(), 1, 16,
                                image_.stack_top, image_.stack_base);
    coreinit_.begin_preinit(threads_.front().cpu);
}

CPUContext& Machine::main_context() { return threads_.front().cpu; }

uint32_t Machine::main_thread_address() const {
    return coreinit_.main_thread_address();
}

uint32_t Machine::add_thread(CPUContext context, int32_t priority) {
    if (priority < 0 || priority > 31) {
        throw std::invalid_argument("thread priority must be 0..31");
    }
    if (threads_.size() != 1) {
        throw std::runtime_error("Cafe runtime thread metadata exhausted");
    }
    const uint32_t address = abi::kSecondThread;
    threads_.push_back({std::move(context), address, priority,
                        ThreadState::ready, 7, 0, enqueue_sequence_++});
    coreinit_.initialize_thread(address, 2, priority,
                                threads_.back().cpu.gpr[1], 0);
    return address;
}
void Machine::add_suspended_thread(uint32_t address, CPUContext context,
                                   int32_t priority, uint8_t affinity) {
    if (priority < 0 || priority > 31) {
        throw std::invalid_argument("thread priority must be 0..31");
    }
    if (affinity == 0 || (affinity & ~uint8_t{7}) != 0) {
        throw std::invalid_argument("thread affinity must select a Cafe core");
    }
    // A joined/exited OSThread may be re-created in place (UE3 does this for
    // its pooled worker threads): reuse the moribund slot rather than erase,
    // because cur() points into threads_.
    for (auto& existing : threads_) {
        if (existing.address == address) {
            if (existing.state != ThreadState::moribund) {
                throw std::invalid_argument("duplicate Cafe thread");
            }
            existing = Thread{std::move(context), address, priority,
                              ThreadState::suspended, affinity, 0, 0};
            return;
        }
    }
    threads_.push_back({std::move(context), address, priority,
                        ThreadState::suspended, affinity, 0, 0});
}

bool Machine::has_thread(uint32_t address) const {
    return std::any_of(
        threads_.begin(), threads_.end(),
        [address](const Thread& candidate) { return candidate.address == address; });
}
void Machine::set_thread_affinity(uint32_t thread_address, uint8_t affinity) {
    thread(thread_address).affinity = affinity & 7;
}
void Machine::resume_thread(uint32_t thread_address) {
    auto& resumed = thread(thread_address);
    if (resumed.state == ThreadState::suspended) {
        enqueue(resumed);
    }
}




void Machine::dump_threads(std::ostream& output) const {
    for (const auto& item : threads_) {
        output << "THREAD address=0x" << std::hex << item.address
               << " pc=0x" << item.cpu.pc << " lr=0x" << item.cpu.lr
               << " entry=0x"
               << image_.memory.read32(item.address + abi::kOsThreadEntryOffset,
                                       item.cpu.pc)
               << " priority=" << std::dec << item.priority
               << " affinity=" << unsigned(item.affinity)
               << " state=" << static_cast<int>(item.state)
               << " wait=0x" << std::hex << item.wait_object
               << " sleeping=" << item.sleeping << '\n';
    }
}

Machine::Thread& Machine::thread(uint32_t address) {
    const auto found = std::find_if(
        threads_.begin(), threads_.end(),
        [address](const Thread& candidate) { return candidate.address == address; });
    if (found == threads_.end()) {
        throw std::invalid_argument("unknown Cafe thread");
    }
    return *found;
}

const Machine::Thread& Machine::thread(uint32_t address) const {
    const auto found = std::find_if(
        threads_.begin(), threads_.end(),
        [address](const Thread& candidate) { return candidate.address == address; });
    if (found == threads_.end()) {
        throw std::invalid_argument("unknown Cafe thread");
    }
    return *found;
}

CPUContext& Machine::context(uint32_t thread_address) {
    return thread(thread_address).cpu;
}

const CPUContext& Machine::context(uint32_t thread_address) const {
    return thread(thread_address).cpu;
}

Machine::Thread* Machine::select_ready(uint32_t excluded_thread) {
    const auto schedule = [this](Thread& thread, uint32_t core) {
        cur_core() = core;
        scheduled_threads_[scheduler_slot_++] = thread.address;
        if (scheduler_slot_ == scheduled_threads_.size()) {
            scheduler_slot_ = 0;
            scheduled_threads_.fill(0);
        }
        return &thread;
    };

    const auto core_available = [this](const Thread& thread, uint32_t core) {
        return pinned_threads_[core] == 0 ||
               pinned_threads_[core] == thread.address;
    };
    if (cross_core_wake_ != 0 && cross_core_wake_ != excluded_thread) {
        auto& woken = thread(cross_core_wake_);
        if (woken.state == ThreadState::ready && woken.affinity != 0) {
            cross_core_wake_ = 0;
            if (woken.pinned_core < 3 &&
                core_available(woken, woken.pinned_core)) {
                return schedule(woken, woken.pinned_core);
            }
            for (uint32_t core = 0; core < 3; ++core) {
                if ((woken.affinity & (1u << core)) != 0 &&
                    core != cur_core() && core_available(woken, core)) {
                    return schedule(woken, core);
                }
            }
            if (core_available(woken, cur_core())) {
                return schedule(woken, cur_core());
            }
        }
        cross_core_wake_ = 0;
    }

    constexpr std::array<uint32_t, 3> core_order{1, 0, 2};
    for (uint32_t tried = 0; tried < 6; ++tried) {
        const uint32_t core = core_order[scheduler_slot_];
        Thread* selected = nullptr;
        for (auto& candidate : threads_) {
            if (candidate.state != ThreadState::ready ||
                (candidate.affinity & (1u << core)) == 0 ||
                (candidate.pinned_core < 3 &&
                 candidate.pinned_core != core) ||
                !core_available(candidate, core) ||
                candidate.address == excluded_thread ||
                std::find(scheduled_threads_.begin(),
                          scheduled_threads_.end(),
                          candidate.address) != scheduled_threads_.end()) {
                continue;
            }
            if (selected == nullptr ||
                candidate.priority < selected->priority ||
                (candidate.priority == selected->priority &&
                 candidate.enqueue_sequence < selected->enqueue_sequence)) {
                selected = &candidate;
            }
        }
        if (selected != nullptr) {
            return schedule(*selected, core);
        }
        scheduled_threads_[scheduler_slot_++] = 0;
        if (scheduler_slot_ == scheduled_threads_.size()) {
            scheduler_slot_ = 0;
            scheduled_threads_.fill(0);
        }
    }
    return nullptr;
}

void Machine::enqueue(Thread& candidate) {
    candidate.state = ThreadState::ready;
    candidate.wait_object = 0;
    candidate.sleeping = false;
    candidate.enqueue_sequence = enqueue_sequence_++;
    coreinit_.set_thread_state(candidate.address, abi::kThreadReady);
    if (wall_clock_) {
        candidate.ready_since_ns = steady_ns();
        host_sync::kick();
    }
}
void Machine::queue_callback(uint32_t function, uint32_t argument,
                             bool external) {
    if (function == 0) {
        return;
    }
    if (callback_return_address_ == 0) {
        const auto found = std::find_if(
            image_.imports.begin(), image_.imports.end(),
            [](const auto& item) {
                return item.second.module == "coreinit" &&
                       item.second.symbol == "OSExitThread";
            });
        if (found == image_.imports.end()) {
            throw std::logic_error("guest callback requires OSExitThread");
        }
        callback_return_address_ = found->first;
    }
    pending_callbacks_.push_back(
        {function, argument,
         external || cur() == nullptr ? 0 : cur()->address});
}

void Machine::queue_callback_args(uint32_t function,
                                  const std::array<uint32_t, 4>& arguments) {
    if (function == 0) {
        return;
    }
    queue_callback(function, arguments[0], true);
    pending_callbacks_.back().extra = {arguments[1], arguments[2],
                                       arguments[3]};
}

void Machine::set_thread_priority(uint32_t thread_address, int32_t priority) {
    if (has_thread(thread_address)) {
        thread(thread_address).priority = priority;
    }
}

bool Machine::suspend_thread(uint32_t thread_address) {
    if (!has_thread(thread_address)) {
        return false;
    }
    auto& target = thread(thread_address);
    if (target.state == ThreadState::moribund) {
        return false;
    }
    if (&target == cur()) {
        target.state = ThreadState::suspended;
        return true;
    }
    // Parallel: a thread running on another core stops at its slice end
    // (its worker requeues only threads still marked running).
    if (target.state == ThreadState::ready ||
        (target.on_cpu && target.state == ThreadState::running)) {
        target.state = ThreadState::suspended;
    }
    return false;
}

bool Machine::thread_exited(uint32_t thread_address) const {
    return has_thread(thread_address) &&
           thread(thread_address).state == ThreadState::moribund;
}

void Machine::begin_callback(Thread& candidate) {
    const auto callback = pending_callbacks_.front();
    pending_callbacks_.pop_front();
    cafe_runtime_.prepare_audio_frame(callback.argument);
    if (std::getenv("NWIIU_CALLBACK_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "CALLBACK begin function=%08X thread=%08X saved_pc=%08X "
                     "sp=%08X r3=%08X\n",
                     callback.function, candidate.address, candidate.cpu.pc,
                     candidate.cpu.gpr[1], candidate.cpu.gpr[3]);
    }
    callback_saved_context_ = candidate.cpu;
    active_callback_ = callback;
    callback_thread_address_ = candidate.address;
    candidate.cpu.pc = callback.function;
    candidate.cpu.lr = callback_return_address_;
    // The interrupted code may sit between `stw r0,4(r1)` and `stwu`, so the
    // LR-save word above r1 is live. Run the callback in a fresh frame well
    // below it; otherwise the callback's prologue overwrites the interrupted
    // function's saved LR with the OSExitThread trampoline and that thread
    // "returns" into thread exit later.
    {
        const uint32_t old_sp = candidate.cpu.gpr[1];
        const uint32_t new_sp = (old_sp - 0x400u) & ~0xFu;
        image_.memory.write32(new_sp, old_sp, candidate.cpu.pc);
        image_.memory.write32(new_sp + 4, 0, candidate.cpu.pc);
        candidate.cpu.gpr[1] = new_sp;
    }
    candidate.cpu.gpr[3] = callback.argument;
    candidate.cpu.gpr[4] = callback.extra[0];
    candidate.cpu.gpr[5] = callback.extra[1];
    candidate.cpu.gpr[6] = callback.extra[2];
}

bool Machine::return_from_callback() {
    if (cur() == nullptr || !callback_saved_context_ ||
        cur()->address != callback_thread_address_) {
        return false;
    }
    if (std::getenv("NWIIU_CALLBACK_TRACE") != nullptr) {
        std::fprintf(stderr,
                     "CALLBACK return thread=%08X callback_pc=%08X "
                     "restore_pc=%08X sp=%08X r3=%08X\n",
                     cur()->address, cur()->cpu.pc,
                     callback_saved_context_->pc,
                     callback_saved_context_->gpr[1],
                     callback_saved_context_->gpr[3]);
    }
    cafe_runtime_.capture_audio_frame(active_callback_->argument);
    const uint64_t instruction_count = cur()->cpu.instruction_count;
    cur()->cpu = std::move(*callback_saved_context_);
    cur()->cpu.instruction_count = instruction_count;
    callback_saved_context_.reset();
    active_callback_.reset();
    callback_thread_address_ = 0;
    enqueue(*cur());
    return true;
}


ExecutionStop Machine::scheduler_stop(StopCategory category, const char* reason,
                                      uint64_t instruction_count) const {
    ExecutionStop stop;
    stop.category = category;
    stop.reason = reason;
    stop.instruction_count = instruction_count;
    if (!threads_.empty()) {
        const auto& cpu = last_active_thread_ == 0
                              ? threads_.front().cpu
                              : thread(last_active_thread_).cpu;
        stop.pc = cpu.pc;
        stop.lr = cpu.lr;
        stop.history_size = cpu.history_size;
        const size_t oldest =
            (cpu.history_cursor + cpu.pc_history.size() - cpu.history_size) %
            cpu.pc_history.size();
        for (size_t index = 0; index < cpu.history_size; ++index) {
            stop.history[index] =
                cpu.pc_history[(oldest + index) % cpu.pc_history.size()];
        }
    }
    executor_.snapshot_trace(stop, last_active_thread_);
    // NWIIU_THREAD_DUMP=1: every guest thread's state, wait object and stack
    // back chain at stop. Finds which thread a stalled loading screen is
    // waiting on.
    // The windowed runner calls run() once per frame, so only the latest
    // dump is kept and it is printed at exit.
    static const bool thread_dump = std::getenv("NWIIU_THREAD_DUMP") != nullptr;
    if (thread_dump) {
        static std::string* latest = [] {
            auto* text = new std::string();
            static auto* printed = text;
            std::atexit([] { std::fputs(printed->c_str(), stderr); });
            return text;
        }();
        static const char* kStates[] = {"suspended", "ready", "running",
                                        "waiting", "moribund"};
        latest->clear();
        char line[256];
        for (const auto& t : threads_) {
            std::snprintf(line, sizeof line,
                          "THREAD %08X prio=%d state=%s wait=%08X sleep=%d "
                          "pc=%08X lr=%08X r3=%08X sp=%08X instr=%llu\n"
                          "  chain:",
                          t.address, t.priority,
                          kStates[static_cast<int>(t.state)], t.wait_object,
                          t.sleeping ? 1 : 0, t.cpu.pc, t.cpu.lr, t.cpu.gpr[3],
                          t.cpu.gpr[1],
                          static_cast<unsigned long long>(
                              t.cpu.instruction_count));
            *latest += line;
            uint32_t sp = t.cpu.gpr[1];
            for (int depth = 0; depth < 24 && sp != 0; ++depth) {
                try {
                    const uint32_t next = image_.memory.read32(sp, 0);
                    if (next == 0 || next <= sp) {
                        break;
                    }
                    std::snprintf(line, sizeof line, " %08X",
                                  image_.memory.read32(next + 4, 0));
                    *latest += line;
                    sp = next;
                } catch (...) {
                    break;
                }
            }
            *latest += "\n";
        }
        // NWIIU_THREAD_DUMP_FILE=<path>: rewrite the dump there every 200
        // calls (~200M instructions windowed), readable while a run hangs.
        static const char* dump_file = std::getenv("NWIIU_THREAD_DUMP_FILE");
        static uint64_t calls = 0;
        if (dump_file != nullptr && ++calls % 200 == 0) {
            if (FILE* out = std::fopen(dump_file, "w")) {
                std::fprintf(out, "calls=%llu\n",
                             static_cast<unsigned long long>(calls));
                std::fputs(latest->c_str(), out);
                std::fclose(out);
            }
        }
    }
    return stop;
}

ExecutionStop Machine::run(uint64_t global_instruction_budget,
                           uint64_t quantum) {
    if (quantum == 0) {
        throw std::invalid_argument("scheduler quantum must be positive");
    }
    uint64_t executed = 0;
    if (global_instruction_budget == 0) {
        return scheduler_stop(StopCategory::instruction_budget,
                              "global instruction budget exhausted", 0);
    }
    if (host_sync::parallel()) {
        return run_parallel(global_instruction_budget, quantum);
    }

    while (executed < global_instruction_budget) {
        wake_elapsed_sleepers();
        cur() = select_ready();
        if (cur() == nullptr) {
            if (wake_earliest_sleeper()) {
                continue;
            }
            return scheduler_stop(StopCategory::deadlock,
                                  "Cafe scheduler deadlock", executed);
        }
        if (!pending_callbacks_.empty() && !callback_saved_context_) {
            const uint32_t owner = pending_callbacks_.front().owner_thread;
            const bool owner_blocked =
                owner == 0 || !has_thread(owner) ||
                (thread(owner).state != ThreadState::ready &&
                 thread(owner).state != ThreadState::running);
            if (owner_blocked) {
                if (auto* callback_thread = select_ready(owner)) {
                    cur() = callback_thread;
                    begin_callback(*cur());
                }
            }
        }
        cur()->state = ThreadState::running;
        coreinit_.set_thread_state(cur()->address, abi::kThreadRunning);
        last_active_thread_ = cur()->address;

        const uint64_t remaining = global_instruction_budget - executed;
        const uint64_t allowance = std::min(quantum, remaining);
        const uint64_t before = cur()->cpu.instruction_count;
        cur_slice_start() = before;
        const uint64_t endpoint =
            before > std::numeric_limits<uint64_t>::max() - allowance
                ? std::numeric_limits<uint64_t>::max()
                : before + allowance;
        auto slice = executor_.run_slice(cur()->cpu, endpoint,
                                         cur()->address);
        const uint64_t delta = cur()->cpu.instruction_count - before;
        executed += delta;
        coreinit_.advance_ticks(delta);

        if (slice.category == SliceCategory::terminal) {
            if (slice.terminal->category == StopCategory::guest_exit) {
                cur()->state = ThreadState::moribund;
                coreinit_.set_thread_state(cur()->address,
                                           abi::kThreadMoribund);
            }
            cur() = nullptr;
            slice.terminal->instruction_count = executed;
            return *slice.terminal;
        }
        if (cur()->state == ThreadState::running) {
            enqueue(*cur());
        }
        cur() = nullptr;
    }

    return scheduler_stop(StopCategory::instruction_budget,
                          "global instruction budget exhausted", executed);
}

uint32_t Machine::current_thread_address() const {
    if (cur() == nullptr) {
        throw std::logic_error("no current Cafe thread");
    }
    return cur()->address;
}

uint32_t Machine::current_core_id() const {
    if (cur() == nullptr) {
        throw std::logic_error("no current Cafe thread");
    }
    return cur_core();
}

void Machine::pin_current_core() {
    if (cur() == nullptr) {
        throw std::logic_error("no current Cafe thread");
    }
    if ((cur()->pinned_core < 3 &&
         cur()->pinned_core != cur_core()) ||
        (pinned_threads_[cur_core()] != 0 &&
         pinned_threads_[cur_core()] != cur()->address)) {
        throw std::logic_error("Cafe core already pinned");
    }
    cur()->pinned_core = static_cast<uint8_t>(cur_core());
    pinned_threads_[cur_core()] = cur()->address;
}

void Machine::unpin_current_core() {
    if (cur() == nullptr || cur()->pinned_core != cur_core() ||
        pinned_threads_[cur_core()] != cur()->address) {
        throw std::logic_error("Cafe thread is not pinned to this core");
    }
    pinned_threads_[cur_core()] = 0;
    cur()->pinned_core = 0xFF;
}

int32_t Machine::current_thread_priority() const {
    if (cur() == nullptr) {
        throw std::logic_error("no current Cafe thread");
    }
    return cur()->priority;
}

uint64_t Machine::current_time_ticks() const {
    if (wall_clock_) {
        return std::max(wall_ticks(), coreinit_.ticks());
    }
    if (cur() == nullptr) {
        return coreinit_.ticks();
    }
    return coreinit_.ticks() +
           (cur()->cpu.instruction_count -
            cur_slice_start());
}

void Machine::block_current(uint32_t wait_object) {
    if (cur() == nullptr) {
        throw std::logic_error("no current Cafe thread");
    }
    block_thread(cur()->address, wait_object);
}

void Machine::exit_current(int32_t value) {
    if (cur() == nullptr) {
        throw std::logic_error("no current Cafe thread");
    }
    image_.memory.write32(
        cur()->address + abi::kOsThreadExitValueOffset,
        static_cast<uint32_t>(value), cur()->cpu.pc);
    cur()->state = ThreadState::moribund;
    coreinit_.set_thread_state(cur()->address, abi::kThreadMoribund);
    // OSJoinThread blocks on the joined thread's address.
    wake_waiters(cur()->address);
}

void Machine::block_thread(uint32_t thread_address, uint32_t wait_object) {
    auto& blocked = thread(thread_address);
    if (blocked.state == ThreadState::moribund) {
        throw std::invalid_argument("cannot block moribund Cafe thread");
    }
    blocked.state = ThreadState::waiting;
    blocked.wait_object = wait_object;
    coreinit_.set_thread_state(blocked.address, abi::kThreadWaiting);
}

void Machine::wake_thread(uint32_t thread_address) {
    auto& woken = thread(thread_address);
    if (woken.state != ThreadState::waiting) {
        throw std::invalid_argument("Cafe thread is not waiting");
    }
    const bool can_run_on_another_core =
        cur() != nullptr &&
        (woken.affinity &
         static_cast<uint8_t>(7u & ~(1u << current_core_id()))) != 0;
    enqueue(woken);
    if (can_run_on_another_core && cross_core_wake_ == 0) {
        cross_core_wake_ = thread_address;
    }
}

void Machine::wake_waiters(uint32_t wait_object) {
    for (auto& candidate : threads_) {
        if (candidate.state == ThreadState::waiting &&
            candidate.wait_object == wait_object) {
            enqueue(candidate);
        }
    }
}

void Machine::sleep_current(uint64_t delta_ticks) {
    if (cur() == nullptr) {
        throw std::logic_error("no current Cafe thread");
    }
    cur()->state = ThreadState::waiting;
    cur()->wait_object = 0;
    cur()->sleeping = true;
    cur()->wake_tick = current_time_ticks() + delta_ticks;
    coreinit_.set_thread_state(cur()->address, abi::kThreadWaiting);
}

void Machine::wake_elapsed_sleepers() {
    const uint64_t now = wall_clock_ ? current_time_ticks() : coreinit_.ticks();
    for (auto& candidate : threads_) {
        if (candidate.state == ThreadState::waiting && candidate.sleeping &&
            candidate.wake_tick <= now) {
            enqueue(candidate);
        }
    }
}

bool Machine::wake_earliest_sleeper() {
    Thread* earliest = nullptr;
    for (auto& candidate : threads_) {
        if (candidate.state == ThreadState::waiting && candidate.sleeping &&
            (earliest == nullptr ||
             candidate.wake_tick < earliest->wake_tick)) {
            earliest = &candidate;
        }
    }
    if (earliest == nullptr) {
        return false;
    }
    if (earliest->wake_tick > coreinit_.ticks()) {
        coreinit_.advance_ticks(earliest->wake_tick - coreinit_.ticks());
    }
    enqueue(*earliest);
    return true;
}
// ---------------------------------------------------------------------------
// NWIIU_PARALLEL: the three Cafe cores on host threads.
//
// The main thread (slot 0) and two worker threads (slots 1, 2) each pick a
// free core and the best ready thread for it, run one slice without the HLE
// lock, then requeue it. Guest time follows the wall clock. The main thread
// returns from run() every few milliseconds so the runner can pump SDL.

uint64_t Machine::wall_ticks() const {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - wall_start_)
                             .count();
    const uint64_t ns = static_cast<uint64_t>(elapsed);
    // 62.15625 MHz: ticks = ns * 62156250 / 1e9 = ns * 1989 / 32000, split to
    // avoid overflow.
    const uint64_t ticks = ns / 32000 * 1989 + (ns % 32000) * 1989 / 32000;
    return tick_base_ + tick_skip_ + ticks;
}

void Machine::sync_wall_ticks() {
    const uint64_t now = wall_ticks();
    if (now > coreinit_.ticks()) {
        coreinit_.advance_ticks(now - coreinit_.ticks());
    }
}

Machine::Thread* Machine::select_parallel(bool main_worker, uint32_t& core) {
    static const int64_t starve_ns = [] {
        const char* value = std::getenv("NWIIU_STARVE_MS");
        return static_cast<int64_t>(value ? std::strtoll(value, nullptr, 10)
                                          : 0) *
               1'000'000;
    }();
    const int64_t now = steady_ns();
    Thread* best = nullptr;
    uint32_t best_core = 0;
    // Lower is better: GX2 thread first on the main worker, then any thread
    // that has waited too long, then priority, then FIFO.
    auto rank = [&](const Thread& t) {
        const int gpu = main_worker && t.gpu_affine ? 0 : 1;
        // Off by default (NWIIU_STARVE_MS=0): Cafe OS is strictly priority
        // scheduled and the title relies on it.
        const int starved =
            starve_ns > 0 && now - t.ready_since_ns > starve_ns ? 0 : 1;
        return std::tuple(gpu, starved, t.priority, t.enqueue_sequence);
    };
    for (auto& candidate : threads_) {
        if (candidate.state != ThreadState::ready || candidate.on_cpu ||
            candidate.affinity == 0) {
            continue;
        }
        if (!main_worker && candidate.gpu_affine) {
            continue;
        }
        uint32_t chosen = 3;
        for (uint32_t c = 0; c < 3; ++c) {
            if (core_busy_[c] || (candidate.affinity & (1u << c)) == 0 ||
                (candidate.pinned_core < 3 && candidate.pinned_core != c) ||
                (pinned_threads_[c] != 0 &&
                 pinned_threads_[c] != candidate.address)) {
                continue;
            }
            chosen = c;
            break;
        }
        if (chosen == 3) {
            continue;
        }
        if (best == nullptr || rank(candidate) < rank(*best)) {
            best = &candidate;
            best_core = chosen;
        }
    }
    core = best_core;
    return best;
}

Machine::StepResult Machine::parallel_step(
    uint64_t quantum, uint64_t& executed,
    std::chrono::microseconds& idle_for) {
    static const bool idle_skip = env_flag("NWIIU_IDLE_SKIP");
    static const bool trace = env_flag("NWIIU_PARALLEL_TRACE");
    const bool main_worker = g_worker_slot == 0;
    Thread* running = nullptr;
    uint32_t core = 0;
    {
        host_sync::HleLock lock;
        if (stopping_.load()) {
            return StepResult::stopped;
        }
        sync_wall_ticks();
        wake_elapsed_sleepers();
        running = select_parallel(main_worker, core);
        if (running == nullptr) {
            // Nothing runnable for this worker. Sleep until the earliest
            // sleeper is due (or a kick); with NWIIU_IDLE_SKIP, jump guest
            // time there when every core is idle.
            const bool any_busy =
                core_busy_[0] || core_busy_[1] || core_busy_[2];
            bool any_ready = false;
            const Thread* earliest = nullptr;
            for (const auto& t : threads_) {
                if (t.state == ThreadState::ready || t.on_cpu) {
                    any_ready = true;
                }
                if (t.state == ThreadState::waiting && t.sleeping &&
                    (earliest == nullptr ||
                     t.wake_tick < earliest->wake_tick)) {
                    earliest = &t;
                }
            }
            idle_for = std::chrono::microseconds(2000);
            if (earliest != nullptr) {
                const uint64_t now = current_time_ticks();
                const uint64_t due =
                    earliest->wake_tick > now ? earliest->wake_tick - now : 0;
                if (idle_skip && !any_busy && !any_ready && due > 0) {
                    tick_skip_ += due;
                    sync_wall_ticks();
                    idle_for = std::chrono::microseconds(0);
                } else {
                    idle_for = std::chrono::microseconds(
                        std::clamp<uint64_t>(due * 1'000'000 / kWallTimerHz,
                                             1000, 2000));
                }
            } else if (!any_busy && !any_ready && pending_callbacks_.empty()) {
                terminal_ = scheduler_stop(StopCategory::deadlock,
                                           "Cafe scheduler deadlock",
                                           executed);
                stopping_.store(true);
                host_sync::kick();
                return StepResult::stopped;
            }
            return StepResult::idle;
        }
        if (!pending_callbacks_.empty() && !callback_saved_context_) {
            const uint32_t owner = pending_callbacks_.front().owner_thread;
            const bool owner_blocked =
                owner == 0 || !has_thread(owner) ||
                (thread(owner).state != ThreadState::ready &&
                 thread(owner).state != ThreadState::running);
            if (owner_blocked && running->address != owner) {
                begin_callback(*running);
            }
        }
        running->state = ThreadState::running;
        running->on_cpu = true;
        core_busy_[core] = true;
        cur() = running;
        cur_core() = core;
        cur_slice_start() = running->cpu.instruction_count;
        coreinit_.set_thread_state(running->address, abi::kThreadRunning);
        last_active_thread_ = running->address;
    }

    const uint64_t before = running->cpu.instruction_count;
    const uint64_t endpoint =
        before > std::numeric_limits<uint64_t>::max() - quantum
            ? std::numeric_limits<uint64_t>::max()
            : before + quantum;
    auto slice = executor_.run_slice(running->cpu, endpoint, running->address);

    host_sync::HleLock lock;
    const uint64_t delta = running->cpu.instruction_count - before;
    executed += delta;
    running->on_cpu = false;
    core_busy_[core] = false;
    if (host_sync::take_marshalled() && !main_worker &&
        !running->gpu_affine) {
        running->gpu_affine = true;
        if (trace) {
            std::fprintf(stderr, "PARALLEL gx2 thread %08X -> main worker\n",
                         running->address);
        }
    }
    cur() = nullptr;
    if (slice.category == SliceCategory::terminal) {
        if (slice.terminal->category == StopCategory::guest_exit) {
            running->state = ThreadState::moribund;
            coreinit_.set_thread_state(running->address,
                                       abi::kThreadMoribund);
        }
        if (!terminal_) {
            slice.terminal->instruction_count = executed;
            terminal_ = std::move(*slice.terminal);
        }
        stopping_.store(true);
        host_sync::kick();
        return StepResult::stopped;
    }
    if (running->state == ThreadState::running) {
        enqueue(*running);
    } else {
        // Its core is free now; let an idle worker take it.
        host_sync::kick();
    }
    return StepResult::ran;
}

void Machine::worker_main(int slot) {
    g_worker_slot = slot;
    uint64_t executed = 0;
    while (!stopping_.load()) {
        try {
            std::chrono::microseconds idle_for{};
            const auto result =
                parallel_step(parallel_quantum_, executed, idle_for);
            if (result == StepResult::stopped) {
                break;
            }
            if (result == StepResult::idle && idle_for.count() > 0) {
                host_sync::idle_wait(idle_for);
            }
        } catch (...) {
            host_sync::HleLock lock;
            if (!worker_error_) {
                worker_error_ = std::current_exception();
            }
            stopping_.store(true);
            host_sync::kick();
            break;
        }
    }
    live_workers_.fetch_sub(1);
    host_sync::kick();
}

ExecutionStop Machine::run_parallel(uint64_t budget, uint64_t quantum) {
    if (workers_.empty() && !stopping_.load()) {
        {
            host_sync::HleLock lock;
            parallel_quantum_ = quantum;
            tick_base_ = coreinit_.ticks();
            wall_start_ = std::chrono::steady_clock::now();
            wall_clock_ = true;
            const int64_t now = steady_ns();
            for (auto& t : threads_) {
                t.ready_since_ns = now;
            }
        }
        static const int worker_count = [] {
            const char* value = std::getenv("NWIIU_PARALLEL_WORKERS");
            const int count = value ? std::atoi(value) : 2;
            return std::clamp(count, 0, 2);
        }();
        for (int slot = 1; slot <= worker_count; ++slot) {
            live_workers_.fetch_add(1);
            workers_.emplace_back([this, slot] { worker_main(slot); });
        }
        std::fprintf(stderr, "PARALLEL: %d host workers + main thread\n",
                     worker_count);
    }

    static const auto slice_time = std::chrono::microseconds([] {
        const char* value = std::getenv("NWIIU_PARALLEL_PUMP_US");
        return value ? std::strtoll(value, nullptr, 10) : 4000;
    }());
    const auto deadline = std::chrono::steady_clock::now() + slice_time;
    uint64_t executed = 0;
    while (true) {
        host_sync::service_main();
        if (stopping_.load()) {
            host_sync::HleLock lock;
            if (worker_error_) {
                std::rethrow_exception(worker_error_);
            }
            if (terminal_) {
                return *terminal_;
            }
            return scheduler_stop(StopCategory::deadlock,
                                  "parallel scheduler stopped", executed);
        }
        std::chrono::microseconds idle_for{};
        StepResult result;
        try {
            result = parallel_step(quantum, executed, idle_for);
        } catch (...) {
            stopping_.store(true);
            host_sync::kick();
            throw;
        }
        if (result == StepResult::stopped) {
            continue;
        }
        if (executed >= budget ||
            std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if (result == StepResult::idle && idle_for.count() > 0) {
            const auto left =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    deadline - std::chrono::steady_clock::now());
            host_sync::idle_wait(std::max(std::chrono::microseconds(0),
                                          std::min(idle_for, left)));
        }
    }
    host_sync::HleLock lock;
    return scheduler_stop(StopCategory::instruction_budget,
                          "global instruction budget exhausted", executed);
}

void Machine::stop_workers() {
    if (workers_.empty()) {
        return;
    }
    stopping_.store(true);
    // A worker may be waiting for the main thread to run a GX2 event for it.
    while (live_workers_.load() > 0) {
        host_sync::service_main();
        host_sync::kick();
        std::this_thread::yield();
    }
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
}

Machine::~Machine() {
    stop_workers();
    host_sync::set_shutdown(nullptr);
}
} // namespace nwii::runtime
