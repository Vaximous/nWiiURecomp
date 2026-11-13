// coreinit imports beyond the WWHD set, first needed by Devil's Third (UE3):
// fast mutexes/conditions, semaphores, timed waits, joins, block heaps, the
// locked cache, calendar time, and assorted system queries.
#include "runtime/cafe_coreinit.h"
#include "runtime/cafe_runtime.h"
#include "runtime/machine.h"
#include "runtime/rpx_exports.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <string>

namespace nwii::runtime {
namespace {
constexpr uint32_t kOsTimerHz = abi::kBusClockSpeed / 4;
constexpr uint32_t kFastMutexTag = 0x664D7458;  // 'fMtX'
constexpr uint32_t kFastCondTag = 0x664E6456;   // 'fCnD'
constexpr uint32_t kCondTag = 0x634E6456;       // 'cNdV'
constexpr uint32_t kSemaphoreTag = 0x73506852;  // 'sPhR'
constexpr uint32_t kSemaphoreCountOffset = 0x0C;
// Free words in the Cafe system block (WWHD state ends at 0x01000DC0, the
// GHS iob begins at 0x01010000).
constexpr uint32_t kSystemQueue = 0x01000E00;
constexpr uint32_t kSystemQueueMessages = 0x01000E40;
constexpr uint32_t kSystemQueueDepth = 16;
constexpr uint32_t kLockedCacheSize = 0x4000;
constexpr uint32_t kPreinitHeapVars = 0x01000F00; // MEM1, FG, MEM2 handles
constexpr uint32_t kPreinitReturn = 0xC1000010;

std::string read_string(const GuestMemory& memory, uint32_t address,
                        uint32_t pc, size_t limit = 1024) {
    std::string text;
    if (address == 0) {
        return text;
    }
    for (size_t index = 0; index < limit; ++index) {
        const char c = static_cast<char>(memory.read8(
            address + static_cast<uint32_t>(index), pc));
        if (c == '\0') {
            break;
        }
        text.push_back(c);
    }
    return text;
}

uint64_t pair64(uint32_t high, uint32_t low) {
    return (uint64_t{high} << 32) | low;
}

void set_pair64(CPUContext& cpu, uint64_t value) {
    cpu.gpr[3] = static_cast<uint32_t>(value >> 32);
    cpu.gpr[4] = static_cast<uint32_t>(value);
}

// Howard Hinnant's civil-date algorithms.
int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<int64_t>(doe) - 719468;
}

void civil_from_days(int64_t z, int64_t& y, unsigned& m, unsigned& d) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = static_cast<unsigned>(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = static_cast<int64_t>(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp < 10 ? mp + 3 : mp - 9;
    y += m <= 2;
}

const int64_t kEpoch2000Days = days_from_civil(2000, 1, 1);
} // namespace

bool CafeCoreinit::post_message(uint32_t queue, uint32_t message,
                                CPUContext& cpu) {
    CPUContext scratch = cpu;
    scratch.gpr[3] = queue;
    scratch.gpr[4] = message;
    scratch.gpr[5] = 0;
    send_message(scratch);
    return scratch.gpr[3] != 0;
}

HleAction CafeCoreinit::acquire_lock(MutexState& state, uint32_t address,
                                     CPUContext& cpu) {
    const uint32_t current = machine_->current_thread_address();
    if (state.owner == 0) {
        state.owner = current;
        state.recursion = 1;
        state.granted = 0;
        return HleAction::return_to_lr;
    }
    if (state.owner == current) {
        if (state.granted == current) {
            state.granted = 0;
        } else {
            ++state.recursion;
        }
        return HleAction::return_to_lr;
    }
    if (std::none_of(state.waiters.begin(), state.waiters.end(),
                     [current](const MutexWaiter& waiter) {
                         return waiter.thread == current;
                     })) {
        state.waiters.push_back(
            {current, machine_->current_thread_priority(), wait_sequence_++});
    }
    (void)cpu;
    machine_->block_current(address);
    return HleAction::reschedule;
}

uint32_t CafeCoreinit::release_lock(MutexState& state) {
    const uint32_t recursion = state.recursion;
    if (state.waiters.empty()) {
        state.owner = 0;
        state.recursion = 0;
        state.granted = 0;
        return recursion;
    }
    const auto next = std::min_element(state.waiters.begin(),
                                       state.waiters.end(), waiter_before);
    const uint32_t woken = next->thread;
    state.waiters.erase(next);
    state.owner = woken;
    state.recursion = 1;
    state.granted = woken;
    machine_->wake_thread(woken);
    return recursion;
}

HleAction CafeCoreinit::cond_wait(CPUContext& cpu, bool fast) {
    const uint32_t cond = cpu.gpr[3];
    const uint32_t mutex_address = cpu.gpr[4];
    const uint32_t current = machine_->current_thread_address();
    MutexState& state = fast ? fast_mutexes_[mutex_address]
                             : mutex(mutex_address, cpu.pc);
    auto found = cond_waits_.find(current);
    if (found == cond_waits_.end()) {
        if (state.owner != current) {
            fault("condition wait without owning mutex", mutex_address, 4,
                  cpu.pc);
        }
        CondWait wait{cond, mutex_address, 0, fast, false};
        wait.recursion = release_lock(state);
        if (!fast) {
            mirror_mutex(mutex_address, state, cpu.pc);
        }
        cond_waits_.emplace(current, wait);
        cond_waiters_[cond].push_back(current);
        machine_->block_current(cond);
        return HleAction::reschedule;
    }
    CondWait& wait = found->second;
    if (!wait.signaled) {
        machine_->block_current(cond);
        return HleAction::reschedule;
    }
    const HleAction action = acquire_lock(state, mutex_address, cpu);
    if (!fast) {
        mirror_mutex(mutex_address, state, cpu.pc);
    }
    if (action != HleAction::return_to_lr) {
        return action;
    }
    state.recursion = wait.recursion;
    if (!fast) {
        mirror_mutex(mutex_address, state, cpu.pc);
    }
    cond_waits_.erase(found);
    return HleAction::return_to_lr;
}

void CafeCoreinit::cond_signal(uint32_t cond) {
    auto found = cond_waiters_.find(cond);
    if (found == cond_waiters_.end()) {
        return;
    }
    for (const uint32_t thread : found->second) {
        cond_waits_[thread].signaled = true;
    }
    cond_waiters_.erase(found);
    machine_->wake_waiters(cond);
}

void CafeCoreinit::begin_preinit(CPUContext& main) {
    const uint32_t preinit = rpx_export_address("__preinit_user");
    if (preinit == 0) {
        return;
    }
    memory_.write32(kPreinitHeapVars + 0, base_heap(0), 0);
    memory_.write32(kPreinitHeapVars + 4, base_heap(8), 0);
    memory_.write32(kPreinitHeapVars + 8, base_heap(1), 0);
    main.pc = preinit;
    main.lr = kPreinitReturn;
    main.gpr[3] = kPreinitHeapVars + 0;
    main.gpr[4] = kPreinitHeapVars + 4;
    main.gpr[5] = kPreinitHeapVars + 8;
}

void CafeCoreinit::register_extra_handlers(CafeRuntime& runtime) {
    const auto reg = [&runtime](const char* symbol, HleHandler handler) {
        runtime.register_handler("coreinit", symbol, std::move(handler));
    };
    const auto ret = [&reg](const char* symbol, uint32_t value) {
        reg(symbol, [value](CPUContext& cpu, GuestMemory&) {
            cpu.gpr[3] = value;
            return HleAction::return_to_lr;
        });
    };
    const auto noop = [&reg](const char* symbol) {
        reg(symbol, [](CPUContext&, GuestMemory&) {
            return HleAction::return_to_lr;
        });
    };

    image_.imports.try_emplace(
        kPreinitReturn, ImportTarget{"coreinit", "__nwiiu_preinit_return"});
    reg("__nwiiu_preinit_return", [this](CPUContext& cpu, GuestMemory&) {
        base_overrides_[0] = memory_.read32(kPreinitHeapVars + 0, cpu.pc);
        base_overrides_[2] = memory_.read32(kPreinitHeapVars + 4, cpu.pc);
        base_overrides_[1] = memory_.read32(kPreinitHeapVars + 8, cpu.pc);
        if (std::getenv("NWIIU_HEAP_TRACE") != nullptr) {
            std::fprintf(stderr, "PREINIT done mem1=%08X fg=%08X mem2=%08X\n",
                         base_overrides_[0], base_overrides_[2],
                         base_overrides_[1]);
        }
        cpu.gpr[3] = 0;
        cpu.gpr[4] = 0;
        cpu.lr = image_.entry_point;
        return HleAction::return_to_lr;
    });

    // --- Fast mutexes and conditions (host-side state only).
    reg("OSFastMutex_Init", [this](CPUContext& cpu, GuestMemory& memory) {
        memory.write32(cpu.gpr[3], kFastMutexTag, cpu.pc);
        memory.write32(cpu.gpr[3] + 4, cpu.gpr[4], cpu.pc);
        fast_mutexes_.insert_or_assign(cpu.gpr[3], MutexState{});
        return HleAction::return_to_lr;
    });
    reg("OSFastMutex_Lock", [this](CPUContext& cpu, GuestMemory&) {
        return acquire_lock(fast_mutexes_[cpu.gpr[3]], cpu.gpr[3], cpu);
    });
    reg("OSFastMutex_TryLock", [this](CPUContext& cpu, GuestMemory&) {
        auto& state = fast_mutexes_[cpu.gpr[3]];
        const uint32_t current = machine_->current_thread_address();
        if (state.owner != 0 && state.owner != current) {
            cpu.gpr[3] = 0;
            return HleAction::return_to_lr;
        }
        if (state.owner == 0) {
            state.owner = current;
            state.recursion = 1;
            state.granted = 0;
        } else {
            ++state.recursion;
        }
        cpu.gpr[3] = 1;
        return HleAction::return_to_lr;
    });
    reg("OSFastMutex_Unlock", [this](CPUContext& cpu, GuestMemory&) {
        auto& state = fast_mutexes_[cpu.gpr[3]];
        if (state.owner != machine_->current_thread_address()) {
            fault("fast mutex unlock by non-owner", cpu.gpr[3], 4, cpu.pc);
        }
        if (state.recursion > 1) {
            --state.recursion;
            return HleAction::return_to_lr;
        }
        release_lock(state);
        return HleAction::return_to_lr;
    });
    reg("OSFastCond_Init", [](CPUContext& cpu, GuestMemory& memory) {
        memory.write32(cpu.gpr[3], kFastCondTag, cpu.pc);
        memory.write32(cpu.gpr[3] + 4, cpu.gpr[4], cpu.pc);
        return HleAction::return_to_lr;
    });
    reg("OSFastCond_Wait", [this](CPUContext& cpu, GuestMemory&) {
        return cond_wait(cpu, true);
    });
    reg("OSFastCond_Signal", [this](CPUContext& cpu, GuestMemory&) {
        cond_signal(cpu.gpr[3]);
        return HleAction::return_to_lr;
    });
    reg("OSInitCond", [](CPUContext& cpu, GuestMemory& memory) {
        memory.write32(cpu.gpr[3], kCondTag, cpu.pc);
        return HleAction::return_to_lr;
    });
    reg("OSWaitCond", [this](CPUContext& cpu, GuestMemory&) {
        return cond_wait(cpu, false);
    });
    reg("OSSignalCond", [this](CPUContext& cpu, GuestMemory&) {
        cond_signal(cpu.gpr[3]);
        return HleAction::return_to_lr;
    });

    // --- Semaphores: count lives in the guest struct; waiters retry.
    reg("OSInitSemaphore", [](CPUContext& cpu, GuestMemory& memory) {
        memory.write32(cpu.gpr[3], kSemaphoreTag, cpu.pc);
        memory.write32(cpu.gpr[3] + 4, 0, cpu.pc);
        memory.write32(cpu.gpr[3] + 8, 0, cpu.pc);
        memory.write32(cpu.gpr[3] + kSemaphoreCountOffset, cpu.gpr[4],
                       cpu.pc);
        return HleAction::return_to_lr;
    });
    reg("OSWaitSemaphore", [this](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t sem = cpu.gpr[3];
        const int32_t count = static_cast<int32_t>(
            memory.read32(sem + kSemaphoreCountOffset, cpu.pc));
        if (count <= 0) {
            machine_->block_current(sem);
            return HleAction::reschedule;
        }
        memory.write32(sem + kSemaphoreCountOffset,
                       static_cast<uint32_t>(count - 1), cpu.pc);
        cpu.gpr[3] = static_cast<uint32_t>(count);
        return HleAction::return_to_lr;
    });
    reg("OSTryWaitSemaphore", [](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t sem = cpu.gpr[3];
        const int32_t count = static_cast<int32_t>(
            memory.read32(sem + kSemaphoreCountOffset, cpu.pc));
        if (count > 0) {
            memory.write32(sem + kSemaphoreCountOffset,
                           static_cast<uint32_t>(count - 1), cpu.pc);
        }
        cpu.gpr[3] = static_cast<uint32_t>(count);
        return HleAction::return_to_lr;
    });
    reg("OSSignalSemaphore", [this](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t sem = cpu.gpr[3];
        const uint32_t count =
            memory.read32(sem + kSemaphoreCountOffset, cpu.pc);
        memory.write32(sem + kSemaphoreCountOffset, count + 1, cpu.pc);
        machine_->wake_waiters(sem);
        cpu.gpr[3] = count;
        return HleAction::return_to_lr;
    });

    // --- Events with timeout: poll in short sleeps until signaled/expired.
    reg("OSWaitEventWithTimeout", [this](CPUContext& cpu, GuestMemory&) {
        const uint32_t address = cpu.gpr[3];
        auto& state = event(address, cpu.pc);
        const uint32_t current = machine_->current_thread_address();
        const auto granted =
            std::find(state.granted.begin(), state.granted.end(), current);
        if (granted != state.granted.end()) {
            state.granted.erase(granted);
            timed_waits_.erase(current);
            cpu.gpr[3] = 1;
            return HleAction::return_to_lr;
        }
        if (abi::read_u32(memory_, address, abi::kEventValueOffset, cpu.pc) !=
            0) {
            if (abi::read_u32(memory_, address, abi::kEventModeOffset,
                              cpu.pc) == 1) {
                abi::write_u32(memory_, address, abi::kEventValueOffset, 0,
                               cpu.pc);
            }
            timed_waits_.erase(current);
            cpu.gpr[3] = 1;
            return HleAction::return_to_lr;
        }
        const uint64_t now = machine_->current_time_ticks();
        auto deadline = timed_waits_.find(current);
        if (deadline == timed_waits_.end()) {
            const uint64_t ns = pair64(cpu.gpr[5], cpu.gpr[6]);
            const uint64_t ticks =
                ns / 1000 * kOsTimerHz / 1000000 + 1;
            deadline = timed_waits_.emplace(current, now + ticks).first;
        }
        if (now >= deadline->second) {
            timed_waits_.erase(deadline);
            cpu.gpr[3] = 0;
            return HleAction::return_to_lr;
        }
        machine_->sleep_current(
            std::min<uint64_t>(deadline->second - now, kOsTimerHz / 2000));
        return HleAction::reschedule;
    });

    // --- Threads.
    reg("OSJoinThread", [this](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t thread = cpu.gpr[3];
        if (!machine_->has_thread(thread)) {
            cpu.gpr[3] = 0;
            return HleAction::return_to_lr;
        }
        if (!machine_->thread_exited(thread)) {
            machine_->block_current(thread);
            return HleAction::reschedule;
        }
        if (cpu.gpr[4] != 0) {
            memory.write32(cpu.gpr[4],
                           abi::read_u32(memory_, thread,
                                         abi::kOsThreadExitValueOffset,
                                         cpu.pc),
                           cpu.pc);
        }
        cpu.gpr[3] = 1;
        return HleAction::return_to_lr;
    });
    noop("OSDetachThread");
    reg("OSSetThreadPriority", [this](CPUContext& cpu, GuestMemory&) {
        const uint32_t thread = cpu.gpr[3];
        const int32_t priority = static_cast<int32_t>(cpu.gpr[4]);
        if (priority < 0 || priority > 31) {
            cpu.gpr[3] = 0;
            return HleAction::return_to_lr;
        }
        abi::write_u32(memory_, thread, abi::kOsThreadPriorityOffset,
                       static_cast<uint32_t>(priority + 64), cpu.pc);
        abi::write_u32(memory_, thread, abi::kOsThreadBasePriorityOffset,
                       static_cast<uint32_t>(priority + 64), cpu.pc);
        machine_->set_thread_priority(thread, priority);
        cpu.gpr[3] = 1;
        cpu.pc = cpu.lr;
        return HleAction::reschedule;
    });
    reg("OSSuspendThread", [this](CPUContext& cpu, GuestMemory&) {
        const uint32_t thread = cpu.gpr[3];
        const uint32_t old =
            abi::read_u32(memory_, thread, abi::kOsThreadSuspendOffset, cpu.pc);
        abi::write_u32(memory_, thread, abi::kOsThreadSuspendOffset, old + 1,
                       cpu.pc);
        cpu.gpr[3] = old;
        if (old == 0 && machine_->suspend_thread(thread)) {
            cpu.pc = cpu.lr;
            return HleAction::reschedule;
        }
        return HleAction::return_to_lr;
    });

    // --- Alarms: the scheduler never fires them yet, so cancel is a flag.
    reg("OSCancelAlarm", [this](CPUContext& cpu, GuestMemory&) {
        abi::write_u32(memory_, cpu.gpr[3], abi::kAlarmStateOffset, 0, cpu.pc);
        cpu.gpr[3] = 1;
        return HleAction::return_to_lr;
    });
    noop("OSCancelAlarms");

    // --- Atomics and time.
    reg("OSAddAtomic64", [](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t pointer = cpu.gpr[3];
        const uint64_t old = pair64(memory.read32(pointer, cpu.pc),
                                    memory.read32(pointer + 4, cpu.pc));
        const uint64_t updated = old + pair64(cpu.gpr[5], cpu.gpr[6]);
        memory.write32(pointer, static_cast<uint32_t>(updated >> 32), cpu.pc);
        memory.write32(pointer + 4, static_cast<uint32_t>(updated), cpu.pc);
        set_pair64(cpu, old);
        return HleAction::return_to_lr;
    });
    reg("OSTicksToCalendarTime", [](CPUContext& cpu, GuestMemory& memory) {
        const int64_t ticks =
            static_cast<int64_t>(pair64(cpu.gpr[3], cpu.gpr[4]));
        const uint32_t out = cpu.gpr[5];
        int64_t seconds = ticks / kOsTimerHz;
        int64_t remainder = ticks % kOsTimerHz;
        if (remainder < 0) {
            remainder += kOsTimerHz;
            --seconds;
        }
        const int64_t usec_total = remainder * 1000000 / kOsTimerHz;
        int64_t days = seconds / 86400;
        int64_t day_seconds = seconds % 86400;
        if (day_seconds < 0) {
            day_seconds += 86400;
            --days;
        }
        int64_t year;
        unsigned month;
        unsigned day;
        const int64_t absolute = days + kEpoch2000Days;
        civil_from_days(absolute, year, month, day);
        const int64_t yday = absolute - days_from_civil(year, 1, 1);
        const int64_t wday = ((absolute % 7) + 7 + 4) % 7; // 1970-01-01 Thu
        const int32_t fields[10] = {
            static_cast<int32_t>(day_seconds % 60),
            static_cast<int32_t>(day_seconds / 60 % 60),
            static_cast<int32_t>(day_seconds / 3600),
            static_cast<int32_t>(day),
            static_cast<int32_t>(month - 1),
            static_cast<int32_t>(year),
            static_cast<int32_t>(wday),
            static_cast<int32_t>(yday),
            static_cast<int32_t>(usec_total / 1000),
            static_cast<int32_t>(usec_total % 1000)};
        for (uint32_t index = 0; index < 10; ++index) {
            memory.write32(out + index * 4,
                           static_cast<uint32_t>(fields[index]), cpu.pc);
        }
        return HleAction::return_to_lr;
    });
    reg("OSCalendarTimeToTicks", [](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t in = cpu.gpr[3];
        const auto field = [&](uint32_t index) {
            return static_cast<int32_t>(memory.read32(in + index * 4, cpu.pc));
        };
        const int64_t days =
            days_from_civil(field(5), static_cast<unsigned>(field(4) + 1),
                            static_cast<unsigned>(field(3))) -
            kEpoch2000Days;
        const int64_t seconds =
            days * 86400 + field(2) * 3600 + field(1) * 60 + field(0);
        const int64_t usec = int64_t{field(8)} * 1000 + field(9);
        set_pair64(cpu, static_cast<uint64_t>(
                            seconds * kOsTimerHz +
                            usec * kOsTimerHz / 1000000));
        return HleAction::return_to_lr;
    });

    // --- Memory.
    reg("OSGetMemBound", [this](CPUContext& cpu, GuestMemory& memory) {
        uint32_t start = 0;
        uint32_t size = 0;
        if (cpu.gpr[3] == 1) {
            start = abi::kMem1Heap;
            size = abi::kMem1End - abi::kMem1Heap;
        } else if (cpu.gpr[3] == 2) {
            start = mem2_heap_;
            size = abi::kMem2End - 0x02000000 - mem2_heap_;
        } else {
            cpu.gpr[3] = static_cast<uint32_t>(-1);
            return HleAction::return_to_lr;
        }
        if (cpu.gpr[4] != 0) {
            memory.write32(cpu.gpr[4], start, cpu.pc);
        }
        if (cpu.gpr[5] != 0) {
            memory.write32(cpu.gpr[5], size, cpu.pc);
        }
        cpu.gpr[3] = 0;
        return HleAction::return_to_lr;
    });
    reg("MEMFreeToFrmHeap", [this](CPUContext& cpu, GuestMemory&) {
        auto& heap = frame_heap(cpu.gpr[3], cpu.pc);
        if ((cpu.gpr[4] & 1) != 0) {
            heap.head = heap.start;
        }
        if ((cpu.gpr[4] & 2) != 0) {
            heap.tail = heap.end;
        }
        mirror_frame(heap);
        return HleAction::return_to_lr;
    });
    // Shrinks the heap to its highest live allocation and returns the new
    // size measured from the handle. UE3's __preinit_user then claims
    // [bound start + that size, bound end) of MEM2 for its own block heap.
    reg("MEMAdjustExpHeap", [this](CPUContext& cpu, GuestMemory&) {
        auto& heap = expanded_heap(cpu.gpr[3], cpu.pc);
        uint32_t used_end = heap.start;
        for (const auto& [address, allocation] : heap.allocated) {
            used_end = std::max(used_end, allocation.raw_address +
                                              allocation.raw_size);
        }
        used_end = (used_end + 3) & ~uint32_t{3};
        std::vector<FreeBlock> kept;
        for (const auto& block : heap.free) {
            if (block.address >= used_end) {
                continue;
            }
            kept.push_back({block.address,
                            std::min(block.size, used_end - block.address)});
        }
        heap.free = std::move(kept);
        heap.end = used_end;
        abi::write_u32(memory_, heap.handle, abi::kHeapDataEndOffset,
                       heap.end, cpu.pc);
        mirror_expanded(heap, cpu.pc);
        cpu.gpr[3] = heap.end - heap.handle;
        return HleAction::return_to_lr;
    });
    reg("MEMGetSizeForMBlockExpHeap", [this](CPUContext& cpu, GuestMemory&) {
        for (const auto& [handle, heap] : expanded_heaps_) {
            const auto found = heap.allocated.find(cpu.gpr[3]);
            if (found != heap.allocated.end()) {
                cpu.gpr[3] = found->second.size;
                return HleAction::return_to_lr;
            }
        }
        cpu.gpr[3] = 0;
        return HleAction::return_to_lr;
    });
    noop("MEMDumpHeap");
    reg("MEMInitBlockHeap", [this](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t tracking = cpu.gpr[3];
        if (std::getenv("NWIIU_HEAP_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "BLOCKHEAP init heap=%08X start=%08X end=%08X "
                         "track=%08X tracksize=%08X flags=%X lr=%08X\n",
                         tracking, cpu.gpr[4], cpu.gpr[5], cpu.gpr[6],
                         cpu.gpr[7], cpu.gpr[8], cpu.lr);
        }
        memory.write32(tracking, 0x424C4B48, cpu.pc); // 'BLKH'
        block_heaps_.insert_or_assign(tracking,
                                      BlockHeap{cpu.gpr[4], cpu.gpr[5], {}});
        cpu.gpr[3] = tracking;
        return HleAction::return_to_lr;
    });
    reg("MEMAllocFromBlockHeapEx", [this](CPUContext& cpu, GuestMemory&) {
        auto found = block_heaps_.find(cpu.gpr[3]);
        const uint32_t size = std::max<uint32_t>(cpu.gpr[4], 4);
        const int32_t align_raw = static_cast<int32_t>(cpu.gpr[5]);
        const uint32_t align = std::max<uint32_t>(
            4, static_cast<uint32_t>(align_raw < 0 ? -align_raw : align_raw));
        if (found == block_heaps_.end()) {
            if (std::getenv("NWIIU_HEAP_TRACE") != nullptr) {
                std::fprintf(stderr,
                             "BLOCKHEAP alloc unknown heap=%08X size=%08X "
                             "lr=%08X\n",
                             cpu.gpr[3], size, cpu.lr);
            }
            cpu.gpr[3] = 0;
            return HleAction::return_to_lr;
        }
        cpu.gpr[3] = 0;
        auto& heap = found->second;
        uint64_t cursor = heap.start;
        const auto try_at = [&](uint64_t candidate, uint64_t limit) {
            const uint64_t aligned = (candidate + align - 1) / align * align;
            return aligned + size <= limit ? aligned : 0;
        };
        for (auto it = heap.used.begin();; ++it) {
            const uint64_t limit = it == heap.used.end() ? heap.end : it->first;
            if (const uint64_t at = try_at(cursor, limit); at != 0) {
                heap.used.emplace(static_cast<uint32_t>(at), size);
                cpu.gpr[3] = static_cast<uint32_t>(at);
                break;
            }
            if (it == heap.used.end()) {
                break;
            }
            cursor = uint64_t{it->first} + it->second;
        }
        if (std::getenv("NWIIU_HEAP_TRACE") != nullptr) {
            std::fprintf(stderr,
                         "BLOCKHEAP alloc heap=%08X size=%08X align=%d -> "
                         "%08X lr=%08X\n",
                         found->first, size, align_raw, cpu.gpr[3], cpu.lr);
        }
        return HleAction::return_to_lr;
    });
    reg("MEMFreeToBlockHeap", [this](CPUContext& cpu, GuestMemory&) {
        auto found = block_heaps_.find(cpu.gpr[3]);
        if (found != block_heaps_.end()) {
            found->second.used.erase(cpu.gpr[4]);
        }
        return HleAction::return_to_lr;
    });
    reg("MEMGetTotalFreeSizeForBlockHeap",
        [this](CPUContext& cpu, GuestMemory&) {
            auto found = block_heaps_.find(cpu.gpr[3]);
            uint32_t free = 0;
            if (found != block_heaps_.end()) {
                free = found->second.end - found->second.start;
                for (const auto& [address, size] : found->second.used) {
                    free -= size;
                }
            }
            cpu.gpr[3] = free;
            return HleAction::return_to_lr;
        });

    // --- Locked cache: backed by ordinary MEM2; "DMA" is a synchronous copy.
    reg("LCAlloc", [this](CPUContext& cpu, GuestMemory&) {
        const uint32_t size = std::min(cpu.gpr[3], kLockedCacheSize);
        cpu.gpr[3] = allocate_expanded(private_heap_, size, 512, cpu.pc);
        return HleAction::return_to_lr;
    });
    reg("LCDealloc", [this](CPUContext& cpu, GuestMemory&) {
        if (cpu.gpr[3] != 0) {
            free_expanded(private_heap_, cpu.gpr[3], cpu.pc);
        }
        return HleAction::return_to_lr;
    });
    ret("LCEnableDMA", 1);
    noop("LCDisableDMA");
    noop("LCWaitDMAQueue");
    const auto lc_copy = [](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t blocks = cpu.gpr[5] == 0 ? 128 : cpu.gpr[5];
        const uint32_t bytes = blocks * 32;
        std::vector<uint8_t> buffer(bytes);
        memory.read_bytes(cpu.gpr[4], buffer, cpu.pc);
        memory.write_bytes(cpu.gpr[3], buffer, cpu.pc);
        return HleAction::return_to_lr;
    };
    reg("LCLoadDMABlocks", lc_copy);
    reg("LCStoreDMABlocks", lc_copy);

    // --- System queries.
    reg("OSGetSystemMessageQueue", [this](CPUContext& cpu, GuestMemory&) {
        if (system_message_queue_ == 0) {
            CPUContext scratch = cpu;
            scratch.gpr[3] = kSystemQueue;
            scratch.gpr[4] = kSystemQueueMessages;
            scratch.gpr[5] = kSystemQueueDepth;
            validate_range(kSystemQueue, abi::kOsMessageQueueSize, cpu.pc,
                           MemoryAccess::write);
            abi::write_u32(memory_, kSystemQueue, abi::kMessageQueueTagOffset,
                           abi::kOsMessageQueueTag, cpu.pc);
            for (const uint32_t offset : {abi::kMessageQueueSendQueueOffset,
                                          abi::kMessageQueueRecvQueueOffset}) {
                abi::write_u32(memory_, kSystemQueue,
                               offset + abi::kThreadQueueParentOffset,
                               kSystemQueue, cpu.pc);
            }
            abi::write_u32(memory_, kSystemQueue,
                           abi::kMessageQueueMessagesOffset,
                           kSystemQueueMessages, cpu.pc);
            abi::write_u32(memory_, kSystemQueue, abi::kMessageQueueSizeOffset,
                           kSystemQueueDepth, cpu.pc);
            message_queues_.insert_or_assign(kSystemQueue,
                                             MessageQueueState{});
            system_message_queue_ = kSystemQueue;
        }
        cpu.gpr[3] = system_message_queue_;
        return HleAction::return_to_lr;
    });
    reg("OSIsHomeButtonMenuEnabled", [this](CPUContext& cpu, GuestMemory&) {
        cpu.gpr[3] = home_button_menu_enabled_;
        return HleAction::return_to_lr;
    });
    reg("IMGetTimeBeforeAPD", [](CPUContext& cpu, GuestMemory& memory) {
        if (cpu.gpr[3] != 0) {
            memory.write32(cpu.gpr[3], 3600, cpu.pc);
        }
        cpu.gpr[3] = 0;
        return HleAction::return_to_lr;
    });
    ret("MCP_Open", 1);
    ret("MCP_Close", 0);
    reg("MCP_GetSysProdSettings", [](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t out = cpu.gpr[4];
        for (uint32_t offset = 0; offset < 0x46; ++offset) {
            memory.write8(out + offset, 0, cpu.pc);
        }
        memory.write8(out + 3, 2, cpu.pc); // platform region: USA
        memory.write8(out + 8, 2, cpu.pc); // game region: USA
        cpu.gpr[3] = 0;
        return HleAction::return_to_lr;
    });
    reg("bspGetHardwareVersion", [](CPUContext& cpu, GuestMemory& memory) {
        if (cpu.gpr[3] != 0) {
            memory.write32(cpu.gpr[3], 0x26100010, cpu.pc);
        }
        cpu.gpr[3] = 0;
        return HleAction::return_to_lr;
    });
    ret("OSDriver_Register", 0);
    ret("OSReadRegister16", 0);
    ret("__OSReadRegister32Ex", 0);
    noop("OSWriteRegister16");
    noop("__OSWriteRegister32Ex");
    noop("OSEnforceInorderIO");
    noop("OSBlockThreadsOnExit");
    noop("OSEnableForegroundExit");
    noop("OSSavesDone_ReadyToRelease");
    noop("OSDynLoad_Release");
    reg("__gh_errno_ptr", [](CPUContext& cpu, GuestMemory&) {
        cpu.gpr[3] = abi::kGhsErrno;
        return HleAction::return_to_lr;
    });
    reg("__ghs_flock_create", [](CPUContext& cpu, GuestMemory& memory) {
        if (cpu.gpr[3] != 0) {
            memory.write32(cpu.gpr[3], 0, cpu.pc);
        }
        return HleAction::return_to_lr;
    });

    // --- Diagnostics and termination.
    reg("OSConsoleWrite", [](CPUContext& cpu, GuestMemory& memory) {
        std::string text;
        for (uint32_t index = 0; index < cpu.gpr[4] && index < 4096; ++index) {
            text.push_back(static_cast<char>(memory.read8(cpu.gpr[3] + index,
                                                          cpu.pc)));
        }
        std::fputs(text.c_str(), stderr);
        return HleAction::return_to_lr;
    });
    reg("OSFatal", [this](CPUContext& cpu, GuestMemory& memory) -> HleAction {
        std::fprintf(stderr, "OSFatal: %s\n",
                     read_string(memory, cpu.gpr[3], cpu.pc).c_str());
        fault("OSFatal", cpu.gpr[3], 4, cpu.pc);
    });
    reg("OSPanic", [this](CPUContext& cpu, GuestMemory& memory) -> HleAction {
        std::fprintf(stderr, "OSPanic: %s:%u: %s\n",
                     read_string(memory, cpu.gpr[3], cpu.pc).c_str(),
                     cpu.gpr[4],
                     read_string(memory, cpu.gpr[5], cpu.pc).c_str());
        fault("OSPanic", cpu.gpr[3], 4, cpu.pc);
    });
    reg("_Exit", [this](CPUContext& cpu, GuestMemory&) {
        // A guest exit is usually UE3 bailing on a failed subsystem; print the
        // back chain so the caller can be located without a debugger.
        std::fprintf(stderr, "_Exit(%d) lr=%08X backtrace:", int(cpu.gpr[3]),
                     cpu.lr);
        uint32_t sp = cpu.gpr[1];
        for (int i = 0; i < 24 && sp != 0 && sp % 4 == 0; ++i) {
            try {
                const uint32_t next = memory_.read32(sp, cpu.pc);
                if (next == 0 || next <= sp) break;
                std::fprintf(stderr, " %08X", memory_.read32(next + 4, cpu.pc));
                sp = next;
            } catch (...) {
                break;
            }
        }
        std::fprintf(stderr, "\n");
        return HleAction::exit;
    });
}
} // namespace nwii::runtime
