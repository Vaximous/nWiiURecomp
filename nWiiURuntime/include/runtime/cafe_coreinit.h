#pragma once

#include "runtime/cafe_abi.h"
#include "runtime/execution_image.h"
#include "runtime/executor.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <vector>

namespace nwii::runtime {
class CafeRuntime;
class Machine;

class CafeCoreinit {
public:
    explicit CafeCoreinit(
        ExecutionImage& image,
        std::optional<std::filesystem::path> shared_font = std::nullopt);
    CafeCoreinit(const CafeCoreinit&) = delete;
    CafeCoreinit& operator=(const CafeCoreinit&) = delete;
    CafeCoreinit(CafeCoreinit&&) = delete;
    CafeCoreinit& operator=(CafeCoreinit&&) = delete;

    void attach_machine(Machine& machine);
    void register_handlers(CafeRuntime& runtime);
    // Imports beyond WWHD's set (cafe_coreinit_extra.cpp).
    void register_extra_handlers(CafeRuntime& runtime);
    // Non-blocking OSSendMessage of the 16-byte OSMessage at `message`;
    // returns false when the queue is full.
    bool post_message(uint32_t queue, uint32_t message, CPUContext& cpu);
    uint32_t mem2_heap() const { return mem2_heap_; }
    // Points the main thread at the image's __preinit_user export (if any),
    // returning into the entry point afterwards, as the Cafe loader does.
    void begin_preinit(CPUContext& main);

    uint32_t base_heap(uint32_t type, uint32_t pc = 0) const;
    uint32_t create_expanded_heap(uint32_t base, uint32_t size,
                                  uint32_t flags, uint32_t pc);
    uint32_t destroy_expanded_heap(uint32_t handle, uint32_t pc);
    uint32_t create_frame_heap(uint32_t base, uint32_t size, uint32_t flags,
                               uint32_t pc);
    uint32_t destroy_frame_heap(uint32_t handle, uint32_t pc);
    uint32_t allocate_frame(uint32_t handle, uint32_t size,
                            int32_t alignment, uint32_t pc);
    uint32_t frame_allocatable(uint32_t handle, int32_t alignment,
                               uint32_t pc) const;
    uint32_t allocate_expanded(uint32_t handle, uint32_t size,
                               int32_t alignment, uint32_t pc);
    void free_expanded(uint32_t handle, uint32_t address, uint32_t pc);
    uint32_t expanded_allocatable(uint32_t handle, int32_t alignment,
                                  uint32_t pc) const;
    uint32_t expanded_total_free(uint32_t handle, uint32_t pc) const;

    void init_mutex(uint32_t address, uint32_t pc);
    void initialize_thread(uint32_t address, uint16_t id, int32_t priority,
                           uint32_t stack_start, uint32_t stack_end);
    void set_thread_state(uint32_t address, uint8_t state);

    uint32_t main_thread_address() const { return abi::kMainThread; }
    uint64_t ticks() const { return ticks_; }
    void advance_ticks(uint64_t delta) { ticks_ += delta; }
    bool home_button_menu_enabled() const {
        return home_button_menu_enabled_;
    }

    friend struct CafeCoreinitTestAccess;

private:
    struct FrameHeap {
        uint32_t handle;
        uint32_t start;
        uint32_t end;
        uint32_t head;
        uint32_t tail;
        uint32_t flags;
    };

    struct FreeBlock {
        uint32_t address;
        uint32_t size;
    };

    struct ExpandedAllocation {
        uint32_t raw_address;
        uint32_t raw_size;
        uint32_t size;
        bool from_end;
    };

    struct ExpandedHeap {
        uint32_t handle;
        uint32_t start;
        uint32_t end;
        std::vector<FreeBlock> free;
        std::map<uint32_t, ExpandedAllocation> allocated;
        uint32_t flags;
        bool mirror_blocks;
    };

    struct MutexWaiter {
        uint32_t thread;
        int32_t priority;
        uint64_t sequence;
    };

    struct MutexState {
        uint32_t owner{};
        uint32_t recursion{};
        uint32_t granted{};
        std::vector<MutexWaiter> waiters;
    };

    struct MessageQueueState {
        // ponytail: ring first/used/size live only in the guest-mirrored
        // OSMessageQueue fields; host state is the waiter bookkeeping.
        std::vector<MutexWaiter> send_waiters;
        std::vector<MutexWaiter> recv_waiters;
    };

    struct EventState {
        std::vector<MutexWaiter> waiters;
        std::vector<uint32_t> granted;
    };

    struct BlockHeap {
        uint32_t start;
        uint32_t end;
        std::map<uint32_t, uint32_t> used; // address -> size
    };

    struct CondWait {
        uint32_t cond;
        uint32_t mutex;
        uint32_t recursion;
        bool fast;
        bool signaled;
    };

    HleAction acquire_lock(MutexState& state, uint32_t address,
                           CPUContext& cpu);
    uint32_t release_lock(MutexState& state);
    HleAction cond_wait(CPUContext& cpu, bool fast);
    void cond_signal(uint32_t cond);

    [[noreturn]] void fault(const char* reason, uint32_t address,
                            uint32_t width, uint32_t pc,
                            MemoryAccess access = MemoryAccess::read) const;
    static uint32_t alignment_magnitude(int32_t alignment, uint32_t pc,
                                        uint32_t address);
    static uint32_t align_up(uint32_t value, uint32_t alignment,
                             uint32_t pc, uint32_t address);
    static uint32_t align_down(uint32_t value, uint32_t alignment);
    void validate_range(uint32_t address, uint32_t size, uint32_t pc,
                        MemoryAccess access) const;
    bool& interrupt_state(uint32_t pc);
    FrameHeap& frame_heap(uint32_t handle, uint32_t pc);
    const FrameHeap& frame_heap(uint32_t handle, uint32_t pc) const;
    ExpandedHeap& expanded_heap(uint32_t handle, uint32_t pc);
    const ExpandedHeap& expanded_heap(uint32_t handle, uint32_t pc) const;
    MutexState& mutex(uint32_t address, uint32_t pc);
    void mirror_frame(const FrameHeap& heap);
    void mirror_expanded(const ExpandedHeap& heap, uint32_t pc);
    void mirror_mutex(uint32_t address, const MutexState& state, uint32_t pc);
    HleAction lock_mutex(uint32_t address, CPUContext& cpu);
    HleAction unlock_mutex(uint32_t address, CPUContext& cpu);
    HleAction try_lock_mutex(uint32_t address, CPUContext& cpu);
    uint32_t pointer_to_mutex(CPUContext& cpu, GuestMemory& memory) const;
    MessageQueueState& message_queue(uint32_t address, uint32_t pc);
    void mirror_thread_queue(uint32_t base, uint32_t head_offset,
                             const std::vector<MutexWaiter>& waiters,
                             uint32_t pc);
    HleAction send_message(CPUContext& cpu);
    HleAction receive_message(CPUContext& cpu);
    EventState& event(uint32_t address, uint32_t pc);
    HleAction wait_event(CPUContext& cpu);
    HleAction signal_event(CPUContext& cpu, bool all);
    static bool waiter_before(const MutexWaiter& left,
                              const MutexWaiter& right);

    ExecutionImage& image_;
    GuestMemory& memory_;
    // MEM2 default heap handle: abi::kMem2Heap for images that fit below it
    // (WWHD), otherwise the first 64 KiB boundary after the image's data.
    uint32_t mem2_heap_;
    Machine* machine_{};
    std::map<uint32_t, FrameHeap> frame_heaps_;
    std::map<uint32_t, ExpandedHeap> expanded_heaps_;
    std::map<uint32_t, MutexState> mutexes_;
    std::map<uint32_t, MessageQueueState> message_queues_;
    std::vector<uint8_t> shared_font_;
    uint32_t dynload_alloc_{};
    uint32_t dynload_free_{};
    uint32_t shared_font_address_{};
    std::array<uint32_t, 15> exception_callbacks_{};
    uint16_t next_thread_id_{2};
    uint64_t ticks_{};
    std::map<uint32_t, EventState> events_;
    std::map<uint32_t, MutexState> fast_mutexes_;
    std::map<uint32_t, std::vector<uint32_t>> cond_waiters_;
    std::map<uint32_t, CondWait> cond_waits_; // by thread
    std::map<uint32_t, uint64_t> timed_waits_; // thread -> deadline tick
    std::map<uint32_t, BlockHeap> block_heaps_;
    uint32_t system_message_queue_{};
    uint32_t private_heap_{};
    // MEM1, MEM2, FG handles a title installed via __preinit_user.
    std::array<uint32_t, 3> base_overrides_{};
    uint64_t wait_sequence_{};
    bool home_button_menu_enabled_{true};
    bool dim_enabled_{true};
    // ponytail: models the mask only; dispatch pending IRQs when devices emit
    // guest interrupts.
    std::array<bool, 3> interrupts_enabled_{true, true, true};
};
} // namespace nwii::runtime
