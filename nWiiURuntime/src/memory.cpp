#include "runtime/memory.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace nwii::runtime {
namespace {
// Cached: getenv takes a lock and scans the environment on every call,
// and these sit on per-access / per-dispatch paths.
inline bool env_nwiiu_matrix_trace() {
    static const bool enabled = std::getenv("NWIIU_MATRIX_TRACE") != nullptr;
    return enabled;
}
inline bool env_nwiiu_pool_trace() {
    static const bool enabled = std::getenv("NWIIU_POOL_TRACE") != nullptr;
    return enabled;
}
} // namespace
namespace {
// 4 GiB of guest space plus slack so an access at 0xFFFFFFFC of up to
// eight bytes (or a 32-byte cache-line op) never leaves the reservation.
constexpr uint64_t kHostFlatSize = (uint64_t{1} << 32) + 0x10000;

uint8_t* host_reserve() {
#if defined(_WIN32)
    return static_cast<uint8_t*>(
        VirtualAlloc(nullptr, kHostFlatSize, MEM_RESERVE, PAGE_NOACCESS));
#else
    void* base = mmap(nullptr, kHostFlatSize, PROT_NONE,
                      MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    return base == MAP_FAILED ? nullptr : static_cast<uint8_t*>(base);
#endif
}

void host_release(uint8_t* base) {
#if defined(_WIN32)
    VirtualFree(base, 0, MEM_RELEASE);
#else
    munmap(base, kHostFlatSize);
#endif
}

bool host_commit(uint8_t* base, uint32_t address, uint64_t size) {
    const uint64_t start = address & ~uint64_t{0xFFFF};
    const uint64_t end = (uint64_t{address} + size + 0xFFFF) & ~uint64_t{0xFFFF};
#if defined(_WIN32)
    return VirtualAlloc(base + start, end - start, MEM_COMMIT,
                        PAGE_READWRITE) != nullptr;
#else
    return mprotect(base + start, end - start, PROT_READ | PROT_WRITE) == 0;
#endif
}

bool permits(MemoryPermissions permissions, MemoryAccess access) {
    switch (access) {
    case MemoryAccess::read:
        return permissions.readable;
    case MemoryAccess::write:
        return permissions.writable;
    case MemoryAccess::execute:
        return permissions.executable;
    }
    return false;
}
bool overlaps_matrix_trace(uint32_t address, uint64_t size) {
    const uint64_t end = static_cast<uint64_t>(address) + size;
    return (address < 0x3B77A5C0u && end > 0x3B77A590u) ||
           (address < 0x17F80434u && end > 0x17F80404u) ||
           (address < 0x17F80708u && end > 0x17F806D8u);
}
}

GuestFault::GuestFault(std::string reason, uint32_t address, uint32_t width,
                       uint32_t pc, MemoryAccess access)
    : std::runtime_error(std::move(reason)), address(address), width(width),
      pc(pc), access(access) {}

GuestMemory::GuestMemory(GuestMemoryCallbacks callbacks)
    : callbacks_(callbacks) {
    if (callbacks.read8 == nullptr || callbacks.read16 == nullptr ||
        callbacks.read32 == nullptr || callbacks.read64 == nullptr ||
        callbacks.write8 == nullptr || callbacks.write16 == nullptr ||
        callbacks.write32 == nullptr || callbacks.write64 == nullptr ||
        callbacks.read_bytes == nullptr || callbacks.write_bytes == nullptr) {
        throw std::invalid_argument("incomplete guest memory callbacks");
    }
    // Only adopt a mapping that spans the whole 32-bit guest space; a partial
    // one would need a per-access bounds check and defeat the fast path.
    if (callbacks.flat_base != nullptr &&
        callbacks.flat_size >= (uint64_t{1} << 32)) {
        flat_base_ = callbacks.flat_base;
    }
}

void GuestMemory::map(uint32_t address, uint32_t size,
                      MemoryPermissions permissions,
                      std::span<const uint8_t> initial) {
    if (external()) {
        throw std::logic_error("cannot map callback-backed guest memory");
    }
    if (size == 0) {
        throw std::runtime_error("zero-sized memory map");
    }
    const uint64_t end = static_cast<uint64_t>(address) + size;
    if (end > uint64_t{1} << std::numeric_limits<uint32_t>::digits) {
        throw std::runtime_error("memory map wraps address space");
    }
    if (initial.size() > size) {
        throw std::runtime_error("initial data exceeds memory map");
    }

    const auto next = std::lower_bound(
        mappings_.begin(), mappings_.end(), address,
        [](const Mapping& mapping, uint32_t value) {
            return mapping.address < value;
        });
    if (next != mappings_.end() && end > next->address) {
        throw std::runtime_error("overlapping memory map");
    }
    if (next != mappings_.begin()) {
        const auto& previous = *(next - 1);
        const uint64_t previous_end =
            static_cast<uint64_t>(previous.address) + previous.size;
        if (previous_end > address) {
            throw std::runtime_error("overlapping memory map");
        }
    }

    if (flat_owner_ != nullptr &&
        !host_commit(flat_owner_.get(), address, size)) {
        throw std::runtime_error("host could not commit guest memory");
    }
    mappings_.insert(next, Mapping{address, size, permissions});
    for (std::size_t copied = 0; copied < initial.size();) {
        const uint32_t current = address + static_cast<uint32_t>(copied);
        const auto count = std::min<std::size_t>(
            initial.size() - copied, kPageSize - (current & (kPageSize - 1)));
        std::copy_n(initial.data() + copied, count, page(current));
        copied += count;
    }
}

uint64_t GuestMemory::mapped_bytes() const {
    uint64_t total{};
    for (const auto& mapping : mappings_) {
        total += mapping.size;
    }
    return total;
}

uint64_t GuestMemory::resident_bytes() const {
    if (flat_owner_ != nullptr) {
        return mapped_bytes();
    }
    return static_cast<uint64_t>(pages_.size()) * kPageSize;
}

bool GuestMemory::enable_host_flat() {
    if (flat_owner_ != nullptr) {
        return true;
    }
    if (external()) {
        return false;
    }
    uint8_t* base = host_reserve();
    if (base == nullptr) {
        return false;
    }
    std::shared_ptr<uint8_t> owner(base, host_release);
    for (const auto& mapping : mappings_) {
        if (!host_commit(base, mapping.address, mapping.size)) {
            return false;
        }
    }
    for (const auto& [number, data] : pages_) {
        std::memcpy(base + (uint64_t{number} << 12), data.data(), kPageSize);
    }
    pages_.clear();
    flat_owner_ = std::move(owner);
    flat_base_ = base;
    return true;
}

void GuestMemory::check(uint32_t address, uint32_t width, uint32_t pc,
                        MemoryAccess access, bool check_permissions) const {
    if (external()) {
        return;
    }
    const uint64_t end = static_cast<uint64_t>(address) + width;
    static thread_local const GuestMemory* cached_memory{};
    static thread_local size_t cached_index{};
    if (cached_memory == this && cached_index < mappings_.size()) {
        const auto& cached = mappings_[cached_index];
        const uint64_t cached_end =
            static_cast<uint64_t>(cached.address) + cached.size;
        if (address >= cached.address && end <= cached_end) {
            if (check_permissions && !permits(cached.permissions, access)) {
                throw GuestFault("guest memory permission fault", address,
                                 width, pc, access);
            }
            return;
        }
    }

    auto next = std::lower_bound(
        mappings_.begin(), mappings_.end(), address,
        [](const Mapping& mapping, uint32_t value) {
            return mapping.address < value;
        });
    if (next == mappings_.end() || next->address != address) {
        if (next == mappings_.begin()) {
            throw GuestFault("unmapped guest memory access", address, width, pc,
                             access);
        }
        --next;
    }

    const uint64_t mapping_end =
        static_cast<uint64_t>(next->address) + next->size;
    if (address < next->address || end > mapping_end) {
        throw GuestFault("out-of-bounds guest memory access", address, width,
                         pc, access);
    }
    cached_memory = this;
    cached_index = static_cast<size_t>(next - mappings_.begin());
    if (check_permissions && !permits(next->permissions, access)) {
        throw GuestFault("guest memory permission fault", address, width, pc,
                         access);
    }
}

void GuestMemory::validate(uint32_t address, uint32_t width, uint32_t pc,
                           MemoryAccess access) const {
    check(address, width, pc, access);
}

void GuestMemory::validate_range(uint32_t address, uint32_t size, uint32_t pc,
                                 MemoryAccess access) const {
    if (external()) {
        return;
    }
    if (size == 0) {
        return;
    }

    const uint64_t end = static_cast<uint64_t>(address) + size;
    if (end > uint64_t{1} << std::numeric_limits<uint32_t>::digits) {
        throw GuestFault("out-of-bounds guest memory access", address, size, pc,
                         access);
    }

    auto mapping = std::upper_bound(
        mappings_.begin(), mappings_.end(), address,
        [](uint32_t value, const Mapping& candidate) {
            return value < candidate.address;
        });
    if (mapping == mappings_.begin()) {
        throw GuestFault("unmapped guest memory access", address, size, pc,
                         access);
    }
    --mapping;

    uint64_t current = address;
    while (current < end) {
        const uint64_t mapping_end =
            static_cast<uint64_t>(mapping->address) + mapping->size;
        if (current < mapping->address || current >= mapping_end) {
            throw GuestFault("out-of-bounds guest memory access", address,
                             size, pc, access);
        }
        if (!permits(mapping->permissions, access)) {
            throw GuestFault("guest memory permission fault", address, size,
                             pc, access);
        }

        current = std::min(end, mapping_end);
        if (current == end) {
            return;
        }
        ++mapping;
        if (mapping == mappings_.end() || mapping->address != current) {
            throw GuestFault("out-of-bounds guest memory access", address,
                             size, pc, access);
        }
    }
}

namespace {
// One-entry lookup cache in front of pages_ (an unordered_map hit per guest
// access dominated standalone runs, which have no flat mapping). Pages are
// never erased and unordered_map element addresses survive rehashing, so a
// cached pointer stays valid; the size tag guards a new GuestMemory reusing
// a destroyed one's address.
struct PageCache {
    const void* owner{};
    size_t generation{};
    uint32_t page{0xFFFFFFFFu};
    uint8_t* data{};
};
thread_local PageCache g_page_cache;
} // namespace

const uint8_t* GuestMemory::page(uint32_t address) const {
    if (flat_owner_ != nullptr) {
        return flat_base_ + address;
    }
    auto& cache = g_page_cache;
    const uint32_t number = address >> 12;
    if (cache.owner == this && cache.page == number &&
        cache.generation == pages_.size()) {
        return cache.data + (address & (kPageSize - 1));
    }
    const auto found = pages_.find(number);
    if (found == pages_.end()) {
        return nullptr;
    }
    cache = {this, pages_.size(), number,
             const_cast<uint8_t*>(found->second.data())};
    return found->second.data() + (address & (kPageSize - 1));
}

uint8_t* GuestMemory::page(uint32_t address) {
    if (flat_owner_ != nullptr) {
        return mutable_flat() + address;
    }
    auto& cache = g_page_cache;
    const uint32_t number = address >> 12;
    if (cache.owner == this && cache.page == number &&
        cache.generation == pages_.size()) {
        return cache.data + (address & (kPageSize - 1));
    }
    auto [found, inserted] = pages_.try_emplace(number);
    if (inserted) {
        found->second.fill(0);
    }
    cache = {this, pages_.size(), number, found->second.data()};
    return found->second.data() + (address & (kPageSize - 1));
}

uint8_t GuestMemory::read8_slow(uint32_t address, uint32_t pc) const {
    if (external()) {
        return callbacks_.read8(callbacks_.context, address);
    }
    check(address, 1, pc, MemoryAccess::read);
    const auto* bytes = page(address);
    return bytes == nullptr ? 0 : bytes[0];
}

uint16_t GuestMemory::read16_slow(uint32_t address, uint32_t pc) const {
    if (external()) {
        return callbacks_.read16(callbacks_.context, address);
    }
    check(address, 2, pc, MemoryAccess::read);
    if ((address & (kPageSize - 1)) + 2 <= kPageSize) {
        const auto* bytes = page(address);
        return bytes == nullptr
                   ? 0
                   : static_cast<uint16_t>(bytes[0]) << 8 | bytes[1];
    }

    uint16_t value{};
    for (uint32_t index = 0; index < 2; ++index) {
        const auto* byte = page(address + index);
        value = static_cast<uint16_t>(value << 8 |
                                      (byte == nullptr ? 0 : byte[0]));
    }
    return value;
}

uint32_t GuestMemory::read32_slow(uint32_t address, uint32_t pc) const {
    if (external()) {
        return callbacks_.read32(callbacks_.context, address);
    }
    check(address, 4, pc, MemoryAccess::read);
    if ((address & (kPageSize - 1)) + 4 <= kPageSize) {
        const auto* bytes = page(address);
        return bytes == nullptr
                   ? 0
                   : static_cast<uint32_t>(bytes[0]) << 24 |
                         static_cast<uint32_t>(bytes[1]) << 16 |
                         static_cast<uint32_t>(bytes[2]) << 8 | bytes[3];
    }

    uint32_t value{};
    for (uint32_t index = 0; index < 4; ++index) {
        const auto* byte = page(address + index);
        value = value << 8 | (byte == nullptr ? 0 : byte[0]);
    }
    return value;
}

uint64_t GuestMemory::read64_slow(uint32_t address, uint32_t pc) const {
    if (external()) {
        return callbacks_.read64(callbacks_.context, address);
    }
    check(address, 8, pc, MemoryAccess::read);
    if ((address & (kPageSize - 1)) + 8 <= kPageSize) {
        const auto* bytes = page(address);
        if (bytes == nullptr) {
            return 0;
        }
        uint64_t value{};
        for (uint32_t index = 0; index < 8; ++index) {
            value = value << 8 | bytes[index];
        }
        return value;
    }

    uint64_t value{};
    for (uint32_t index = 0; index < 8; ++index) {
        const auto* byte = page(address + index);
        value = value << 8 | (byte == nullptr ? 0 : byte[0]);
    }
    return value;
}

void GuestMemory::write8_slow(uint32_t address, uint8_t value, uint32_t pc) {
    if (external()) {
        callbacks_.write8(callbacks_.context, address, value);
        return;
    }
    check(address, 1, pc, MemoryAccess::write);
    if ((address == 0x39DE6FE0u || address == 0x39DE7000u) &&
        env_nwiiu_pool_trace()) {
        std::fprintf(stderr, "POOLBYTE pc=%08X address=%08X value=%02X\n",
                     pc, address, value);
    }
    page(address)[0] = value;
}

void GuestMemory::write16_slow(uint32_t address, uint16_t value, uint32_t pc) {
    if (external()) {
        callbacks_.write16(callbacks_.context, address, value);
        return;
    }
    check(address, 2, pc, MemoryAccess::write);
    if (env_nwiiu_pool_trace() &&
        ((address <= 0x39DE6FE0u && 0x39DE6FE0u - address < 2) ||
         (address <= 0x39DE7000u && 0x39DE7000u - address < 2))) {
        std::fprintf(stderr, "POOL16 pc=%08X address=%08X value=%04X\n",
                     pc, address, value);
    }
    if ((address & (kPageSize - 1)) + 2 <= kPageSize) {
        auto* bytes = page(address);
        bytes[0] = static_cast<uint8_t>(value >> 8);
        bytes[1] = static_cast<uint8_t>(value);
        return;
    }

    for (uint32_t index = 0; index < 2; ++index) {
        page(address + index)[0] =
            static_cast<uint8_t>(value >> (8 - index * 8));
    }
}

void GuestMemory::write32_slow(uint32_t address, uint32_t value, uint32_t pc) {
    if (external()) {
        callbacks_.write32(callbacks_.context, address, value);
        return;
    }
    check(address, 4, pc, MemoryAccess::write);
    if (env_nwiiu_pool_trace() &&
        (address == 0x39DE6FE0u || address == 0x39DE7000u) &&
        value > 0x100000u) {
        std::fprintf(stderr, "POOL32 pc=%08X address=%08X value=%08X\n",
                     pc, address, value);
    }
    if (env_nwiiu_matrix_trace() &&
        overlaps_matrix_trace(address, 4)) {
        std::fprintf(stderr, "MATRIX32 pc=%08X address=%08X value=%08X\n",
                     pc, address, value);
    }
    if ((address & (kPageSize - 1)) + 4 <= kPageSize) {
        auto* bytes = page(address);
        bytes[0] = static_cast<uint8_t>(value >> 24);
        bytes[1] = static_cast<uint8_t>(value >> 16);
        bytes[2] = static_cast<uint8_t>(value >> 8);
        bytes[3] = static_cast<uint8_t>(value);
        return;
    }

    for (uint32_t index = 0; index < 4; ++index) {
        page(address + index)[0] =
            static_cast<uint8_t>(value >> (24 - index * 8));
    }
}

void GuestMemory::write64_slow(uint32_t address, uint64_t value, uint32_t pc) {
    if (external()) {
        callbacks_.write64(callbacks_.context, address, value);
        return;
    }
    check(address, 8, pc, MemoryAccess::write);
    if (env_nwiiu_pool_trace() &&
        ((address <= 0x39DE6FE0u && 0x39DE6FE0u - address < 8) ||
         (address <= 0x39DE7000u && 0x39DE7000u - address < 8))) {
        std::fprintf(stderr, "POOL64 pc=%08X address=%08X value=%016llX\n",
                     pc, address, static_cast<unsigned long long>(value));
    }
    if (env_nwiiu_matrix_trace() &&
        overlaps_matrix_trace(address, 8)) {
        std::fprintf(stderr, "MATRIX64 pc=%08X address=%08X value=%016llX\n",
                     pc, address, static_cast<unsigned long long>(value));
    }
    if ((address & (kPageSize - 1)) + 8 <= kPageSize) {
        auto* bytes = page(address);
        for (uint32_t index = 0; index < 8; ++index) {
            bytes[index] = static_cast<uint8_t>(value >> (56 - index * 8));
        }
        return;
    }

    for (uint32_t index = 0; index < 8; ++index) {
        page(address + index)[0] =
            static_cast<uint8_t>(value >> (56 - index * 8));
    }
}

void GuestMemory::read_bytes(uint32_t address, std::span<uint8_t> output,
                             uint32_t pc) const {
    if (flat_base_ != nullptr) {
        std::memcpy(output.data(), flat_base_ + address, output.size());
        return;
    }
    if (external()) {
        callbacks_.read_bytes(callbacks_.context, address, output.data(),
                              static_cast<uint32_t>(output.size()));
        return;
    }
    validate_range(address, static_cast<uint32_t>(output.size()), pc,
                   MemoryAccess::read);
    for (std::size_t copied = 0; copied < output.size();) {
        const uint32_t current = address + static_cast<uint32_t>(copied);
        const auto count = std::min<std::size_t>(
            output.size() - copied, kPageSize - (current & (kPageSize - 1)));
        const auto* source = page(current);
        if (source == nullptr) {
            std::fill_n(output.data() + copied, count, uint8_t{});
        } else {
            std::memcpy(output.data() + copied, source, count);
        }
        copied += count;
    }
}

void GuestMemory::write_bytes(uint32_t address,
                              std::span<const uint8_t> input, uint32_t pc) {
    if (flat_base_ != nullptr) {
        std::memcpy(mutable_flat() + address, input.data(), input.size());
        return;
    }
    if (external()) {
        callbacks_.write_bytes(callbacks_.context, address, input.data(),
                               static_cast<uint32_t>(input.size()));
        return;
    }
    validate_range(address, static_cast<uint32_t>(input.size()), pc,
                   MemoryAccess::write);
    if (env_nwiiu_pool_trace() &&
        ((address <= 0x39DE6FE0u &&
          0x39DE6FE0u - address < input.size()) ||
         (address <= 0x39DE7000u &&
          0x39DE7000u - address < input.size()))) {
        std::fprintf(stderr, "POOLBYTES pc=%08X address=%08X size=%zu\n",
                     pc, address, input.size());
    }
    if (env_nwiiu_matrix_trace() &&
        overlaps_matrix_trace(address, input.size())) {
        std::fprintf(stderr, "MATRIXBYTES pc=%08X address=%08X size=%zu\n",
                     pc, address, input.size());
    }
    for (std::size_t copied = 0; copied < input.size();) {
        const uint32_t current = address + static_cast<uint32_t>(copied);
        const auto count = std::min<std::size_t>(
            input.size() - copied, kPageSize - (current & (kPageSize - 1)));
        std::memcpy(page(current), input.data() + copied, count);
        copied += count;
    }
}

void GuestMemory::fill(uint32_t address, uint32_t size,
                       std::span<const uint8_t> pattern, uint32_t pc) {
    if (external()) {
        if (size != 0 && pattern.empty()) {
            throw std::invalid_argument("guest memory fill pattern is empty");
        }
        for (uint32_t index = 0; index < size; ++index) {
            callbacks_.write8(callbacks_.context, address + index,
                              pattern[index % pattern.size()]);
        }
        return;
    }
    if (size == 0) {
        return;
    }
    if (pattern.empty()) {
        throw std::invalid_argument("guest memory fill pattern is empty");
    }
    validate_range(address, size, pc, MemoryAccess::write);
    if (env_nwiiu_pool_trace() &&
        ((address <= 0x39DE6FE0u && 0x39DE6FE0u - address < size) ||
         (address <= 0x39DE7000u && 0x39DE7000u - address < size))) {
        std::fprintf(stderr, "POOLFILL pc=%08X address=%08X size=%u\n",
                     pc, address, size);
    }
    if (env_nwiiu_matrix_trace() &&
        overlaps_matrix_trace(address, size)) {
        std::fprintf(stderr, "MATRIXFILL pc=%08X address=%08X size=%u\n",
                     pc, address, size);
    }
    for (uint32_t written = 0; written < size;) {
        const uint32_t current = address + written;
        const uint32_t page_count =
            std::min(size - written, kPageSize - (current & (kPageSize - 1)));
        auto* destination = page(current);
        if (pattern.size() == 1) {
            std::memset(destination, pattern.front(), page_count);
        } else {
            for (uint32_t index = 0; index < page_count; ++index) {
                destination[index] =
                    pattern[(written + index) % pattern.size()];
            }
        }
        written += page_count;
    }
}

uint32_t GuestMemory::fetch32(uint32_t address) const {
    if (address % 4 != 0) {
        throw GuestFault("unaligned instruction fetch", address, 4, address,
                         MemoryAccess::execute);
    }
    if (external()) {
        return callbacks_.read32(callbacks_.context, address);
    }
    check(address, 4, address, MemoryAccess::execute);
    const auto* bytes = page(address);
    if (bytes == nullptr) {
        return 0;
    }
    return static_cast<uint32_t>(bytes[0]) << 24 |
           static_cast<uint32_t>(bytes[1]) << 16 |
           static_cast<uint32_t>(bytes[2]) << 8 | bytes[3];
}

void GuestMemory::patch16(uint32_t address, uint16_t value) {
    if (external()) {
        callbacks_.write16(callbacks_.context, address, value);
        return;
    }
    check(address, 2, 0, MemoryAccess::write, false);
    if ((address & (kPageSize - 1)) + 2 <= kPageSize) {
        auto* bytes = page(address);
        bytes[0] = static_cast<uint8_t>(value >> 8);
        bytes[1] = static_cast<uint8_t>(value);
        return;
    }

    for (uint32_t index = 0; index < 2; ++index) {
        page(address + index)[0] =
            static_cast<uint8_t>(value >> (8 - index * 8));
    }
}

void GuestMemory::patch32(uint32_t address, uint32_t value) {
    if (external()) {
        callbacks_.write32(callbacks_.context, address, value);
        return;
    }
    check(address, 4, 0, MemoryAccess::write, false);
    if ((address & (kPageSize - 1)) + 4 <= kPageSize) {
        auto* bytes = page(address);
        bytes[0] = static_cast<uint8_t>(value >> 24);
        bytes[1] = static_cast<uint8_t>(value >> 16);
        bytes[2] = static_cast<uint8_t>(value >> 8);
        bytes[3] = static_cast<uint8_t>(value);
        return;
    }

    for (uint32_t index = 0; index < 4; ++index) {
        page(address + index)[0] =
            static_cast<uint8_t>(value >> (24 - index * 8));
    }
}
}
