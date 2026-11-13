#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

namespace nwii::runtime {
enum class MemoryAccess { read, write, execute };

struct MemoryPermissions {
    bool readable{};
    bool writable{};
    bool executable{};
};

class GuestFault final : public std::runtime_error {
public:
    GuestFault(std::string reason, uint32_t address, uint32_t width,
               uint32_t pc, MemoryAccess access);
    uint32_t address;
    uint32_t width;
    uint32_t pc;
    MemoryAccess access;
};

#if defined(_MSC_VER)
#define NWII_BSWAP16(v) _byteswap_ushort(v)
#define NWII_BSWAP32(v) _byteswap_ulong(v)
#define NWII_BSWAP64(v) _byteswap_uint64(v)
#else
#define NWII_BSWAP16(v) __builtin_bswap16(v)
#define NWII_BSWAP32(v) __builtin_bswap32(v)
#define NWII_BSWAP64(v) __builtin_bswap64(v)
#endif

struct GuestMemoryCallbacks {
    void* context{};
    uint8_t (*read8)(void*, uint32_t){};
    uint16_t (*read16)(void*, uint32_t){};
    uint32_t (*read32)(void*, uint32_t){};
    uint64_t (*read64)(void*, uint32_t){};
    void (*write8)(void*, uint32_t, uint8_t){};
    void (*write16)(void*, uint32_t, uint16_t){};
    void (*write32)(void*, uint32_t, uint32_t){};
    void (*write64)(void*, uint32_t, uint64_t){};
    void (*read_bytes)(void*, uint32_t, uint8_t*, uint32_t){};
    void (*write_bytes)(void*, uint32_t, const uint8_t*, uint32_t){};
    // Flat big-endian mapping of the guest space. Only adopted when it covers
    // the whole 32-bit range, which makes a per-access bounds check provably
    // dead for uint32_t addresses.
    const uint8_t* flat_base{};
    uint64_t flat_size{};
};

class GuestMemory {
public:
    GuestMemory() = default;
    explicit GuestMemory(GuestMemoryCallbacks callbacks);
    void map(uint32_t address, uint32_t size, MemoryPermissions permissions,
             std::span<const uint8_t> initial = {});
    uint64_t mapped_bytes() const;
    uint64_t resident_bytes() const;
    // Moves an internally backed memory onto one reserved 4 GiB host range:
    // every existing mapping is committed and its pages copied over, later
    // maps commit in place, and the inline flat fast paths take over. That
    // makes concurrent access from several host threads safe (no lazily
    // grown page table) and lets stwcx. use a real host compare-exchange.
    // Accesses outside a mapping no longer raise GuestFault. Returns false
    // when the host could not reserve the range (memory stays paged).
    bool enable_host_flat();
    bool host_flat() const { return flat_owner_ != nullptr; }
    // Base of the host-flat range (null unless host_flat() or a full-range
    // callback mapping was adopted).
    uint8_t* flat_data() { return mutable_flat(); }
    // Atomic when flat: stores `desired` only if the word still equals
    // `expected`. Used by stwcx. so reservations hold across host threads.
    bool compare_exchange32(uint32_t address, uint32_t expected,
                            uint32_t desired, uint32_t pc) {
        if (flat_base_ != nullptr) {
            uint32_t raw_expected = NWII_BSWAP32(expected);
            const uint32_t raw_desired = NWII_BSWAP32(desired);
            return __atomic_compare_exchange_n(
                reinterpret_cast<uint32_t*>(mutable_flat() + address),
                &raw_expected, raw_desired, false, __ATOMIC_SEQ_CST,
                __ATOMIC_SEQ_CST);
        }
        if (read32(address, pc) != expected) {
            return false;
        }
        write32(address, desired, pc);
        return true;
    }
    // Hot paths are inline: the module is built without LTO, so an
    // out-of-line call here would defeat the point of the flat mapping.
    uint8_t read8(uint32_t address, uint32_t pc) const {
        if (flat_base_ != nullptr)
            return flat_base_[address];
        return read8_slow(address, pc);
    }
    uint16_t read16(uint32_t address, uint32_t pc) const {
        if (flat_base_ != nullptr) {
            uint16_t raw;
            std::memcpy(&raw, flat_base_ + address, sizeof raw);
            return NWII_BSWAP16(raw);
        }
        return read16_slow(address, pc);
    }
    uint32_t read32(uint32_t address, uint32_t pc) const {
        if (flat_base_ != nullptr) {
            uint32_t raw;
            std::memcpy(&raw, flat_base_ + address, sizeof raw);
            return NWII_BSWAP32(raw);
        }
        return read32_slow(address, pc);
    }
    uint64_t read64(uint32_t address, uint32_t pc) const {
        if (flat_base_ != nullptr) {
            uint64_t raw;
            std::memcpy(&raw, flat_base_ + address, sizeof raw);
            return NWII_BSWAP64(raw);
        }
        return read64_slow(address, pc);
    }
    void write8(uint32_t address, uint8_t value, uint32_t pc) {
        if (flat_base_ != nullptr) {
            mutable_flat()[address] = value;
            return;
        }
        write8_slow(address, value, pc);
    }
    void write16(uint32_t address, uint16_t value, uint32_t pc) {
        if (flat_base_ != nullptr) {
            const uint16_t raw = NWII_BSWAP16(value);
            std::memcpy(mutable_flat() + address, &raw, sizeof raw);
            return;
        }
        write16_slow(address, value, pc);
    }
    void write32(uint32_t address, uint32_t value, uint32_t pc) {
        if (flat_base_ != nullptr) {
            const uint32_t raw = NWII_BSWAP32(value);
            std::memcpy(mutable_flat() + address, &raw, sizeof raw);
            return;
        }
        write32_slow(address, value, pc);
    }
    void write64(uint32_t address, uint64_t value, uint32_t pc) {
        if (flat_base_ != nullptr) {
            const uint64_t raw = NWII_BSWAP64(value);
            std::memcpy(mutable_flat() + address, &raw, sizeof raw);
            return;
        }
        write64_slow(address, value, pc);
    }
    void read_bytes(uint32_t address, std::span<uint8_t> output,
                    uint32_t pc) const;
    void write_bytes(uint32_t address, std::span<const uint8_t> input,
                     uint32_t pc);
    void fill(uint32_t address, uint32_t size,
              std::span<const uint8_t> pattern, uint32_t pc);
    void validate(uint32_t address, uint32_t width, uint32_t pc,
                  MemoryAccess access) const;
    void validate_range(uint32_t address, uint32_t size, uint32_t pc,
                        MemoryAccess access) const;
    uint32_t fetch32(uint32_t address) const;
    void patch16(uint32_t address, uint16_t value);
    void patch32(uint32_t address, uint32_t value);

private:
    static constexpr uint32_t kPageSize = 0x1000;

    struct Mapping {
        uint32_t address;
        uint32_t size;
        MemoryPermissions permissions;
    };
    using Page = std::array<uint8_t, kPageSize>;

    void check(uint32_t address, uint32_t width, uint32_t pc,
               MemoryAccess access, bool check_permissions = true) const;
    const uint8_t* page(uint32_t address) const;
    uint8_t* page(uint32_t address);

    bool external() const { return callbacks_.read8 != nullptr; }

    // flat_base_ is only set from a host mapping the guest may write through.
    uint8_t* mutable_flat() { return const_cast<uint8_t*>(flat_base_); }

    uint8_t read8_slow(uint32_t address, uint32_t pc) const;
    uint16_t read16_slow(uint32_t address, uint32_t pc) const;
    uint32_t read32_slow(uint32_t address, uint32_t pc) const;
    uint64_t read64_slow(uint32_t address, uint32_t pc) const;
    void write8_slow(uint32_t address, uint8_t value, uint32_t pc);
    void write16_slow(uint32_t address, uint16_t value, uint32_t pc);
    void write32_slow(uint32_t address, uint32_t value, uint32_t pc);
    void write64_slow(uint32_t address, uint64_t value, uint32_t pc);

    GuestMemoryCallbacks callbacks_{};
    const uint8_t* flat_base_{};
    // Owns the host reservation behind enable_host_flat(); shared so that a
    // copied GuestMemory never frees the range under the original.
    std::shared_ptr<uint8_t> flat_owner_;
    std::vector<Mapping> mappings_;
    std::unordered_map<uint32_t, Page> pages_;
};

// What recompiled blocks access guest memory through: the host-flat range as
// plain big-endian loads and stores, with no paged fallback and no mapping or
// permission checks. The instruction semantics are templates over the memory
// type, so these instantiations never collide with the interpreter's
// GuestMemory ones. Only valid once GuestMemory::enable_host_flat succeeded
// (the registry refuses to install blocks otherwise).
class FlatMemory {
public:
    explicit FlatMemory(GuestMemory& memory) : base_(memory.flat_data()) {}
    uint8_t read8(uint32_t address, uint32_t) const { return base_[address]; }
    uint16_t read16(uint32_t address, uint32_t) const {
        uint16_t raw;
        std::memcpy(&raw, base_ + address, sizeof raw);
        return NWII_BSWAP16(raw);
    }
    uint32_t read32(uint32_t address, uint32_t) const {
        uint32_t raw;
        std::memcpy(&raw, base_ + address, sizeof raw);
        return NWII_BSWAP32(raw);
    }
    uint64_t read64(uint32_t address, uint32_t) const {
        uint64_t raw;
        std::memcpy(&raw, base_ + address, sizeof raw);
        return NWII_BSWAP64(raw);
    }
    void write8(uint32_t address, uint8_t value, uint32_t) {
        base_[address] = value;
    }
    void write16(uint32_t address, uint16_t value, uint32_t) {
        const uint16_t raw = NWII_BSWAP16(value);
        std::memcpy(base_ + address, &raw, sizeof raw);
    }
    void write32(uint32_t address, uint32_t value, uint32_t) {
        const uint32_t raw = NWII_BSWAP32(value);
        std::memcpy(base_ + address, &raw, sizeof raw);
    }
    void write64(uint32_t address, uint64_t value, uint32_t) {
        const uint64_t raw = NWII_BSWAP64(value);
        std::memcpy(base_ + address, &raw, sizeof raw);
    }
    bool compare_exchange32(uint32_t address, uint32_t expected,
                            uint32_t desired, uint32_t) {
        uint32_t raw_expected = NWII_BSWAP32(expected);
        return __atomic_compare_exchange_n(
            reinterpret_cast<uint32_t*>(base_ + address), &raw_expected,
            NWII_BSWAP32(desired), false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    }
    void validate(uint32_t, uint32_t, uint32_t, MemoryAccess) const {}
    void validate_range(uint32_t, uint32_t, uint32_t, MemoryAccess) const {}

private:
    uint8_t* base_;
};
}
