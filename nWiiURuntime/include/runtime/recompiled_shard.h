#pragma once

// The only header a generated shard includes (and its precompiled header).
// Keep it to what translated blocks need: CPU state, the flat memory view and
// the instruction semantics. In particular it must not pull in executor.h,
// so executor changes stay a relink.

#include "runtime/cpu_context.h"
#include "runtime/memory.h"
#include "runtime/ppc_semantics.h"

#include <cstdint>

namespace nwii::runtime {
// The block may run under a host that loaded the same image (no relocated
// words inside it).
inline constexpr uint8_t kRecompiledExternalSafe = 1;
// Such a host should still verify the first word, in case it patched code.
inline constexpr uint8_t kRecompiledExternalCheck = 2;

struct RecompiledBlock {
    uint32_t address;
    uint32_t instruction_count;
    void (*thunk)(CPUContext&, GuestMemory&);
    uint32_t first_word;
    uint8_t flags;
};
} // namespace nwii::runtime
