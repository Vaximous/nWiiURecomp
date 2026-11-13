#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>

namespace nwiiu::recomp {
// How a block is emitted. The defaults give a self-contained extern "C"
// function over GuestMemory (the fixture/test form). Project shards use the
// compact form: internal linkage (the shard's block table is the only
// reference), guest memory accessed through runtime::FlatMemory (no paged
// fallback or per-access checks), one instruction-count update per block,
// and no per-function includes (the shard includes them once).
struct NativeEmitOptions {
    bool internal_linkage{false};
    bool flat_memory{false};
    bool batch_instruction_count{false};
    bool include_headers{true};
};

std::string generate_native_block(
    std::string_view name, uint32_t start_address,
    std::span<const uint32_t> instructions,
    const std::map<uint32_t, uint32_t>& branch_overrides,
    const NativeEmitOptions& options);
std::string generate_native_block(
    std::string_view name, uint32_t start_address,
    std::span<const uint32_t> instructions,
    const std::map<uint32_t, uint32_t>& branch_overrides);
std::string generate_native_function(
    std::string_view name, uint32_t start_address,
    std::span<const uint32_t> instructions,
    const std::map<uint32_t, uint32_t>& branch_overrides);
} // namespace nwiiu::recomp
