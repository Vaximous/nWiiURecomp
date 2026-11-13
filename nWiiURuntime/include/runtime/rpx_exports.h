#pragma once

#include <cstdint>
#include <string_view>

namespace nwii::runtime {
// Function exports (.fexports) of the most recently built ExecutionImage.
// Kept out of execution_image.h on purpose: every generated shard includes
// that header, so growing ExecutionImage costs a full module rebuild.
// Returns 0 when the image does not export `name`.
uint32_t rpx_export_address(std::string_view name);
} // namespace nwii::runtime
