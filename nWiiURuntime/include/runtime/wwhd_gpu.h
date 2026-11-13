#pragma once

#include "runtime/latte_shader.h"

#include <SDL3/SDL.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <span>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace nwii::runtime {

class WwhdGpu {
public:
    struct Texture {
        SDL_GPUTexture* handle{};
        uint32_t width{};
        uint32_t height{};
    };

    struct VertexStream {
        uint32_t location{};
        SDL_GPUVertexElementFormat format{};
        uint32_t pitch{};
        std::vector<uint8_t> data;
    };

    WwhdGpu() {
        // Debug mode enables the Vulkan validation layers when installed,
        // which cost several ms per frame; opt in with NWIIU_GPU_DEBUG.
        device_ = SDL_CreateGPUDevice(SDL_GPU_SHADERFORMAT_SPIRV,
                                      std::getenv("NWIIU_GPU_DEBUG") != nullptr,
                                      "vulkan");
        if (device_ == nullptr) {
            fail("SDL_CreateGPUDevice");
        }
        SDL_GPUSamplerCreateInfo info{};
        info.min_filter = SDL_GPU_FILTER_LINEAR;
        info.mag_filter = SDL_GPU_FILTER_LINEAR;
        info.mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_LINEAR;
        info.address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        info.address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        info.address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_REPEAT;
        sampler_ = SDL_CreateGPUSampler(device_, &info);
        if (sampler_ == nullptr) {
            SDL_DestroyGPUDevice(device_);
            device_ = nullptr;
            fail("SDL_CreateGPUSampler");
        }
    }

    SDL_GPUDevice* device() const { return device_; }

    void wait_idle() {
        flush();
        if (!SDL_WaitForGPUIdle(device_)) {
            fail("SDL_WaitForGPUIdle");
        }
    }

    ~WwhdGpu() {
        if (pending_ != nullptr) {
            SDL_SubmitGPUCommandBuffer(pending_);
            pending_ = nullptr;
        }
        SDL_WaitForGPUIdle(device_);
        for (const auto& [key, pipeline] : pipelines_) {
            static_cast<void>(key);
            SDL_ReleaseGPUGraphicsPipeline(device_, pipeline);
        }
        if (arena_buffer_ != nullptr) {
            SDL_ReleaseGPUBuffer(device_, arena_buffer_);
            SDL_ReleaseGPUTransferBuffer(device_, arena_transfer_);
        }
        SDL_ReleaseGPUSampler(device_, sampler_);
        SDL_DestroyGPUDevice(device_);
    }

    WwhdGpu(const WwhdGpu&) = delete;
    WwhdGpu& operator=(const WwhdGpu&) = delete;

    Texture create_target(uint32_t width, uint32_t height,
                          std::span<const uint8_t> pixels = {}) {
        SDL_GPUTextureCreateInfo info{};
        info.type = SDL_GPU_TEXTURETYPE_2D;
        info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        info.usage = SDL_GPU_TEXTUREUSAGE_COLOR_TARGET |
                     SDL_GPU_TEXTUREUSAGE_SAMPLER;
        info.width = width;
        info.height = height;
        info.layer_count_or_depth = 1;
        info.num_levels = 1;
        info.sample_count = SDL_GPU_SAMPLECOUNT_1;
        auto* texture = SDL_CreateGPUTexture(device_, &info);
        if (texture == nullptr) {
            fail("SDL_CreateGPUTexture target");
        }
        Texture result{texture, width, height};
        if (!pixels.empty()) {
            upload(result, pixels);
        }
        return result;
    }

    Texture create_texture(uint32_t width, uint32_t height,
                           std::span<const uint8_t> pixels) {
        SDL_GPUTextureCreateInfo info{};
        info.type = SDL_GPU_TEXTURETYPE_2D;
        info.format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM;
        info.usage = SDL_GPU_TEXTUREUSAGE_SAMPLER;
        info.width = width;
        info.height = height;
        info.layer_count_or_depth = 1;
        info.num_levels = 1;
        info.sample_count = SDL_GPU_SAMPLECOUNT_1;
        auto* texture = SDL_CreateGPUTexture(device_, &info);
        if (texture == nullptr) {
            fail("SDL_CreateGPUTexture sampled");
        }
        Texture result{texture, width, height};
        upload(result, pixels);
        return result;
    }

    void upload(Texture texture, std::span<const uint8_t> pixels) {
        auto* transfer = transfer_buffer(SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                         pixels.size());
        void* mapped = SDL_MapGPUTransferBuffer(device_, transfer, false);
        if (mapped == nullptr) {
            SDL_ReleaseGPUTransferBuffer(device_, transfer);
            fail("SDL_MapGPUTransferBuffer texture");
        }
        std::memcpy(mapped, pixels.data(), pixels.size());
        SDL_UnmapGPUTransferBuffer(device_, transfer);
        auto* command = command_buffer();
        auto* copy = SDL_BeginGPUCopyPass(command);
        const SDL_GPUTextureTransferInfo source{
            transfer, 0, texture.width, texture.height};
        const SDL_GPUTextureRegion destination{
            texture.handle, 0, 0, 0, 0, 0, texture.width, texture.height, 1};
        SDL_UploadToGPUTexture(copy, &source, &destination, false);
        SDL_EndGPUCopyPass(copy);
        submit(command);
        // No idle wait: SDL defers the transfer buffer's release until the
        // submitted copy has consumed it.
        SDL_ReleaseGPUTransferBuffer(device_, transfer);
    }

    // GPU-side copy between equally sized RGBA8 textures; no CPU round trip
    // and no idle wait (the old snapshot path downloaded + re-uploaded).
    void copy(Texture source, Texture destination) {
        auto* command = command_buffer();
        auto* pass = SDL_BeginGPUCopyPass(command);
        const SDL_GPUTextureLocation from{source.handle, 0, 0, 0, 0, 0};
        const SDL_GPUTextureLocation to{destination.handle, 0, 0, 0, 0, 0};
        SDL_CopyGPUTextureToTexture(
            pass, &from, &to, std::min(source.width, destination.width),
            std::min(source.height, destination.height), 1, false);
        SDL_EndGPUCopyPass(pass);
        submit(command);
    }

    void release(Texture texture) {
        if (texture.handle != nullptr) {
            SDL_ReleaseGPUTexture(device_, texture.handle);
        }
    }

    void clear(Texture target, const std::array<float, 4>& color) {
        auto* command = command_buffer();
        const SDL_GPUColorTargetInfo info{
            target.handle, 0, 0,
            {color[0], color[1], color[2], color[3]},
            SDL_GPU_LOADOP_CLEAR, SDL_GPU_STOREOP_STORE,
            nullptr, 0, 0, false, false, 0, 0};
        auto* pass = SDL_BeginGPURenderPass(command, &info, 1, nullptr);
        SDL_EndGPURenderPass(pass);
        submit(command);
    }

    using PipelineKey = std::tuple<uint32_t, uint32_t, uint32_t, uint64_t>;

    // The behaviour before blend state was plumbed through: straight alpha
    // blending compiled in but disabled, i.e. plain overwrite.
    static SDL_GPUColorTargetBlendState default_blend() {
        return {SDL_GPU_BLENDFACTOR_SRC_ALPHA,
                SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                SDL_GPU_BLENDOP_ADD,
                SDL_GPU_BLENDFACTOR_ONE,
                SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                SDL_GPU_BLENDOP_ADD,
                0, false, false, 0, 0};
    }

    void draw(const std::tuple<uint32_t, uint32_t, uint32_t>& key,
              const LatteTranslation& translated, Texture target,
              std::span<const VertexStream> streams,
              std::span<const uint32_t> indices,
              std::span<const uint32_t> vertex_uniforms,
              std::span<const uint32_t> fragment_uniforms,
              std::span<const Texture> sampled,
              const SDL_GPUColorTargetBlendState& blend = default_blend()) {
        // The pipeline bakes in blend state and the vertex layout, so both
        // are part of its cache key alongside the shader addresses.
        uint64_t state = 1469598103934665603ull;
        const auto mix = [&state](uint64_t value) {
            state = (state ^ value) * 1099511628211ull;
        };
        mix(blend.src_color_blendfactor);
        mix(blend.dst_color_blendfactor);
        mix(blend.color_blend_op);
        mix(blend.src_alpha_blendfactor);
        mix(blend.dst_alpha_blendfactor);
        mix(blend.alpha_blend_op);
        mix(blend.color_write_mask);
        mix(blend.enable_blend);
        mix(blend.enable_color_write_mask);
        for (const auto& stream : streams) {
            mix(stream.location);
            mix(stream.format);
            mix(stream.pitch);
        }
        auto* pipeline = graphics_pipeline(
            {std::get<0>(key), std::get<1>(key), std::get<2>(key), state},
            translated, streams, blend);
        std::vector<SDL_GPUBuffer*> buffers;
        std::vector<SDL_GPUTransferBuffer*> transfers;
        std::vector<SDL_GPUBufferBinding> vertex_bindings;
        vertex_bindings.reserve(streams.size());
        const auto index_bytes = std::span<const uint8_t>{
            reinterpret_cast<const uint8_t*>(indices.data()),
            indices.size_bytes()};
        SDL_GPUBufferBinding index_binding{};

        auto* command = command_buffer();
        auto* copy = SDL_BeginGPUCopyPass(command);
        // Suballocate from the per-frame arena: creating and releasing a
        // Vulkan buffer plus a transfer buffer per stream per draw dominated
        // scene frames. Oversized draws fall back to dedicated buffers.
        size_t needed = align(index_bytes.size());
        for (const auto& stream : streams) {
            needed += align(stream.data.size());
        }
        if (needed <= kArenaBytes) {
            arena_reserve(needed);
            for (const auto& stream : streams) {
                vertex_bindings.push_back(
                    {arena_buffer_, arena_upload(copy, stream.data)});
            }
            index_binding = {arena_buffer_, arena_upload(copy, index_bytes)};
        } else {
            buffers.reserve(streams.size() + 1);
            transfers.reserve(streams.size() + 1);
            for (const auto& stream : streams) {
                auto [buffer, transfer] = upload_buffer(
                    copy, SDL_GPU_BUFFERUSAGE_VERTEX, stream.data);
                buffers.push_back(buffer);
                transfers.push_back(transfer);
                vertex_bindings.push_back({buffer, 0});
            }
            auto [index_buffer, index_transfer] = upload_buffer(
                copy, SDL_GPU_BUFFERUSAGE_INDEX, index_bytes);
            buffers.push_back(index_buffer);
            transfers.push_back(index_transfer);
            index_binding = {index_buffer, 0};
        }
        SDL_EndGPUCopyPass(copy);

        SDL_PushGPUVertexUniformData(
            command, 0, vertex_uniforms.data(),
            static_cast<uint32_t>(vertex_uniforms.size_bytes()));
        SDL_PushGPUFragmentUniformData(
            command, 0, fragment_uniforms.data(),
            static_cast<uint32_t>(fragment_uniforms.size_bytes()));
        const SDL_GPUColorTargetInfo color{
            target.handle, 0, 0, {}, SDL_GPU_LOADOP_LOAD,
            SDL_GPU_STOREOP_STORE, nullptr, 0, 0, false, false, 0, 0};
        auto* pass = SDL_BeginGPURenderPass(command, &color, 1, nullptr);
        SDL_BindGPUGraphicsPipeline(pass, pipeline);
        SDL_BindGPUVertexBuffers(pass, 0, vertex_bindings.data(),
                                 vertex_bindings.size());
        SDL_BindGPUIndexBuffer(pass, &index_binding,
                               SDL_GPU_INDEXELEMENTSIZE_32BIT);
        std::vector<SDL_GPUTextureSamplerBinding> sampler_bindings;
        sampler_bindings.reserve(sampled.size());
        for (const auto texture : sampled) {
            sampler_bindings.push_back({texture.handle, sampler_});
        }
        if (!sampler_bindings.empty()) {
            SDL_BindGPUFragmentSamplers(pass, 0, sampler_bindings.data(),
                                        sampler_bindings.size());
        }
        SDL_DrawGPUIndexedPrimitives(pass, indices.size(), 1, 0, 0, 0);
        SDL_EndGPURenderPass(pass);
        submit(command);
        for (auto* transfer : transfers) {
            SDL_ReleaseGPUTransferBuffer(device_, transfer);
        }
        for (auto* buffer : buffers) {
            SDL_ReleaseGPUBuffer(device_, buffer);
        }
    }

    std::vector<uint8_t> download(Texture texture) {
        const size_t size = static_cast<size_t>(texture.width) *
                            texture.height * 4;
        auto* transfer = transfer_buffer(SDL_GPU_TRANSFERBUFFERUSAGE_DOWNLOAD,
                                         size);
        auto* command = command_buffer();
        auto* copy = SDL_BeginGPUCopyPass(command);
        const SDL_GPUTextureRegion source{texture.handle, 0, 0, 0, 0, 0,
                                          texture.width, texture.height, 1};
        const SDL_GPUTextureTransferInfo destination{
            transfer, 0, texture.width, texture.height};
        SDL_DownloadFromGPUTexture(copy, &source, &destination);
        SDL_EndGPUCopyPass(copy);
        flush();
        if (!SDL_WaitForGPUIdle(device_)) {
            SDL_ReleaseGPUTransferBuffer(device_, transfer);
            fail("SDL_WaitForGPUIdle");
        }
        void* mapped = SDL_MapGPUTransferBuffer(device_, transfer, false);
        if (mapped == nullptr) {
            SDL_ReleaseGPUTransferBuffer(device_, transfer);
            fail("SDL_MapGPUTransferBuffer download");
        }
        std::vector<uint8_t> pixels(size);
        std::memcpy(pixels.data(), mapped, size);
        SDL_UnmapGPUTransferBuffer(device_, transfer);
        SDL_ReleaseGPUTransferBuffer(device_, transfer);
        return pixels;
    }

private:
    [[noreturn]] static void fail(const char* operation) {
        throw std::runtime_error(std::string{operation} + ": " +
                                 SDL_GetError());
    }

    // All work records into one shared command buffer that is submitted by
    // flush() (per frame, before readbacks, or every kFlushOps operations).
    // Submitting per draw cost a queue submission each, ~120 per UE3 frame.
    SDL_GPUCommandBuffer* command_buffer() {
        if (pending_ == nullptr) {
            pending_ = SDL_AcquireGPUCommandBuffer(device_);
            if (pending_ == nullptr) {
                fail("SDL_AcquireGPUCommandBuffer");
            }
        }
        return pending_;
    }

    void submit(SDL_GPUCommandBuffer* command) {
        static_cast<void>(command);
        if (++pending_ops_ >= kFlushOps) {
            flush();
        }
    }

public:
    void flush() {
        if (pending_ != nullptr) {
            auto* command = pending_;
            pending_ = nullptr;
            pending_ops_ = 0;
            if (!SDL_SubmitGPUCommandBuffer(command)) {
                fail("SDL_SubmitGPUCommandBuffer");
            }
        }
    }

private:
    static constexpr uint32_t kFlushOps = 512;
    SDL_GPUCommandBuffer* pending_{};
    uint32_t pending_ops_{};

    // One vertex+index buffer and one upload buffer, filled front to back.
    // When full, the next fill starts at 0 with cycle=true so SDL hands out
    // fresh backing memory while earlier recorded draws keep the old one.
    static constexpr size_t kArenaBytes = 64u << 20;
    SDL_GPUBuffer* arena_buffer_{};
    SDL_GPUTransferBuffer* arena_transfer_{};
    size_t arena_offset_{};
    bool arena_cycle_{true};

    static size_t align(size_t bytes) { return (bytes + 15) & ~size_t{15}; }

    void arena_reserve(size_t bytes) {
        if (arena_buffer_ == nullptr) {
            SDL_GPUBufferCreateInfo buffer_info{};
            buffer_info.usage =
                SDL_GPU_BUFFERUSAGE_VERTEX | SDL_GPU_BUFFERUSAGE_INDEX;
            buffer_info.size = static_cast<uint32_t>(kArenaBytes);
            arena_buffer_ = SDL_CreateGPUBuffer(device_, &buffer_info);
            if (arena_buffer_ == nullptr) {
                fail("SDL_CreateGPUBuffer arena");
            }
            arena_transfer_ = transfer_buffer(
                SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD, kArenaBytes);
        }
        if (arena_offset_ + bytes > kArenaBytes) {
            arena_offset_ = 0;
            arena_cycle_ = true;
        }
    }

    uint32_t arena_upload(SDL_GPUCopyPass* copy,
                          std::span<const uint8_t> bytes) {
        const auto offset = static_cast<uint32_t>(arena_offset_);
        auto* mapped = static_cast<uint8_t*>(
            SDL_MapGPUTransferBuffer(device_, arena_transfer_, arena_cycle_));
        if (mapped == nullptr) {
            fail("SDL_MapGPUTransferBuffer arena");
        }
        if (!bytes.empty()) {
            std::memcpy(mapped + offset, bytes.data(), bytes.size());
        }
        SDL_UnmapGPUTransferBuffer(device_, arena_transfer_);
        if (!bytes.empty()) {
            const SDL_GPUTransferBufferLocation source{arena_transfer_,
                                                       offset};
            const SDL_GPUBufferRegion destination{
                arena_buffer_, offset, static_cast<uint32_t>(bytes.size())};
            SDL_UploadToGPUBuffer(copy, &source, &destination, arena_cycle_);
            arena_cycle_ = false;
        }
        arena_offset_ += align(bytes.size());
        return offset;
    }

    SDL_GPUTransferBuffer* transfer_buffer(SDL_GPUTransferBufferUsage usage,
                                            size_t size) {
        SDL_GPUTransferBufferCreateInfo info{};
        info.usage = usage;
        info.size = static_cast<uint32_t>(size);
        auto* transfer = SDL_CreateGPUTransferBuffer(device_, &info);
        if (transfer == nullptr) {
            fail("SDL_CreateGPUTransferBuffer");
        }
        return transfer;
    }

    std::pair<SDL_GPUBuffer*, SDL_GPUTransferBuffer*> upload_buffer(
        SDL_GPUCopyPass* copy, SDL_GPUBufferUsageFlags usage,
        std::span<const uint8_t> bytes) {
        SDL_GPUBufferCreateInfo buffer_info{};
        buffer_info.usage = usage;
        buffer_info.size = static_cast<uint32_t>(bytes.size());
        auto* buffer = SDL_CreateGPUBuffer(device_, &buffer_info);
        if (buffer == nullptr) {
            fail("SDL_CreateGPUBuffer");
        }
        auto* transfer = transfer_buffer(SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
                                         bytes.size());
        void* mapped = SDL_MapGPUTransferBuffer(device_, transfer, false);
        if (mapped == nullptr) {
            SDL_ReleaseGPUTransferBuffer(device_, transfer);
            SDL_ReleaseGPUBuffer(device_, buffer);
            fail("SDL_MapGPUTransferBuffer buffer");
        }
        std::memcpy(mapped, bytes.data(), bytes.size());
        SDL_UnmapGPUTransferBuffer(device_, transfer);
        const SDL_GPUTransferBufferLocation source{transfer, 0};
        const SDL_GPUBufferRegion destination{
            buffer, 0, static_cast<uint32_t>(bytes.size())};
        SDL_UploadToGPUBuffer(copy, &source, &destination, false);
        return {buffer, transfer};
    }

    SDL_GPUGraphicsPipeline* graphics_pipeline(
        const PipelineKey& key, const LatteTranslation& translated,
        std::span<const VertexStream> streams,
        const SDL_GPUColorTargetBlendState& blend) {
        if (const auto found = pipelines_.find(key);
            found != pipelines_.end()) {
            return found->second;
        }
        const SDL_GPUShaderCreateInfo vertex_info{
            translated.vertex_spirv.size() * sizeof(uint32_t),
            reinterpret_cast<const uint8_t*>(translated.vertex_spirv.data()),
            "main", SDL_GPU_SHADERFORMAT_SPIRV, SDL_GPU_SHADERSTAGE_VERTEX,
            0, 0, 0, 1, 0};
        const SDL_GPUShaderCreateInfo fragment_info{
            translated.fragment_spirv.size() * sizeof(uint32_t),
            reinterpret_cast<const uint8_t*>(translated.fragment_spirv.data()),
            "main", SDL_GPU_SHADERFORMAT_SPIRV, SDL_GPU_SHADERSTAGE_FRAGMENT,
            static_cast<uint32_t>(translated.ps_sampler_slots.size()),
            0, 0, 1, 0};
        auto* vertex = SDL_CreateGPUShader(device_, &vertex_info);
        auto* fragment = SDL_CreateGPUShader(device_, &fragment_info);
        if (vertex == nullptr || fragment == nullptr) {
            SDL_ReleaseGPUShader(device_, vertex);
            SDL_ReleaseGPUShader(device_, fragment);
            fail("SDL_CreateGPUShader");
        }
        std::vector<SDL_GPUVertexBufferDescription> descriptions;
        std::vector<SDL_GPUVertexAttribute> attributes;
        descriptions.reserve(streams.size());
        attributes.reserve(streams.size());
        for (uint32_t slot = 0; slot < streams.size(); ++slot) {
            descriptions.push_back({slot, streams[slot].pitch,
                                    SDL_GPU_VERTEXINPUTRATE_VERTEX, 0});
            attributes.push_back({streams[slot].location, slot,
                                  streams[slot].format, 0});
        }
        const SDL_GPUColorTargetDescription color{
            SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM, blend};
        SDL_GPUGraphicsPipelineCreateInfo info{};
        info.vertex_shader = vertex;
        info.fragment_shader = fragment;
        info.vertex_input_state = {descriptions.data(),
                                   static_cast<uint32_t>(descriptions.size()),
                                   attributes.data(),
                                   static_cast<uint32_t>(attributes.size())};
        info.primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST;
        info.rasterizer_state.fill_mode = SDL_GPU_FILLMODE_FILL;
        info.rasterizer_state.cull_mode = SDL_GPU_CULLMODE_NONE;
        info.rasterizer_state.front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE;
        info.rasterizer_state.enable_depth_clip = true;
        info.multisample_state.sample_count = SDL_GPU_SAMPLECOUNT_1;
        info.target_info = {&color, 1, SDL_GPU_TEXTUREFORMAT_INVALID,
                            false, 0, 0, 0};
        auto* pipeline = SDL_CreateGPUGraphicsPipeline(device_, &info);
        SDL_ReleaseGPUShader(device_, vertex);
        SDL_ReleaseGPUShader(device_, fragment);
        if (pipeline == nullptr) {
            fail("SDL_CreateGPUGraphicsPipeline");
        }
        pipelines_.emplace(key, pipeline);
        return pipeline;
    }

    SDL_GPUDevice* device_{};
    SDL_GPUSampler* sampler_{};
    std::map<PipelineKey, SDL_GPUGraphicsPipeline*> pipelines_;
};

} // namespace nwii::runtime
