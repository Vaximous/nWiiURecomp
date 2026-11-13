// Second-wave Cafe libraries, first needed by Devil's Third (UE3).
//
// Online services (nsysnet, nlibcurl, nn_ac/act/fp/ndm, NSSL) report "no
// network" so the title takes its offline paths. FS/SAVE async calls run the
// synchronous handler and deliver an FSAsyncResult the way the SDK does
// (message queue first, else callback). zlib125 marshals the guest z_stream
// onto host zlib. GX2 state the in-tree renderer does not model is accepted
// and ignored.
#include "runtime/cafe_runtime.h"
#include "runtime/machine.h"

#include <zlib.h>

#include <array>
#include <cstdio>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nwii::runtime {
namespace {
// nn::Result values. Bit 31 set = failure.
constexpr uint32_t kResultSuccess = 0;
constexpr uint32_t kResultNotConnected = 0xA0B12C80; // nn_ac: not connected
constexpr uint32_t kResultActNoNetwork = 0xC0712200;
constexpr uint32_t kResultFpOffline = 0xC0C12C80;

constexpr uint32_t kFsAsyncResultOffset = 0xA00; // unused tail of FSCmdBlock
constexpr uint32_t kFsMessageType = 8;

// Guest z_stream (32-bit, big-endian).
constexpr uint32_t kZNextIn = 0x00;
constexpr uint32_t kZAvailIn = 0x04;
constexpr uint32_t kZTotalIn = 0x08;
constexpr uint32_t kZNextOut = 0x0C;
constexpr uint32_t kZAvailOut = 0x10;
constexpr uint32_t kZTotalOut = 0x14;
constexpr uint32_t kZMsg = 0x18;
constexpr uint32_t kZAdler = 0x30;

std::map<uint32_t, std::unique_ptr<z_stream>>& inflate_streams() {
    static std::map<uint32_t, std::unique_ptr<z_stream>> streams;
    return streams;
}

uint32_t stack_argument(const CPUContext& cpu, const GuestMemory& memory,
                        unsigned index) {
    // Arguments 0..7 are r3..r10; the rest start at sp+8.
    if (index < 8) {
        return cpu.gpr[3 + index];
    }
    return memory.read32(cpu.gpr[1] + 8 + (index - 8) * 4, cpu.pc);
}
} // namespace

void CafeRuntime::register_extra_handlers() {
    const auto add = [this](const char* module, const char* symbol,
                            HleHandler handler) {
        if (find_handler(module, symbol) == nullptr) {
            register_handler(module, symbol, std::move(handler));
        }
    };
    const auto ret = [&add](const char* module, const char* symbol,
                            uint32_t value) {
        add(module, symbol, [value](CPUContext& cpu, GuestMemory&) {
            cpu.gpr[3] = value;
            return HleAction::return_to_lr;
        });
    };
    const auto noop = [&add](const char* module, const char* symbol) {
        add(module, symbol, [](CPUContext&, GuestMemory&) {
            return HleAction::return_to_lr;
        });
    };
    // Writes `value` through the pointer argument in r(3+arg), returns `rc`.
    const auto out32 = [&add](const char* module, const char* symbol,
                              unsigned arg, uint32_t value, uint32_t rc) {
        add(module, symbol, [arg, value, rc](CPUContext& cpu,
                                             GuestMemory& memory) {
            if (cpu.gpr[3 + arg] != 0) {
                memory.write32(cpu.gpr[3 + arg], value, cpu.pc);
            }
            cpu.gpr[3] = rc;
            return HleAction::return_to_lr;
        });
    };

    // ---------------------------------------------------------------- FS
    const auto deliver = [this](CPUContext& cpu, GuestMemory& memory,
                                uint32_t client, uint32_t block,
                                uint32_t status, uint32_t params) {
        const uint32_t callback = params ? memory.read32(params, cpu.pc) : 0;
        const uint32_t context =
            params ? memory.read32(params + 4, cpu.pc) : 0;
        const uint32_t queue = params ? memory.read32(params + 8, cpu.pc) : 0;
        const uint32_t result = block + kFsAsyncResultOffset;
        memory.write32(result + 0x00, callback, cpu.pc);
        memory.write32(result + 0x04, context, cpu.pc);
        memory.write32(result + 0x08, queue, cpu.pc);
        memory.write32(result + 0x0C, result, cpu.pc);
        memory.write32(result + 0x10, 0, cpu.pc);
        memory.write32(result + 0x14, 0, cpu.pc);
        memory.write32(result + 0x18, kFsMessageType, cpu.pc);
        memory.write32(result + 0x1C, client, cpu.pc);
        memory.write32(result + 0x20, block, cpu.pc);
        memory.write32(result + 0x24, status, cpu.pc);
        if (machine_ == nullptr) {
            return;
        }
        if (queue != 0) {
            machine_->coreinit().post_message(queue, result + 0x0C, cpu);
        } else if (callback != 0) {
            machine_->queue_callback_args(callback,
                                          {client, block, status, context});
        }
    };
    // FSReadFileWithPos/FSWriteFileWithPos = FSSetPosFile + the plain call.
    const auto with_pos = [this](const char* plain) {
        return [this, plain](CPUContext& cpu, GuestMemory& memory) {
            // (client, block, buf, size, count, pos, handle, flag, err)
            const uint32_t err = stack_argument(cpu, memory, 8);
            CPUContext seek = cpu;
            seek.gpr[5] = cpu.gpr[9];
            seek.gpr[6] = cpu.gpr[8];
            seek.gpr[7] = err;
            (*find_handler("coreinit", "FSSetPosFile"))(seek, memory);
            if (static_cast<int32_t>(seek.gpr[3]) < 0) {
                cpu.gpr[3] = seek.gpr[3];
                return HleAction::return_to_lr;
            }
            CPUContext io = cpu;
            io.gpr[8] = cpu.gpr[9];
            io.gpr[9] = cpu.gpr[10];
            io.gpr[10] = err;
            (*find_handler("coreinit", plain))(io, memory);
            cpu.gpr[3] = io.gpr[3];
            return HleAction::return_to_lr;
        };
    };
    add("coreinit", "FSReadFileWithPos", with_pos("FSReadFile"));
    add("coreinit", "FSWriteFileWithPos", with_pos("FSWriteFile"));
    const auto async = [this, &add, deliver](const char* module,
                                             const char* name,
                                             const char* sync,
                                             unsigned sync_args) {
        add(module, name,
            [this, module = std::string(module), sync = std::string(sync),
             sync_args, deliver](CPUContext& cpu, GuestMemory& memory) {
                const HleHandler* handler = find_handler(module, sync);
                if (handler == nullptr) {
                    throw GuestFault("async wrapper without sync handler",
                                     cpu.pc, 4, cpu.pc, MemoryAccess::execute);
                }
                const uint32_t params =
                    stack_argument(cpu, memory, sync_args);
                CPUContext call = cpu;
                (*handler)(call, memory);
                deliver(cpu, memory, cpu.gpr[3], cpu.gpr[4], call.gpr[3],
                        params);
                cpu.gpr[3] = 0;
                return HleAction::return_to_lr;
            });
    };
    async("coreinit", "FSCloseFileAsync", "FSCloseFile", 4);
    async("coreinit", "FSGetStatFileAsync", "FSGetStatFile", 5);
    async("coreinit", "FSReadFileAsync", "FSReadFile", 8);
    async("coreinit", "FSWriteFileAsync", "FSWriteFile", 8);
    async("coreinit", "FSReadFileWithPosAsync", "FSReadFileWithPos", 9);
    async("coreinit", "FSWriteFileWithPosAsync", "FSWriteFileWithPos", 9);
    async("coreinit", "FSOpenFileAsync", "FSOpenFile", 6);
    async("coreinit", "FSGetStatAsync", "FSGetStat", 5);
    async("nn_save", "SAVEOpenFileAsync", "SAVEOpenFile", 7);
    async("nn_save", "SAVEFlushQuotaAsync", "SAVEFlushQuota", 4);
    add("coreinit", "FSGetAsyncResult", [](CPUContext& cpu,
                                           GuestMemory& memory) {
        cpu.gpr[3] = memory.read32(cpu.gpr[3], cpu.pc);
        return HleAction::return_to_lr;
    });

    // ---------------------------------------------------------------- DMAE
    add("dmae", "DMAECopyMem", [this](CPUContext& cpu, GuestMemory& memory) {
        const uint32_t bytes = cpu.gpr[5] * 4;
        std::vector<uint8_t> buffer(bytes);
        memory.read_bytes(cpu.gpr[4], buffer, cpu.pc);
        if (cpu.gpr[6] == 2) { // DMAE_ENDIAN_32: byte-swap each word
            for (size_t i = 0; i + 3 < buffer.size(); i += 4) {
                std::swap(buffer[i], buffer[i + 3]);
                std::swap(buffer[i + 1], buffer[i + 2]);
            }
        } else if (cpu.gpr[6] == 1) { // DMAE_ENDIAN_16
            for (size_t i = 0; i + 1 < buffer.size(); i += 2) {
                std::swap(buffer[i], buffer[i + 1]);
            }
        }
        memory.write_bytes(cpu.gpr[3], buffer, cpu.pc);
        const uint64_t now = machine_ ? machine_->current_time_ticks() : 0;
        cpu.gpr[3] = static_cast<uint32_t>(now >> 32);
        cpu.gpr[4] = static_cast<uint32_t>(now);
        return HleAction::return_to_lr;
    });
    add("dmae", "DMAESemaphore", [this](CPUContext& cpu, GuestMemory& memory) {
        // op 0 = signal (increment), 1 = wait (decrement).
        const uint32_t value = memory.read32(cpu.gpr[3], cpu.pc);
        memory.write32(cpu.gpr[3] + 0,
                       cpu.gpr[4] == 0 ? value + 1 : value - 1, cpu.pc);
        const uint64_t now = machine_ ? machine_->current_time_ticks() : 0;
        cpu.gpr[3] = static_cast<uint32_t>(now >> 32);
        cpu.gpr[4] = static_cast<uint32_t>(now);
        return HleAction::return_to_lr;
    });
    ret("dmae", "DMAEWaitDone", 1);

    // ---------------------------------------------------------------- GX2
    for (const char* symbol :
         {"GX2DebugTagUserString", "GX2DebugTagUserStringVA",
          "GX2SetDRCGamma", "GX2SetTVGamma", "GX2SetDepthOnlyControl",
          "GX2SetLineWidth", "GX2SetMaxTessellationLevel",
          "GX2SetMinTessellationLevel", "GX2SetPointLimits",
          "GX2SetPointSize", "GX2SetPolygonOffset",
          "GX2SetPrimitiveRestartIndex", "GX2SetRasterizerClipControl",
          "GX2SetRasterizerClipControlEx", "GX2SetSemaphore",
          "GX2SetStreamOutEnable", "GX2SetTessellation",
          "GX2SetVertexTexture", "GX2InitPerfMetric", "GX2QueryBegin",
          "GX2QueryEnd", "GX2FreeTilingAperture",
          "GX2ConvertDepthBufferToTextureSurface", "_GX2ResetCounters",
          "_GX2SampleCounters", "_GX2SetCounterInfo", "_GX2StartCounters",
          "_GX2StopCounters"}) {
        noop("gx2", symbol);
    }
    for (const char* symbol : {"GX2GetTVGamma", "GX2GetDRCGamma"}) {
        add("gx2", symbol, [](CPUContext& cpu, GuestMemory& memory) {
            if (cpu.gpr[3] != 0) {
                memory.write32(cpu.gpr[3], 0x3F800000, cpu.pc); // 1.0f
            }
            return HleAction::return_to_lr;
        });
    }
    for (const char* symbol :
         {"GX2GetLastSubmittedTimeStamp", "GX2GetRetiredTimeStamp"}) {
        add("gx2", symbol, [this](CPUContext& cpu, GuestMemory&) {
            const uint64_t now =
                machine_ ? machine_->current_time_ticks() : 0;
            cpu.gpr[3] = static_cast<uint32_t>(now >> 32);
            cpu.gpr[4] = static_cast<uint32_t>(now);
            return HleAction::return_to_lr;
        });
    }
    ret("gx2", "GX2WaitTimeStamp", 1);
    ret("gx2", "GX2GetPerfMetricU64", 0);
    ret("gx2", "GX2GetPerfMetricF32", 0);
    ret("gx2", "GX2GetSurfaceFormatBits", 32);
    add("gx2", "GX2QueryGetOcclusionResult",
        [](CPUContext& cpu, GuestMemory& memory) {
            if (cpu.gpr[4] != 0) {
                memory.write32(cpu.gpr[4], 0, cpu.pc);
                memory.write32(cpu.gpr[4] + 4, 1000, cpu.pc);
            }
            cpu.gpr[3] = 1;
            return HleAction::return_to_lr;
        });
    // (surface, level, slice, endian, u32* handle, u32* address): hand back
    // the raw image, i.e. no detiling — enough for CPU writes to land.
    add("gx2", "GX2AllocateTilingApertureEx",
        [](CPUContext& cpu, GuestMemory& memory) {
            if (cpu.gpr[7] != 0) {
                memory.write32(cpu.gpr[7], 1, cpu.pc);
            }
            if (cpu.gpr[8] != 0) {
                memory.write32(cpu.gpr[8],
                               memory.read32(cpu.gpr[3] + 0x24, cpu.pc),
                               cpu.pc);
            }
            return HleAction::return_to_lr;
        });

    // ---------------------------------------------------------- audio/input
    if (const HleHandler* acquire = find_handler("snd_core",
                                                 "AXAcquireVoiceEx")) {
        add("snd_core", "AXAcquireVoice", *acquire);
    }
    out32("snd_core", "AXGetDRCVSMode", 0, 0, 0);
    out32("snd_core", "AXGetDeviceMode", 1, 0, 0);
    ret("snd_core", "AXIsInit", 1);
    ret("snd_core", "AXSetDRCVSMode", 0);
    noop("snd_core", "AXSetDefaultMixerSelect");
    noop("snd_core", "AXSetVoiceRmtOn");
    noop("snd_core", "AXSetVoiceSrcType");
    noop("snd_user", "MIXInit");
    noop("snd_user", "MIXQuit");
    noop("padscore", "KPADInit");
    noop("padscore", "KPADReset");
    noop("padscore", "WPADControlMotor");
    ret("padscore", "WPADControlSpeaker", 0);
    ret("padscore", "WPADGetBatteryLevel", 4);
    ret("padscore", "WPADGetStatus", 3);
    ret("padscore", "WPADIsMotorEnabled", 0);
    noop("padscore", "WPADResetAutoSleepTimeCount");
    noop("padscore", "WPADSetAutoSleepTime");
    ret("vpad", "VPADControlMotor", 0);
    noop("vpad", "VPADInitGyroDirReviseParam");
    ret("vpad", "VPADSetButtonProcMode", 0);
    ret("nsyskbd", "KBDInit", 0);
    ret("nsyskbd", "KBDSetCountry", 0);
    ret("nsyskbd", "KBDSetLedsRetry", 0);
    ret("nsyskbd", "KBDSetModState", 0);
    ret("nsyskbd", "KBDGetModState", 0);
    // 0 means "an event was written to r4". Devil's Third drains keys with
    // `while (KBDGetKey(ch, &ev) == 0)`, so a 0 here spins forever on stale
    // stack data (492M calls per 100B instructions). Report an empty queue.
    ret("nsyskbd", "KBDGetKey", 1);
    out32("nsyskbd", "KBDGetChannelStatus", 1, 0, 0);
    noop("proc_ui", "ProcUIDrawDoneRelease");
    noop("proc_ui", "ProcUIShutdown");
    ret("sysapp", "SYSLaunchSettings", 0);

    // ------------------------------------------------------------ network
    for (const char* symbol : {"socket_lib_init", "socket_lib_finish"}) {
        ret("nsysnet", symbol, 0);
    }
    for (const char* symbol :
         {"socket", "accept", "bind", "connect", "listen", "recv",
          "recvfrom", "send", "sendto", "setsockopt", "getsockopt",
          "getsockname", "shutdown", "socketclose", "select",
          "getaddrinfo", "NSSLCreateContext", "NSSLCreateConnection",
          "NSSLDoHandshake", "NSSLRead", "NSSLWrite",
          "NSSLAddServerPKIGroups"}) {
        ret("nsysnet", symbol, static_cast<uint32_t>(-1));
    }
    ret("nsysnet", "socketlasterr", 50); // ENETDOWN
    ret("nsysnet", "gethostbyname", 0);
    noop("nsysnet", "freeaddrinfo");
    for (const char* symbol : {"NSSLInit", "NSSLFinish",
                               "NSSLDestroyContext",
                               "NSSLDestroyConnection"}) {
        ret("nsysnet", symbol, 0);
    }
    // Guest is big-endian: network order is host order.
    for (const char* symbol : {"htonl", "htons", "ntohl", "ntohs"}) {
        add("nsysnet", symbol, [](CPUContext&, GuestMemory&) {
            return HleAction::return_to_lr;
        });
    }
    ret("nsysnet", "inet_pton", 0);
    ret("nsysnet", "inet_ntop", 0);
    ret("nsysnet", "inet_ntoa", 0);

    ret("nlibcurl", "curl_global_init", 0);
    noop("nlibcurl", "curl_global_cleanup");
    ret("nlibcurl", "curl_easy_init", 0);
    noop("nlibcurl", "curl_easy_cleanup");
    ret("nlibcurl", "curl_easy_setopt", 0);
    ret("nlibcurl", "curl_easy_perform", 7); // CURLE_COULDNT_CONNECT
    ret("nlibcurl", "curl_easy_getinfo", 0);
    ret("nlibcurl", "curl_multi_init", 0);
    ret("nlibcurl", "curl_multi_cleanup", 0);
    ret("nlibcurl", "curl_multi_add_handle", 0);
    ret("nlibcurl", "curl_multi_remove_handle", 0);
    out32("nlibcurl", "curl_multi_perform", 1, 0, 0);
    ret("nlibcurl", "curl_multi_info_read", 0);
    ret("nlibcurl", "curl_slist_append", 0);
    noop("nlibcurl", "curl_slist_free_all");

    ret("nn_ac", "Initialize__Q2_2nn2acFv", kResultSuccess);
    ret("nn_ac", "Finalize__Q2_2nn2acFv", kResultSuccess);
    ret("nn_ac", "Connect__Q2_2nn2acFv", kResultNotConnected);
    ret("nn_ac", "ConnectAsync__Q2_2nn2acFv", kResultNotConnected);
    ret("nn_ac", "Close__Q2_2nn2acFv", kResultSuccess);
    out32("nn_ac", "GetConnectStatus__Q2_2nn2acFPQ3_2nn2ac6Status", 0, 0,
          kResultSuccess);
    out32("nn_ac", "IsApplicationConnected__Q2_2nn2acFPb", 0, 0,
          kResultSuccess);
    out32("nn_ac", "GetAssignedAddress__Q2_2nn2acFPUl", 0, 0,
          kResultNotConnected);
    out32("nn_ac", "GetAssignedSubnet__Q2_2nn2acFPUl", 0, 0,
          kResultNotConnected);

    ret("nn_act", "Finalize__Q2_2nn3actFv", kResultSuccess);
    ret("nn_act", "GetPrincipalId__Q2_2nn3actFv", 0);
    ret("nn_act", "IsNetworkAccount__Q2_2nn3actFv", 0);
    ret("nn_act", "HasNfsAccount__Q2_2nn3actFv", 0);
    ret("nn_act", "GetErrorCode__Q2_2nn3actFRCQ2_2nn6Result", 0);
    for (const char* symbol : {"GetUtcOffset__Q2_2nn3actFv"}) {
        add("nn_act", symbol, [](CPUContext& cpu, GuestMemory&) {
            cpu.gpr[3] = 0;
            cpu.gpr[4] = 0;
            return HleAction::return_to_lr;
        });
    }
    add("nn_act", "GetTransferableId__Q2_2nn3actFUi",
        [](CPUContext& cpu, GuestMemory&) {
            cpu.gpr[3] = 0x00000000;
            cpu.gpr[4] = 0x00C0FFEE;
            return HleAction::return_to_lr;
        });
    add("nn_act", "GetAccountId__Q2_2nn3actFPc",
        [](CPUContext& cpu, GuestMemory& memory) {
            const char name[] = "Player";
            for (uint32_t i = 0; i < sizeof(name); ++i) {
                memory.write8(cpu.gpr[3] + i, static_cast<uint8_t>(name[i]),
                              cpu.pc);
            }
            cpu.gpr[3] = kResultSuccess;
            return HleAction::return_to_lr;
        });
    for (const char* symbol :
         {"AcquireEcServiceToken__Q2_2nn3actFPc",
          "AcquireIndependentServiceToken__Q2_2nn3actFPcPCc",
          "AcquireNexServiceToken__Q2_2nn3actFP26ACTNexAuthenticationResultUi"}) {
        ret("nn_act", symbol, kResultActNoNetwork);
    }

    ret("nn_fp", "Initialize__Q2_2nn2fpFv", kResultSuccess);
    ret("nn_fp", "Finalize__Q2_2nn2fpFv", kResultSuccess);
    ret("nn_fp", "IsInitialized__Q2_2nn2fpFv", 1);
    ret("nn_fp", "IsOnline__Q2_2nn2fpFv", 0);
    out32("nn_fp", "GetFriendList__Q2_2nn2fpFPUiT1UiT3", 1, 0,
          kResultSuccess);
    out32("nn_fp", "GetBlackList__Q2_2nn2fpFPUiT1UiT3", 1, 0, kResultSuccess);
    ret("nn_fp",
        "AddRecentPlayRecord__Q2_2nn2fpFPCQ3_2nn2fp16RecentPlayRecordUi",
        kResultSuccess);
    ret("nn_fp", "UpdateGameModeDescription__Q2_2nn2fpFPCw", kResultSuccess);
    ret("nn_fp", "RegisterAccountAsync__Q2_2nn2fpFPFQ2_2nn6ResultPv_vPv",
        kResultFpOffline);
    ret("nn_fp", "ResultToErrorCode__Q2_2nn2fpFQ2_2nn6Result", 0);
    for (const char* symbol :
         {"Initialize__Q2_2nn3ndmFv", "Finalize__Q2_2nn3ndmFv",
          "ResumeDaemons__Q2_2nn3ndmFUi", "SuspendDaemons__Q2_2nn3ndmFUi"}) {
        ret("nn_ndm", symbol, kResultSuccess);
    }

    // ---------------------------------------------------------- zlib125
    const auto inflate_call = [](CPUContext& cpu, GuestMemory& memory,
                                 z_stream& stream, int flush) {
        const uint32_t guest = cpu.gpr[3];
        const uint32_t next_in = memory.read32(guest + kZNextIn, cpu.pc);
        const uint32_t avail_in = memory.read32(guest + kZAvailIn, cpu.pc);
        const uint32_t next_out = memory.read32(guest + kZNextOut, cpu.pc);
        const uint32_t avail_out = memory.read32(guest + kZAvailOut, cpu.pc);
        std::vector<uint8_t> input(avail_in);
        std::vector<uint8_t> output(avail_out);
        memory.read_bytes(next_in, input, cpu.pc);
        stream.next_in = input.data();
        stream.avail_in = avail_in;
        stream.next_out = output.data();
        stream.avail_out = avail_out;
        const int rc = inflate(&stream, flush);
        const uint32_t consumed = avail_in - stream.avail_in;
        const uint32_t produced = avail_out - stream.avail_out;
        memory.write_bytes(next_out, std::span(output.data(), produced),
                           cpu.pc);
        memory.write32(guest + kZNextIn, next_in + consumed, cpu.pc);
        memory.write32(guest + kZAvailIn, stream.avail_in, cpu.pc);
        memory.write32(guest + kZTotalIn,
                       memory.read32(guest + kZTotalIn, cpu.pc) + consumed,
                       cpu.pc);
        memory.write32(guest + kZNextOut, next_out + produced, cpu.pc);
        memory.write32(guest + kZAvailOut, stream.avail_out, cpu.pc);
        memory.write32(guest + kZTotalOut,
                       memory.read32(guest + kZTotalOut, cpu.pc) + produced,
                       cpu.pc);
        memory.write32(guest + kZAdler, static_cast<uint32_t>(stream.adler),
                       cpu.pc);
        cpu.gpr[3] = static_cast<uint32_t>(rc);
    };
    add("zlib125", "inflateInit_", [](CPUContext& cpu, GuestMemory& memory) {
        auto stream = std::make_unique<z_stream>();
        const int rc = inflateInit(stream.get());
        memory.write32(cpu.gpr[3] + kZTotalIn, 0, cpu.pc);
        memory.write32(cpu.gpr[3] + kZTotalOut, 0, cpu.pc);
        memory.write32(cpu.gpr[3] + kZMsg, 0, cpu.pc);
        inflate_streams()[cpu.gpr[3]] = std::move(stream);
        cpu.gpr[3] = static_cast<uint32_t>(rc);
        return HleAction::return_to_lr;
    });
    add("zlib125", "inflateInit2_", [](CPUContext& cpu, GuestMemory& memory) {
        auto stream = std::make_unique<z_stream>();
        const int rc = inflateInit2(stream.get(),
                                    static_cast<int32_t>(cpu.gpr[4]));
        memory.write32(cpu.gpr[3] + kZTotalIn, 0, cpu.pc);
        memory.write32(cpu.gpr[3] + kZTotalOut, 0, cpu.pc);
        inflate_streams()[cpu.gpr[3]] = std::move(stream);
        cpu.gpr[3] = static_cast<uint32_t>(rc);
        return HleAction::return_to_lr;
    });
    add("zlib125", "inflate",
        [inflate_call](CPUContext& cpu, GuestMemory& memory) {
            auto found = inflate_streams().find(cpu.gpr[3]);
            if (found == inflate_streams().end()) {
                cpu.gpr[3] = static_cast<uint32_t>(Z_STREAM_ERROR);
                return HleAction::return_to_lr;
            }
            inflate_call(cpu, memory, *found->second,
                         static_cast<int>(cpu.gpr[4]));
            return HleAction::return_to_lr;
        });
    add("zlib125", "inflateReset", [](CPUContext& cpu, GuestMemory& memory) {
        auto found = inflate_streams().find(cpu.gpr[3]);
        int rc = Z_STREAM_ERROR;
        if (found != inflate_streams().end()) {
            rc = inflateReset(found->second.get());
            memory.write32(cpu.gpr[3] + kZTotalIn, 0, cpu.pc);
            memory.write32(cpu.gpr[3] + kZTotalOut, 0, cpu.pc);
        }
        cpu.gpr[3] = static_cast<uint32_t>(rc);
        return HleAction::return_to_lr;
    });
    add("zlib125", "inflateEnd", [](CPUContext& cpu, GuestMemory&) {
        auto found = inflate_streams().find(cpu.gpr[3]);
        int rc = Z_STREAM_ERROR;
        if (found != inflate_streams().end()) {
            rc = inflateEnd(found->second.get());
            inflate_streams().erase(found);
        }
        cpu.gpr[3] = static_cast<uint32_t>(rc);
        return HleAction::return_to_lr;
    });
    // uncompress(dest, destLen*, src, srcLen) / compress2(..., level)
    const auto one_shot = [](bool compressing) {
        return [compressing](CPUContext& cpu, GuestMemory& memory) {
            uLongf dest_len = memory.read32(cpu.gpr[4], cpu.pc);
            std::vector<uint8_t> source(cpu.gpr[6]);
            std::vector<uint8_t> dest(dest_len);
            memory.read_bytes(cpu.gpr[5], source, cpu.pc);
            const int level = compressing && cpu.gpr[7] <= 9
                                  ? static_cast<int>(cpu.gpr[7])
                                  : Z_DEFAULT_COMPRESSION;
            const int rc =
                compressing
                    ? compress2(dest.data(), &dest_len, source.data(),
                                source.size(), level)
                    : uncompress(dest.data(), &dest_len, source.data(),
                                 source.size());
            if (rc == Z_OK) {
                memory.write_bytes(cpu.gpr[3],
                                   std::span(dest.data(), dest_len), cpu.pc);
                memory.write32(cpu.gpr[4], static_cast<uint32_t>(dest_len),
                               cpu.pc);
            }
            cpu.gpr[3] = static_cast<uint32_t>(rc);
            return HleAction::return_to_lr;
        };
    };
    add("zlib125", "uncompress", one_shot(false));
    add("zlib125", "compress2", one_shot(true));
    add("zlib125", "compress", [one_shot](CPUContext& cpu,
                                          GuestMemory& memory) {
        cpu.gpr[7] = static_cast<uint32_t>(Z_DEFAULT_COMPRESSION);
        return one_shot(true)(cpu, memory);
    });
}
} // namespace nwii::runtime
