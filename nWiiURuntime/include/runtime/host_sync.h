#pragma once

// Host-side synchronisation for running the three Cafe cores on separate host
// threads (NWIIU_PARALLEL=1).
//
// - One recursive "HLE lock" serialises every HLE import and all scheduler
//   bookkeeping. Guest code itself runs unlocked; guest atomics go through
//   GuestMemory::compare_exchange32.
// - The SDL window and GPU renderer belong to the main host thread. The main
//   thread is itself one of the core workers and prefers the guest thread that
//   drives GX2, so most render events stay on it. A GX2 call that lands on
//   another worker is handed to the main thread with run_on_main(), which
//   blocks the caller (still holding the HLE lock) until it has run.
// - Because a caller may hold the HLE lock while it waits on the main thread,
//   the main thread never blocks on that lock: HleLock spins on try_lock and
//   services handed-over work in between.

#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>

namespace nwii::runtime::host_sync {

// NWIIU_PARALLEL=1, read once.
bool parallel();

// Records the calling thread as the main (SDL) thread. Called from the
// Machine constructor, which runs on it.
void mark_main_thread();
bool on_main_thread();

std::recursive_mutex& hle_mutex();

// Acquires the HLE lock; deadlock-safe on the main thread (see above). A
// no-op unless parallel().
class HleLock {
public:
    HleLock();
    ~HleLock();
    HleLock(const HleLock&) = delete;
    HleLock& operator=(const HleLock&) = delete;

private:
    bool held_{};
};

// Runs `work` on the main thread and waits for it. Inline when already on
// the main thread or when not parallel.
void run_on_main(const std::function<void()>& work);

// True once (then cleared) if the calling host thread handed work to the main
// thread since the last call. The scheduler uses it to move the guest thread
// that did so onto the main worker.
bool take_marshalled();

// Main thread: run every pending handed-over request.
void service_main();

// Wakes every idle worker (a thread became ready, or a stop was requested).
void kick();

// Waits, without holding the HLE lock, until kick(), a handed-over request
// (main thread only, which it then services) or the timeout.
void idle_wait(std::chrono::microseconds timeout);

// The Machine registers how to stop its core workers; the renderer calls
// shutdown() first thing in its destructor so no worker hands it an event
// after it is gone.
void set_shutdown(std::function<void()> stop);
void shutdown();

} // namespace nwii::runtime::host_sync
