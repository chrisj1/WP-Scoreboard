#pragma once

#include <obs-module.h>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>

// Shared state for the replay buffer filter.
// One instance exists per filter attachment.
struct ReplayFilterCtx {
	obs_source_t        *filter_source = nullptr;
	std::string          parent_name;
	std::deque<obs_source_frame *> frames;
	std::mutex           frames_mutex;
	uint64_t             max_ns = 60ULL * 1000000000ULL;
	bool                 parent_resolved = false;
};

// Global registry: source name → filter context.
// Populated by replay-filter.cpp, consumed by replay-source.cpp.
extern std::mutex g_replay_registry_mutex;
extern std::unordered_map<std::string, ReplayFilterCtx *> g_replay_registry;

void register_replay_filter();
void register_replay_source();

// Call this to snapshot all active replay sources right now.
// Triggered by WebSocket "trigger_replay" command or hotkey.
void trigger_replay_snapshot();

// Ensure the filter for `source_name` has at least `required_ns` of buffer.
// Called by replay sources so they auto-configure their camera's filter.
// No-op if the filter isn't registered yet (called again at trigger time).
void ensure_filter_buffer(const std::string &source_name, uint64_t required_ns);
