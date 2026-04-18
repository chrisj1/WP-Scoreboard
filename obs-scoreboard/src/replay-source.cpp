// replay-source.cpp
// "Instant Replay Camera" OBS async-video source.
//
// On show(): looks up the configured source name in g_replay_registry,
// snapshots the last N seconds of buffered frames, and plays them back at
// the chosen speed via obs_source_output_video.

#include "replay-shared.h"

#include <obs-module.h>
#include <util/platform.h>

#include <string>
#include <deque>
#include <vector>
#include <mutex>
#include <algorithm>
#include <cstring>

#define DEFAULT_REPLAY_DURATION_SEC 10.0
#define MIN_REPLAY_DURATION_SEC     1.0
#define MAX_REPLAY_DURATION_SEC     60.0

// ─── Source state ─────────────────────────────────────────────────────────────

struct ReplaySourceCtx {
	obs_source_t *source = nullptr;

	// Settings
	std::string source_name;   // name of the source with Replay Buffer filter
	float       playback_speed = 0.5f;
	double      replay_duration_sec = DEFAULT_REPLAY_DURATION_SEC;

	// Snapshot — captured on trigger, played on show.
	// snapshot_mutex must be held when reading or writing snapshot,
	// play_index, accum_ns, or playing.
	std::mutex  snapshot_mutex;
	std::vector<obs_source_frame *> snapshot; // frames to play back
	bool        snapshot_ready = false;       // true after trigger_replay_snapshot()
	size_t      play_index = 0;               // current position in snapshot

	// Accumulated time (nanoseconds) used to pace playback
	uint64_t    accum_ns   = 0;
	bool        playing    = false;

	// Dimensions (from snapshot frames, or fallback)
	uint32_t    width  = 1920;
	uint32_t    height = 1080;
};

// ─── Global list of all active replay source instances ────────────────────────

static std::mutex              g_sources_mutex;
static std::vector<ReplaySourceCtx *> g_sources;

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Caller must hold ctx->snapshot_mutex.
static void free_snapshot_locked(ReplaySourceCtx *ctx)
{
	for (auto *f : ctx->snapshot)
		obs_source_frame_destroy(f);
	ctx->snapshot.clear();
	ctx->play_index = 0;
	ctx->accum_ns   = 0;
	ctx->playing    = false;
}

// Take a deep copy of the last `replay_duration_sec` seconds from the filter's
// frame deque. Holds ctx->snapshot_mutex for the full duration so video_tick
// cannot read a partially-replaced snapshot.
static void take_snapshot(ReplaySourceCtx *ctx)
{
	// Build the new snapshot into a temporary vector so we hold
	// snapshot_mutex for as short a time as possible.
	std::string source_name;
	double replay_sec;
	float  playback_speed;
	{
		std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);
		source_name    = ctx->source_name;
		replay_sec     = ctx->replay_duration_sec;
		playback_speed = ctx->playback_speed;
	}

	// Grab frames from the filter registry (hold registry + frames locks).
	std::vector<obs_source_frame *> new_frames;
	uint32_t new_width = 0, new_height = 0;

	{
		std::lock_guard<std::mutex> rlock(g_replay_registry_mutex);
		auto it = g_replay_registry.find(source_name);
		if (it == g_replay_registry.end()) {
			blog(LOG_WARNING,
			     "[ReplaySource] Source '%s' not found in registry. "
			     "Make sure the 'Replay Buffer' filter is attached to it.",
			     source_name.c_str());
			return;
		}

		ReplayFilterCtx *fctx = it->second;
		std::lock_guard<std::mutex> flock(fctx->frames_mutex);

		if (fctx->frames.empty()) {
			blog(LOG_WARNING, "[ReplaySource] Buffer is empty for '%s'",
			     source_name.c_str());
			return;
		}

		uint64_t want_ns = (uint64_t)(replay_sec * 1e9);
		uint64_t newest  = fctx->frames.back()->timestamp;
		uint64_t cutoff  = (newest >= want_ns) ? (newest - want_ns) : 0;

		size_t start = 0;
		for (size_t i = 0; i < fctx->frames.size(); ++i) {
			if (fctx->frames[i]->timestamp >= cutoff) {
				start = i;
				break;
			}
		}

		for (size_t i = start; i < fctx->frames.size(); ++i) {
			obs_source_frame *src  = fctx->frames[i];
			obs_source_frame *copy = obs_source_frame_create(
				src->format, src->width, src->height);
			if (copy) {
				obs_source_frame_copy(copy, src);
				copy->timestamp = src->timestamp;
				new_frames.push_back(copy);
			}
		}
	}

	if (new_frames.empty())
		return;

	new_width  = new_frames[0]->width;
	new_height = new_frames[0]->height;

	// Swap in the new snapshot atomically under snapshot_mutex.
	{
		std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);
		free_snapshot_locked(ctx);
		ctx->snapshot   = std::move(new_frames);
		ctx->width      = new_width;
		ctx->height     = new_height;
		ctx->playing    = true;
		ctx->play_index = 0;
		ctx->accum_ns   = 0;
	}

	blog(LOG_INFO,
	     "[ReplaySource] Snapshot: %zu frames from '%s' (speed=%.2f)",
	     ctx->snapshot.size(), source_name.c_str(),
	     (double)playback_speed);
}

// ─── OBS source callbacks ─────────────────────────────────────────────────────

static const char *replay_source_get_name(void *)
{
	return "Instant Replay Camera";
}

static void *replay_source_create(obs_data_t *settings, obs_source_t *source)
{
	auto *ctx      = new ReplaySourceCtx();
	ctx->source    = source;
	ctx->source_name        = obs_data_get_string(settings, "source_name");
	ctx->playback_speed     = (float)obs_data_get_double(settings, "playback_speed");
	ctx->replay_duration_sec = obs_data_get_double(settings, "replay_duration_sec");
	if (ctx->playback_speed <= 0.0f)
		ctx->playback_speed = 0.5f;
	if (ctx->replay_duration_sec <= 0.0)
		ctx->replay_duration_sec = DEFAULT_REPLAY_DURATION_SEC;

	std::lock_guard<std::mutex> lock(g_sources_mutex);
	g_sources.push_back(ctx);
	return ctx;
}

static void replay_source_destroy(void *data)
{
	auto *ctx = static_cast<ReplaySourceCtx *>(data);

	{
		std::lock_guard<std::mutex> lock(g_sources_mutex);
		g_sources.erase(
			std::remove(g_sources.begin(), g_sources.end(), ctx),
			g_sources.end());
	}

	{
		std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);
		free_snapshot_locked(ctx);
	}
	delete ctx;
}

static void push_buffer_requirement(ReplaySourceCtx *ctx)
{
	if (ctx->source_name.empty()) return;
	// Buffer just needs to hold replay_duration_sec of real-time footage
	uint64_t required_ns = (uint64_t)(ctx->replay_duration_sec * 1e9);
	ensure_filter_buffer(ctx->source_name, required_ns);
}

static void replay_source_update(void *data, obs_data_t *settings)
{
	auto *ctx = static_cast<ReplaySourceCtx *>(data);
	ctx->source_name         = obs_data_get_string(settings, "source_name");
	ctx->playback_speed      = (float)obs_data_get_double(settings, "playback_speed");
	ctx->replay_duration_sec = obs_data_get_double(settings, "replay_duration_sec");
	if (ctx->playback_speed <= 0.0f)
		ctx->playback_speed = 0.5f;
	if (ctx->replay_duration_sec <= 0.0)
		ctx->replay_duration_sec = DEFAULT_REPLAY_DURATION_SEC;

	// Auto-configure the camera filter's buffer to fit our needs
	push_buffer_requirement(ctx);
}

// show() fires when the source becomes visible → start playback of saved snapshot.
// If no snapshot has been triggered yet, take one now as a fallback.
static void replay_source_show(void *data)
{
	auto *ctx = static_cast<ReplaySourceCtx *>(data);
	bool has_snapshot;
	{
		std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);
		has_snapshot = ctx->snapshot_ready && !ctx->snapshot.empty();
	}
	if (!has_snapshot) {
		// Fallback: capture now (no prior trigger)
		take_snapshot(ctx);
		std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);
		ctx->snapshot_ready = !ctx->snapshot.empty();
	} else {
		// Restart playback from the saved snapshot
		std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);
		ctx->play_index = 0;
		ctx->accum_ns   = 0;
		ctx->playing    = !ctx->snapshot.empty();
	}
}

// video_tick: output one frame per tick, advance to next frame only when
// enough wall-clock time has elapsed (source frame duration / playback speed).
// This guarantees every captured frame is displayed — no frames are skipped.
static void replay_source_video_tick(void *data, float seconds)
{
	auto *ctx = static_cast<ReplaySourceCtx *>(data);

	std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);

	if (!ctx->playing || ctx->snapshot.empty())
		return;

	if (ctx->play_index >= ctx->snapshot.size()) {
		ctx->play_index = 0;
		ctx->accum_ns   = 0;
	}

	// Always output the current frame this tick so OBS has something to show.
	obs_source_frame out = *ctx->snapshot[ctx->play_index];
	out.timestamp = os_gettime_ns();
	obs_source_output_video(ctx->source, &out);

	// Accumulate real wall-clock time.
	ctx->accum_ns += (uint64_t)(seconds * 1e9);

	// How long should this frame be displayed?
	// = source frame duration / playback_speed
	// e.g. a 16.6 ms source frame at 0.5x plays for 33.3 ms.
	size_t   next_idx = ctx->play_index + 1;
	uint64_t cur_ts   = ctx->snapshot[ctx->play_index]->timestamp;
	uint64_t next_ts  = (next_idx < ctx->snapshot.size())
	                    ? ctx->snapshot[next_idx]->timestamp
	                    : cur_ts + 33333333ULL; // fallback ~30 fps for last frame
	uint64_t src_dur_ns     = (next_ts > cur_ts) ? (next_ts - cur_ts) : 33333333ULL;
	uint64_t display_dur_ns = (uint64_t)(src_dur_ns / ctx->playback_speed);

	if (ctx->accum_ns >= display_dur_ns) {
		ctx->accum_ns -= display_dur_ns;
		ctx->play_index = next_idx;
		if (ctx->play_index >= ctx->snapshot.size()) {
			// Loop back to start
			ctx->play_index = 0;
			ctx->accum_ns   = 0;
		}
	}
}

static uint32_t replay_source_get_width(void *data)
{
	auto *ctx = static_cast<ReplaySourceCtx *>(data);
	return ctx->width > 0 ? ctx->width : 1920;
}

static uint32_t replay_source_get_height(void *data)
{
	auto *ctx = static_cast<ReplaySourceCtx *>(data);
	return ctx->height > 0 ? ctx->height : 1080;
}

// ─── Properties ───────────────────────────────────────────────────────────────

struct EnumSourcesParam {
	obs_property_t *list;
};

static bool enum_sources_cb(void *param, obs_source_t *source)
{
	auto *p = static_cast<EnumSourcesParam *>(param);
	const char *name = obs_source_get_name(source);
	if (name && *name)
		obs_property_list_add_string(p->list, name, name);
	return true;
}

static obs_properties_t *replay_source_get_properties(void * /*data*/)
{
	obs_properties_t *props = obs_properties_create();

	obs_property_t *list = obs_properties_add_list(
		props, "source_name", "Camera Source (with Replay Buffer filter)",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);

	// Add a blank entry first
	obs_property_list_add_string(list, "(select a source)", "");

	// Enumerate all input sources in the current scene collection
	EnumSourcesParam param{list};
	obs_enum_sources(enum_sources_cb, &param);

	obs_properties_add_float_slider(props, "replay_duration_sec",
		"Replay Duration (seconds)",
		MIN_REPLAY_DURATION_SEC, MAX_REPLAY_DURATION_SEC, 1.0);

	obs_property_t *speed = obs_properties_add_list(
		props, "playback_speed", "Playback Speed",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_FLOAT);
	obs_property_list_add_float(speed, "0.25x  (quarter speed)", 0.25);
	obs_property_list_add_float(speed, "0.5x   (half speed)",    0.5);
	obs_property_list_add_float(speed, "0.75x  (3/4 speed)",     0.75);
	obs_property_list_add_float(speed, "1.0x   (normal speed)",  1.0);

	return props;
}

static void replay_source_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_string(settings, "source_name", "");
	obs_data_set_default_double(settings, "replay_duration_sec", DEFAULT_REPLAY_DURATION_SEC);
	obs_data_set_default_double(settings, "playback_speed", 0.5);
}

// ─── Public API ───────────────────────────────────────────────────────────────

// Called from WebSocket "trigger_replay" command.
// Snapshots the buffer for every active Instant Replay Camera source.
void trigger_replay_snapshot()
{
	std::lock_guard<std::mutex> lock(g_sources_mutex);
	for (auto *ctx : g_sources) {
		// Re-push buffer requirement in case filter was registered after source
		push_buffer_requirement(ctx);

		take_snapshot(ctx);
		size_t frame_count;
		{
			std::lock_guard<std::mutex> slock(ctx->snapshot_mutex);
			ctx->snapshot_ready = !ctx->snapshot.empty();
			frame_count = ctx->snapshot.size();
		}
		blog(LOG_INFO, "[ReplaySource] Snapshot triggered for '%s' (%zu frames)",
		     ctx->source_name.c_str(), frame_count);
	}
}

// ─── Registration ─────────────────────────────────────────────────────────────

void register_replay_source()
{
	struct obs_source_info info = {};
	info.id             = "instant_replay_camera";
	info.type           = OBS_SOURCE_TYPE_INPUT;
	info.output_flags   = OBS_SOURCE_ASYNC_VIDEO | OBS_SOURCE_DO_NOT_DUPLICATE;
	info.get_name       = replay_source_get_name;
	info.create         = replay_source_create;
	info.destroy        = replay_source_destroy;
	info.update         = replay_source_update;
	info.show           = replay_source_show;
	info.video_tick     = replay_source_video_tick;
	info.get_width      = replay_source_get_width;
	info.get_height     = replay_source_get_height;
	info.get_properties = replay_source_get_properties;
	info.get_defaults   = replay_source_get_defaults;

	obs_register_source(&info);
	blog(LOG_INFO, "[ReplaySource] Registered source: instant_replay_camera");
}
