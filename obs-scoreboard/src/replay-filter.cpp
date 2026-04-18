// replay-filter.cpp
// "Replay Buffer" OBS filter.
//
// Attach this filter to any video source (camera, capture card, etc.).
// It deep-copies every frame into a rolling deque and registers itself in
// g_replay_registry so that replay-source.cpp can find it by source name.

#include "replay-shared.h"

#include <obs-module.h>
#include <util/platform.h>

#include <string>
#include <mutex>
#include <unordered_map>

// ─── Global registry definition ───────────────────────────────────────────────

std::mutex g_replay_registry_mutex;
std::unordered_map<std::string, ReplayFilterCtx *> g_replay_registry;

// ─── Filter callbacks ─────────────────────────────────────────────────────────

static const char *replay_filter_get_name(void *)
{
	return "Replay Buffer";
}

static void *replay_filter_create(obs_data_t *settings, obs_source_t *filter)
{
	auto *ctx          = new ReplayFilterCtx();
	ctx->filter_source = filter;

	uint64_t dur_sec = (uint64_t)obs_data_get_int(settings, "buffer_duration");
	if (dur_sec < 10)  dur_sec = 60;
	ctx->max_ns = dur_sec * 1000000000ULL;

	// We cannot call obs_filter_get_parent() here because the filter has not
	// been attached to its parent yet at create time. We resolve lazily in
	// filter_video on the first call.

	blog(LOG_INFO, "[ReplayFilter] Created (duration=%llus)",
	     (unsigned long long)dur_sec);
	return ctx;
}

static void replay_filter_destroy(void *data)
{
	auto *ctx = static_cast<ReplayFilterCtx *>(data);

	// Unregister from global map
	{
		std::lock_guard<std::mutex> lock(g_replay_registry_mutex);
		if (!ctx->parent_name.empty())
			g_replay_registry.erase(ctx->parent_name);
	}

	// Free all buffered frames
	{
		std::lock_guard<std::mutex> lock(ctx->frames_mutex);
		for (auto *f : ctx->frames)
			obs_source_frame_destroy(f);
		ctx->frames.clear();
	}

	blog(LOG_INFO, "[ReplayFilter] Destroyed (source=%s)",
	     ctx->parent_name.c_str());
	delete ctx;
}

static void replay_filter_update(void *data, obs_data_t *settings)
{
	auto *ctx = static_cast<ReplayFilterCtx *>(data);
	uint64_t dur_sec = (uint64_t)obs_data_get_int(settings, "buffer_duration");
	if (dur_sec < 10) dur_sec = 60;

	std::lock_guard<std::mutex> lock(ctx->frames_mutex);
	ctx->max_ns = dur_sec * 1000000000ULL;
}

// Pass-through filter: copy each frame into the ring buffer and forward it
// unchanged to the next filter / source output.
static struct obs_source_frame *replay_filter_video(
	void *data, struct obs_source_frame *frame)
{
	auto *ctx = static_cast<ReplayFilterCtx *>(data);

	// Lazily resolve the parent source name on the first video call.
	if (!ctx->parent_resolved) {
		obs_source_t *parent = obs_filter_get_parent(ctx->filter_source);
		if (parent) {
			const char *name = obs_source_get_name(parent);
			if (name && name[0] != '\0') {
				ctx->parent_name     = name;
				ctx->parent_resolved = true;

				// Register in global map
				std::lock_guard<std::mutex> rlock(g_replay_registry_mutex);
				g_replay_registry[ctx->parent_name] = ctx;
				blog(LOG_INFO,
				     "[ReplayFilter] Registered for source '%s'",
				     ctx->parent_name.c_str());
			}
		}
	}

	if (frame) {
		// Deep-copy the frame
		obs_source_frame *copy = obs_source_frame_create(
			frame->format, frame->width, frame->height);
		if (copy) {
			obs_source_frame_copy(copy, frame);
			copy->timestamp = frame->timestamp;

			std::lock_guard<std::mutex> lock(ctx->frames_mutex);

			ctx->frames.push_back(copy);

			// Trim frames older than max_ns from the front
			while (ctx->frames.size() > 1) {
				uint64_t newest = ctx->frames.back()->timestamp;
				uint64_t oldest = ctx->frames.front()->timestamp;
				if (newest >= oldest &&
				    (newest - oldest) > ctx->max_ns) {
					obs_source_frame_destroy(ctx->frames.front());
					ctx->frames.pop_front();
				} else {
					break;
				}
			}
		}
	}

	// Pass through – do not consume the frame
	return frame;
}

// ─── Public API ───────────────────────────────────────────────────────────────

void ensure_filter_buffer(const std::string &source_name, uint64_t required_ns)
{
	std::lock_guard<std::mutex> rlock(g_replay_registry_mutex);
	auto it = g_replay_registry.find(source_name);
	if (it == g_replay_registry.end()) return;

	ReplayFilterCtx *fctx = it->second;
	std::lock_guard<std::mutex> flock(fctx->frames_mutex);
	if (fctx->max_ns < required_ns) {
		fctx->max_ns = required_ns;
		blog(LOG_INFO,
		     "[ReplayFilter] Auto-expanded buffer for '%s' to %llus",
		     source_name.c_str(),
		     (unsigned long long)(required_ns / 1000000000ULL));
	}
}

// ─── Properties ───────────────────────────────────────────────────────────────

static obs_properties_t *replay_filter_get_properties(void * /*data*/)
{
	obs_properties_t *props = obs_properties_create();
	obs_properties_add_int(props, "buffer_duration",
	                       "Min Buffer Duration (sec) — auto-expanded by replay sources",
	                       10, 300, 1);
	return props;
}

static void replay_filter_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "buffer_duration", 60);
}

// ─── Registration ─────────────────────────────────────────────────────────────

void register_replay_filter()
{
	struct obs_source_info info = {};
	info.id             = "replay_camera_filter";
	info.type           = OBS_SOURCE_TYPE_FILTER;
	info.output_flags   = OBS_SOURCE_VIDEO | OBS_SOURCE_ASYNC;
	info.get_name       = replay_filter_get_name;
	info.create         = replay_filter_create;
	info.destroy        = replay_filter_destroy;
	info.update         = replay_filter_update;
	info.filter_video   = replay_filter_video;
	info.get_properties = replay_filter_get_properties;
	info.get_defaults   = replay_filter_get_defaults;

	obs_register_source(&info);
	blog(LOG_INFO, "[ReplayFilter] Registered filter: replay_camera_filter");
}
