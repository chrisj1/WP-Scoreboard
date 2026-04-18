#include <obs-module.h>
#include <graphics/image-file.h>
#include <util/platform.h>
#include <util/dstr.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>

#ifdef USE_CNN_OCR
#include "clock-ocr-engine.h"
#ifdef __APPLE__
#undef NO
#undef YES
#endif
#include <opencv2/opencv.hpp>
#endif

struct averaged_frame_viz_source {
	obs_source_t *source;
	gs_texture_t *texture;

	uint32_t width;
	uint32_t height;

#ifdef USE_CNN_OCR
	// Pre-rendered BGRA bytes produced on the Qt/update side.
	// The render callback (graphics thread) only needs to upload them to GPU.
	std::vector<uint8_t> pending_bytes;
	uint32_t             pending_w = 0;
	uint32_t             pending_h = 0;
	std::mutex           data_mutex;   // protects pending_bytes/w/h
	bool                 needs_update = false;
#endif

	bool data_updated = false;
};

#ifdef USE_CNN_OCR
static void register_averaged_frame_viz_source(averaged_frame_viz_source* context);
static void unregister_averaged_frame_viz_source(averaged_frame_viz_source* context);
#endif

static const char *averaged_frame_viz_source_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Clock Averaged Frame Visualization";
}

static void averaged_frame_viz_source_update(void *data, obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
	struct averaged_frame_viz_source *context = (struct averaged_frame_viz_source *)data;
	context->width  = 800;
	context->height = 400;
}

static void *averaged_frame_viz_source_create(obs_data_t *settings, obs_source_t *source)
{
	// Use new so C++ members (std::mutex, std::vector) are properly constructed.
	struct averaged_frame_viz_source *context = new averaged_frame_viz_source();
	context->source  = source;
	context->width   = 800;
	context->height  = 400;
	context->texture = nullptr;

#ifdef USE_CNN_OCR
	register_averaged_frame_viz_source(context);
#endif

	UNUSED_PARAMETER(settings);
	return context;
}

static void averaged_frame_viz_source_destroy(void *data)
{
	struct averaged_frame_viz_source *context = (struct averaged_frame_viz_source *)data;

#ifdef USE_CNN_OCR
	unregister_averaged_frame_viz_source(context);
#endif

	if (context->texture) {
		obs_enter_graphics();
		gs_texture_destroy(context->texture);
		obs_leave_graphics();
	}

	delete context;
}

static void averaged_frame_viz_source_get_defaults(obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
}

static obs_properties_t *averaged_frame_viz_source_get_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();
	obs_properties_add_text(props, "info",
		"Shows the averaged frames before CNN processing",
		OBS_TEXT_INFO);
	return props;
}

#ifdef USE_CNN_OCR
static void draw_frame_with_label(cv::Mat& dst, const cv::Mat& frame,
                                  int x, int y, int w, int h,
                                  const std::string& label)
{
	cv::rectangle(dst, cv::Rect(x-2, y-2, w+4, h+4), cv::Scalar(100,100,100,255), 2);
	cv::putText(dst, label, cv::Point(x, y-10),
	            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255,255), 2);

	if (!frame.empty()) {
		cv::Mat resized;
		cv::resize(frame, resized, cv::Size(w, h));
		cv::Mat bgra;
		if (resized.channels() == 1)
			cv::cvtColor(resized, bgra, cv::COLOR_GRAY2BGRA);
		else if (resized.channels() == 3)
			cv::cvtColor(resized, bgra, cv::COLOR_BGR2BGRA);
		else
			bgra = resized;

		cv::Rect roi(x, y, w, h);
		if (roi.x >= 0 && roi.y >= 0 &&
		    roi.x + roi.width  <= dst.cols &&
		    roi.y + roi.height <= dst.rows)
			bgra.copyTo(dst(roi));
	} else {
		cv::rectangle(dst, cv::Rect(x, y, w, h), cv::Scalar(50,50,50,255), -1);
		cv::putText(dst, "No Data", cv::Point(x + w/2 - 50, y + h/2),
		            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(200,200,200,255), 2);
	}
}
#endif

static void averaged_frame_viz_source_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct averaged_frame_viz_source *context = (struct averaged_frame_viz_source *)data;
	if (!context) return;

#ifdef USE_CNN_OCR
	if (context->needs_update) {
		// Try to grab pre-rendered bytes without blocking the graphics thread.
		// If the update side currently holds the lock, skip this frame.
		std::unique_lock<std::mutex> lock(context->data_mutex, std::try_to_lock);
		if (lock.owns_lock() && !context->pending_bytes.empty()) {
			uint32_t w = context->pending_w;
			uint32_t h = context->pending_h;
			std::vector<uint8_t> bytes = std::move(context->pending_bytes);
			context->needs_update = false;
			lock.unlock(); // release before GPU work

			// We are already on the graphics thread — no obs_enter/leave needed.
			if (context->texture) {
				gs_texture_destroy(context->texture);
				context->texture = nullptr;
			}
			const uint8_t *ptr = bytes.data();
			context->texture = gs_texture_create(w, h, GS_BGRA, 1, &ptr, 0);
		}
	}
#endif

	if (context->texture) {
		gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		if (default_effect) {
			gs_effect_set_texture(
				gs_effect_get_param_by_name(default_effect, "image"),
				context->texture);
			while (gs_effect_loop(default_effect, "Draw"))
				gs_draw_sprite(context->texture, 0, context->width, context->height);
		}
	}
}

static uint32_t averaged_frame_viz_source_get_width(void *data)
{
	return ((struct averaged_frame_viz_source *)data)->width;
}

static uint32_t averaged_frame_viz_source_get_height(void *data)
{
	return ((struct averaged_frame_viz_source *)data)->height;
}

struct obs_source_info averaged_frame_viz_source_info = {};

void init_averaged_frame_viz_source_info()
{
	averaged_frame_viz_source_info.id             = "averaged_frame_viz_source";
	averaged_frame_viz_source_info.type           = OBS_SOURCE_TYPE_INPUT;
	averaged_frame_viz_source_info.output_flags   = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
	averaged_frame_viz_source_info.get_name       = averaged_frame_viz_source_get_name;
	averaged_frame_viz_source_info.create         = averaged_frame_viz_source_create;
	averaged_frame_viz_source_info.destroy        = averaged_frame_viz_source_destroy;
	averaged_frame_viz_source_info.update         = averaged_frame_viz_source_update;
	averaged_frame_viz_source_info.get_properties = averaged_frame_viz_source_get_properties;
	averaged_frame_viz_source_info.get_defaults   = averaged_frame_viz_source_get_defaults;
	averaged_frame_viz_source_info.video_render   = averaged_frame_viz_source_render;
	averaged_frame_viz_source_info.get_width      = averaged_frame_viz_source_get_width;
	averaged_frame_viz_source_info.get_height     = averaged_frame_viz_source_get_height;
}

#ifdef USE_CNN_OCR
static std::vector<averaged_frame_viz_source*> g_averaged_frame_viz_sources;
static std::mutex g_averaged_frame_viz_mutex;

static void register_averaged_frame_viz_source(averaged_frame_viz_source* context)
{
	std::lock_guard<std::mutex> lock(g_averaged_frame_viz_mutex);
	g_averaged_frame_viz_sources.push_back(context);
}

static void unregister_averaged_frame_viz_source(averaged_frame_viz_source* context)
{
	std::lock_guard<std::mutex> lock(g_averaged_frame_viz_mutex);
	auto it = std::find(g_averaged_frame_viz_sources.begin(),
	                    g_averaged_frame_viz_sources.end(), context);
	if (it != g_averaged_frame_viz_sources.end())
		g_averaged_frame_viz_sources.erase(it);
}

void update_averaged_frame_viz_data(
	const cv::Mat& shot_averaged_frame,
	const cv::Mat& game_averaged_frame)
{
	// Build the composite image entirely on the calling (Qt) thread —
	// no OpenCV on the graphics thread, no heavy work inside the mutex.
	const int IMG_W = 800, IMG_H = 400;
	cv::Mat img(IMG_H, IMG_W, CV_8UC4, cv::Scalar(30, 30, 30, 255));

	const int margin       = 20;
	const int frame_width  = 360;
	const int frame_height = 320;

	draw_frame_with_label(img, shot_averaged_frame,
	                      margin, margin + 30, frame_width, frame_height,
	                      "Shot Clock Averaged Frame");
	draw_frame_with_label(img, game_averaged_frame,
	                      margin + frame_width + margin, margin + 30,
	                      frame_width, frame_height,
	                      "Game Clock Averaged Frame");
	cv::putText(img, "CNN Input Frames (5-Frame Average)", cv::Point(250, 25),
	            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255,255), 2);

	// Convert to a flat BGRA byte buffer.
	std::vector<uint8_t> bytes(IMG_W * IMG_H * 4);
	// img is already CV_8UC4 BGRA — just copy row by row to handle stride.
	for (int row = 0; row < IMG_H; row++)
		memcpy(bytes.data() + row * IMG_W * 4, img.ptr(row), IMG_W * 4);

	// Push bytes to all active sources under a brief lock.
	std::lock_guard<std::mutex> global_lock(g_averaged_frame_viz_mutex);
	for (auto* ctx : g_averaged_frame_viz_sources) {
		if (!ctx) continue;
		std::lock_guard<std::mutex> lock(ctx->data_mutex);
		ctx->pending_bytes  = bytes;          // copy (cheap for small image)
		ctx->pending_w      = IMG_W;
		ctx->pending_h      = IMG_H;
		ctx->needs_update   = true;
	}
}
#endif
