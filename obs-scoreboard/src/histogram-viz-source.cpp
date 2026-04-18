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

struct histogram_viz_source {
	obs_source_t *source;
	gs_texture_t *texture;

	uint32_t width;
	uint32_t height;

#ifdef USE_CNN_OCR
	// Pre-rendered BGRA bytes produced on the Qt/update side.
	std::vector<uint8_t> pending_bytes;
	uint32_t             pending_w = 0;
	uint32_t             pending_h = 0;
	std::mutex           data_mutex;
	bool                 needs_update = false;
#endif

	bool data_updated = false;
};

static const char *histogram_viz_source_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Clock Detection Histogram Visualization";
}

static void histogram_viz_source_update(void *data, obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
	struct histogram_viz_source *context = (struct histogram_viz_source *)data;
	context->width  = 1200;
	context->height = 800;
}

static void *histogram_viz_source_create(obs_data_t *settings, obs_source_t *source)
{
	struct histogram_viz_source *context = new histogram_viz_source();
	context->source  = source;
	context->width   = 1200;
	context->height  = 800;
	context->texture = nullptr;

	histogram_viz_source_update(context, settings);
	return context;
}

static void histogram_viz_source_destroy(void *data)
{
	struct histogram_viz_source *context = (struct histogram_viz_source *)data;
	if (context) {
		if (context->texture) {
			obs_enter_graphics();
			gs_texture_destroy(context->texture);
			obs_leave_graphics();
		}
		delete context;
	}
}

static uint32_t histogram_viz_source_get_width(void *data)
{
	return ((struct histogram_viz_source *)data)->width;
}

static uint32_t histogram_viz_source_get_height(void *data)
{
	return ((struct histogram_viz_source *)data)->height;
}

#ifdef USE_CNN_OCR
static void draw_histogram_cv(cv::Mat& img, const std::vector<float>& data,
                               int x, int y, int w, int h,
                               const cv::Scalar& color, const std::string& title)
{
	if (data.empty()) return;
	cv::rectangle(img, cv::Point(x, y), cv::Point(x+w, y+h), cv::Scalar(200,200,200), 2);
	cv::putText(img, title, cv::Point(x+10, y+25),
	            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);

	auto max_it = std::max_element(data.begin(), data.end());
	float max_val = *max_it;
	size_t argmax_idx = std::distance(data.begin(), max_it);
	float display_max = std::max(max_val, 0.01f);

	int hist_y = y + 40;
	int hist_h = h - 80;
	int hist_x = x + 10;
	int hist_w = w - 20;

	float bar_width = (float)hist_w / data.size();
	for (size_t i = 0; i < data.size(); i++) {
		float bar_height = (data[i] / display_max) * hist_h;
		int bar_x = hist_x + (int)(i * bar_width);
		int bar_y = hist_y + hist_h - (int)bar_height;

		cv::Scalar bar_color = color;
		if (i == argmax_idx && max_val > 0.001f) {
			bar_color = cv::Scalar(
				std::min(255.0, color[0] * 1.5),
				std::min(255.0, color[1] * 1.5),
				std::min(255.0, color[2] * 1.5),
				color[3]);
		}
		cv::rectangle(img, cv::Point(bar_x, bar_y),
		              cv::Point(bar_x + (int)bar_width - 1, hist_y + hist_h),
		              bar_color, cv::FILLED);
	}

	char max_text[64];
	snprintf(max_text, sizeof(max_text), "Max: %.3f", max_val);
	cv::putText(img, max_text, cv::Point(x+10, y+h-30),
	            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);

	char argmax_text[128];
	snprintf(argmax_text, sizeof(argmax_text), "Argmax: %zu (%.3f)", argmax_idx, max_val);
	cv::putText(img, argmax_text, cv::Point(x+10, y+h-10),
	            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
}

static void draw_zoomed_game_histogram_cv(cv::Mat& img, const std::vector<float>& data,
                                          int x, int y, int w, int h,
                                          const cv::Scalar& color, const std::string& title)
{
	if (data.empty() || data.size() != 480) return;
	cv::rectangle(img, cv::Point(x, y), cv::Point(x+w, y+h), cv::Scalar(200,200,200), 2);

	auto max_it = std::max_element(data.begin(), data.end());
	float max_val = *max_it;
	size_t argmax_idx = std::distance(data.begin(), max_it);

	const int zoom_radius = 20;
	int start_idx = std::max(0, (int)argmax_idx - zoom_radius);
	int end_idx   = std::min((int)data.size() - 1, (int)argmax_idx + zoom_radius);
	int num_bars  = end_idx - start_idx + 1;

	std::vector<float> zoomed(data.begin() + start_idx, data.begin() + end_idx + 1);
	int argmax_minutes = argmax_idx / 60;
	int argmax_seconds = argmax_idx % 60;

	char title_text[256];
	snprintf(title_text, sizeof(title_text), "%s (Argmax: %d:%02d)",
	         title.c_str(), argmax_minutes, argmax_seconds);
	cv::putText(img, title_text, cv::Point(x+10, y+25),
	            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);

	float display_max = std::max(max_val, 0.01f);
	int hist_y = y + 40;
	int hist_h = h - 100;
	int hist_x = x + 10;
	int hist_w = w - 20;

	float bar_width = (float)hist_w / num_bars;
	for (int i = 0; i < num_bars; i++) {
		int actual_idx = start_idx + i;
		float value = zoomed[i];
		float bar_height = (value / display_max) * hist_h;
		int bar_x = hist_x + (int)(i * bar_width);
		int bar_y = hist_y + hist_h - (int)bar_height;

		cv::Scalar bar_color = color;
		if (actual_idx == (int)argmax_idx && max_val > 0.001f) {
			bar_color = cv::Scalar(
				std::min(255.0, color[0] * 1.5),
				std::min(255.0, color[1] * 1.5),
				std::min(255.0, color[2] * 1.5),
				color[3]);
		}
		cv::rectangle(img, cv::Point(bar_x, bar_y),
		              cv::Point(bar_x + (int)bar_width - 1, hist_y + hist_h),
		              bar_color, cv::FILLED);

		if (i % 5 == 0 || actual_idx == (int)argmax_idx) {
			int minutes = actual_idx / 60;
			int seconds = actual_idx % 60;
			char time_text[16];
			snprintf(time_text, sizeof(time_text), "%d:%02d", minutes, seconds);
			cv::putText(img, time_text, cv::Point(bar_x, y+h-5),
			            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255,255,255), 1);
		}
	}

	char stats_text[128];
	snprintf(stats_text, sizeof(stats_text),
	         "Max: %.4f | Range: %d:%02d - %d:%02d",
	         max_val, start_idx/60, start_idx%60, end_idx/60, end_idx%60);
	cv::putText(img, stats_text, cv::Point(x+10, y+h-25),
	            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255,255,255), 1);
}
#endif

static void histogram_viz_source_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct histogram_viz_source *context = (struct histogram_viz_source *)data;
	if (!context) return;

#ifdef USE_CNN_OCR
	if (context->needs_update) {
		std::unique_lock<std::mutex> lock(context->data_mutex, std::try_to_lock);
		if (lock.owns_lock() && !context->pending_bytes.empty()) {
			uint32_t w = context->pending_w;
			uint32_t h = context->pending_h;
			std::vector<uint8_t> bytes = std::move(context->pending_bytes);
			context->needs_update = false;
			lock.unlock(); // release before GPU work

			// Already on the graphics thread — no obs_enter/leave needed.
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
		gs_effect_t *eff = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		gs_eparam_t *image = gs_effect_get_param_by_name(eff, "image");
		gs_effect_set_texture(image, context->texture);
		while (gs_effect_loop(eff, "Draw"))
			gs_draw_sprite(context->texture, 0, context->width, context->height);
	}
}

static obs_properties_t *histogram_viz_source_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();
#ifdef USE_CNN_OCR
	obs_properties_add_text(props, "info",
		"This source displays Bayesian filter histograms:\n"
		"Top row: Shot Clock (0-30)\n"
		"Bottom row: Game Clock (0:00-7:59)\n"
		"Columns: Prior (Cyan) | CNN Raw (Yellow) | Posterior (Green)",
		OBS_TEXT_INFO);
#else
	obs_properties_add_text(props, "info",
		"CNN support not compiled. This source requires CNN features.",
		OBS_TEXT_INFO);
#endif
	return props;
}

static void histogram_viz_source_defaults(obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
}

struct obs_source_info histogram_viz_source_info = {};

void init_histogram_viz_source_info()
{
	histogram_viz_source_info.id             = "histogram_viz_source";
	histogram_viz_source_info.type           = OBS_SOURCE_TYPE_INPUT;
	histogram_viz_source_info.output_flags   = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
	histogram_viz_source_info.get_name       = histogram_viz_source_get_name;
	histogram_viz_source_info.create         = histogram_viz_source_create;
	histogram_viz_source_info.destroy        = histogram_viz_source_destroy;
	histogram_viz_source_info.update         = histogram_viz_source_update;
	histogram_viz_source_info.get_properties = histogram_viz_source_properties;
	histogram_viz_source_info.get_defaults   = histogram_viz_source_defaults;
	histogram_viz_source_info.video_render   = histogram_viz_source_render;
	histogram_viz_source_info.get_width      = histogram_viz_source_get_width;
	histogram_viz_source_info.get_height     = histogram_viz_source_get_height;
}

#ifdef USE_CNN_OCR
void update_histogram_viz_data(
	const ClockPrediction& shot_pred,
	const ClockPrediction& game_pred)
{
	// Build the full histogram image on the calling (Qt) thread.
	const int IMG_W = 1200, IMG_H = 800;
	cv::Mat img(IMG_H, IMG_W, CV_8UC4, cv::Scalar(20, 20, 20, 255));

	const int margin     = 20;
	const int col_width  = 380;
	const int row_height = 350;
	const int y_shot     = margin;
	const int y_game     = margin + row_height + margin;

	draw_histogram_cv(img, shot_pred.prior,
	                  margin, y_shot, col_width, row_height,
	                  cv::Scalar(255,255,0,255), "Shot Prior");
	draw_histogram_cv(img, shot_pred.cnn_raw,
	                  margin + col_width + margin, y_shot, col_width, row_height,
	                  cv::Scalar(0,255,255,255), "Shot CNN");
	draw_histogram_cv(img, shot_pred.probabilities,
	                  margin + 2*(col_width+margin), y_shot, col_width, row_height,
	                  cv::Scalar(0,255,0,255), "Shot Posterior");

	draw_zoomed_game_histogram_cv(img, game_pred.prior,
	                              margin, y_game, col_width, row_height,
	                              cv::Scalar(255,255,0,255), "Game Prior");
	draw_zoomed_game_histogram_cv(img, game_pred.cnn_raw,
	                              margin + col_width + margin, y_game, col_width, row_height,
	                              cv::Scalar(0,255,255,255), "Game CNN");
	draw_zoomed_game_histogram_cv(img, game_pred.probabilities,
	                              margin + 2*(col_width+margin), y_game, col_width, row_height,
	                              cv::Scalar(0,255,0,255), "Game Posterior");

	// Flatten to BGRA bytes.
	std::vector<uint8_t> bytes(IMG_W * IMG_H * 4);
	for (int row = 0; row < IMG_H; row++)
		memcpy(bytes.data() + row * IMG_W * 4, img.ptr(row), IMG_W * 4);

	// Push to all active sources under a brief lock.
	struct PushData {
		std::vector<uint8_t> *bytes;
		uint32_t w, h;
	} push = { &bytes, (uint32_t)IMG_W, (uint32_t)IMG_H };

	obs_enum_sources([](void *param, obs_source_t *source) -> bool {
		const char *id = obs_source_get_id(source);
		if (strcmp(id, "histogram_viz_source") == 0) {
			auto *ctx  = (struct histogram_viz_source *)obs_obj_get_data(source);
			auto *push = (struct PushData *)param;
			if (ctx && push) {
				std::lock_guard<std::mutex> lock(ctx->data_mutex);
				ctx->pending_bytes = *push->bytes;
				ctx->pending_w     = push->w;
				ctx->pending_h     = push->h;
				ctx->needs_update  = true;
			}
		}
		return true;
	}, &push);
}
#endif
