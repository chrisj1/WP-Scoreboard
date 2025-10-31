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
#include <opencv2/opencv.hpp>
#endif

struct histogram_viz_source {
	obs_source_t *source;
	gs_texture_t *texture;
	
	uint32_t width;
	uint32_t height;
	
#ifdef USE_CNN_OCR
	// Histogram data
	std::vector<float> shot_prior;
	std::vector<float> shot_cnn;
	std::vector<float> shot_posterior;
	std::vector<float> game_prior;
	std::vector<float> game_cnn;
	std::vector<float> game_posterior;
	
	// Predicted values for display
	std::string shot_value;
	float shot_confidence;
	std::string game_value;
	float game_confidence;
	
	std::mutex data_mutex;
	bool needs_update;
#endif
	
	bool data_updated;
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
	
	// Fixed size for histogram display
	context->width = 1200;
	context->height = 800;
}

static void *histogram_viz_source_create(obs_data_t *settings, obs_source_t *source)
{
	struct histogram_viz_source *context = (struct histogram_viz_source *)bzalloc(sizeof(struct histogram_viz_source));
	context->source = source;
	context->width = 1200;
	context->height = 800;
	context->data_updated = false;
	context->texture = nullptr;
	
#ifdef USE_CNN_OCR
	context->needs_update = true;
	
	// Initialize empty vectors
	context->shot_prior.resize(31, 0.0f);
	context->shot_cnn.resize(31, 0.0f);
	context->shot_posterior.resize(31, 0.0f);
	context->game_prior.resize(480, 0.0f);
	context->game_cnn.resize(480, 0.0f);
	context->game_posterior.resize(480, 0.0f);
	context->shot_value = "00";
	context->shot_confidence = 0.0f;
	context->game_value = "0:00";
	context->game_confidence = 0.0f;
#endif
	
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
		bfree(context);
	}
}

static uint32_t histogram_viz_source_get_width(void *data)
{
	struct histogram_viz_source *context = (struct histogram_viz_source *)data;
	return context->width;
}

static uint32_t histogram_viz_source_get_height(void *data)
{
	struct histogram_viz_source *context = (struct histogram_viz_source *)data;
	return context->height;
}

#ifdef USE_CNN_OCR
// Helper function to draw a single histogram using OpenCV
static void draw_histogram_cv(cv::Mat& img, const std::vector<float>& data, 
                               int x, int y, int w, int h,
                               const cv::Scalar& color, const std::string& title)
{
	if (data.empty()) return;
	
	// Draw border
	cv::rectangle(img, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(200, 200, 200), 2);
	
	// Draw title in white for better readability
	cv::putText(img, title, cv::Point(x + 10, y + 25), 
	            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
	
	// Find max value and its index for scaling and argmax
	auto max_it = std::max_element(data.begin(), data.end());
	float max_val = *max_it;
	size_t argmax_idx = std::distance(data.begin(), max_it);
	
	// For very flat distributions, use a minimum scale to make them visible
	float display_max = std::max(max_val, 0.01f);
	
	// Drawing area (leave margin for title and padding)
	int hist_y = y + 40;
	int hist_h = h - 80; // More space for argmax text
	int hist_x = x + 10;
	int hist_w = w - 20;
	
	// Draw bars
	float bar_width = (float)hist_w / data.size();
	for (size_t i = 0; i < data.size(); i++) {
		float bar_height = (data[i] / display_max) * hist_h;
		int bar_x = hist_x + (int)(i * bar_width);
		int bar_y = hist_y + hist_h - (int)bar_height;
		
		// Highlight the argmax bar with brighter color
		cv::Scalar bar_color = color;
		if (i == argmax_idx && max_val > 0.001f) {
			bar_color = cv::Scalar(
				std::min(255.0, color[0] * 1.5),
				std::min(255.0, color[1] * 1.5), 
				std::min(255.0, color[2] * 1.5),
				color[3]
			);
		}
		
		cv::rectangle(img, 
		              cv::Point(bar_x, bar_y),
		              cv::Point(bar_x + (int)bar_width - 1, hist_y + hist_h),
		              bar_color, cv::FILLED);
	}
	
	// Draw max value indicator in white
	char max_text[64];
	snprintf(max_text, sizeof(max_text), "Max: %.3f", max_val);
	cv::putText(img, max_text, cv::Point(x + 10, y + h - 30), 
	            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
	
	// Draw argmax value and index in white
	char argmax_text[128];
	snprintf(argmax_text, sizeof(argmax_text), "Argmax: %zu (%.3f)", argmax_idx, max_val);
	cv::putText(img, argmax_text, cv::Point(x + 10, y + h - 10), 
	            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
}

static void draw_zoomed_game_histogram_cv(cv::Mat& img, const std::vector<float>& data, 
                                          int x, int y, int w, int h,
                                          const cv::Scalar& color, const std::string& title)
{
	if (data.empty() || data.size() != 480) return; // Expect 480 game clock states
	
	// Draw border
	cv::rectangle(img, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(200, 200, 200), 2);
	
	// Find argmax in full data
	auto max_it = std::max_element(data.begin(), data.end());
	float max_val = *max_it;
	size_t argmax_idx = std::distance(data.begin(), max_it);
	
	// Calculate zoom window: 20 bars on each side of argmax (41 total bars)
	const int zoom_radius = 20;
	int start_idx = std::max(0, (int)argmax_idx - zoom_radius);
	int end_idx = std::min((int)data.size() - 1, (int)argmax_idx + zoom_radius);
	int num_bars = end_idx - start_idx + 1;
	
	// Create zoomed data slice
	std::vector<float> zoomed_data(data.begin() + start_idx, data.begin() + end_idx + 1);
	
	// Convert argmax to time for display
	int argmax_minutes = argmax_idx / 60;
	int argmax_seconds = argmax_idx % 60;
	
	// Draw title with argmax time info
	char title_text[256];
	snprintf(title_text, sizeof(title_text), "%s (Argmax: %d:%02d)", 
	         title.c_str(), argmax_minutes, argmax_seconds);
	cv::putText(img, title_text, cv::Point(x + 10, y + 25), 
	            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
	
	// For very flat distributions, use a minimum scale
	float display_max = std::max(max_val, 0.01f);
	
	// Drawing area
	int hist_y = y + 40;
	int hist_h = h - 100; // More space for time labels
	int hist_x = x + 10;
	int hist_w = w - 20;
	
	// Draw individual bars for each time value
	float bar_width = (float)hist_w / num_bars;
	for (int i = 0; i < num_bars; i++) {
		int actual_idx = start_idx + i;
		float value = zoomed_data[i];
		float bar_height = (value / display_max) * hist_h;
		int bar_x = hist_x + (int)(i * bar_width);
		int bar_y = hist_y + hist_h - (int)bar_height;
		
		// Highlight the argmax bar
		cv::Scalar bar_color = color;
		if (actual_idx == (int)argmax_idx && max_val > 0.001f) {
			bar_color = cv::Scalar(
				std::min(255.0, color[0] * 1.5),
				std::min(255.0, color[1] * 1.5), 
				std::min(255.0, color[2] * 1.5),
				color[3]
			);
		}
		
		cv::rectangle(img, 
		              cv::Point(bar_x, bar_y),
		              cv::Point(bar_x + (int)bar_width - 1, hist_y + hist_h),
		              bar_color, cv::FILLED);
		
		// Draw time labels for key positions (every 5th bar to avoid crowding)
		if (i % 5 == 0 || actual_idx == (int)argmax_idx) {
			int minutes = actual_idx / 60;
			int seconds = actual_idx % 60;
			char time_text[16];
			snprintf(time_text, sizeof(time_text), "%d:%02d", minutes, seconds);
			
			// Use smaller font and white color
			cv::putText(img, time_text, cv::Point(bar_x, y + h - 5), 
			            cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);
		}
	}
	
	// Draw statistics
	char stats_text[128];
	snprintf(stats_text, sizeof(stats_text), "Max: %.4f | Range: %d:%02d - %d:%02d", 
	         max_val, start_idx/60, start_idx%60, end_idx/60, end_idx%60);
	cv::putText(img, stats_text, cv::Point(x + 10, y + h - 25), 
	            cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
}

#endif

static void histogram_viz_source_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct histogram_viz_source *context = (struct histogram_viz_source *)data;
	
	if (!context) return;
	
#ifdef USE_CNN_OCR
	// Update texture if needed
	if (context->needs_update) {
		std::lock_guard<std::mutex> lock(context->data_mutex);
		
		// Create OpenCV image
		cv::Mat img(800, 1200, CV_8UC4, cv::Scalar(20, 20, 20, 255)); // Dark gray background
		
		// Layout: 3x2 grid
		// Row 1: Shot Clock (Prior, CNN, Posterior)
		// Row 2: Game Clock (Prior, CNN, Posterior)
		int margin = 20;
		int col_width = 380;
		int row_height = 350;
		
		// Shot Clock Row (y=20)
		int y_shot = margin;
		
		// Shot Prior (cyan)
		draw_histogram_cv(img, context->shot_prior, 
		                  margin, y_shot, col_width, row_height,
		                  cv::Scalar(255, 255, 0, 255), "Shot Prior");
		
		// Shot CNN (yellow)
		draw_histogram_cv(img, context->shot_cnn,
		                  margin + col_width + margin, y_shot, col_width, row_height,
		                  cv::Scalar(0, 255, 255, 255), "Shot CNN");
		
		// Shot Posterior (green)
		draw_histogram_cv(img, context->shot_posterior,
		                  margin + 2 * (col_width + margin), y_shot, col_width, row_height,
		                  cv::Scalar(0, 255, 0, 255), "Shot Posterior");
		
		// Game Clock Row (y=390) - Use zoomed histograms showing individual time bars
		int y_game = margin + row_height + margin;
		
		// Use the new zoomed histogram function that shows individual time bars
		// around the argmax instead of sampling every 8th value
		
		// Game Prior (cyan) - Zoomed view
		draw_zoomed_game_histogram_cv(img, context->game_prior,
		                             margin, y_game, col_width, row_height,
		                             cv::Scalar(255, 255, 0, 255), "Game Prior");
		
		// Game CNN (yellow) - Zoomed view
		draw_zoomed_game_histogram_cv(img, context->game_cnn,
		                             margin + col_width + margin, y_game, col_width, row_height,
		                             cv::Scalar(0, 255, 255, 255), "Game CNN");
		
		// Game Posterior (green) - Zoomed view
		draw_zoomed_game_histogram_cv(img, context->game_posterior,
		                             margin + 2 * (col_width + margin), y_game, col_width, row_height,
		                             cv::Scalar(0, 255, 0, 255), "Game Posterior");
		
		// Convert cv::Mat to gs_texture_t
		// OBS expects BGRA format, OpenCV uses BGRA by default (CV_8UC4)
		obs_enter_graphics();
		
		// Destroy old texture if exists
		if (context->texture) {
			gs_texture_destroy(context->texture);
			context->texture = nullptr;
		}
		
		// Create texture from cv::Mat data
		context->texture = gs_texture_create(img.cols, img.rows, GS_BGRA, 1, 
		                                      (const uint8_t **)&img.data, 0);
		
		obs_leave_graphics();
		
		context->needs_update = false;
	}
	
	// Draw the texture
	if (context->texture) {
		gs_effect_t *effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		gs_eparam_t *image = gs_effect_get_param_by_name(effect, "image");
		gs_effect_set_texture(image, context->texture);
		
		while (gs_effect_loop(effect, "Draw")) {
			gs_draw_sprite(context->texture, 0, context->width, context->height);
		}
	}
#else
	// Draw "CNN not enabled" message
	gs_effect_t *solid = obs_get_base_effect(OBS_EFFECT_SOLID);
	if (solid) {
		gs_technique_t *tech = gs_effect_get_technique(solid, "Solid");
		gs_eparam_t *param_color = gs_effect_get_param_by_name(solid, "color");
		
		struct vec4 bg_color;
		vec4_set(&bg_color, 0.3f, 0.0f, 0.0f, 1.0f);
		gs_effect_set_vec4(param_color, &bg_color);
		
		gs_technique_begin(tech);
		gs_technique_begin_pass(tech, 0);
		
		gs_render_start(false);
		gs_vertex2f(0, 0);
		gs_vertex2f(context->width, 0);
		gs_vertex2f(context->width, context->height);
		gs_vertex2f(0, context->height);
		gs_vertbuffer_t *bg = gs_render_save();
		
		gs_load_vertexbuffer(bg);
		gs_draw(GS_TRISTRIP, 0, 4);
		gs_vertexbuffer_destroy(bg);
		
		gs_technique_end_pass(tech);
		gs_technique_end(tech);
	}
#endif
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

// Public API to update histogram data
#ifdef USE_CNN_OCR
void update_histogram_viz_data(
	const ClockPrediction& shot_pred,
	const ClockPrediction& game_pred)
{
	// Create a struct to pass both predictions through the void* parameter
	struct PredPair {
		const ClockPrediction* shot;
		const ClockPrediction* game;
	};
	
	PredPair pair = { &shot_pred, &game_pred };
	
	// Find the histogram source and update it
	obs_enum_sources([](void *param, obs_source_t *source) -> bool {
		const char *id = obs_source_get_id(source);
		if (strcmp(id, "histogram_viz_source") == 0) {
			struct histogram_viz_source *context = 
				(struct histogram_viz_source *)obs_obj_get_data(source);
			
			if (context) {
				std::lock_guard<std::mutex> lock(context->data_mutex);
				
				PredPair *pair = (PredPair *)param;
				
				// Update shot clock data
				context->shot_prior = pair->shot->prior;
				context->shot_cnn = pair->shot->cnn_raw;
				context->shot_posterior = pair->shot->probabilities;
				context->shot_value = pair->shot->value;
				context->shot_confidence = pair->shot->confidence;
				
				// Update game clock data
				context->game_prior = pair->game->prior;
				context->game_cnn = pair->game->cnn_raw;
				context->game_posterior = pair->game->probabilities;
				context->game_value = pair->game->value;
				context->game_confidence = pair->game->confidence;
				
				context->data_updated = true;
				context->needs_update = true; // Trigger texture regeneration
			}
		}
		return true;
	}, (void*)&pair);
}
#endif

struct obs_source_info histogram_viz_source_info = {};

void init_histogram_viz_source_info()
{
	histogram_viz_source_info.id = "histogram_viz_source";
	histogram_viz_source_info.type = OBS_SOURCE_TYPE_INPUT;
	histogram_viz_source_info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
	histogram_viz_source_info.get_name = histogram_viz_source_get_name;
	histogram_viz_source_info.create = histogram_viz_source_create;
	histogram_viz_source_info.destroy = histogram_viz_source_destroy;
	histogram_viz_source_info.update = histogram_viz_source_update;
	histogram_viz_source_info.get_properties = histogram_viz_source_properties;
	histogram_viz_source_info.get_defaults = histogram_viz_source_defaults;
	histogram_viz_source_info.video_render = histogram_viz_source_render;
	histogram_viz_source_info.get_width = histogram_viz_source_get_width;
	histogram_viz_source_info.get_height = histogram_viz_source_get_height;
}
