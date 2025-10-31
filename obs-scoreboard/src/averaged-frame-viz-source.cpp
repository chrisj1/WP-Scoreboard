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

struct averaged_frame_viz_source {
	obs_source_t *source;
	gs_texture_t *texture;
	
	uint32_t width;
	uint32_t height;
	
#ifdef USE_CNN_OCR
	// Averaged frame data
	cv::Mat shot_averaged_frame;
	cv::Mat game_averaged_frame;
	
	std::mutex data_mutex;
	bool needs_update;
#endif
	
	bool data_updated;
};

#ifdef USE_CNN_OCR
// Forward declarations for registration functions
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
	
	// Fixed size for frame display
	context->width = 800;
	context->height = 400;
}

static void *averaged_frame_viz_source_create(obs_data_t *settings, obs_source_t *source)
{
	struct averaged_frame_viz_source *context = (struct averaged_frame_viz_source *)bzalloc(sizeof(struct averaged_frame_viz_source));
	context->source = source;
	context->width = 800;
	context->height = 400;
	context->data_updated = false;
	context->texture = nullptr;
	
#ifdef USE_CNN_OCR
	context->needs_update = true;
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
	
	bfree(context);
}

static void averaged_frame_viz_source_get_defaults(obs_data_t *settings)
{
	UNUSED_PARAMETER(settings);
}

static obs_properties_t *averaged_frame_viz_source_get_properties(void *data)
{
	UNUSED_PARAMETER(data);
	obs_properties_t *props = obs_properties_create();
	
	obs_properties_add_text(props, "info", "Shows the averaged frames before CNN processing",
	                        OBS_TEXT_INFO);
	
	return props;
}

#ifdef USE_CNN_OCR
static void draw_frame_with_label(cv::Mat& dst, const cv::Mat& frame, int x, int y, int w, int h, const std::string& label)
{
	// Draw frame border
	cv::rectangle(dst, cv::Rect(x-2, y-2, w+4, h+4), cv::Scalar(100, 100, 100, 255), 2);
	
	// Draw label
	cv::putText(dst, label, cv::Point(x, y-10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255, 255), 2);
	
	if (!frame.empty()) {
		// Resize frame to fit the display area
		cv::Mat resized_frame;
		cv::resize(frame, resized_frame, cv::Size(w, h));
		
		// Convert to BGRA if needed
		cv::Mat bgra_frame;
		if (resized_frame.channels() == 1) {
			cv::cvtColor(resized_frame, bgra_frame, cv::COLOR_GRAY2BGRA);
		} else if (resized_frame.channels() == 3) {
			cv::cvtColor(resized_frame, bgra_frame, cv::COLOR_BGR2BGRA);
		} else {
			bgra_frame = resized_frame;
		}
		
		// Copy to destination
		cv::Rect roi(x, y, w, h);
		if (roi.x >= 0 && roi.y >= 0 && 
		    roi.x + roi.width <= dst.cols && 
		    roi.y + roi.height <= dst.rows) {
			bgra_frame.copyTo(dst(roi));
		}
	} else {
		// Draw "No Data" if frame is empty
		cv::rectangle(dst, cv::Rect(x, y, w, h), cv::Scalar(50, 50, 50, 255), -1);
		cv::putText(dst, "No Data", cv::Point(x + w/2 - 50, y + h/2), 
		           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(200, 200, 200, 255), 2);
	}
}
#endif

static void averaged_frame_viz_source_render(void *data, gs_effect_t *effect)
{
	UNUSED_PARAMETER(effect);
	struct averaged_frame_viz_source *context = (struct averaged_frame_viz_source *)data;
	
	if (!context) return;
	
#ifdef USE_CNN_OCR
	// Update texture if needed
	if (context->needs_update) {
		std::lock_guard<std::mutex> lock(context->data_mutex);
		
		// Create OpenCV image
		cv::Mat img(400, 800, CV_8UC4, cv::Scalar(30, 30, 30, 255)); // Dark background
		
		// Layout: 2 columns
		int margin = 20;
		int frame_width = 360;
		int frame_height = 320;
		
		// Shot Clock Frame (left)
		draw_frame_with_label(img, context->shot_averaged_frame,
		                     margin, margin + 30, frame_width, frame_height,
		                     "Shot Clock Averaged Frame");
		
		// Game Clock Frame (right)
		draw_frame_with_label(img, context->game_averaged_frame,
		                     margin + frame_width + margin, margin + 30, frame_width, frame_height,
		                     "Game Clock Averaged Frame");
		
		// Add title
		cv::putText(img, "CNN Input Frames (5-Frame Average)", cv::Point(250, 25),
		           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255, 255), 2);
		
		// Update texture
		obs_enter_graphics();
		
		if (context->texture) {
			gs_texture_destroy(context->texture);
		}
		
		context->texture = gs_texture_create(img.cols, img.rows, GS_BGRA, 1, 
		                                    (const uint8_t**)&img.data, 0);
		
		obs_leave_graphics();
		
		context->needs_update = false;
	}
#endif
	
	// Render the texture
	if (context->texture) {
		// Get the default effect for rendering textures
		gs_effect_t *default_effect = obs_get_base_effect(OBS_EFFECT_DEFAULT);
		
		if (default_effect) {
			gs_effect_set_texture(gs_effect_get_param_by_name(default_effect, "image"), context->texture);
			
			while (gs_effect_loop(default_effect, "Draw"))
				gs_draw_sprite(context->texture, 0, context->width, context->height);
		} else {
			// Fallback: just draw the sprite without effects
			gs_draw_sprite(context->texture, 0, context->width, context->height);
		}
	}
}

static uint32_t averaged_frame_viz_source_get_width(void *data)
{
	struct averaged_frame_viz_source *context = (struct averaged_frame_viz_source *)data;
	return context->width;
}

static uint32_t averaged_frame_viz_source_get_height(void *data)
{
	struct averaged_frame_viz_source *context = (struct averaged_frame_viz_source *)data;
	return context->height;
}

struct obs_source_info averaged_frame_viz_source_info = {};

void init_averaged_frame_viz_source_info()
{
	averaged_frame_viz_source_info.id = "averaged_frame_viz_source";
	averaged_frame_viz_source_info.type = OBS_SOURCE_TYPE_INPUT;
	averaged_frame_viz_source_info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
	averaged_frame_viz_source_info.get_name = averaged_frame_viz_source_get_name;
	averaged_frame_viz_source_info.create = averaged_frame_viz_source_create;
	averaged_frame_viz_source_info.destroy = averaged_frame_viz_source_destroy;
	averaged_frame_viz_source_info.update = averaged_frame_viz_source_update;
	averaged_frame_viz_source_info.get_properties = averaged_frame_viz_source_get_properties;
	averaged_frame_viz_source_info.get_defaults = averaged_frame_viz_source_get_defaults;
	averaged_frame_viz_source_info.video_render = averaged_frame_viz_source_render;
	averaged_frame_viz_source_info.get_width = averaged_frame_viz_source_get_width;
	averaged_frame_viz_source_info.get_height = averaged_frame_viz_source_get_height;
}

#ifdef USE_CNN_OCR
// Global storage for data updates
static std::vector<averaged_frame_viz_source*> g_averaged_frame_viz_sources;
static std::mutex g_averaged_frame_viz_mutex;

// Register source instance
static void register_averaged_frame_viz_source(averaged_frame_viz_source* context)
{
	std::lock_guard<std::mutex> lock(g_averaged_frame_viz_mutex);
	g_averaged_frame_viz_sources.push_back(context);
}

// Unregister source instance
static void unregister_averaged_frame_viz_source(averaged_frame_viz_source* context)
{
	std::lock_guard<std::mutex> lock(g_averaged_frame_viz_mutex);
	auto it = std::find(g_averaged_frame_viz_sources.begin(), g_averaged_frame_viz_sources.end(), context);
	if (it != g_averaged_frame_viz_sources.end()) {
		g_averaged_frame_viz_sources.erase(it);
	}
}

void update_averaged_frame_viz_data(
	const cv::Mat& shot_averaged_frame,
	const cv::Mat& game_averaged_frame)
{
	std::lock_guard<std::mutex> global_lock(g_averaged_frame_viz_mutex);
	
	blog(LOG_INFO, "update_averaged_frame_viz_data called: shot %dx%d (empty: %s), game %dx%d (empty: %s), %zu sources",
	     shot_averaged_frame.cols, shot_averaged_frame.rows, shot_averaged_frame.empty() ? "true" : "false",
	     game_averaged_frame.cols, game_averaged_frame.rows, game_averaged_frame.empty() ? "true" : "false",
	     g_averaged_frame_viz_sources.size());
	
	// Update all active averaged frame viz sources
	for (auto* context : g_averaged_frame_viz_sources) {
		if (!context) continue;
		
		std::lock_guard<std::mutex> lock(context->data_mutex);
		
		// Clone the frames to avoid data corruption
		if (!shot_averaged_frame.empty()) {
			context->shot_averaged_frame = shot_averaged_frame.clone();
		}
		if (!game_averaged_frame.empty()) {
			context->game_averaged_frame = game_averaged_frame.clone();
		}
		
		context->needs_update = true;
	}
}
#endif