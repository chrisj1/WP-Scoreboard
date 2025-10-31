#include <obs-module.h>
#include <graphics/image-file.h>
#include <util/platform.h>
#include <util/dstr.h>
#include <string>

// Windows GDI+ for text rendering
#ifdef _WIN32
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
using namespace Gdiplus;

// GDI+ initialization
static ULONG_PTR gdiplusToken = 0;

void init_gdiplus()
{
	if (gdiplusToken == 0) {
		GdiplusStartupInput gdiplusStartupInput;
		GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);
	}
}

void shutdown_gdiplus()
{
	if (gdiplusToken != 0) {
		GdiplusShutdown(gdiplusToken);
		gdiplusToken = 0;
	}
}
#endif

struct scoreboard_source {
	obs_source_t *source;
	
	// Scoreboard data
	int home_score;
	int away_score;
	int shot_clock;  // 30 seconds for water polo
	int game_clock_minutes;
	int game_clock_seconds;
	std::string home_team;
	std::string away_team;
	
	// Water polo specific
	int period;  // 1-4 quarters
	int home_exclusions;  // Number of exclusions
	int away_exclusions;
	int home_timeouts;  // Timeouts remaining
	int away_timeouts;
	
	// Rendering
	gs_texture_t *texture;
	uint32_t width;
	uint32_t height;
	
	// Colors
	uint32_t bg_color;
	uint32_t text_color;
	
	// Visibility options
	bool show_game_clock;
	bool show_shot_clock;
};

// Global pointer to the active scoreboard source (for WebSocket updates)
static scoreboard_source *g_active_scoreboard = nullptr;

static const char *scoreboard_source_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Water Polo Scoreboard";
}

static void scoreboard_source_update(void *data, obs_data_t *settings)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	
	// Get settings
	context->home_score = (int)obs_data_get_int(settings, "home_score");
	context->away_score = (int)obs_data_get_int(settings, "away_score");
	context->shot_clock = (int)obs_data_get_int(settings, "shot_clock");
	context->game_clock_minutes = (int)obs_data_get_int(settings, "game_minutes");
	context->game_clock_seconds = (int)obs_data_get_int(settings, "game_seconds");
	
	// Water polo specific
	context->period = (int)obs_data_get_int(settings, "period");
	context->home_exclusions = (int)obs_data_get_int(settings, "home_exclusions");
	context->away_exclusions = (int)obs_data_get_int(settings, "away_exclusions");
	context->home_timeouts = (int)obs_data_get_int(settings, "home_timeouts");
	context->away_timeouts = (int)obs_data_get_int(settings, "away_timeouts");
	
	const char *home_team = obs_data_get_string(settings, "home_team");
	const char *away_team = obs_data_get_string(settings, "away_team");
	
	if (home_team)
		context->home_team = home_team;
	if (away_team)
		context->away_team = away_team;
	
	context->bg_color = (uint32_t)obs_data_get_int(settings, "bg_color");
	context->text_color = (uint32_t)obs_data_get_int(settings, "text_color");
	
	context->width = (uint32_t)obs_data_get_int(settings, "width");
	context->height = (uint32_t)obs_data_get_int(settings, "height");
	
	// Visibility options
	context->show_game_clock = obs_data_get_bool(settings, "show_game_clock");
	context->show_shot_clock = obs_data_get_bool(settings, "show_shot_clock");
	
	blog(LOG_INFO, "Scoreboard update: show_game_clock=%s, show_shot_clock=%s", 
	     context->show_game_clock ? "true" : "false", 
	     context->show_shot_clock ? "true" : "false");
	
	// Recalculate dimensions based on visible content
	bool effective_show_shot_clock = context->show_shot_clock;
	int game_clock_total = context->game_clock_minutes * 60 + context->game_clock_seconds;
	if (context->shot_clock <= 0 || game_clock_total < context->shot_clock) {
		effective_show_shot_clock = false;
	}
	
	// Calculate actual content dimensions
	int scoreBoxHeight = 120;
	int scoreBoxWidth = 200;
	int margin = 20;
	int clockWidth = 250; // Width of game clock
	int shotClockWidth = 160; // Width of shot clock
	
	// Calculate required width: left margin + home box + spacing + center clocks + spacing + away box + right margin
	int centerClockWidth = clockWidth; // Start with game clock width
	if (effective_show_shot_clock && shotClockWidth > centerClockWidth) {
		centerClockWidth = shotClockWidth; // Use shot clock width if larger and visible
	}
	int requiredSpacing = 50; // Spacing between team boxes and center section
	int actual_board_width = margin + scoreBoxWidth + requiredSpacing + centerClockWidth + requiredSpacing + scoreBoxWidth + margin;
	
	int centerY = context->height / 2 - scoreBoxHeight / 2;
	int clock_section_height = 0;
	if (context->show_game_clock) clock_section_height += 80 + 35; // game clock + period label
	if (effective_show_shot_clock) clock_section_height += 60 + 25 + 15; // shot clock + label + gap
	
	// Calculate the actual board dimensions needed
	int actual_board_height = centerY + scoreBoxHeight + clock_section_height + 50;
	
	// Update context dimensions to match actual content
	context->height = actual_board_height;
	context->width = actual_board_width;
	
	blog(LOG_INFO, "Scoreboard dimensions updated to: %dx%d", context->width, context->height);
}

static void *scoreboard_source_create(obs_data_t *settings, obs_source_t *source)
{
	struct scoreboard_source *context = (struct scoreboard_source *)bzalloc(sizeof(struct scoreboard_source));
	context->source = source;
	
	// Default values
	context->home_score = 0;
	context->away_score = 0;
	context->shot_clock = 30;  // Water polo shot clock
	context->game_clock_minutes = 8;  // 8 minute quarters
	context->game_clock_seconds = 0;
	context->home_team = "HOME";
	context->away_team = "AWAY";
	context->period = 1;
	context->home_exclusions = 0;
	context->away_exclusions = 0;
	context->home_timeouts = 2;  // 2 timeouts per half
	context->away_timeouts = 2;
	context->show_game_clock = true;
	context->show_shot_clock = true;
	context->width = 1920;
	context->height = 200;
	context->bg_color = 0xFF000000; // Black
	context->text_color = 0xFFFFFFFF; // White
	context->show_game_clock = true;
	context->show_shot_clock = true;
	
	scoreboard_source_update(context, settings);
	
	// Set as active scoreboard
	g_active_scoreboard = context;
	
	return context;
}

static void scoreboard_source_destroy(void *data)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	
	// Clear global reference if this was the active scoreboard
	if (g_active_scoreboard == context) {
		g_active_scoreboard = nullptr;
	}
	
	if (context->texture) {
		obs_enter_graphics();
		gs_texture_destroy(context->texture);
		obs_leave_graphics();
	}
	
	bfree(context);
}

static obs_properties_t *scoreboard_source_properties(void *data)
{
	UNUSED_PARAMETER(data);
	
	obs_properties_t *props = obs_properties_create();
	
	// Team names
	obs_properties_add_text(props, "home_team", "Home Team", OBS_TEXT_DEFAULT);
	obs_properties_add_text(props, "away_team", "Away Team", OBS_TEXT_DEFAULT);
	
	// Scores
	obs_properties_add_int(props, "home_score", "Home Score", 0, 999, 1);
	obs_properties_add_int(props, "away_score", "Away Score", 0, 999, 1);
	
	// Shot clock
	obs_properties_add_int(props, "shot_clock", "Shot Clock", 0, 30, 1);
	
	// Game clock
	obs_properties_add_int(props, "game_minutes", "Game Clock Minutes", 0, 8, 1);
	obs_properties_add_int(props, "game_seconds", "Game Clock Seconds", 0, 59, 1);
	
	// Period
	obs_properties_add_int(props, "period", "Period", 1, 4, 1);
	
	// Exclusions
	obs_properties_add_int(props, "home_exclusions", "Home Exclusions", 0, 10, 1);
	obs_properties_add_int(props, "away_exclusions", "Away Exclusions", 0, 10, 1);
	
	// Timeouts
	obs_properties_add_int(props, "home_timeouts", "Home Timeouts", 0, 2, 1);
	obs_properties_add_int(props, "away_timeouts", "Away Timeouts", 0, 2, 1);
	
	// Dimensions
	obs_properties_add_int(props, "width", "Width", 100, 3840, 10);
	obs_properties_add_int(props, "height", "Height", 50, 2160, 10);
	
	// Colors
	obs_properties_add_color(props, "bg_color", "Background Color");
	obs_properties_add_color(props, "text_color", "Text Color");
	
	// Visibility options
	obs_properties_add_bool(props, "show_game_clock", "Show Game Clock", true);
	obs_properties_add_bool(props, "show_shot_clock", "Show Shot Clock", true);
	
	return props;
}

static void scoreboard_source_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "home_score", 0);
	obs_data_set_default_int(settings, "away_score", 0);
	obs_data_set_default_int(settings, "shot_clock", 30);
	obs_data_set_default_int(settings, "game_minutes", 8);
	obs_data_set_default_int(settings, "game_seconds", 0);
	obs_data_set_default_int(settings, "period", 1);
	obs_data_set_default_int(settings, "home_exclusions", 0);
	obs_data_set_default_int(settings, "away_exclusions", 0);
	obs_data_set_default_int(settings, "home_timeouts", 2);
	obs_data_set_default_int(settings, "away_timeouts", 2);
	obs_data_set_default_string(settings, "home_team", "HOME");
	obs_data_set_default_string(settings, "away_team", "AWAY");
	obs_data_set_default_int(settings, "width", 1920);
	obs_data_set_default_int(settings, "height", 200);
	obs_data_set_default_int(settings, "bg_color", 0xFF000000);
	obs_data_set_default_int(settings, "text_color", 0xFFFFFFFF);
	obs_data_set_default_bool(settings, "show_game_clock", true);
	obs_data_set_default_bool(settings, "show_shot_clock", true);
}

static uint32_t scoreboard_source_get_width(void *data)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	return context->width;
}

static uint32_t scoreboard_source_get_height(void *data)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	return context->height;
}

static void scoreboard_source_render(void *data, gs_effect_t *effect)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	
	blog(LOG_INFO, "Scoreboard render called - size: %dx%d, show_game_clock=%s, show_shot_clock=%s", 
	     context->width, context->height,
	     context->show_game_clock ? "true" : "false", 
	     context->show_shot_clock ? "true" : "false");
	
	// Pre-calculate effective shot clock visibility (same logic as update function)
	bool effective_show_shot_clock = context->show_shot_clock;
	int game_clock_total = context->game_clock_minutes * 60 + context->game_clock_seconds;
	if (context->shot_clock <= 0 || game_clock_total < context->shot_clock) {
		effective_show_shot_clock = false;
	}
	
	// Recreate texture on every render to show updates
	obs_enter_graphics();
	
#ifdef _WIN32
	try {
		// Create a GDI+ bitmap
		Bitmap bitmap(context->width, context->height, PixelFormat32bppARGB);
		Graphics graphics(&bitmap);
		
		blog(LOG_INFO, "GDI+ bitmap created successfully");
		
		// Set high quality rendering
		graphics.SetTextRenderingHint(TextRenderingHintAntiAlias);
		graphics.SetSmoothingMode(SmoothingModeAntiAlias);
		
		// Fill background
		Color bgColor((context->bg_color >> 24) & 0xFF,
		              (context->bg_color >> 16) & 0xFF,
		              (context->bg_color >> 8) & 0xFF,
		              context->bg_color & 0xFF);
		SolidBrush bgBrush(bgColor);
		graphics.FillRectangle(&bgBrush, 0, 0, context->width, context->height);
		
		blog(LOG_INFO, "Background filled");
	
	// Setup text colors
	Color textColor((context->text_color >> 24) & 0xFF,
	                (context->text_color >> 16) & 0xFF,
	                (context->text_color >> 8) & 0xFF,
	                context->text_color & 0xFF);
	SolidBrush textBrush(textColor);
	
	// Accent color for boxes
	Color accentColor(255, 0, 128, 255); // Blue
	SolidBrush accentBrush(accentColor);
	Pen accentPen(&accentBrush, 3.0f);
	
	// Create fonts
	FontFamily fontFamily(L"Arial");
	Font scoreFont(&fontFamily, 72, FontStyleBold, UnitPixel);
	Font labelFont(&fontFamily, 24, FontStyleBold, UnitPixel);
	Font clockFont(&fontFamily, 48, FontStyleBold, UnitPixel);
	Font smallFont(&fontFamily, 20, FontStyleRegular, UnitPixel);
	
	// Layout
	int margin = 20;
	int scoreBoxWidth = 200;
	int scoreBoxHeight = 120;
	int centerY = context->height / 2 - scoreBoxHeight / 2;
	
	// Home team section (left)
	RectF homeBox(margin, centerY, scoreBoxWidth, scoreBoxHeight);
	graphics.DrawRectangle(&accentPen, homeBox);
	
	// Home team name
	StringFormat centerFormat;
	centerFormat.SetAlignment(StringAlignmentCenter);
	centerFormat.SetLineAlignment(StringAlignmentCenter);
	
	std::wstring homeTeamW(context->home_team.begin(), context->home_team.end());
	RectF homeNameRect(margin, centerY - 35, scoreBoxWidth, 30);
	graphics.DrawString(homeTeamW.c_str(), -1, &labelFont, homeNameRect, &centerFormat, &textBrush);
	
	// Home score
	wchar_t homeScoreStr[16];
	swprintf_s(homeScoreStr, L"%d", context->home_score);
	graphics.DrawString(homeScoreStr, -1, &scoreFont, homeBox, &centerFormat, &textBrush);
	
	// Home exclusions (small text below score)
	wchar_t homeExclStr[32];
	swprintf_s(homeExclStr, L"Excl: %d", context->home_exclusions);
	RectF homeExclRect(margin, centerY + scoreBoxHeight + 5, scoreBoxWidth, 25);
	graphics.DrawString(homeExclStr, -1, &smallFont, homeExclRect, &centerFormat, &textBrush);
	
	// Home timeouts
	wchar_t homeToStr[32];
	swprintf_s(homeToStr, L"TO: %d", context->home_timeouts);
	RectF homeToRect(margin, centerY + scoreBoxHeight + 30, scoreBoxWidth, 25);
	graphics.DrawString(homeToStr, -1, &smallFont, homeToRect, &centerFormat, &textBrush);
	
	// Away team section (right)
	int awayX = context->width - margin - scoreBoxWidth;
	RectF awayBox(awayX, centerY, scoreBoxWidth, scoreBoxHeight);
	graphics.DrawRectangle(&accentPen, awayBox);
	
	// Away team name
	std::wstring awayTeamW(context->away_team.begin(), context->away_team.end());
	RectF awayNameRect(awayX, centerY - 35, scoreBoxWidth, 30);
	graphics.DrawString(awayTeamW.c_str(), -1, &labelFont, awayNameRect, &centerFormat, &textBrush);
	
	// Away score
	wchar_t awayScoreStr[16];
	swprintf_s(awayScoreStr, L"%d", context->away_score);
	graphics.DrawString(awayScoreStr, -1, &scoreFont, awayBox, &centerFormat, &textBrush);
	
	// Away exclusions
	wchar_t awayExclStr[32];
	swprintf_s(awayExclStr, L"Excl: %d", context->away_exclusions);
	RectF awayExclRect(awayX, centerY + scoreBoxHeight + 5, scoreBoxWidth, 25);
	graphics.DrawString(awayExclStr, -1, &smallFont, awayExclRect, &centerFormat, &textBrush);
	
	// Away timeouts
	wchar_t awayToStr[32];
	swprintf_s(awayToStr, L"TO: %d", context->away_timeouts);
	RectF awayToRect(awayX, centerY + scoreBoxHeight + 30, scoreBoxWidth, 25);
	graphics.DrawString(awayToStr, -1, &smallFont, awayToRect, &centerFormat, &textBrush);
	
	// Center section - clocks
	int centerX = context->width / 2;
	int clockWidth = 250;
	int clockHeight = 80;
	int clockY = 30;
	
	// Game clock box
	RectF gameClockBox(centerX - clockWidth / 2, clockY, clockWidth, clockHeight);
	
	if (context->show_game_clock) {
		graphics.FillRectangle(&accentBrush, gameClockBox);
		
		// Game clock text
		wchar_t gameClockStr[32];
		swprintf_s(gameClockStr, L"%02d:%02d", context->game_clock_minutes, context->game_clock_seconds);
		SolidBrush whiteTextBrush(Color(255, 255, 255, 255));
		graphics.DrawString(gameClockStr, -1, &clockFont, gameClockBox, &centerFormat, &whiteTextBrush);
		
		// Period label
		wchar_t periodStr[32];
		swprintf_s(periodStr, L"Period %d", context->period);
		RectF periodRect(centerX - clockWidth / 2, clockY - 35, clockWidth, 30);
		graphics.DrawString(periodStr, -1, &labelFont, periodRect, &centerFormat, &textBrush);
	}
	
	// Shot clock
	int shotClockY = clockY + clockHeight + 15;
	RectF shotClockBox(centerX - 80, shotClockY, 160, 60);

    if (effective_show_shot_clock) {
        // Shot clock color changes when low
        Color shotClockColor = context->shot_clock <= 5 ? Color(255, 255, 0, 0) : Color(255, 60, 60, 60);
        SolidBrush shotClockBrush(shotClockColor);
        graphics.FillRectangle(&shotClockBrush, shotClockBox);

        wchar_t shotClockStr[16];
        swprintf_s(shotClockStr, L"%d", context->shot_clock);
        Font shotClockFont(&fontFamily, 40, FontStyleBold, UnitPixel);
        SolidBrush whiteTextBrush(Color(255, 255, 255, 255));
        graphics.DrawString(shotClockStr, -1, &shotClockFont, shotClockBox, &centerFormat, &whiteTextBrush);

        // Shot clock label
        wchar_t shotLabelStr[] = L"SHOT CLOCK";
        RectF shotLabelRect(centerX - 80, shotClockY + 65, 160, 25);
        graphics.DrawString(shotLabelStr, -1, &smallFont, shotLabelRect, &centerFormat, &textBrush);
    }

	// Lock bitmap and copy to OBS texture
	BitmapData bitmapData;
	Rect rect(0, 0, context->width, context->height);
	Status lockStatus = bitmap.LockBits(&rect, ImageLockModeRead, PixelFormat32bppARGB, &bitmapData);
	
	if (lockStatus == Ok) {
		blog(LOG_INFO, "Bitmap locked: %dx%d, stride=%d", bitmapData.Width, bitmapData.Height, bitmapData.Stride);
		
		// Destroy old texture
		if (context->texture) {
			gs_texture_destroy(context->texture);
			context->texture = nullptr;
		}
		
		// Copy bitmap data to a contiguous buffer (GDI+ may have padding)
		uint32_t *pixels = (uint32_t *)bzalloc(context->width * context->height * sizeof(uint32_t));
		uint8_t *src = (uint8_t *)bitmapData.Scan0;
		
		for (uint32_t y = 0; y < context->height; y++) {
			uint32_t *srcRow = (uint32_t *)(src + y * bitmapData.Stride);
			uint32_t *dstRow = pixels + y * context->width;
			
			for (uint32_t x = 0; x < context->width; x++) {
				// GDI+ uses ARGB, OBS uses BGRA - need to swap R and B
				uint32_t pixel = srcRow[x];
				uint8_t a = (pixel >> 24) & 0xFF;
				uint8_t r = (pixel >> 16) & 0xFF;
				uint8_t g = (pixel >> 8) & 0xFF;
				uint8_t b = pixel & 0xFF;
				dstRow[x] = (a << 24) | (r << 16) | (g << 8) | b;
			}
		}
		
		// Create OBS texture from our contiguous buffer
		context->texture = gs_texture_create(context->width, context->height,
		                                     GS_BGRA, 1, (const uint8_t **)&pixels, 0);
		
		bfree(pixels);
		
		if (context->texture) {
			blog(LOG_INFO, "Texture created successfully");
		} else {
			blog(LOG_ERROR, "Failed to create texture!");
		}
		
		bitmap.UnlockBits(&bitmapData);
	} else {
		blog(LOG_ERROR, "Failed to lock bitmap bits!");
	}
	} catch (const std::exception& e) {
		blog(LOG_ERROR, "GDI+ exception in render: %s", e.what());
	} catch (...) {
		blog(LOG_ERROR, "Unknown exception in GDI+ render");
	}
#else
	// Non-Windows fallback (simple colored rectangle)
	uint32_t *pixels = (uint32_t *)bzalloc(context->width * context->height * sizeof(uint32_t));
	for (uint32_t i = 0; i < context->width * context->height; i++) {
		pixels[i] = context->bg_color;
	}
	
	if (context->texture) {
		gs_texture_destroy(context->texture);
	}
	
	context->texture = gs_texture_create(context->width, context->height,
	                                     GS_RGBA, 1, (const uint8_t **)&pixels, 0);
	bfree(pixels);
#endif
	
	// Draw the texture using OBS's default image drawing
	if (context->texture && effect) {
		// Use the effect's "image" parameter
		gs_eparam_t *image = gs_effect_get_param_by_name(effect, "image");
		if (image) {
			gs_effect_set_texture(image, context->texture);
			
			// Draw using the provided effect
			while (gs_effect_loop(effect, "Draw")) {
				gs_draw_sprite(context->texture, 0, context->width, context->height);
			}
		}
	}
	
	obs_leave_graphics();
}

static void scoreboard_source_tick(void *data, float seconds)
{
	UNUSED_PARAMETER(data);
	UNUSED_PARAMETER(seconds);
	// TODO: Update animations, etc.
}

struct obs_source_info scoreboard_source_info = {};

bool register_scoreboard_source()
{
	// Initialize the source info struct
	scoreboard_source_info.id = "waterpolo_scoreboard_source";
	scoreboard_source_info.type = OBS_SOURCE_TYPE_INPUT;
	scoreboard_source_info.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW;
	scoreboard_source_info.get_name = scoreboard_source_get_name;
	scoreboard_source_info.create = scoreboard_source_create;
	scoreboard_source_info.destroy = scoreboard_source_destroy;
	scoreboard_source_info.update = scoreboard_source_update;
	scoreboard_source_info.get_defaults = scoreboard_source_get_defaults;
	scoreboard_source_info.get_properties = scoreboard_source_properties;
	scoreboard_source_info.get_width = scoreboard_source_get_width;
	scoreboard_source_info.get_height = scoreboard_source_get_height;
	scoreboard_source_info.video_render = scoreboard_source_render;
	scoreboard_source_info.video_tick = scoreboard_source_tick;
	
	obs_register_source(&scoreboard_source_info);
	return true;
}

// Function to get the active scoreboard (used by WebSocket server and GUI)
scoreboard_source* get_active_scoreboard()
{
	return g_active_scoreboard;
}

// Function to update scoreboard from external sources (WebSocket/GUI)
void update_scoreboard_data(obs_data_t *data)
{
	blog(LOG_INFO, "*** UPDATE_SCOREBOARD_DATA FUNCTION CALLED ***");
	blog(LOG_ERROR, "*** UPDATE_SCOREBOARD_DATA ERROR LEVEL TEST ***");
	
	if (!g_active_scoreboard) {
		blog(LOG_WARNING, "g_active_scoreboard is NULL");
		return;
	}
	
	if (!g_active_scoreboard->source) {
		blog(LOG_WARNING, "g_active_scoreboard->source is NULL");
		return;
	}
	
	blog(LOG_INFO, "Active scoreboard found, proceeding with update");
	
	// Get current settings
	obs_data_t *settings = obs_source_get_settings(g_active_scoreboard->source);
	
	// Update with new data
	if (obs_data_has_user_value(data, "home_score"))
		obs_data_set_int(settings, "home_score", obs_data_get_int(data, "home_score"));
	if (obs_data_has_user_value(data, "away_score"))
		obs_data_set_int(settings, "away_score", obs_data_get_int(data, "away_score"));
	if (obs_data_has_user_value(data, "shot_clock"))
		obs_data_set_int(settings, "shot_clock", obs_data_get_int(data, "shot_clock"));
	if (obs_data_has_user_value(data, "game_minutes"))
		obs_data_set_int(settings, "game_minutes", obs_data_get_int(data, "game_minutes"));
	if (obs_data_has_user_value(data, "game_seconds"))
		obs_data_set_int(settings, "game_seconds", obs_data_get_int(data, "game_seconds"));
	if (obs_data_has_user_value(data, "period"))
		obs_data_set_int(settings, "period", obs_data_get_int(data, "period"));
	if (obs_data_has_user_value(data, "home_exclusions"))
		obs_data_set_int(settings, "home_exclusions", obs_data_get_int(data, "home_exclusions"));
	if (obs_data_has_user_value(data, "away_exclusions"))
		obs_data_set_int(settings, "away_exclusions", obs_data_get_int(data, "away_exclusions"));
	if (obs_data_has_user_value(data, "home_timeouts"))
		obs_data_set_int(settings, "home_timeouts", obs_data_get_int(data, "home_timeouts"));
	if (obs_data_has_user_value(data, "away_timeouts"))
		obs_data_set_int(settings, "away_timeouts", obs_data_get_int(data, "away_timeouts"));
	if (obs_data_has_user_value(data, "home_team"))
		obs_data_set_string(settings, "home_team", obs_data_get_string(data, "home_team"));
	if (obs_data_has_user_value(data, "away_team"))
		obs_data_set_string(settings, "away_team", obs_data_get_string(data, "away_team"));
	if (obs_data_has_user_value(data, "show_game_clock")) {
		bool value = obs_data_get_bool(data, "show_game_clock");
		blog(LOG_INFO, "Setting show_game_clock to %s", value ? "true" : "false");
		obs_data_set_bool(settings, "show_game_clock", value);
	}
	if (obs_data_has_user_value(data, "show_shot_clock")) {
		bool value = obs_data_get_bool(data, "show_shot_clock");
		blog(LOG_INFO, "Setting show_shot_clock to %s", value ? "true" : "false");
		obs_data_set_bool(settings, "show_shot_clock", value);
	}
	
	// Apply the updated settings
	blog(LOG_INFO, "Applying updated settings to source");
	obs_source_update(g_active_scoreboard->source, settings);
	
	obs_data_release(settings);
}
