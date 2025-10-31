#include <obs-module.h>
#include <graphics/image-file.h>
#include <util/platform.h>
#include <util/dstr.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
using namespace Gdiplus;

static ULONG_PTR g_gdiplusToken = 0;

void init_gdiplus_simple()
{
	if (g_gdiplusToken == 0) {
		GdiplusStartupInput gdiplusStartupInput;
		GdiplusStartup(&g_gdiplusToken, &gdiplusStartupInput, NULL);
		blog(LOG_INFO, "GDI+ initialized for text rendering");
	}
}

void shutdown_gdiplus_simple()
{
	if (g_gdiplusToken != 0) {
		GdiplusShutdown(g_gdiplusToken);
		g_gdiplusToken = 0;
	}
}
#endif

struct scoreboard_source {
	obs_source_t *source;
	gs_texture_t *texture;
	bool needs_update;
	
	// Scoreboard data
	int home_score;
	int away_score;
	int shot_clock;
	int game_clock_minutes;
	int game_clock_seconds;
	std::string home_team;
	std::string away_team;
	
	// Team logos
	std::string home_logo_path;
	std::string away_logo_path;
	
	// Next game preview
	std::string next_home_team;
	std::string next_away_team;
	std::string next_home_logo_path;
	std::string next_away_logo_path;
	
	// Water polo specific
	int period;
	std::string period_text; // For "Final", "5th", "Shootout"
	int home_exclusions;
	int away_exclusions;
	int home_timeouts;
	int away_timeouts;
	
	// Man-up indicators
	bool home_manup;
	bool away_manup;
	
	// Team colors (ARGB format)
	uint32_t home_color;
	uint32_t away_color;
	uint32_t home_text_color;
	uint32_t away_text_color;
	
	// Configuration
	std::string config_dir;
	
	// Display settings
	uint32_t width;
	uint32_t height;
	
	// Clock visibility options
	bool show_game_clock;
	bool show_shot_clock;
};

// Global instance for updates
static struct scoreboard_source *g_scoreboard = nullptr;

static const char *scoreboard_source_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Water Polo Scoreboard (Simple)";
}

static void *scoreboard_source_create(obs_data_t *settings, obs_source_t *source)
{
	struct scoreboard_source *context = (struct scoreboard_source *)bzalloc(sizeof(struct scoreboard_source));
	context->source = source;
	
	// Initialize default values
	context->home_score = 0;
	context->away_score = 0;
	context->shot_clock = 30;
	context->game_clock_minutes = 8;
	context->game_clock_seconds = 0;
	context->home_team = "Home";
	context->away_team = "Away";
	context->period = 1;
	context->period_text = "";
	context->home_exclusions = 0;
	context->away_exclusions = 0;
	context->home_timeouts = 3;
	context->away_timeouts = 3;
	context->home_manup = false;
	context->away_manup = false;
	context->home_color = 0xFF0080FF; // Blue
	context->away_color = 0xFFFF8000; // Orange
	context->home_text_color = 0xFFFFFFFF; // White
	context->away_text_color = 0xFFFFFFFF; // White
	context->width = 1520;
	context->height = 120;
	context->texture = nullptr;
	context->needs_update = true;
	context->show_game_clock = true;
	context->show_shot_clock = true;
	
	// Load settings
	const char *config_dir = obs_data_get_string(settings, "config_dir");
	if (config_dir && *config_dir) {
		context->config_dir = config_dir;
	}
	
	context->home_color = (uint32_t)obs_data_get_int(settings, "home_color");
	context->away_color = (uint32_t)obs_data_get_int(settings, "away_color");
	context->home_text_color = (uint32_t)obs_data_get_int(settings, "home_text_color");
	context->away_text_color = (uint32_t)obs_data_get_int(settings, "away_text_color");
	
	if (context->home_color == 0) context->home_color = 0xFF0080FF;
	if (context->away_color == 0) context->away_color = 0xFFFF8000;
	if (context->home_text_color == 0) context->home_text_color = 0xFFFFFFFF;
	if (context->away_text_color == 0) context->away_text_color = 0xFFFFFFFF;
	
	g_scoreboard = context;
	
	blog(LOG_INFO, "Water Polo Scoreboard (Simple) created - %dx%d", context->width, context->height);
	
	return context;
}

static void scoreboard_source_destroy(void *data)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	
	if (context->texture) {
		obs_enter_graphics();
		gs_texture_destroy(context->texture);
		obs_leave_graphics();
	}
	
	if (g_scoreboard == context) {
		g_scoreboard = nullptr;
	}
	
	bfree(context);
}

static void scoreboard_source_update(void *data, obs_data_t *settings)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	
	// Update config directory
	const char *config_dir = obs_data_get_string(settings, "config_dir");
	if (config_dir && *config_dir) {
		context->config_dir = config_dir;
	}
	
	// Update colors
	uint32_t home_color = (uint32_t)obs_data_get_int(settings, "home_color");
	uint32_t away_color = (uint32_t)obs_data_get_int(settings, "away_color");
	uint32_t home_text_color = (uint32_t)obs_data_get_int(settings, "home_text_color");
	uint32_t away_text_color = (uint32_t)obs_data_get_int(settings, "away_text_color");
	
	if (home_color != 0 && home_color != context->home_color) {
		context->home_color = home_color;
		context->needs_update = true;
	}
	
	if (away_color != 0 && away_color != context->away_color) {
		context->away_color = away_color;
		context->needs_update = true;
	}
	
	if (home_text_color != 0 && home_text_color != context->home_text_color) {
		context->home_text_color = home_text_color;
		context->needs_update = true;
	}
	
	if (away_text_color != 0 && away_text_color != context->away_text_color) {
		context->away_text_color = away_text_color;
		context->needs_update = true;
	}
}

static void scoreboard_source_render(void *data, gs_effect_t *effect)
{
	struct scoreboard_source *context = (struct scoreboard_source *)data;
	
	UNUSED_PARAMETER(effect);
	
	if (!context) {
		return;
	}
	
	// Recreate texture if needed
	if (!context->texture || context->needs_update) {
		context->needs_update = false;
		
		obs_enter_graphics();
		
#ifdef _WIN32
		// Use GDI+ to render with text - ESPN-style horizontal layout
		Bitmap bitmap(context->width, context->height, PixelFormat32bppARGB);
		Graphics graphics(&bitmap);
		
		graphics.SetTextRenderingHint(TextRenderingHintAntiAlias);
		graphics.SetSmoothingMode(SmoothingModeAntiAlias);
		
		// Transparent background (no background fill - fully transparent)
		graphics.Clear(Color(0, 0, 0, 0));
		
		// Fonts - ESPN style
		FontFamily fontFamily(L"Arial");
		Font teamFont(&fontFamily, 22, FontStyleBold, UnitPixel);
		Font scoreFont(&fontFamily, 48, FontStyleBold, UnitPixel);
		Font clockFont(&fontFamily, 32, FontStyleBold, UnitPixel);
		Font smallFont(&fontFamily, 16, FontStyleBold, UnitPixel);
		Font tinyFont(&fontFamily, 12, FontStyleRegular, UnitPixel);
		
		StringFormat leftFormat, centerFormat, rightFormat;
		leftFormat.SetAlignment(StringAlignmentNear);
		leftFormat.SetLineAlignment(StringAlignmentCenter);
		centerFormat.SetAlignment(StringAlignmentCenter);
		centerFormat.SetLineAlignment(StringAlignmentCenter);
		rightFormat.SetAlignment(StringAlignmentFar);
		rightFormat.SetLineAlignment(StringAlignmentCenter);
		
		SolidBrush whiteBrush(Color(255, 255, 255, 255));
		SolidBrush lightGrayBrush(Color(255, 200, 200, 200));
		SolidBrush darkBrush(Color(255, 40, 40, 40));
		
		// Create text color brushes
		uint8_t home_text_a = (context->home_text_color >> 24) & 0xFF;
		uint8_t home_text_r = (context->home_text_color >> 16) & 0xFF;
		uint8_t home_text_g = (context->home_text_color >> 8) & 0xFF;
		uint8_t home_text_b = context->home_text_color & 0xFF;
		SolidBrush homeTextBrush(Color(home_text_a, home_text_r, home_text_g, home_text_b));
		
		uint8_t away_text_a = (context->away_text_color >> 24) & 0xFF;
		uint8_t away_text_r = (context->away_text_color >> 16) & 0xFF;
		uint8_t away_text_g = (context->away_text_color >> 8) & 0xFF;
		uint8_t away_text_b = context->away_text_color & 0xFF;
		SolidBrush awayTextBrush(Color(away_text_a, away_text_r, away_text_g, away_text_b));
		
		// Team background colors
		uint8_t home_a = (context->home_color >> 24) & 0xFF;
		uint8_t home_r = (context->home_color >> 16) & 0xFF;
		uint8_t home_g = (context->home_color >> 8) & 0xFF;
		uint8_t home_b = context->home_color & 0xFF;
		SolidBrush homeBrush(Color(home_a, home_r, home_g, home_b));
		
		uint8_t away_a = (context->away_color >> 24) & 0xFF;
		uint8_t away_r = (context->away_color >> 16) & 0xFF;
		uint8_t away_g = (context->away_color >> 8) & 0xFF;
		uint8_t away_b = context->away_color & 0xFF;
		SolidBrush awayBrush(Color(away_a, away_r, away_g, away_b));
		
		// Layout: [HOME] [CLOCKS] [AWAY] | [NEXT GAME]
		// Horizontal strip at top of screen
		int leftMargin = 10;
		int yPos = 10;
		int rowHeight = 100;
		
		// HOME TEAM BOX (left side) - with rounded corners and gradient
		int homeBoxWidth = 230;
		RectF homeBoxRect(leftMargin, yPos, homeBoxWidth, rowHeight);
		
		// Create gradient brush for home team
		uint8_t home_r_dark = home_r > 30 ? home_r - 30 : 0;
		uint8_t home_g_dark = home_g > 30 ? home_g - 30 : 0;
		uint8_t home_b_dark = home_b > 30 ? home_b - 30 : 0;
		
		LinearGradientBrush homeGradientBrush(
			PointF((REAL)leftMargin, (REAL)yPos), 
			PointF((REAL)leftMargin, (REAL)(yPos + rowHeight)),
			Color(home_a, home_r, home_g, home_b),
			Color(home_a, home_r_dark, home_g_dark, home_b_dark)
		);
		
		// Draw rounded rectangle for home box
		GraphicsPath homePath;
		int cornerRadius = 8;
		homePath.AddArc((REAL)homeBoxRect.X, (REAL)homeBoxRect.Y, (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 180.0f, 90.0f);
		homePath.AddArc((REAL)(homeBoxRect.X + homeBoxRect.Width - cornerRadius * 2), (REAL)homeBoxRect.Y, (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 270.0f, 90.0f);
		homePath.AddArc((REAL)(homeBoxRect.X + homeBoxRect.Width - cornerRadius * 2), (REAL)(homeBoxRect.Y + homeBoxRect.Height - cornerRadius * 2), (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 0.0f, 90.0f);
		homePath.AddArc((REAL)homeBoxRect.X, (REAL)(homeBoxRect.Y + homeBoxRect.Height - cornerRadius * 2), (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 90.0f, 90.0f);
		homePath.CloseFigure();
		graphics.FillPath(&homeGradientBrush, &homePath);
		
		// Home team logo (left side of box)
		int logoSize = 70;
		int logoMargin = 10;
		if (!context->home_logo_path.empty()) {
			std::wstring logoPath(context->home_logo_path.begin(), context->home_logo_path.end());
			Image* homeLogo = Image::FromFile(logoPath.c_str());
			if (homeLogo && homeLogo->GetLastStatus() == Ok) {
				graphics.DrawImage(homeLogo, leftMargin + logoMargin, yPos + (rowHeight - logoSize) / 2, logoSize, logoSize);
				delete homeLogo;
			} else {
				blog(LOG_WARNING, "Failed to load home logo: %s", context->home_logo_path.c_str());
			}
		}
		
		// Calculate text area (between logo and end of box)
		int textStartX = leftMargin + logoMargin + logoSize + 10;
		int textWidth = homeBoxWidth - logoMargin - logoSize - 20;
		
		// Home team name (centered in available space)
		std::wstring home_team_w(context->home_team.begin(), context->home_team.end());
		RectF homeNameRect(textStartX, yPos + 15, textWidth, 30);
		graphics.DrawString(home_team_w.c_str(), -1, &teamFont, homeNameRect, &centerFormat, &homeTextBrush);
		
		// Home score (centered under name)
		std::wstring home_score_w = std::to_wstring(context->home_score);
		RectF homeScoreRect(textStartX, yPos + 40, textWidth, 50);
		graphics.DrawString(home_score_w.c_str(), -1, &scoreFont, homeScoreRect, &centerFormat, &homeTextBrush);
		
		// Home man-up indicator
		if (context->home_manup) {
			SolidBrush manupBrush(Color(220, 255, 215, 0)); // Bright yellow with transparency
			graphics.FillRectangle(&manupBrush, leftMargin + 5, yPos + rowHeight - 25, homeBoxWidth - 10, 20);
			Font manupFont(&fontFamily, 14, FontStyleBold, UnitPixel);
			SolidBrush manupTextBrush(Color(255, 0, 0, 0)); // Black text
			RectF manupRect(leftMargin + 5, yPos + rowHeight - 25, homeBoxWidth - 10, 20);
			graphics.DrawString(L"MAN UP", -1, &manupFont, manupRect, &centerFormat, &manupTextBrush);
		}
		
		// CENTER INFO SECTION (clocks and period) - dynamic width based on visible elements
		int centerX = leftMargin + homeBoxWidth + 10;
		
		// Calculate center width based on visible clocks
		int gameClockWidth = context->show_game_clock ? 160 : 0; // 150 + 10 spacing
		int periodWidth = 90;
		int shotClockWidth = context->show_shot_clock ? 160 : 0; // 150 + 10 spacing
		int centerPadding = 30; // 15 on each side
		int centerWidth = centerPadding + gameClockWidth + periodWidth + shotClockWidth;
		
		// Semi-transparent dark background for center section with rounded corners
		RectF centerRect((REAL)centerX, (REAL)yPos, (REAL)centerWidth, (REAL)rowHeight);
		LinearGradientBrush centerGradientBrush(
			PointF((REAL)centerX, (REAL)yPos), 
			PointF((REAL)centerX, (REAL)(yPos + rowHeight)),
			Color(200, 30, 30, 30),
			Color(200, 10, 10, 10)
		);
		
		GraphicsPath centerPath;
		int centerRadius = 8;
		centerPath.AddArc((REAL)centerRect.X, (REAL)centerRect.Y, (REAL)(centerRadius * 2), (REAL)(centerRadius * 2), 180.0f, 90.0f);
		centerPath.AddArc((REAL)(centerRect.X + centerRect.Width - centerRadius * 2), (REAL)centerRect.Y, (REAL)(centerRadius * 2), (REAL)(centerRadius * 2), 270.0f, 90.0f);
		centerPath.AddArc((REAL)(centerRect.X + centerRect.Width - centerRadius * 2), (REAL)(centerRect.Y + centerRect.Height - centerRadius * 2), (REAL)(centerRadius * 2), (REAL)(centerRadius * 2), 0.0f, 90.0f);
		centerPath.AddArc((REAL)centerRect.X, (REAL)(centerRect.Y + centerRect.Height - centerRadius * 2), (REAL)(centerRadius * 2), (REAL)(centerRadius * 2), 90.0f, 90.0f);
		centerPath.CloseFigure();
		graphics.FillPath(&centerGradientBrush, &centerPath);
		
		// Game Clock (left part of center) - only show if enabled
		if (context->show_game_clock) {
			int gameClockX = centerX + 15;
			int gameClockDisplayWidth = 150;
			std::wstringstream game_clock_ss;
			game_clock_ss << std::setw(2) << std::setfill(L'0') << context->game_clock_minutes 
			              << L":" << std::setw(2) << std::setfill(L'0') << context->game_clock_seconds;
			
			// Center the clock text vertically in the box
			RectF gameClockRect(gameClockX, yPos + (rowHeight - 50) / 2, gameClockDisplayWidth, 50);
			graphics.DrawString(game_clock_ss.str().c_str(), -1, &clockFont, gameClockRect, &centerFormat, &whiteBrush);
			
			// Label above game clock (also centered)
			RectF gameClockLabelRect(gameClockX, yPos + 10, gameClockDisplayWidth, 20);
			graphics.DrawString(L"GAME CLOCK", -1, &tinyFont, gameClockLabelRect, &centerFormat, &lightGrayBrush);
		}
		
		// Period (center of center section) - center it based on available space and vertically center
		int periodX = centerX + centerPadding/2 + gameClockWidth;
		
		// Period display - check for special text first
		std::wstringstream period_ss;
		if (!context->period_text.empty()) {
			// Use special period text (Final, 5th, Shootout)
			std::wstring period_text_w(context->period_text.begin(), context->period_text.end());
			period_ss << period_text_w;
		} else {
			// Regular quarter display
			period_ss << L"Q" << context->period;
		}
		
		// Center the period text vertically
		RectF periodRect(periodX, yPos + (rowHeight - 60) / 2, periodWidth, 60);
		Font periodFont(&fontFamily, 28, FontStyleBold, UnitPixel);
		graphics.DrawString(period_ss.str().c_str(), -1, &periodFont, periodRect, &centerFormat, &whiteBrush);
		
		// Shot Clock (right part of center) - only show if enabled
		if (context->show_shot_clock) {
			int shotClockX = periodX + periodWidth + 10;
			int shotClockDisplayWidth = 150;
			std::wstring shot_clock_w = std::to_wstring(context->shot_clock);
			
			// Center the shot clock text vertically
			RectF shotClockRect(shotClockX, yPos + (rowHeight - 50) / 2, shotClockDisplayWidth, 50);
			
			if (context->shot_clock <= 5) {
				// Create rounded rectangle for urgent shot clock
				RectF alertRect((REAL)(shotClockX - 5), (REAL)(yPos + 5), (REAL)(shotClockDisplayWidth + 10), (REAL)(rowHeight - 10));
				LinearGradientBrush redGradientBrush(
					PointF((REAL)(shotClockX - 5), (REAL)(yPos + 5)), 
					PointF((REAL)(shotClockX - 5), (REAL)(yPos + rowHeight - 5)),
					Color(255, 255, 0, 0),
					Color(255, 200, 0, 0)
				);
				
				GraphicsPath alertPath;
				int alertRadius = 6;
				alertPath.AddArc((REAL)alertRect.X, (REAL)alertRect.Y, (REAL)(alertRadius * 2), (REAL)(alertRadius * 2), 180.0f, 90.0f);
				alertPath.AddArc((REAL)(alertRect.X + alertRect.Width - alertRadius * 2), (REAL)alertRect.Y, (REAL)(alertRadius * 2), (REAL)(alertRadius * 2), 270.0f, 90.0f);
				alertPath.AddArc((REAL)(alertRect.X + alertRect.Width - alertRadius * 2), (REAL)(alertRect.Y + alertRect.Height - alertRadius * 2), (REAL)(alertRadius * 2), (REAL)(alertRadius * 2), 0.0f, 90.0f);
				alertPath.AddArc((REAL)alertRect.X, (REAL)(alertRect.Y + alertRect.Height - alertRadius * 2), (REAL)(alertRadius * 2), (REAL)(alertRadius * 2), 90.0f, 90.0f);
				alertPath.CloseFigure();
				graphics.FillPath(&redGradientBrush, &alertPath);
				graphics.DrawString(shot_clock_w.c_str(), -1, &clockFont, shotClockRect, &centerFormat, &whiteBrush);
			} else {
				graphics.DrawString(shot_clock_w.c_str(), -1, &clockFont, shotClockRect, &centerFormat, &whiteBrush);
			}
			
			// Label above shot clock (centered)
			RectF shotClockLabelRect(shotClockX, yPos + 10, shotClockDisplayWidth, 20);
			graphics.DrawString(L"SHOT CLOCK", -1, &tinyFont, shotClockLabelRect, &centerFormat, &lightGrayBrush);
		}
		
		// AWAY TEAM BOX (right side) - with rounded corners and gradient
		int awayBoxX = centerX + centerWidth + 10;
		int awayBoxWidth = 230;
		RectF awayBoxRect(awayBoxX, yPos, awayBoxWidth, rowHeight);
		
		// Create gradient brush for away team
		uint8_t away_r_dark = away_r > 30 ? away_r - 30 : 0;
		uint8_t away_g_dark = away_g > 30 ? away_g - 30 : 0;
		uint8_t away_b_dark = away_b > 30 ? away_b - 30 : 0;
		
		LinearGradientBrush awayGradientBrush(
			PointF((REAL)awayBoxX, (REAL)yPos), 
			PointF((REAL)awayBoxX, (REAL)(yPos + rowHeight)),
			Color(away_a, away_r, away_g, away_b),
			Color(away_a, away_r_dark, away_g_dark, away_b_dark)
		);
		
		// Draw rounded rectangle for away box
		GraphicsPath awayPath;
		awayPath.AddArc((REAL)awayBoxRect.X, (REAL)awayBoxRect.Y, (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 180.0f, 90.0f);
		awayPath.AddArc((REAL)(awayBoxRect.X + awayBoxRect.Width - cornerRadius * 2), (REAL)awayBoxRect.Y, (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 270.0f, 90.0f);
		awayPath.AddArc((REAL)(awayBoxRect.X + awayBoxRect.Width - cornerRadius * 2), (REAL)(awayBoxRect.Y + awayBoxRect.Height - cornerRadius * 2), (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 0.0f, 90.0f);
		awayPath.AddArc((REAL)awayBoxRect.X, (REAL)(awayBoxRect.Y + awayBoxRect.Height - cornerRadius * 2), (REAL)(cornerRadius * 2), (REAL)(cornerRadius * 2), 90.0f, 90.0f);
		awayPath.CloseFigure();
		graphics.FillPath(&awayGradientBrush, &awayPath);
		
		// Away team logo (left side of box, vertically centered)
		if (!context->away_logo_path.empty()) {
			std::wstring logoPath(context->away_logo_path.begin(), context->away_logo_path.end());
			Image* awayLogo = Image::FromFile(logoPath.c_str());
			if (awayLogo && awayLogo->GetLastStatus() == Ok) {
				graphics.DrawImage(awayLogo, awayBoxX + logoMargin, yPos + (rowHeight - logoSize) / 2, logoSize, logoSize);
				delete awayLogo;
			} else {
				blog(LOG_WARNING, "Failed to load away logo: %s", context->away_logo_path.c_str());
			}
		}
		
		// Calculate text area for away team (between logo and end of box)
		int awayTextStartX = awayBoxX + logoMargin + logoSize + 10;
		int awayTextWidth = awayBoxWidth - logoMargin - logoSize - 20;
		
		// Away team name (centered in available space)
		std::wstring away_team_w(context->away_team.begin(), context->away_team.end());
		RectF awayNameRect(awayTextStartX, yPos + 15, awayTextWidth, 30);
		graphics.DrawString(away_team_w.c_str(), -1, &teamFont, awayNameRect, &centerFormat, &awayTextBrush);
		
		// Away score (centered under name)
		std::wstring away_score_w = std::to_wstring(context->away_score);
		RectF awayScoreRect(awayTextStartX, yPos + 40, awayTextWidth, 50);
		graphics.DrawString(away_score_w.c_str(), -1, &scoreFont, awayScoreRect, &centerFormat, &awayTextBrush);
		
		// Away man-up indicator
		if (context->away_manup) {
			SolidBrush manupBrush(Color(220, 255, 215, 0)); // Bright yellow with transparency
			graphics.FillRectangle(&manupBrush, awayBoxX + 5, yPos + rowHeight - 25, awayBoxWidth - 10, 20);
			Font manupFont(&fontFamily, 14, FontStyleBold, UnitPixel);
			SolidBrush manupTextBrush(Color(255, 0, 0, 0)); // Black text
			RectF manupRect(awayBoxX + 5, yPos + rowHeight - 25, awayBoxWidth - 10, 20);
			graphics.DrawString(L"MAN UP", -1, &manupFont, manupRect, &centerFormat, &manupTextBrush);
		}
		
		// NEXT GAME PREVIEW (right side with separator)
		int nextGameX = awayBoxX + awayBoxWidth + 20;
		int nextGameWidth = 240;
		
		// Semi-transparent background for next game
		SolidBrush nextGameBrush(Color(150, 30, 30, 30));
		graphics.FillRectangle(&nextGameBrush, nextGameX, yPos, nextGameWidth, rowHeight);
		
		// "NEXT GAME" label at top
		RectF nextGameLabelRect(nextGameX, yPos + 8, nextGameWidth, 18);
		graphics.DrawString(L"NEXT GAME", -1, &tinyFont, nextGameLabelRect, &centerFormat, &lightGrayBrush);
		
		// Next game logos side by side
		int nextLogoSize = 50;
		int nextLogoY = yPos + 32;
		int spacing = 10;
		int totalWidth = (nextLogoSize * 2) + spacing;
		int startX = nextGameX + (nextGameWidth - totalWidth) / 2;
		
		// Next home logo
		if (!context->next_home_logo_path.empty()) {
			std::wstring nextHomePath(context->next_home_logo_path.begin(), context->next_home_logo_path.end());
			Image* nextHomeLogo = Image::FromFile(nextHomePath.c_str());
			if (nextHomeLogo && nextHomeLogo->GetLastStatus() == Ok) {
				graphics.DrawImage(nextHomeLogo, startX, nextLogoY, nextLogoSize, nextLogoSize);
				delete nextHomeLogo;
			}
		}
		
		// "VS" text between logos
		RectF vsRect(startX + nextLogoSize, nextLogoY, spacing, nextLogoSize);
		Font vsFont(&fontFamily, 14, FontStyleBold, UnitPixel);
		graphics.DrawString(L"vs", -1, &vsFont, vsRect, &centerFormat, &lightGrayBrush);
		
		// Next away logo
		if (!context->next_away_logo_path.empty()) {
			std::wstring nextAwayPath(context->next_away_logo_path.begin(), context->next_away_logo_path.end());
			Image* nextAwayLogo = Image::FromFile(nextAwayPath.c_str());
			if (nextAwayLogo && nextAwayLogo->GetLastStatus() == Ok) {
				graphics.DrawImage(nextAwayLogo, startX + nextLogoSize + spacing, nextLogoY, nextLogoSize, nextLogoSize);
				delete nextAwayLogo;
			}
		}
		
		// Subtle border lines for definition (no border for transparent look)
		// Pen borderPen(Color(255, 80, 80, 80), 2);
		// graphics.DrawRectangle(&borderPen, 0, 0, context->width - 1, context->height - 1);
		
		// Lock bitmap and copy to texture
		BitmapData bitmapData;
		Rect rect(0, 0, context->width, context->height);
		Status lockStatus = bitmap.LockBits(&rect, ImageLockModeRead, PixelFormat32bppARGB, &bitmapData);
		
		if (lockStatus == Ok) {
			// Destroy old texture
			if (context->texture) {
				gs_texture_destroy(context->texture);
				context->texture = nullptr;
			}
			
			// Copy to contiguous buffer
			uint32_t *pixels = (uint32_t *)bzalloc(context->width * context->height * sizeof(uint32_t));
			uint8_t *src = (uint8_t *)bitmapData.Scan0;
			
			for (uint32_t y = 0; y < context->height; y++) {
				uint32_t *srcRow = (uint32_t *)(src + y * bitmapData.Stride);
				uint32_t *dstRow = pixels + y * context->width;
				
				for (uint32_t x = 0; x < context->width; x++) {
					// GDI+ uses ARGB, keep as-is for BGRA
					uint32_t pixel = srcRow[x];
					uint8_t a = (pixel >> 24) & 0xFF;
					uint8_t r = (pixel >> 16) & 0xFF;
					uint8_t g = (pixel >> 8) & 0xFF;
					uint8_t b = pixel & 0xFF;
					dstRow[x] = (a << 24) | (r << 16) | (g << 8) | b;
				}
			}
			
			context->texture = gs_texture_create(context->width, context->height,
			                                     GS_BGRA, 1, (const uint8_t **)&pixels, 0);
			
			bfree(pixels);
			bitmap.UnlockBits(&bitmapData);
		}
#else
		// Simple colored texture for non-Windows
		uint32_t *pixels = (uint32_t *)bzalloc(context->width * context->height * sizeof(uint32_t));
		for (uint32_t i = 0; i < context->width * context->height; i++) {
			pixels[i] = 0xFF1a1a1a;
		}
		
		if (context->texture) {
			gs_texture_destroy(context->texture);
		}
		
		context->texture = gs_texture_create(context->width, context->height,
		                                     GS_BGRA, 1, (const uint8_t **)&pixels, 0);
		bfree(pixels);
#endif
		
		obs_leave_graphics();
	}
	
	// Draw the texture
	if (context->texture) {
		obs_source_draw(context->texture, 0, 0, context->width, context->height, false);
	}
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

void update_scoreboard_data(obs_data_t *data)
{
	if (!g_scoreboard || !data) {
		return;
	}
	
	// Extract data from obs_data_t
	g_scoreboard->home_score = (int)obs_data_get_int(data, "home_score");
	g_scoreboard->away_score = (int)obs_data_get_int(data, "away_score");
	g_scoreboard->shot_clock = (int)obs_data_get_int(data, "shot_clock");
	g_scoreboard->game_clock_minutes = (int)obs_data_get_int(data, "game_clock_minutes");
	g_scoreboard->game_clock_seconds = (int)obs_data_get_int(data, "game_clock_seconds");
	
	const char *home_team = obs_data_get_string(data, "home_team");
	const char *away_team = obs_data_get_string(data, "away_team");
	if (home_team && *home_team) g_scoreboard->home_team = home_team;
	if (away_team && *away_team) g_scoreboard->away_team = away_team;
	
	// Update logo paths if provided
	const char *home_logo = obs_data_get_string(data, "home_logo_path");
	const char *away_logo = obs_data_get_string(data, "away_logo_path");
	if (home_logo && *home_logo) g_scoreboard->home_logo_path = home_logo;
	if (away_logo && *away_logo) g_scoreboard->away_logo_path = away_logo;
	
	// Update next game data if provided
	const char *next_home_team = obs_data_get_string(data, "next_home_team");
	const char *next_away_team = obs_data_get_string(data, "next_away_team");
	const char *next_home_logo = obs_data_get_string(data, "next_home_logo_path");
	const char *next_away_logo = obs_data_get_string(data, "next_away_logo_path");
	if (next_home_team && *next_home_team) g_scoreboard->next_home_team = next_home_team;
	if (next_away_team && *next_away_team) g_scoreboard->next_away_team = next_away_team;
	if (next_home_logo && *next_home_logo) g_scoreboard->next_home_logo_path = next_home_logo;
	if (next_away_logo && *next_away_logo) g_scoreboard->next_away_logo_path = next_away_logo;
	
	g_scoreboard->period = (int)obs_data_get_int(data, "period");
	
	// Update period text if provided (for Final, 5th, Shootout)
	const char *period_text = obs_data_get_string(data, "period_text");
	if (period_text) {
		g_scoreboard->period_text = period_text;
	}
	
	g_scoreboard->home_exclusions = (int)obs_data_get_int(data, "home_exclusions");
	g_scoreboard->away_exclusions = (int)obs_data_get_int(data, "away_exclusions");
	g_scoreboard->home_timeouts = (int)obs_data_get_int(data, "home_timeouts");
	g_scoreboard->away_timeouts = (int)obs_data_get_int(data, "away_timeouts");
	
	// Update man-up indicators
	g_scoreboard->home_manup = obs_data_get_bool(data, "home_manup");
	g_scoreboard->away_manup = obs_data_get_bool(data, "away_manup");
	
	// Update colors if provided
	uint32_t home_color = (uint32_t)obs_data_get_int(data, "home_color");
	uint32_t away_color = (uint32_t)obs_data_get_int(data, "away_color");
	uint32_t home_text_color = (uint32_t)obs_data_get_int(data, "home_text_color");
	uint32_t away_text_color = (uint32_t)obs_data_get_int(data, "away_text_color");
	
	if (home_color != 0) g_scoreboard->home_color = home_color;
	if (away_color != 0) g_scoreboard->away_color = away_color;
	if (home_text_color != 0) g_scoreboard->home_text_color = home_text_color;
	if (away_text_color != 0) g_scoreboard->away_text_color = away_text_color;
	
	// Update clock visibility if provided
	if (obs_data_has_user_value(data, "show_game_clock")) {
		bool value = obs_data_get_bool(data, "show_game_clock");
		blog(LOG_INFO, "Simple: Setting show_game_clock to %s", value ? "true" : "false");
		g_scoreboard->show_game_clock = value;
	}
	if (obs_data_has_user_value(data, "show_shot_clock")) {
		bool value = obs_data_get_bool(data, "show_shot_clock");
		blog(LOG_INFO, "Simple: Setting show_shot_clock to %s", value ? "true" : "false");
		g_scoreboard->show_shot_clock = value;
	}
	
	// Mark for update on next render
	g_scoreboard->needs_update = true;
	
	blog(LOG_INFO, "Scoreboard updated: %s %d - %d %s", 
	     g_scoreboard->home_team.c_str(), g_scoreboard->home_score,
	     g_scoreboard->away_score, g_scoreboard->away_team.c_str());
}

static obs_properties_t *scoreboard_source_get_properties(void *data)
{
	UNUSED_PARAMETER(data);
	
	obs_properties_t *props = obs_properties_create();
	
	obs_properties_add_path(props, "config_dir", "Configuration Directory",
	                        OBS_PATH_DIRECTORY, nullptr, nullptr);
	
	obs_properties_add_color(props, "home_color", "Home Team Color");
	obs_properties_add_color(props, "away_color", "Away Team Color");
	
	return props;
}

static void scoreboard_source_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_int(settings, "home_color", 0xFF0080FF); // Blue
	obs_data_set_default_int(settings, "away_color", 0xFFFF8000); // Orange
}

void register_scoreboard_source()
{
	struct obs_source_info info = {};
	info.id = "water_polo_scoreboard_simple";
	info.type = OBS_SOURCE_TYPE_INPUT;
	info.output_flags = OBS_SOURCE_VIDEO;
	info.get_name = scoreboard_source_get_name;
	info.create = scoreboard_source_create;
	info.destroy = scoreboard_source_destroy;
	info.update = scoreboard_source_update;
	info.get_properties = scoreboard_source_get_properties;
	info.get_defaults = scoreboard_source_get_defaults;
	info.video_render = scoreboard_source_render;
	info.get_width = scoreboard_source_get_width;
	info.get_height = scoreboard_source_get_height;
	
	obs_register_source(&info);
	
	blog(LOG_INFO, "Registered water_polo_scoreboard_simple source");
	
#ifdef USE_CNN_OCR
	// Register histogram visualization source
	extern struct obs_source_info histogram_viz_source_info;
	extern void init_histogram_viz_source_info();
	init_histogram_viz_source_info();
	obs_register_source(&histogram_viz_source_info);
	blog(LOG_INFO, "Registered histogram_viz_source");
	
	// Register averaged frame visualization source
	extern struct obs_source_info averaged_frame_viz_source_info;
	extern void init_averaged_frame_viz_source_info();
	init_averaged_frame_viz_source_info();
	obs_register_source(&averaged_frame_viz_source_info);
	blog(LOG_INFO, "Registered averaged_frame_viz_source");
#endif
}
