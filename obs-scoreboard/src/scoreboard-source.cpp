#include <obs-module.h>
#include <graphics/image-file.h>
#include <util/platform.h>
#include <util/dstr.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <vector>

#ifndef _WIN32
#include <QImage>
#include <QPainter>
#include <QPainterPath>
#include <QLinearGradient>
#include <QtSvg/QSvgRenderer>
#include <QString>
#include <QFont>
#endif

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

#include "scoreboard-source.h"

// Global instance for updates
static struct scoreboard_source *g_scoreboard = nullptr;

// Getter function for external access
struct scoreboard_source *get_global_scoreboard() {
	return g_scoreboard;
}

// Calculate team record (wins-losses-ties) from schedule
void calculate_team_record(const std::string& team_name, const std::string& config_dir, int& wins, int& losses, int& ties) {
	wins = 0;
	losses = 0;
	ties = 0;
	
	if (team_name.empty() || config_dir.empty()) return;
	
	std::string schedule_path = config_dir + "/schedule.csv";
	std::ifstream file(schedule_path);
	if (!file.is_open()) {
		blog(LOG_WARNING, "[Scoreboard] Could not open schedule.csv to calculate record");
		return;
	}
	
	std::string line;
	bool first_line = true;
	
	while (std::getline(file, line)) {
		if (first_line) {
			first_line = false;
			continue;
		}
		if (line.empty()) continue;
		
		// Parse CSV line
		std::vector<std::string> fields;
		std::string field;
		bool in_quotes = false;
		
		for (size_t i = 0; i < line.size(); i++) {
			char c = line[i];
			if (c == '"') {
				in_quotes = !in_quotes;
			} else if (c == ',' && !in_quotes) {
				fields.push_back(field);
				field.clear();
			} else {
				field += c;
			}
		}
		fields.push_back(field);
		
		if (fields.size() >= 6) {
			std::string home = fields[1];
			std::string away = fields[2];
			std::string home_score_str = fields[3];
			std::string away_score_str = fields[4];
			std::string winner = fields[5];
			
			// Check if this team played and if there's a result
			if ((home == team_name || away == team_name)) {
				// Check if game has been played (has scores)
				if (!home_score_str.empty() && !away_score_str.empty()) {
					if (winner.empty() || winner == "Tie") {
						// Tie game
						ties++;
					} else if (winner == team_name) {
						wins++;
					} else {
						losses++;
					}
				}
			}
		}
	}
	
	blog(LOG_INFO, "[Scoreboard] Team %s record: %d-%d-%d", team_name.c_str(), wins, losses, ties);
}

static const char *scoreboard_source_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Water Polo Scoreboard";
}

static void *scoreboard_source_create(obs_data_t *settings, obs_source_t *source)
{
	struct scoreboard_source *context = (struct scoreboard_source *)bzalloc(sizeof(struct scoreboard_source));
	context->source = source;
	
	// Non-zero defaults (bzalloc zeroes the rest)
	context->shot_clock         = 30;
	context->game_clock_minutes = 8;
	context->home_team          = "Home";
	context->away_team          = "Away";
	context->period             = 1;
	context->home_timeouts      = 3;
	context->away_timeouts      = 3;
	context->home_color         = 0xFF0080FF; // Blue
	context->away_color         = 0xFFFF8000; // Orange
	context->home_text_color    = 0xFFFFFFFF; // White
	context->away_text_color    = 0xFFFFFFFF; // White
	context->width              = 1520;
	context->height             = 120;
	context->needs_update       = true;
	context->show_game_clock    = true;
	context->show_shot_clock    = true;
	context->show_records       = true;
	
	// Load settings
	const char *config_dir = obs_data_get_string(settings, "config_dir");
	if (config_dir && *config_dir) {
		context->config_dir = config_dir;
	}
	
	context->show_records = obs_data_get_bool(settings, "show_records");
	context->home_color = (uint32_t)obs_data_get_int(settings, "home_color");
	context->away_color = (uint32_t)obs_data_get_int(settings, "away_color");
	context->home_text_color = (uint32_t)obs_data_get_int(settings, "home_text_color");
	context->away_text_color = (uint32_t)obs_data_get_int(settings, "away_text_color");
	
	if (context->home_color == 0) context->home_color = 0xFF0080FF;
	if (context->away_color == 0) context->away_color = 0xFFFF8000;
	if (context->home_text_color == 0) context->home_text_color = 0xFFFFFFFF;
	if (context->away_text_color == 0) context->away_text_color = 0xFFFFFFFF;
	
	g_scoreboard = context;
	
	blog(LOG_INFO, "[Scoreboard] Created %dx%d", context->width, context->height);
	
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
	
	// Update show records preference
	bool show_records = obs_data_get_bool(settings, "show_records");
	if (show_records != context->show_records) {
		context->show_records = show_records;
		context->needs_update = true;
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

    if (!context) {
            return;
    }

    uint64_t current_time = os_gettime_ns();
    bool properties_changed = false;

    if (context->home_manup && (current_time - context->home_manup_start_ns) > 30000000000ULL) {
        context->home_manup = false;
        properties_changed = true;
    }
    if (context->away_manup && (current_time - context->away_manup_start_ns) > 30000000000ULL) {
        context->away_manup = false;
        properties_changed = true;
    }
    if (properties_changed) {
        context->needs_update = true;
    }

	
	UNUSED_PARAMETER(effect);
	
	
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
	Font recordFont(&fontFamily, 15, FontStyleBold, UnitPixel);
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
		int homeBoxWidth = 170;
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
		int logoMargin = 5;
		if (!context->home_logo_path.empty()) {
			std::wstring logoPath(context->home_logo_path.begin(), context->home_logo_path.end());
			Image* homeLogo = Image::FromFile(logoPath.c_str());
			if (homeLogo && homeLogo->GetLastStatus() == Ok) {
				graphics.DrawImage(homeLogo, leftMargin + logoMargin, yPos + (rowHeight - logoSize) / 2 - 8, logoSize, logoSize);
				delete homeLogo;
			} else {
				blog(LOG_WARNING, "Failed to load home logo: %s", context->home_logo_path.c_str());
			}
		}
		
		// Home team record under logo (if enabled)
		if (context->show_records) {
			std::wstring home_record = L"(" + std::to_wstring(context->home_wins) + L"-" + std::to_wstring(context->home_losses) + L"-" + std::to_wstring(context->home_ties) + L")";
			RectF homeRecordRect(leftMargin + logoMargin, yPos + (rowHeight - logoSize) / 2 + logoSize - 3, logoSize, 18);
			graphics.DrawString(home_record.c_str(), -1, &recordFont, homeRecordRect, &centerFormat, &homeTextBrush);
		}
		
		// Calculate text area (between logo and end of box)
		int textStartX = leftMargin + logoMargin + logoSize + 5;
		int textWidth = homeBoxWidth - logoMargin - logoSize - 10;
		
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
			std::wstring shot_clock_w = (context->shot_clock < 10 ? L"0" : L"") + std::to_wstring(context->shot_clock);
			
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
		int awayBoxWidth = 170;
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
				graphics.DrawImage(awayLogo, awayBoxX + logoMargin, yPos + (rowHeight - logoSize) / 2 - 8, logoSize, logoSize);
				delete awayLogo;
			} else {
				blog(LOG_WARNING, "Failed to load away logo: %s", context->away_logo_path.c_str());
			}
		}
		
		// Away team record under logo (if enabled)
		if (context->show_records) {
			std::wstring away_record = L"(" + std::to_wstring(context->away_wins) + L"-" + std::to_wstring(context->away_losses) + L"-" + std::to_wstring(context->away_ties) + L")";
			RectF awayRecordRect(awayBoxX + logoMargin, yPos + (rowHeight - logoSize) / 2 + logoSize - 3, logoSize, 18);
			graphics.DrawString(away_record.c_str(), -1, &recordFont, awayRecordRect, &centerFormat, &awayTextBrush);
		}
		
		// Calculate text area for away team (between logo and end of box)
		int awayTextStartX = awayBoxX + logoMargin + logoSize + 5;
		int awayTextWidth = awayBoxWidth - logoMargin - logoSize - 10;
		
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
		
		// NEXT GAME PREVIEW (right side) — only shown when a next game exists
		if (!context->next_home_team.empty()) {
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
               // macOS/Linux: QPainter rendering — mirrors the GDI+ layout above
               {
                       QImage image(context->width, context->height, QImage::Format_ARGB32);
                       image.fill(Qt::transparent);
                       QPainter painter(&image);
                       painter.setRenderHint(QPainter::Antialiasing);
                       painter.setRenderHint(QPainter::TextAntialiasing);

                       QFont teamFont("Arial", 22, QFont::Bold);
                       QFont scoreFont("Arial", 48, QFont::Bold);
                       QFont clockFont("Arial", 32, QFont::Bold);
                       QFont recordFont("Arial", 15, QFont::Bold);
                       QFont tinyFont("Arial", 12, QFont::Normal);
                       QFont periodFont("Arial", 28, QFont::Bold);
                       QFont manupFont("Arial", 14, QFont::Bold);
                       QFont vsFont("Arial", 14, QFont::Bold);

                       // Unpack stored ARGB colors
                       auto unpack = [](uint32_t c) -> QColor {
                               return QColor(
                                       (c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF,
                                       (c >> 24) & 0xFF
                               );
                       };
                       QColor homeColor    = unpack(context->home_color);
                       QColor awayColor    = unpack(context->away_color);
                       QColor homeText     = unpack(context->home_text_color);
                       QColor awayText     = unpack(context->away_text_color);

                       auto darker = [](QColor c) -> QColor {
                               return QColor(
                                       std::max(0, c.red()   - 30),
                                       std::max(0, c.green() - 30),
                                       std::max(0, c.blue()  - 30),
                                       c.alpha()
                               );
                       };

                       // Draw a rounded-rect gradient fill
                       auto fillRounded = [&](int x, int y, int w, int h, QColor top, QColor bot) {
                               QPainterPath p;
                               p.addRoundedRect(x, y, w, h, 8, 8);
                               QLinearGradient g(x, y, x, y + h);
                               g.setColorAt(0, top); g.setColorAt(1, bot);
                               painter.fillPath(p, QBrush(g));
                       };

                       // Draw text centred (horizontally and vertically) inside a rect
                       auto drawText = [&](const QString &text, int x, int y, int w, int h,
                                           const QFont &font, const QColor &color,
                                           Qt::Alignment halign = Qt::AlignHCenter) {
                               painter.setFont(font);
                               painter.setPen(color);
                               painter.drawText(QRect(x, y, w, h),
                                                halign | Qt::AlignVCenter, text);
                       };

                       // Draw a logo (SVG or raster) into a rect, preserving aspect ratio
                       auto drawLogo = [&](const std::string &path, int x, int y, int w, int h) {
                               if (path.empty()) return;
                               QString qp = QString::fromStdString(path);
                               if (qp.toLower().endsWith(".svg")) {
                                       QSvgRenderer r(qp);
                                       if (r.isValid()) {
                                               QSize svgSize = r.defaultSize();
                                               QRectF target(x, y, w, h);
                                               if (svgSize.isValid() && svgSize.width() > 0 && svgSize.height() > 0) {
                                                       double ar = (double)svgSize.width() / svgSize.height();
                                                       double tw = w, th = h;
                                                       if (ar > 1.0) th = w / ar;
                                                       else tw = h * ar;
                                                       target = QRectF(x + (w - tw) / 2.0, y + (h - th) / 2.0, tw, th);
                                               }
                                               r.render(&painter, target);
                                       }
                               } else {
                                       QImage img(qp);
                                       if (!img.isNull()) {
                                               QImage scaled = img.scaled(w, h, Qt::KeepAspectRatio, Qt::SmoothTransformation);
                                               int ox = x + (w - scaled.width()) / 2;
                                               int oy = y + (h - scaled.height()) / 2;
                                               painter.drawImage(QRect(ox, oy, scaled.width(), scaled.height()), scaled);
                                       }
                               }
                       };

                       // ── LAYOUT: TBS NCAA-style dark bar ──────────────────
                       const int W = (int)context->width;
                       const int H = (int)context->height;
                       const int yPos = 0;
                       const int rowHeight = H;
                       const int gapW = 4;

                       // Unified dark background spanning the full scoreboard
                       {
                               QPainterPath bgPath;
                               bgPath.addRoundedRect(0, 0, W, H, 5, 5);
                               QLinearGradient bg(0, 0, 0, H);
                               bg.setColorAt(0, QColor(22, 22, 28, 250));
                               bg.setColorAt(1, QColor(12, 12, 16, 250));
                               painter.fillPath(bgPath, QBrush(bg));
                       }

                       // Reference widths for proportional column sizing
                       int gcW0      = context->show_game_clock ? 160 : 0;
                       int scW0      = context->show_shot_clock ? 100 : 0;
                       int periodW0  = 80;
                       int clockBase = periodW0 + gcW0 + scW0 + 30;
                       int teamBase  = 230;
                       int nextBase  = 180;
                       int totalBase = teamBase + teamBase + clockBase + nextBase;

                       // Proportional column widths: HOME | AWAY | CLOCKS | NEXT
                       int contentAvail = W - 3 * gapW;
                       int homeBoxWidth = contentAvail * teamBase  / totalBase;
                       int awayBoxWidth = homeBoxWidth;
                       int clockWidth   = contentAvail * clockBase / totalBase;
                       int nextW        = contentAvail - homeBoxWidth - awayBoxWidth - clockWidth;

                       int awayBoxX = homeBoxWidth + gapW;
                       int clockX   = awayBoxX + awayBoxWidth + gapW;
                       int nextX    = clockX + clockWidth + gapW;

                       double teamScale = (double)homeBoxWidth / teamBase;
                       int logoSize     = std::min(rowHeight - 22, std::max(20, (int)(68 * teamScale)));
                       int logoMargin   = std::max(4, (int)(6 * teamScale));
                       int logoAreaW    = logoSize + 2 * logoMargin;
                       int scoreAreaW   = std::max(60, (int)(56 * teamScale));

                       // Horizontal split line position (~56% down) — same across all sections
                       // Upper half: large values (team name, score, period, clock)
                       // Lower half: labels and secondary info
                       const int lineY = (int)(rowHeight * 0.56);

                       // Shrink font until text fits within maxW pixels
                       auto fitFont = [&](const QFont &base, const QString &text, int maxW) -> QFont {
                               QFont f = base;
                               QFontMetrics fm(f);
                               while (f.pointSize() > 7 && fm.horizontalAdvance(text) > maxW) {
                                       f.setPointSize(f.pointSize() - 1);
                                       fm = QFontMetrics(f);
                               }
                               return f;
                       };

                       // TBS-style team box:
                       //  [solid-color logo block] [gradient→dark] [team name | SCORE ]
                       //  ──── team-colored 2px separator line ────────────────────────
                       //  [record  EXCL:N  TO:N                 ] [MAN UP badge       ]
                       auto drawTeamBox = [&](int boxX, int boxW,
                                              const std::string &teamName, int score,
                                              int wins, int losses, int ties,
                                              const std::string &logoPath,
                                              QColor tc, QColor textColor,
                                              int exclusions, int timeouts, bool manup) {
                               int nameAreaW = boxW - logoAreaW - scoreAreaW;

                               // Solid team-color logo block (full height)
                               painter.fillRect(boxX, yPos, logoAreaW, rowHeight, tc);

                               // Gradient fade: team color → transparent, into name area
                               {
                                       int gradW = std::min(nameAreaW, 80);
                                       QLinearGradient fadeGrad(boxX + logoAreaW, yPos,
                                                                boxX + logoAreaW + gradW, yPos);
                                       fadeGrad.setColorAt(0, QColor(tc.red(), tc.green(), tc.blue(), 170));
                                       fadeGrad.setColorAt(1, QColor(tc.red(), tc.green(), tc.blue(), 0));
                                       painter.fillRect(boxX + logoAreaW, yPos, gradW, rowHeight,
                                                        QBrush(fadeGrad));
                               }

                               // Logo centered in colored block
                               drawLogo(logoPath, boxX + logoMargin,
                                        yPos + (rowHeight - logoSize) / 2, logoSize, logoSize);

                               // Upper portion: team name (left) | score (right)
                               int nX = boxX + logoAreaW + 6;
                               int nW = nameAreaW - 8;
                               int sX = boxX + logoAreaW + nameAreaW;
                               {
                                       QString name = QString::fromStdString(teamName);
                                       drawText(name, nX, yPos + 6, nW, lineY - 8,
                                                fitFont(teamFont, name, nW), textColor, Qt::AlignLeft);
                               }
                               drawText(QString::number(score),
                                        sX, yPos + 4, scoreAreaW, lineY - 6,
                                        scoreFont, Qt::white);

                               // Team-colored separator line
                               painter.fillRect(boxX + logoAreaW, lineY,
                                                nameAreaW + scoreAreaW, 2, tc);

                               // Lower portion: record + EXCL/TO info | MAN UP badge
                               {
                                       QFont subFont("Arial", 10, QFont::Normal);
                                       int subY = lineY + 4;
                                       int subH = rowHeight - lineY - 4;
                                       QString sub;
                                       if (context->show_records)
                                               sub = QString("%1-%2-%3  ")
                                                       .arg(wins).arg(losses).arg(ties);
                                       sub += QString("EXCL:%1  TO:%2")
                                               .arg(exclusions).arg(timeouts);
                                       drawText(sub, nX, subY, nW, subH,
                                                subFont, QColor(185, 185, 190), Qt::AlignLeft);
                                       if (manup) {
                                               QFont muFont("Arial", 9, QFont::Bold);
                                               int muW = scoreAreaW - 8;
                                               QPainterPath mp;
                                               mp.addRoundedRect(sX + 4, subY + 2, muW, subH - 4, 3, 3);
                                               painter.fillPath(mp, QColor(255, 200, 0));
                                               drawText("MAN UP", sX + 4, subY + 2, muW, subH - 4,
                                                        muFont, Qt::black);
                                       }
                               }
                       };

                       // ── HOME BOX ──────────────────────────────────────────
                       drawTeamBox(0, homeBoxWidth,
                                   context->home_team, context->home_score,
                                   context->home_wins, context->home_losses, context->home_ties,
                                   context->home_logo_path, homeColor, homeText,
                                   context->home_exclusions, context->home_timeouts,
                                   context->home_manup);

                       // ── AWAY BOX ──────────────────────────────────────────
                       drawTeamBox(awayBoxX, awayBoxWidth,
                                   context->away_team, context->away_score,
                                   context->away_wins, context->away_losses, context->away_ties,
                                   context->away_logo_path, awayColor, awayText,
                                   context->away_exclusions, context->away_timeouts,
                                   context->away_manup);

                       // ── CLOCK SECTION ─────────────────────────────────────
                       // Slightly lighter dark panel to distinguish from team boxes
                       painter.fillRect(clockX, yPos, clockWidth, rowHeight, QColor(28, 28, 36, 255));
                       painter.fillRect(clockX, yPos + 8, 1, rowHeight - 16,
                                        QColor(255, 255, 255, 25));
                       {
                               int pInner  = clockWidth * periodW0 / clockBase;
                               int gcInner = context->show_game_clock
                                       ? (clockWidth * gcW0 / clockBase) : 0;
                               int scInner = context->show_shot_clock
                                       ? (clockWidth * scW0 / clockBase) : 0;
                               int hPad    = (clockWidth - pInner - gcInner - scInner) / 2;

                               // Period
                               int pX = clockX + hPad;
                               QString pText = context->period_text.empty()
                                       ? QString("Q%1").arg(context->period)
                                       : QString::fromStdString(context->period_text);
                               drawText(pText, pX, yPos + 4, pInner, lineY - 6, periodFont, Qt::white);
                               painter.fillRect(pX, lineY, pInner, 2, QColor(80, 80, 100));
                               drawText("PERIOD", pX, lineY + 4, pInner, rowHeight - lineY - 4,
                                        tinyFont, QColor(160, 160, 180));
                               painter.fillRect(pX + pInner, yPos + 8, 1, rowHeight - 16,
                                                QColor(255, 255, 255, 25));

                               // Game clock
                               int gcX = pX + pInner + 1;
                               if (context->show_game_clock) {
                                       QString gc = QString("%1:%2")
                                               .arg(context->game_clock_minutes, 2, 10, QChar('0'))
                                               .arg(context->game_clock_seconds, 2, 10, QChar('0'));
                                       drawText(gc, gcX, yPos + 4, gcInner, lineY - 6,
                                                clockFont, Qt::white);
                                       painter.fillRect(gcX, lineY, gcInner, 2, QColor(80, 80, 100));
                                       drawText("GAME CLOCK", gcX, lineY + 4, gcInner,
                                                rowHeight - lineY - 4, tinyFont, QColor(160, 160, 180));
                                       painter.fillRect(gcX + gcInner, yPos + 8, 1, rowHeight - 16,
                                                        QColor(255, 255, 255, 25));
                               }

                               // Shot clock
                               int scX = gcX + gcInner;
                               if (context->show_shot_clock) {
                                       if (context->shot_clock <= 5) {
                                               QPainterPath ap;
                                               ap.addRoundedRect(scX - 3, yPos + 3,
                                                                 scInner + 6, lineY - 6, 4, 4);
                                               QLinearGradient rg(scX, yPos + 3, scX, lineY - 6);
                                               rg.setColorAt(0, QColor(220, 30, 30));
                                               rg.setColorAt(1, QColor(160, 0, 0));
                                               painter.fillPath(ap, QBrush(rg));
                                       }
                                       drawText(QString::number(context->shot_clock),
                                                scX, yPos + 4, scInner, lineY - 6,
                                                clockFont, Qt::white);
                                       painter.fillRect(scX, lineY, scInner, 2,
                                                        context->shot_clock <= 5
                                                        ? QColor(200, 30, 30) : QColor(80, 80, 100));
                                       drawText("SHOT CLK", scX, lineY + 4, scInner,
                                                rowHeight - lineY - 4, tinyFont, QColor(160, 160, 180));
                               }
                       }

                       // ── NEXT GAME PREVIEW ─────────────────────────────────
                       painter.fillRect(nextX, yPos, nextW, rowHeight, QColor(20, 20, 26, 255));
                       painter.fillRect(nextX, yPos + 8, 1, rowHeight - 16, QColor(255, 255, 255, 25));
                       {
                               int nlSize   = std::min(lineY - 8, std::max(26, (int)(38 * teamScale)));
                               int nlSpacing = std::max(4, (int)(6 * teamScale));
                               int nlY      = yPos + (lineY - nlSize) / 2;
                               int nlTotal  = nlSize * 2 + nlSpacing;
                               int nlStartX = nextX + (nextW - nlTotal) / 2;
                               drawLogo(context->next_home_logo_path, nlStartX, nlY, nlSize, nlSize);
                               drawText("vs", nlStartX + nlSize, nlY, nlSpacing, nlSize,
                                        vsFont, QColor(150, 150, 160));
                               drawLogo(context->next_away_logo_path,
                                        nlStartX + nlSize + nlSpacing, nlY, nlSize, nlSize);
                               painter.fillRect(nextX, lineY, nextW, 2, QColor(80, 80, 100));
                               drawText("NEXT GAME", nextX, lineY + 4, nextW, rowHeight - lineY - 4,
                                        tinyFont, QColor(160, 160, 180));
                       }

                       painter.end();

                       // Upload QImage pixels to OBS texture
                       // QImage::Format_ARGB32 on little-endian (ARM/x86) has byte order B,G,R,A
                       // which matches GS_BGRA — no pixel-format conversion needed.
                       if (context->texture) {
                               gs_texture_destroy(context->texture);
                               context->texture = nullptr;
                       }
                       const uint8_t *bits = image.constBits();
                       context->texture = gs_texture_create(context->width, context->height,
                                                            GS_BGRA, 1, &bits, 0);
		}
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

// Defined in control-panel.cpp — syncs UI spinboxes after a WebSocket update
// so clock timers don't overwrite the new state on their next tick.
void sync_control_panel_ui();

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
	if (home_team && *home_team) {
		g_scoreboard->home_team = home_team;
		// Calculate home team record
		calculate_team_record(g_scoreboard->home_team, g_scoreboard->config_dir, 
		                      g_scoreboard->home_wins, g_scoreboard->home_losses, g_scoreboard->home_ties);
	}
	if (away_team && *away_team) {
		g_scoreboard->away_team = away_team;
		// Calculate away team record
		calculate_team_record(g_scoreboard->away_team, g_scoreboard->config_dir, 
		                      g_scoreboard->away_wins, g_scoreboard->away_losses, g_scoreboard->away_ties);
	}
	
	// Update logo paths if provided
	const char *home_logo = obs_data_get_string(data, "home_logo_path");
	const char *away_logo = obs_data_get_string(data, "away_logo_path");
	if (home_logo && *home_logo) g_scoreboard->home_logo_path = home_logo;
	if (away_logo && *away_logo) g_scoreboard->away_logo_path = away_logo;
	
	// Update next game data only when explicitly provided (so score updates
	// from WebSocket don't wipe the next-game panel set by the control panel).
	// Use obs_data_has_user_value so an explicitly sent empty string still clears it.
	if (obs_data_has_user_value(data, "next_home_team"))
		g_scoreboard->next_home_team = obs_data_get_string(data, "next_home_team");
	if (obs_data_has_user_value(data, "next_away_team"))
		g_scoreboard->next_away_team = obs_data_get_string(data, "next_away_team");
	if (obs_data_has_user_value(data, "next_home_logo_path"))
		g_scoreboard->next_home_logo_path = obs_data_get_string(data, "next_home_logo_path");
	if (obs_data_has_user_value(data, "next_away_logo_path"))
		g_scoreboard->next_away_logo_path = obs_data_get_string(data, "next_away_logo_path");
	
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
       bool new_home_manup = obs_data_get_bool(data, "home_manup");
   if (new_home_manup && !g_scoreboard->home_manup) {
       g_scoreboard->home_manup_start_ns = os_gettime_ns();
   }
   g_scoreboard->home_manup = new_home_manup;
       bool new_away_manup = obs_data_get_bool(data, "away_manup");
   if (new_away_manup && !g_scoreboard->away_manup) {
       g_scoreboard->away_manup_start_ns = os_gettime_ns();
   }
   g_scoreboard->away_manup = new_away_manup;
	
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
		blog(LOG_INFO, "[Scoreboard] show_game_clock = %s", value ? "true" : "false");
		g_scoreboard->show_game_clock = value;
	}
	if (obs_data_has_user_value(data, "show_shot_clock")) {
		bool value = obs_data_get_bool(data, "show_shot_clock");
		blog(LOG_INFO, "[Scoreboard] show_shot_clock = %s", value ? "true" : "false");
		g_scoreboard->show_shot_clock = value;
	}
	
	// Mark for update on next render
	g_scoreboard->needs_update = true;

	// Sync control panel spinboxes so clock timers don't overwrite these values
	sync_control_panel_ui();

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
	
	obs_properties_add_bool(props, "show_records", "Show Team Records");
	
	obs_properties_add_color(props, "home_color", "Home Team Color");
	obs_properties_add_color(props, "away_color", "Away Team Color");
	
	return props;
}

static void scoreboard_source_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "show_records", true);
	obs_data_set_default_int(settings, "home_color", 0xFF0080FF); // Blue
	obs_data_set_default_int(settings, "away_color", 0xFFFF8000); // Orange
}

void register_scoreboard_source()
{
	struct obs_source_info info = {};
	info.id = "water_polo_scoreboard";
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
	blog(LOG_INFO, "[Scoreboard] Registered source: water_polo_scoreboard");

	// Register legacy alias so existing scenes with the old simple source don't error
	struct obs_source_info simple_info = info;
	simple_info.id = "water_polo_scoreboard_simple";
	obs_register_source(&simple_info);
	blog(LOG_INFO, "[Scoreboard] Registered legacy alias: water_polo_scoreboard_simple");
	
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

// --- STREAMDECK HOTKEYS ---

#include <obs-frontend-api.h>

void on_add_home_score(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    g_scoreboard->home_score++;
    g_scoreboard->needs_update = true;
}
void on_remove_home_score(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    if (g_scoreboard->home_score > 0) g_scoreboard->home_score--;
    g_scoreboard->needs_update = true;
}
void on_add_away_score(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    g_scoreboard->away_score++;
    g_scoreboard->needs_update = true;
}
void on_remove_away_score(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    if (g_scoreboard->away_score > 0) g_scoreboard->away_score--;
    g_scoreboard->needs_update = true;
}
void on_toggle_home_manup(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    g_scoreboard->home_manup = !g_scoreboard->home_manup;
    if (g_scoreboard->home_manup) g_scoreboard->home_manup_start_ns = os_gettime_ns();
    g_scoreboard->needs_update = true;
}
void on_toggle_away_manup(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    g_scoreboard->away_manup = !g_scoreboard->away_manup;
    if (g_scoreboard->away_manup) g_scoreboard->away_manup_start_ns = os_gettime_ns();
    g_scoreboard->needs_update = true;
}
#include "shared-schedule.h"
void on_next_period(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    if (g_scoreboard->period < 5) g_scoreboard->period++;
    if (g_scoreboard->period > 4) g_scoreboard->period_text = (g_scoreboard->period == 5) ? "OT" : "SO";
    g_scoreboard->needs_update = true;
}
void on_prev_period(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    if (g_scoreboard->period > 1) g_scoreboard->period--;
    if (g_scoreboard->period <= 4) g_scoreboard->period_text = "";
    g_scoreboard->needs_update = true;
}
void on_advance_game(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    if (!g_schedule_data || g_schedule_data->schedule.empty()) return;
    std::string current_home = g_scoreboard->home_team;
    std::string current_away = g_scoreboard->away_team;
    int current_index = -1;
    for(size_t i = 0; i < g_schedule_data->schedule.size(); i++) {
        if(g_schedule_data->schedule[i].home_team == current_home && 
           g_schedule_data->schedule[i].away_team == current_away) {
            current_index = i; break;
        }
    }
    int next_index = (current_index == -1) ? 0 : current_index + 1;
    if(next_index < (int)g_schedule_data->schedule.size()) {
        auto next_game = g_schedule_data->schedule[next_index];
        g_scoreboard->home_team = next_game.home_team;
        g_scoreboard->away_team = next_game.away_team;
        g_scoreboard->home_score = 0;
        g_scoreboard->away_score = 0;
        g_scoreboard->period = 1;
        g_scoreboard->period_text = "";
        g_scoreboard->home_exclusions = 0;
        g_scoreboard->away_exclusions = 0;
        g_scoreboard->home_timeouts = 3;
        g_scoreboard->away_timeouts = 3;
        g_scoreboard->home_manup = false;
        g_scoreboard->away_manup = false;
        calculate_team_record(g_scoreboard->home_team, g_scoreboard->config_dir,
                              g_scoreboard->home_wins, g_scoreboard->home_losses, g_scoreboard->home_ties);
        calculate_team_record(g_scoreboard->away_team, g_scoreboard->config_dir,
                              g_scoreboard->away_wins, g_scoreboard->away_losses, g_scoreboard->away_ties);
        g_scoreboard->needs_update = true;
    }
}
void on_previous_game(void *data, obs_hotkey_id id, obs_hotkey_t *hotkey, bool pressed) {
    if (!pressed || !g_scoreboard) return;
    if (!g_schedule_data || g_schedule_data->schedule.empty()) return;
    std::string current_home = g_scoreboard->home_team;
    std::string current_away = g_scoreboard->away_team;
    int current_index = -1;
    for(size_t i = 0; i < g_schedule_data->schedule.size(); i++) {
        if(g_schedule_data->schedule[i].home_team == current_home && 
           g_schedule_data->schedule[i].away_team == current_away) {
            current_index = i; break;
        }
    }
    if(current_index > 0) {
        auto prev_game = g_schedule_data->schedule[current_index - 1];
        g_scoreboard->home_team = prev_game.home_team;
        g_scoreboard->away_team = prev_game.away_team;
        g_scoreboard->home_score = 0;
        g_scoreboard->away_score = 0;
        g_scoreboard->period = 1;
        g_scoreboard->period_text = "";
        g_scoreboard->home_exclusions = 0;
        g_scoreboard->away_exclusions = 0;
        g_scoreboard->home_timeouts = 3;
        g_scoreboard->away_timeouts = 3;
        g_scoreboard->home_manup = false;
        g_scoreboard->away_manup = false;
        calculate_team_record(g_scoreboard->home_team, g_scoreboard->config_dir,
                              g_scoreboard->home_wins, g_scoreboard->home_losses, g_scoreboard->home_ties);
        calculate_team_record(g_scoreboard->away_team, g_scoreboard->config_dir,
                              g_scoreboard->away_wins, g_scoreboard->away_losses, g_scoreboard->away_ties);
        g_scoreboard->needs_update = true;
    }
}
void init_scoreboard_hotkeys() {
    obs_hotkey_register_frontend("WPScoreboard.AddHome", "Scoreboard: Add Home Score", on_add_home_score, NULL);
    obs_hotkey_register_frontend("WPScoreboard.RemoveHome", "Scoreboard: Remove Home Score", on_remove_home_score, NULL);
    obs_hotkey_register_frontend("WPScoreboard.AddAway", "Scoreboard: Add Away Score", on_add_away_score, NULL);
    obs_hotkey_register_frontend("WPScoreboard.RemoveAway", "Scoreboard: Remove Away Score", on_remove_away_score, NULL);
    obs_hotkey_register_frontend("WPScoreboard.ToggleHomeManUp", "Scoreboard: Toggle Home Man-Up", on_toggle_home_manup, NULL);
    obs_hotkey_register_frontend("WPScoreboard.ToggleAwayManUp", "Scoreboard: Toggle Away Man-Up", on_toggle_away_manup, NULL);
    obs_hotkey_register_frontend("WPScoreboard.NextPeriod", "Scoreboard: Next Period", on_next_period, NULL);
    obs_hotkey_register_frontend("WPScoreboard.PrevPeriod", "Scoreboard: Previous Period", on_prev_period, NULL);
    obs_hotkey_register_frontend("WPScoreboard.NextGame", "Scoreboard: Next Game", on_advance_game, NULL);
    obs_hotkey_register_frontend("WPScoreboard.PrevGame", "Scoreboard: Previous Game", on_previous_game, NULL);
    blog(LOG_INFO, "Registered WP Scoreboard simple hotkeys");
}
