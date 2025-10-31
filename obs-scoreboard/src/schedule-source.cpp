#include <obs-module.h>
#include <graphics/vec3.h>
#include <graphics/matrix4.h>
#include <util/platform.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <cctype>
#include <set>
#include "shared-schedule.h"

#ifdef _WIN32
#include <windows.h>
#include <gdiplus.h>
#include <gdiplusgraphics.h>
#include <gdiplusbrush.h>
#include <gdipluspen.h>
#include <gdipluspath.h>
#include <comdef.h>
#pragma comment(lib, "gdiplus.lib")

using namespace Gdiplus;
#endif

#include <QtCore/QSettings>

// Get saved config directory from control panel settings
std::string get_saved_config_dir() {
	QSettings settings("WaterPoloScoreboard", "ControlPanel");
	QString configDir = settings.value("configDir", "").toString();
	return configDir.toUtf8().constData();
}

// Forward declarations and structures
struct Team {
	std::string code;
	std::string name;
	std::string logo_path;
	uint32_t home_bg;
	uint32_t home_text;
	uint32_t away_bg;
	uint32_t away_text;
};

struct Game {
	std::string date;
	std::string time;
	std::string home_team;
	std::string away_team;
	std::chrono::system_clock::time_point start_time;
};

// Helper function for parsing datetime strings
std::chrono::system_clock::time_point parse_datetime(const std::string& datetime_str) {
	std::tm tm = {};
	std::istringstream ss(datetime_str);
	ss >> std::get_time(&tm, "%Y-%m-%d %H:%M");
	
	if (ss.fail()) {
		blog(LOG_WARNING, "[Schedule] Failed to parse datetime: %s", datetime_str.c_str());
		return std::chrono::system_clock::now();
	}
	
	return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

// Global schedule data (shared between control panel and schedule source)
struct GlobalScheduleData {
	std::vector<Game> schedule;
	std::map<std::string, Team> teams;
	std::string config_dir;
	std::chrono::system_clock::time_point last_update;
	
	GlobalScheduleData() {
		last_update = std::chrono::system_clock::now();
	}
};

static GlobalScheduleData *g_schedule_data = nullptr;

// Initialize global schedule data
void init_global_schedule_data() {
	if (!g_schedule_data) {
		g_schedule_data = new GlobalScheduleData();
		blog(LOG_INFO, "[Schedule] Global schedule data initialized");
	}
}

// Cleanup global schedule data
void cleanup_global_schedule_data() {
	if (g_schedule_data) {
		delete g_schedule_data;
		g_schedule_data = nullptr;
		blog(LOG_INFO, "[Schedule] Global schedule data cleaned up");
	}
}

// Convert hex string to color value
uint32_t hex_to_color(const std::string& hex) {
	std::string clean_hex = hex;
	if (clean_hex.front() == '#') {
		clean_hex = clean_hex.substr(1);
	}
	
	// Convert to uint32_t
	std::stringstream ss;
	ss << std::hex << clean_hex;
	uint32_t result;
	ss >> result;
	
	// Add alpha channel if not present (assume RRGGBB -> FFRRGGBB)
	if (clean_hex.length() == 6) {
		result |= 0xFF000000;
	}
	
	return result;
}

// Update global schedule data (called from control panel)
void update_global_schedule_data(const std::string& config_dir) {
	if (!g_schedule_data) {
		init_global_schedule_data();
	}
	
	g_schedule_data->config_dir = config_dir;
	g_schedule_data->last_update = std::chrono::system_clock::now();
	
	// Load teams from teams.csv
	g_schedule_data->teams.clear();
	
	std::string teams_path;
	if (!config_dir.empty()) {
		teams_path = config_dir + "/teams.csv";
	} else {
		teams_path = "config/teams.csv";
	}
	
	std::ifstream teams_file(teams_path);
	if (teams_file.is_open()) {
		std::string line;
		bool first_line = true;
		
		while (std::getline(teams_file, line)) {
			if (first_line) {
				first_line = false;
				continue; // Skip header: name,home_bg,home_text,away_bg,away_text
			}
			
			if (line.empty()) continue;
			
			std::stringstream ss(line);
			std::string team_name, home_bg_hex, home_text_hex, away_bg_hex, away_text_hex;
			
			if (std::getline(ss, team_name, ',') &&
				std::getline(ss, home_bg_hex, ',') &&
				std::getline(ss, home_text_hex, ',') &&
				std::getline(ss, away_bg_hex, ',') &&
				std::getline(ss, away_text_hex, ',')) {
				
				// Remove quotes if present
				auto remove_quotes = [](std::string& str) {
					if (str.front() == '"' && str.back() == '"') {
						str = str.substr(1, str.length() - 2);
					}
				};
				
				remove_quotes(team_name);
				remove_quotes(home_bg_hex);
				remove_quotes(home_text_hex);
				remove_quotes(away_bg_hex);
				remove_quotes(away_text_hex);
				
				// Create team with colors (using name as code for now)
				Team team;
				team.code = team_name;
				team.name = team_name;
				
				// Create logo path from team name (convert to lowercase for filename)
				std::string logo_name = team_name;
				std::transform(logo_name.begin(), logo_name.end(), logo_name.begin(), ::tolower);
				// Replace spaces with empty string for logo filenames
				logo_name.erase(std::remove(logo_name.begin(), logo_name.end(), ' '), logo_name.end());
				team.logo_path = "logos/" + logo_name + ".svg";
				
				team.home_bg = hex_to_color(home_bg_hex);
				team.home_text = hex_to_color(home_text_hex);
				team.away_bg = hex_to_color(away_bg_hex);
				team.away_text = hex_to_color(away_text_hex);
				
				blog(LOG_INFO, "[Schedule] Team '%s' logo path: %s", team_name.c_str(), team.logo_path.c_str());
				
				g_schedule_data->teams[team_name] = team;
			}
		}
		teams_file.close();
		blog(LOG_INFO, "[Schedule] Loaded %zu teams from %s", g_schedule_data->teams.size(), teams_path.c_str());
	} else {
		blog(LOG_WARNING, "[Schedule] Could not open teams.csv at %s", teams_path.c_str());
		// Add default teams as fallback with default colors
		uint32_t default_home_bg = 0xFF0080FF; // Blue
		uint32_t default_home_text = 0xFFFFFFFF; // White
		uint32_t default_away_bg = 0xFFFF8000; // Orange  
		uint32_t default_away_text = 0xFFFFFFFF; // White
		
		g_schedule_data->teams["RPI"] = {"RPI", "Rensselaer Polytechnic Institute", "logos/rpi.svg", default_home_bg, default_home_text, default_away_bg, default_away_text};
		g_schedule_data->teams["Syracuse"] = {"Syracuse", "Syracuse University", "logos/syracuse.svg", default_home_bg, default_home_text, default_away_bg, default_away_text};
		g_schedule_data->teams["Cornell"] = {"Cornell", "Cornell University", "logos/cornell.svg", default_home_bg, default_home_text, default_away_bg, default_away_text};
		g_schedule_data->teams["NYU"] = {"NYU", "New York University", "logos/nyu.svg", default_home_bg, default_home_text, default_away_bg, default_away_text};
		g_schedule_data->teams["Army"] = {"Army", "United States Military Academy", "logos/army.svg", default_home_bg, default_home_text, default_away_bg, default_away_text};
		g_schedule_data->teams["Columbia"] = {"Columbia", "Columbia University", "logos/columbia.svg", default_home_bg, default_home_text, default_away_bg, default_away_text};
		g_schedule_data->teams["Coast Guard"] = {"Coast Guard", "United States Coast Guard Academy", "logos/coastguard.svg", default_home_bg, default_home_text, default_away_bg, default_away_text};
	}
	
	// Load schedule
	g_schedule_data->schedule.clear();
	
	std::string config_path;
	if (!config_dir.empty()) {
		config_path = config_dir + "/schedule.csv";
	} else {
		config_path = "config/schedule.csv";
	}
	
	std::ifstream file(config_path);
	if (!file.is_open()) {
		blog(LOG_WARNING, "[Schedule] Could not open schedule file: %s", config_path.c_str());
		return;
	}
	
	std::string line;
	bool first_line = true;
	
	while (std::getline(file, line)) {
		if (first_line) {
			first_line = false;
			continue; // Skip header
		}
		
		if (line.empty()) continue;
		
		std::stringstream ss(line);
		std::string start_time_str, home, away;
		
		if (std::getline(ss, start_time_str, ',') &&
			std::getline(ss, home, ',') &&
			std::getline(ss, away)) {
			
			Game game;
			game.start_time = parse_datetime(start_time_str);
			game.home_team = home;
			game.away_team = away;
			
			// Extract date and time
			auto time_t = std::chrono::system_clock::to_time_t(game.start_time);
			auto tm = *std::localtime(&time_t);
			
			std::ostringstream date_ss, time_ss;
			date_ss << std::put_time(&tm, "%Y-%m-%d");
			
			// Convert to 12-hour format with AM/PM
			int hour = tm.tm_hour;
			int minute = tm.tm_min;
			std::string am_pm = (hour >= 12) ? "PM" : "AM";
			if (hour == 0) hour = 12; // Midnight
			else if (hour > 12) hour -= 12; // Convert to 12-hour
			
			time_ss << std::setfill('0') << std::setw(2) << hour << ":" 
			        << std::setfill('0') << std::setw(2) << minute << " " << am_pm;
			
			game.date = date_ss.str();
			game.time = time_ss.str();
			
			g_schedule_data->schedule.push_back(game);
		}
	}
	
	blog(LOG_INFO, "[Schedule] Loaded %zu games from %s", g_schedule_data->schedule.size(), config_path.c_str());
	notify_schedule_data_updated();
}

// Notify that schedule data was updated (for any listening sources)
void notify_schedule_data_updated() {
	// This function can be used to trigger updates in multiple schedule sources
	// For now, it's just a placeholder that sources can call to indicate data changed
	blog(LOG_INFO, "[Schedule] Schedule data updated notification sent");
}

// Forward declarations for OBS callbacks
static const char *schedule_source_get_name(void *unused);
static void *schedule_source_create(obs_data_t *settings, obs_source_t *source);
static void schedule_source_destroy(void *data);
static void schedule_source_update(void *data, obs_data_t *settings);
static obs_properties_t *schedule_source_get_properties(void *data);
static void schedule_source_get_defaults(obs_data_t *settings);
static void schedule_source_render(void *data, gs_effect_t *effect);
static uint32_t schedule_source_get_width(void *data);
static uint32_t schedule_source_get_height(void *data);

// Schedule source context
struct schedule_source_context {
	obs_source_t *source;
	
	// Rendering
	uint32_t width;
	uint32_t height;
	
	// Display preferences - store selected dates instead of day booleans
	std::vector<std::string> selected_dates;
	
	// Auto-rotation settings
	int rotation_seconds;
	std::chrono::steady_clock::time_point last_rotation;
	int current_day_index;
	std::vector<std::string> active_days; // Will store selected dates
	std::chrono::system_clock::time_point last_schedule_update;
	
	// Visual settings
	uint32_t background_color;
	uint32_t text_color;
	uint32_t accent_color;
	int font_size;
	
	// Config
	std::string config_dir;
	
#ifdef _WIN32
	// GDI+ resources
	Graphics *graphics;
	Bitmap *render_target;
	std::map<std::string, Image*> team_logos;
#endif
	
	schedule_source_context() : source(nullptr), width(1280), height(1080),
		rotation_seconds(5), current_day_index(0),
		background_color(0x001A1A1A), text_color(0xFFFFFFFF), accent_color(0xFF0080FF),
		font_size(36)
#ifdef _WIN32
		, graphics(nullptr), render_target(nullptr)
#endif
	{
		last_rotation = std::chrono::steady_clock::now();
		last_schedule_update = (std::chrono::system_clock::time_point::min)();
	}
};

// Update active days based on preferences
void update_active_days(schedule_source_context *context) {
	context->active_days.clear();
	
	// Use selected dates directly
	context->active_days = context->selected_dates;
	
	// Reset rotation index if it's out of bounds
	if (context->current_day_index >= (int)context->active_days.size()) {
		context->current_day_index = 0;
	}
}

// Load schedule from CSV
void load_schedule_data(schedule_source_context *context, const std::string& config_dir) {
	// Update global schedule data instead of context-specific data
	update_global_schedule_data(config_dir);
}

// Get all unique dates from the current schedule
std::vector<std::string> get_schedule_dates() {
	std::vector<std::string> dates;
	std::set<std::string> unique_dates;
	
	if (!g_schedule_data) return dates;
	
	for (const auto& game : g_schedule_data->schedule) {
		unique_dates.insert(game.date);
	}
	
	// Convert set to sorted vector
	for (const auto& date : unique_dates) {
		dates.push_back(date);
	}
	
	return dates;
}

// Update active days based on preferences
// Get games for a specific date
std::vector<Game> get_games_for_day(const std::string& date_str) {
	std::vector<Game> day_games;
	
	if (!g_schedule_data) return day_games;
	
	for (const auto& game : g_schedule_data->schedule) {
		if (game.date == date_str) {
			day_games.push_back(game);
		}
	}
	
	// Sort by time
	std::sort(day_games.begin(), day_games.end(), 
		[](const Game& a, const Game& b) {
			return a.start_time < b.start_time;
		});
	
	return day_games;
}

#ifdef _WIN32
// Load team logo with multiple attempts
Image* load_team_logo(const std::string& logo_path, const std::string& config_dir) {
	std::vector<std::string> paths_to_try;
	
	// Try the provided path first
	if (!config_dir.empty()) {
		paths_to_try.push_back(config_dir + "/logos/" + logo_path);
		paths_to_try.push_back(config_dir + "/" + logo_path); // Also try without logos subdirectory
	}
	
	// Try with .png extension if .svg fails
	std::string png_path = logo_path;
	if (png_path.length() >= 4 && png_path.substr(png_path.length() - 4) == ".svg") {
		png_path = png_path.substr(0, png_path.length() - 4) + ".png";
	}
	if (!config_dir.empty()) {
		paths_to_try.push_back(config_dir + "/logos/" + png_path);
		paths_to_try.push_back(config_dir + "/" + png_path); // Also try without logos subdirectory
	}
	
	for (const auto& full_path : paths_to_try) {
		blog(LOG_INFO, "[Schedule] Attempting to load logo from: %s", full_path.c_str());
		
		// Convert to wide string
		int size_needed = MultiByteToWideChar(CP_UTF8, 0, full_path.c_str(), -1, NULL, 0);
		std::wstring wide_path(size_needed, 0);
		MultiByteToWideChar(CP_UTF8, 0, full_path.c_str(), -1, &wide_path[0], size_needed);
		
		Image* image = Image::FromFile(wide_path.c_str());
		if (image && image->GetLastStatus() == Ok) {
			blog(LOG_INFO, "[Schedule] Successfully loaded logo: %s", full_path.c_str());
			return image;
		} else {
			blog(LOG_WARNING, "[Schedule] Failed to load logo: %s (Status: %d)", full_path.c_str(), 
				 image ? image->GetLastStatus() : -1);
			if (image) {
				delete image;
			}
		}
	}
	
	return nullptr;
}

// Convert date string to day of week
std::string date_to_day_of_week(const std::string& date_str) {
	std::tm tm = {};
	std::istringstream ss(date_str);
	ss >> std::get_time(&tm, "%Y-%m-%d");
	
	if (ss.fail()) {
		return date_str; // Return original if parsing fails
	}
	
	auto time_point = std::chrono::system_clock::from_time_t(std::mktime(&tm));
	auto time_t = std::chrono::system_clock::to_time_t(time_point);
	std::tm* local_tm = std::localtime(&time_t);
	
	const char* days[] = {"Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"};
	return std::string(days[local_tm->tm_wday]) + " (" + date_str + ")";
}

// Create rounded rectangle path
void add_rounded_rectangle(GraphicsPath* path, float x, float y, float width, float height, float radius) {
	if (radius <= 0) {
		path->AddRectangle(RectF(x, y, width, height));
		return;
	}
	
	float diameter = radius * 2;
	
	// Top-left arc
	path->AddArc(x, y, diameter, diameter, 180, 90);
	// Top-right arc
	path->AddArc(x + width - diameter, y, diameter, diameter, 270, 90);
	// Bottom-right arc
	path->AddArc(x + width - diameter, y + height - diameter, diameter, diameter, 0, 90);
	// Bottom-left arc
	path->AddArc(x, y + height - diameter, diameter, diameter, 90, 90);
	
	path->CloseFigure();
}

// Render schedule for a date
void render_day_schedule(schedule_source_context *context, const std::string& date_str) {
	if (!context->graphics) return;
	
	auto games = get_games_for_day(date_str);
	
	// Clear background
	context->graphics->Clear(Color(
		(context->background_color >> 24) & 0xFF,
		(context->background_color >> 16) & 0xFF,
		(context->background_color >> 8) & 0xFF,
		context->background_color & 0xFF
	));
	
	// Set up rendering quality
	context->graphics->SetSmoothingMode(SmoothingModeAntiAlias);
	context->graphics->SetTextRenderingHint(TextRenderingHintAntiAlias);
	
	// Fonts (bigger sizes)
	FontFamily fontFamily(L"Segoe UI");
	Gdiplus::Font titleFont(&fontFamily, (REAL)(context->font_size * 1.8), FontStyleBold, UnitPixel);
	Gdiplus::Font gameFont(&fontFamily, (REAL)(context->font_size * 1.2), FontStyleRegular, UnitPixel);
	Gdiplus::Font timeFont(&fontFamily, (REAL)context->font_size, FontStyleRegular, UnitPixel);
	
	// Colors
	SolidBrush textBrush(Color(
		(context->text_color >> 24) & 0xFF,
		(context->text_color >> 16) & 0xFF,
		(context->text_color >> 8) & 0xFF,
		context->text_color & 0xFF
	));
	
	SolidBrush accentBrush(Color(
		(context->accent_color >> 24) & 0xFF,
		(context->accent_color >> 16) & 0xFF,
		(context->accent_color >> 8) & 0xFF,
		context->accent_color & 0xFF
	));
	
	// Calculate layout with larger sizes to prevent cutoff but fit more games
	float margin = 30.0f; // Further reduced margin
	float titleHeight = 80.0f; // Further reduced title height
	float gameHeight = 100.0f; // Further reduced game height to fit more games
	float logoSize = 70.0f; // Smaller logo size
	
	// Calculate how many games can fit - target 12 games
	float availableHeight = context->height - margin - titleHeight - 15.0f - margin; // Space for title + margins
	int maxGames = std::min(12, (int)(availableHeight / (gameHeight + 10.0f))); // Target 12 games, 10.0f spacing
	
	// Title with day of week
	std::string day_info = date_to_day_of_week(date_str);
	std::wstring title = L"Schedule - " + std::wstring(day_info.begin(), day_info.end());
	RectF titleRect(margin, margin, (REAL)(context->width - 2 * margin), titleHeight);
	
	// Create rounded rectangle for title
	GraphicsPath titlePath;
	add_rounded_rectangle(&titlePath, titleRect.X, titleRect.Y, titleRect.Width, titleRect.Height, 15.0f);
	
	// Gradient background for title
	LinearGradientBrush titleGradient(
		PointF(titleRect.X, titleRect.Y),
		PointF(titleRect.X, titleRect.Y + titleRect.Height),
		Color(120, (context->accent_color >> 16) & 0xFF, (context->accent_color >> 8) & 0xFF, context->accent_color & 0xFF),
		Color(80, (context->accent_color >> 16) & 0xFF, (context->accent_color >> 8) & 0xFF, context->accent_color & 0xFF)
	);
	
	context->graphics->FillPath(&titleGradient, &titlePath);
	
	// Title text
	StringFormat centerFormat;
	centerFormat.SetAlignment(StringAlignmentCenter);
	centerFormat.SetLineAlignment(StringAlignmentCenter);
	
	context->graphics->DrawString(title.c_str(), -1, &titleFont, titleRect, &centerFormat, &textBrush);
	
	// Games
	float currentY = margin + titleHeight + 20.0f;
	
	if (games.empty()) {
		// No games message with day of week
		std::string day_info = date_to_day_of_week(date_str);
		std::wstring noGames = L"No games scheduled for " + std::wstring(day_info.begin(), day_info.end());
		RectF messageRect(margin, currentY, (REAL)(context->width - 2 * margin), gameHeight);
		
		GraphicsPath messagePath;
		add_rounded_rectangle(&messagePath, messageRect.X, messageRect.Y, messageRect.Width, messageRect.Height, 10.0f);
		
		SolidBrush messageBg(Color(30, 255, 255, 255));
		context->graphics->FillPath(&messageBg, &messagePath);
		
		context->graphics->DrawString(noGames.c_str(), -1, &gameFont, messageRect, &centerFormat, &textBrush);
	} else {
		int gameCount = 0;
		for (const auto& game : games) {
			if (gameCount >= maxGames) break;
			
			// Game container
			RectF gameRect(margin, currentY, (REAL)(context->width - 2 * margin), gameHeight);
			
			GraphicsPath gamePath;
			add_rounded_rectangle(&gamePath, gameRect.X, gameRect.Y, gameRect.Width, gameRect.Height, 12.0f);
			
			// Get team colors
			Color homeColor1(80, 100, 150, 200); // Default
			Color homeColor2(40, 50, 100, 150);
			Color awayColor1(40, 50, 100, 150); // Default
			Color awayColor2(80, 100, 150, 200);
			Color homeTextColor(255, 255, 255, 255); // Default white
			Color awayTextColor(255, 255, 255, 255);
			
			// Get home team colors
			if (g_schedule_data && g_schedule_data->teams.find(game.home_team) != g_schedule_data->teams.end()) {
				const auto& homeTeam = g_schedule_data->teams.at(game.home_team);
				uint32_t home_bg = homeTeam.home_bg;
				uint32_t home_text = homeTeam.home_text;
				homeColor1 = Color(200, (home_bg >> 16) & 0xFF, (home_bg >> 8) & 0xFF, home_bg & 0xFF);
				homeColor2 = Color(100, (home_bg >> 16) & 0xFF, (home_bg >> 8) & 0xFF, home_bg & 0xFF);
				homeTextColor = Color((home_text >> 24) & 0xFF, (home_text >> 16) & 0xFF, (home_text >> 8) & 0xFF, home_text & 0xFF);
			}
			
			// Get away team colors
			if (g_schedule_data && g_schedule_data->teams.find(game.away_team) != g_schedule_data->teams.end()) {
				const auto& awayTeam = g_schedule_data->teams.at(game.away_team);
				uint32_t away_bg = awayTeam.away_bg;
				uint32_t away_text = awayTeam.away_text;
				awayColor1 = Color(100, (away_bg >> 16) & 0xFF, (away_bg >> 8) & 0xFF, away_bg & 0xFF);
				awayColor2 = Color(200, (away_bg >> 16) & 0xFF, (away_bg >> 8) & 0xFF, away_bg & 0xFF);
				awayTextColor = Color((away_text >> 24) & 0xFF, (away_text >> 16) & 0xFF, (away_text >> 8) & 0xFF, away_text & 0xFF);
			}
			
			// Create horizontal gradient blending home (left) to away (right) colors
			LinearGradientBrush gameGradient(
				PointF(gameRect.X, gameRect.Y + gameRect.Height / 2),
				PointF(gameRect.X + gameRect.Width, gameRect.Y + gameRect.Height / 2),
				homeColor1,
				awayColor2
			);
			
			// Set blend for smooth transition: home color -> middle fade -> away color
			REAL positions[] = {0.0f, 0.4f, 0.6f, 1.0f};
			Color colors[] = {homeColor1, homeColor2, awayColor1, awayColor2};
			gameGradient.SetInterpolationColors(colors, positions, 4);
			
			context->graphics->FillPath(&gameGradient, &gamePath);
			
			// Subtle border
			Pen borderPen(Color(80, 255, 255, 255), 1.5f);
			context->graphics->DrawPath(&borderPen, &gamePath);
			
			// Center divider and time/VS section
			float centerX = gameRect.X + gameRect.Width / 2;
			float centerWidth = 200.0f; // Wider to fit AM/PM text comfortably
			RectF centerRect(centerX - centerWidth / 2, gameRect.Y, centerWidth, gameHeight);
			
			// Semi-transparent center overlay
			SolidBrush centerOverlay(Color(120, 0, 0, 0));
			GraphicsPath centerPath;
			add_rounded_rectangle(&centerPath, centerRect.X, centerRect.Y, centerRect.Width, centerRect.Height, 8.0f);
			context->graphics->FillPath(&centerOverlay, &centerPath);
			
			// Time at top of center with more vertical space
			std::wstring timeStr = std::wstring(game.time.begin(), game.time.end());
			RectF timeRect(centerRect.X, centerRect.Y + 20, centerRect.Width, 30);
			StringFormat centerFormat;
			centerFormat.SetAlignment(StringAlignmentCenter);
			centerFormat.SetLineAlignment(StringAlignmentCenter);
			SolidBrush timeBrush(Color(255, 255, 255, 255));
			context->graphics->DrawString(timeStr.c_str(), -1, &timeFont, timeRect, &centerFormat, &timeBrush);
			
			// VS below time with more spacing
			RectF vsRect(centerRect.X, centerRect.Y + 55, centerRect.Width, 30);
			SolidBrush vsBrush(Color(200, 255, 255, 255));
			context->graphics->DrawString(L"VS", -1, &gameFont, vsRect, &centerFormat, &vsBrush);
			
			// HOME TEAM (Left side)
			float leftPanelX = gameRect.X + 15;
			float leftPanelWidth = centerX - centerWidth / 2 - leftPanelX - 10;
			float contentY = gameRect.Y + (gameHeight - logoSize) / 2;
			
			// Debug: Log config_dir being used
			if (gameCount == 0) {
				blog(LOG_INFO, "[Schedule] Rendering with config_dir: '%s'", context->config_dir.c_str());
			}
			
			// Home team logo
			if (g_schedule_data && g_schedule_data->teams.find(game.home_team) != g_schedule_data->teams.end()) {
				const auto& team = g_schedule_data->teams.at(game.home_team);
				
				if (context->team_logos.find(team.code) == context->team_logos.end()) {
					context->team_logos[team.code] = load_team_logo(team.logo_path, context->config_dir);
				}
				
				Image* team_logo = context->team_logos[team.code];
				if (team_logo) {
					RectF logoRect(leftPanelX, contentY, logoSize, logoSize);
					context->graphics->DrawImage(team_logo, logoRect);
				}
			}
			
			// Home team name
			std::wstring homeTeam = std::wstring(game.home_team.begin(), game.home_team.end());
			RectF homeTextRect(leftPanelX + logoSize + 12, contentY, leftPanelWidth - logoSize - 12, logoSize);
			StringFormat leftFormat;
			leftFormat.SetAlignment(StringAlignmentNear);
			leftFormat.SetLineAlignment(StringAlignmentCenter);
			leftFormat.SetTrimming(StringTrimmingEllipsisCharacter);
			SolidBrush homeTextBrush(homeTextColor);
			context->graphics->DrawString(homeTeam.c_str(), -1, &gameFont, homeTextRect, &leftFormat, &homeTextBrush);
			
			// AWAY TEAM (Right side)
			float rightPanelEnd = gameRect.X + gameRect.Width - 15;
			float rightPanelWidth = rightPanelEnd - (centerX + centerWidth / 2) - 10;
			float rightContentX = centerX + centerWidth / 2 + 10;
			
			// Away team logo (on the right)
			float awayLogoX = rightPanelEnd - logoSize;
			if (g_schedule_data && g_schedule_data->teams.find(game.away_team) != g_schedule_data->teams.end()) {
				const auto& team = g_schedule_data->teams.at(game.away_team);
				
				if (context->team_logos.find(team.code) == context->team_logos.end()) {
					context->team_logos[team.code] = load_team_logo(team.logo_path, context->config_dir);
				}
				
				Image* team_logo = context->team_logos[team.code];
				if (team_logo) {
					RectF logoRect(awayLogoX, contentY, logoSize, logoSize);
					context->graphics->DrawImage(team_logo, logoRect);
				}
			}
			
			// Away team name (right-aligned, before logo)
			std::wstring awayTeam = std::wstring(game.away_team.begin(), game.away_team.end());
			RectF awayTextRect(rightContentX, contentY, awayLogoX - rightContentX - 12, logoSize);
			StringFormat rightFormat;
			rightFormat.SetAlignment(StringAlignmentFar);
			rightFormat.SetLineAlignment(StringAlignmentCenter);
			rightFormat.SetTrimming(StringTrimmingEllipsisCharacter);
			SolidBrush awayTextBrush(awayTextColor);
			context->graphics->DrawString(awayTeam.c_str(), -1, &gameFont, awayTextRect, &rightFormat, &awayTextBrush);
			
			currentY += gameHeight + 10.0f;
			gameCount++;
		}
	}
}
#endif

// OBS Source callbacks implementation
static const char *schedule_source_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Water Polo Schedule";
}

static void *schedule_source_create(obs_data_t *settings, obs_source_t *source)
{
	auto *context = new schedule_source_context();
	context->source = source;
	
	// Initialize global schedule data if needed
	init_global_schedule_data();

#ifdef _WIN32
	// Create render target
	context->render_target = new Bitmap(context->width, context->height, PixelFormat32bppARGB);
	context->graphics = new Graphics(context->render_target);
#endif
	
	// Update from settings
	schedule_source_update(context, settings);
	
	blog(LOG_INFO, "[Schedule] Source created");
	return context;
}static void schedule_source_destroy(void *data)
{
	auto *context = static_cast<schedule_source_context*>(data);
	
#ifdef _WIN32
	// Clean up logos
	for (auto& pair : context->team_logos) {
		if (pair.second) {
			delete pair.second;
		}
	}
	
	if (context->graphics) {
		delete context->graphics;
	}
	if (context->render_target) {
		delete context->render_target;
	}
#endif
	
	delete context;
	blog(LOG_INFO, "[Schedule] Source destroyed");
}

static void schedule_source_update(void *data, obs_data_t *settings)
{
	auto *context = static_cast<schedule_source_context*>(data);
	
	// Check if config directory setting has changed
	const char *config_directory = obs_data_get_string(settings, "config_directory");
	std::string new_config_dir = config_directory ? config_directory : "";
	
	// If config directory is provided in settings, use it and save it
	if (!new_config_dir.empty() && new_config_dir != context->config_dir) {
		blog(LOG_INFO, "[Schedule] Loading schedule from config directory: %s", new_config_dir.c_str());
		update_global_schedule_data(new_config_dir);
		context->config_dir = new_config_dir;
		
		// Save this directory for future use
		QSettings qsettings("WaterPoloScoreboard", "ControlPanel");
		qsettings.setValue("config_directory", QString::fromStdString(new_config_dir));
		
		context->last_schedule_update = std::chrono::system_clock::now();
		
		// Clear logo cache when config changes
		#ifdef _WIN32
		for (auto& pair : context->team_logos) {
			if (pair.second) {
				delete pair.second;
			}
		}
		context->team_logos.clear();
		#endif
	}
	// Fall back to saved config directory if no setting provided
	else if (new_config_dir.empty() && context->config_dir.empty()) {
		std::string saved_config_dir = get_saved_config_dir();
		if (!saved_config_dir.empty()) {
			blog(LOG_INFO, "[Schedule] Loading schedule from saved config dir: %s", saved_config_dir.c_str());
			update_global_schedule_data(saved_config_dir);
			context->config_dir = saved_config_dir;
		} else {
			blog(LOG_INFO, "[Schedule] No config directory set. Please set config directory in source properties.");
			update_global_schedule_data("");
			context->config_dir = "";
		}
		context->last_schedule_update = std::chrono::system_clock::now();
		
		// Clear logo cache when config changes
		#ifdef _WIN32
		for (auto& pair : context->team_logos) {
			if (pair.second) {
				delete pair.second;
			}
		}
		context->team_logos.clear();
		#endif
	}
	
	// Update selected dates based on checkboxes
	context->selected_dates.clear();
	auto available_dates = get_schedule_dates();
	
	for (const auto& date : available_dates) {
		std::string prop_name = "show_date_" + date;
		if (obs_data_get_bool(settings, prop_name.c_str())) {
			context->selected_dates.push_back(date);
		}
	}
	
	context->rotation_seconds = (int)obs_data_get_int(settings, "rotation_seconds");
	
	// Visual settings
	context->background_color = (uint32_t)obs_data_get_int(settings, "background_color");
	context->text_color = (uint32_t)obs_data_get_int(settings, "text_color");
	context->accent_color = (uint32_t)obs_data_get_int(settings, "accent_color");
	context->font_size = (int)obs_data_get_int(settings, "font_size");
	
	// Update active days
	update_active_days(context);
	
	blog(LOG_INFO, "[Schedule] Settings updated - Active dates: %zu, Rotation: %ds", 
		 context->active_days.size(), context->rotation_seconds);
}

static obs_properties_t *schedule_source_get_properties(void *data)
{
	UNUSED_PARAMETER(data);
	
	obs_properties_t *props = obs_properties_create();
	
	// Add config directory path setting
	obs_properties_add_path(props, "config_directory", "Config Directory (teams.csv, schedule.csv, logos)",
		OBS_PATH_DIRECTORY, nullptr, nullptr);
	
	// Try to load schedule data if not already loaded
	if (!g_schedule_data) {
		std::string saved_config_dir = get_saved_config_dir();
		if (!saved_config_dir.empty()) {
			update_global_schedule_data(saved_config_dir);
		} else {
			update_global_schedule_data("");
		}
	}
	
	// Get available dates from current schedule
	auto available_dates = get_schedule_dates();
	
	if (!available_dates.empty()) {
		obs_properties_add_text(props, "dates_header", "Select Dates to Display:", OBS_TEXT_INFO);
		
		// Add checkbox for each available date
		for (const auto& date : available_dates) {
			std::string prop_name = "show_date_" + date;
			std::string display_name = "Show " + date;
			obs_properties_add_bool(props, prop_name.c_str(), display_name.c_str());
		}
	} else {
		std::string saved_config_dir = get_saved_config_dir();
		if (!saved_config_dir.empty()) {
			std::string msg = "No schedule data found in: " + saved_config_dir + ". Please check that schedule.csv exists.";
			obs_properties_add_text(props, "no_schedule", msg.c_str(), OBS_TEXT_INFO);
		} else {
			obs_properties_add_text(props, "no_schedule", "No schedule data loaded. Please set config directory in control panel first.", OBS_TEXT_INFO);
		}
	}
	
	// Rotation settings
	obs_properties_add_int_slider(props, "rotation_seconds", 
		"Seconds per Date", 2, 30, 1);
	
	// Visual settings
	obs_properties_add_color(props, "background_color", "Background Color");
	obs_properties_add_color(props, "text_color", "Text Color");
	obs_properties_add_color(props, "accent_color", "Accent Color");
	obs_properties_add_int_slider(props, "font_size", "Font Size", 12, 48, 2);
	
	return props;
}

static void schedule_source_get_defaults(obs_data_t *settings)
{
	// Set default config directory from saved settings
	std::string saved_config_dir = get_saved_config_dir();
	if (!saved_config_dir.empty()) {
		obs_data_set_default_string(settings, "config_directory", saved_config_dir.c_str());
	}
	
	// Get available dates and default to all enabled
	auto available_dates = get_schedule_dates();
	for (const auto& date : available_dates) {
		std::string prop_name = "show_date_" + date;
		obs_data_set_default_bool(settings, prop_name.c_str(), true);
	}
	
	// Default rotation
	obs_data_set_default_int(settings, "rotation_seconds", 5);
	
	// Default colors
	obs_data_set_default_int(settings, "background_color", 0x001A1A1A); // Transparent background
	obs_data_set_default_int(settings, "text_color", 0xFFFFFFFF);
	obs_data_set_default_int(settings, "accent_color", 0xFF0080FF);
	obs_data_set_default_int(settings, "font_size", 36);
}

static void schedule_source_render(void *data, gs_effect_t *effect)
{
	auto *context = static_cast<schedule_source_context*>(data);
	
	if (context->active_days.empty()) {
		return; // No days to show
	}
	
	// Handle rotation
	auto now = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - context->last_rotation).count();
	
	if (context->active_days.size() > 1 && elapsed >= context->rotation_seconds) {
		context->current_day_index = (context->current_day_index + 1) % context->active_days.size();
		context->last_rotation = now;
	}
	
	// Get current day to display
	std::string current_day = context->active_days[context->current_day_index];
	
#ifdef _WIN32
	// Render the schedule
	render_day_schedule(context, current_day);
	
	// Convert to OBS texture
	if (context->render_target) {
		BitmapData bitmapData;
		Rect rect(0, 0, context->width, context->height);
		
		if (context->render_target->LockBits(&rect, ImageLockModeRead, PixelFormat32bppARGB, &bitmapData) == Ok) {
			// Create texture
			gs_texture_t *texture = gs_texture_create(context->width, context->height, GS_BGRA, 1, 
				(const uint8_t**)&bitmapData.Scan0, GS_DYNAMIC);
			
			if (texture) {
				gs_effect_set_texture(gs_effect_get_param_by_name(effect, "image"), texture);
				gs_draw_sprite(texture, 0, context->width, context->height);
				gs_texture_destroy(texture);
			}
			
			context->render_target->UnlockBits(&bitmapData);
		}
	}
#endif
}

static uint32_t schedule_source_get_width(void *data)
{
	auto *context = static_cast<schedule_source_context*>(data);
	return context->width;
}

static uint32_t schedule_source_get_height(void *data)
{
	auto *context = static_cast<schedule_source_context*>(data);
	return context->height;
}

// Register the schedule source
void register_schedule_source()
{
	struct obs_source_info info = {};
	info.id = "water_polo_schedule";
	info.type = OBS_SOURCE_TYPE_INPUT;
	info.output_flags = OBS_SOURCE_VIDEO;
	info.get_name = schedule_source_get_name;
	info.create = schedule_source_create;
	info.destroy = schedule_source_destroy;
	info.update = schedule_source_update;
	info.get_properties = schedule_source_get_properties;
	info.get_defaults = schedule_source_get_defaults;
	info.video_render = schedule_source_render;
	info.get_width = schedule_source_get_width;
	info.get_height = schedule_source_get_height;
	
	obs_register_source(&info);
	
	blog(LOG_INFO, "Registered water_polo_schedule source");
}