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

#ifndef _WIN32
#include <QImage>
#include <QPainter>
#include <QPainterPath>
#include <QLinearGradient>
#include <QFont>
#include <QtSvg/QSvgRenderer>
#endif

// Get saved config directory from control panel settings
std::string get_saved_config_dir() {
	QSettings settings("WaterPoloScoreboard", "ControlPanel");
	QString configDir = settings.value("configDir", "").toString();
	return configDir.toUtf8().constData();
}

// Helper function for parsing datetime strings
std::chrono::system_clock::time_point parse_datetime(const std::string& datetime_str) {
	std::tm tm = {};
	std::istringstream ss(datetime_str);
	ss >> std::get_time(&tm, "%Y-%m-%d %H:%M");
	
	if (ss.fail()) {
		blog(LOG_WARNING, "[Schedule] Failed to parse datetime: %s", datetime_str.c_str());
		return std::chrono::system_clock::now();
	}
	
	// Set DST flag to -1 to let mktime determine DST automatically
	// but we'll force it to not apply DST by setting tm_isdst to 0
	tm.tm_isdst = 0;
	
	return std::chrono::system_clock::from_time_t(std::mktime(&tm));
}

// Global schedule data (shared between control panel and schedule source)
GlobalScheduleData::GlobalScheduleData() {
	last_update = std::chrono::system_clock::now();
}

GlobalScheduleData *g_schedule_data = nullptr;

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
	int line_num = 0;
	
	while (std::getline(file, line)) {
		line_num++;
		
		if (first_line) {
			first_line = false;
			continue; // Skip header
		}
		
		if (line.empty()) continue;
		
		// Proper CSV parsing - handle commas inside quotes and multiple columns
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
		fields.push_back(field); // Add last field
		
		// We need at least 3 fields: start_time, home, away
		// Additional fields (home_score, away_score, winner) are optional
		if (fields.size() >= 3) {
			std::string start_time_str = fields[0];
			std::string home = fields[1];
			std::string away = fields[2];
			
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

	// OBS texture (built from QImage / GDI+ bitmap)
	gs_texture_t *texture = nullptr;
	bool needs_update = true;
	int  last_day_index = -1; // detect rotation changes

	// Scroll
	uint32_t max_height  = 600;  // cap; content scrolls if taller
	float    scroll_y    = 0.0f; // current scroll position in pixels
	float    scroll_speed = 40.0f; // px per second
	float    scroll_pause = 2.0f;  // seconds to pause at top/bottom
	float    scroll_pause_timer = 2.0f;
	bool     scroll_dir_down = true;
	uint32_t content_height = 0; // actual rendered content height

#ifdef _WIN32
	// GDI+ resources
	Graphics *graphics;
	Bitmap *render_target;
	std::map<std::string, Image*> team_logos;
#endif
	
	schedule_source_context() : source(nullptr), width(900), height(600),
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
// Resolve placeholder team names like "Winner Game 17" or "Loser Game 17" to display format
// If game has been played: returns just the team name (e.g., "RPI")
// If game hasn't been played yet: returns "Winner: Home vs Away" or "Loser: Home vs Away"
// If for_display=false: always returns just the team name (for roster loading)
std::string resolve_team_placeholder(const std::string& team_name, bool for_display) {
	if (!g_schedule_data) return team_name;
	
	// Check if this is a placeholder
	if (team_name.find("Winner Game ") == 0 || team_name.find("Loser Game ") == 0) {
		bool is_winner = (team_name.find("Winner") == 0);
		std::string prefix = is_winner ? "Winner" : "Loser";
		
		// Extract game number
		size_t pos = team_name.find("Game ") + 5;
		int game_num = std::stoi(team_name.substr(pos));
		
		blog(LOG_INFO, "[Resolve] Attempting to resolve %s (game %d)", team_name.c_str(), game_num);
		
		// Find that game in the schedule (game numbers are 1-indexed)
		if (game_num > 0 && game_num <= (int)g_schedule_data->schedule.size()) {
			const auto& ref_game = g_schedule_data->schedule[game_num - 1];
			
			// Check if that game has been played (has scores in CSV)
			// We need to re-read the CSV to get score info since Game struct doesn't store it
			std::string config_path = g_schedule_data->config_dir + "/schedule.csv";
			std::ifstream file(config_path);
			if (file.is_open()) {
				std::string line;
				bool first_line = true;
				int current_game = 0;
				
				while (std::getline(file, line)) {
					if (first_line) {
						first_line = false;
						continue;
					}
					if (line.empty()) continue;
					
					current_game++;
					if (current_game == game_num) {
						// Parse this game's winner from CSV
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
						
						if (fields.size() >= 6 && !fields[5].empty()) {
							// Game has been played - return just the team name
							std::string winner = fields[5];
							std::string home = fields[1];
							std::string away = fields[2];
							std::string loser = (winner == home) ? away : home;
							std::string team = is_winner ? winner : loser;
							
							blog(LOG_INFO, "[Resolve] Resolved %s -> %s (game completed)", team_name.c_str(), team.c_str());
							return team;
						} else {
							// Game hasn't been played yet
							if (for_display) {
								// Show "Winner: Home vs Away" or "Loser: Home vs Away"
								// But first resolve home and away in case they are also placeholders
								std::string home = fields[1];
								std::string away = fields[2];
								
								blog(LOG_INFO, "[Resolve] Game %d not played yet, home='%s' away='%s'", 
								     game_num, home.c_str(), away.c_str());
								
								// Recursively resolve placeholders in home and away
								home = resolve_team_placeholder(home, false);
								away = resolve_team_placeholder(away, false);
								
								blog(LOG_INFO, "[Resolve] After recursive resolution: home='%s' away='%s'", 
								     home.c_str(), away.c_str());
								
								// Check if both teams are actual teams (not placeholders)
								// If so, just show "Team1 vs Team2" without the Winner/Loser prefix
								bool home_is_placeholder = (home.find("Winner Game ") == 0 || home.find("Loser Game ") == 0);
								bool away_is_placeholder = (away.find("Winner Game ") == 0 || away.find("Loser Game ") == 0);
								
								if (!home_is_placeholder && !away_is_placeholder) {
									// Both resolved to actual teams, no need for Winner/Loser prefix
									std::string result = home + " vs " + away;
									blog(LOG_INFO, "[Resolve] Both teams resolved, showing: %s", result.c_str());
									return result;
								} else {
									// At least one is still a placeholder, show with prefix
									std::string result = prefix + ": " + home + " vs " + away;
									blog(LOG_INFO, "[Resolve] Game not played yet, showing: %s", result.c_str());
									return result;
								}
							} else {
								// For roster/scoreboard: try to resolve even if game not played
								// Return the actual team from the referenced game
								std::string home = fields[1];
								std::string away = fields[2];
								
								// Recursively resolve home and away
								home = resolve_team_placeholder(home, false);
								away = resolve_team_placeholder(away, false);
								
								// Check if both are actual teams (not still placeholders)
								bool home_is_placeholder = (home.find("Winner Game ") == 0 || home.find("Loser Game ") == 0);
								bool away_is_placeholder = (away.find("Winner Game ") == 0 || away.find("Loser Game ") == 0);
								
								if (!home_is_placeholder && !away_is_placeholder) {
									// Both teams are known, pick the appropriate one
									// For "Winner" we don't know yet, but for roster purposes
									// we could default to home team or return empty
									blog(LOG_INFO, "[Resolve] Game %d not played, both teams known: %s vs %s", 
									     game_num, home.c_str(), away.c_str());
									// For roster loading, we can't determine winner/loser yet
									// Return placeholder to indicate roster not available yet
									return team_name;
								} else {
									// At least one team still unknown
									blog(LOG_INFO, "[Resolve] Game %d teams not fully resolved yet", game_num);
									return team_name;
								}
							}
						}
						break;
					}
				}
				file.close();
			}
		}
	}
	
	return team_name; // Return as-is if not a placeholder or can't resolve
}

std::vector<Game> get_games_for_day(const std::string& date_str) {
	std::vector<Game> day_games;
	
	if (!g_schedule_data) return day_games;
	
	for (const auto& game : g_schedule_data->schedule) {
		if (game.date == date_str) {
			// Resolve placeholder team names for display
			Game display_game = game;
			display_game.home_team = resolve_team_placeholder(game.home_team);
			display_game.away_team = resolve_team_placeholder(game.away_team);
			day_games.push_back(display_game);
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
			
			// Check if team name starts with "Winner:" or "Loser:" (unresolved placeholder)
			std::string homeStr = game.home_team;
			bool homeIsUnresolved = (homeStr.find("Winner:") == 0 || homeStr.find("Loser:") == 0);
			
			if (homeIsUnresolved) {
				// Unresolved placeholder - split into two lines with smaller font
				size_t colonPos = homeStr.find(':');
				std::string prefix = homeStr.substr(0, colonPos);
				std::string matchup = homeStr.substr(colonPos + 2); // Skip ": "
				
				// Use smaller font for two-line display
				Gdiplus::Font smallerFont(L"Segoe UI", 22, FontStyleBold, UnitPixel);
				
				// Draw prefix on first line
				std::wstring prefixW = std::wstring(prefix.begin(), prefix.end());
				RectF line1Rect(homeTextRect.X, homeTextRect.Y, homeTextRect.Width, homeTextRect.Height / 2);
				StringFormat topFormat;
				topFormat.SetAlignment(StringAlignmentNear);
				topFormat.SetLineAlignment(StringAlignmentFar);
				topFormat.SetTrimming(StringTrimmingEllipsisCharacter);
				context->graphics->DrawString(prefixW.c_str(), -1, &smallerFont, line1Rect, &topFormat, &homeTextBrush);
				
				// Draw matchup on second line
				std::wstring matchupW = std::wstring(matchup.begin(), matchup.end());
				RectF line2Rect(homeTextRect.X, homeTextRect.Y + homeTextRect.Height / 2, homeTextRect.Width, homeTextRect.Height / 2);
				StringFormat bottomFormat;
				bottomFormat.SetAlignment(StringAlignmentNear);
				bottomFormat.SetLineAlignment(StringAlignmentNear);
				bottomFormat.SetTrimming(StringTrimmingEllipsisCharacter);
				context->graphics->DrawString(matchupW.c_str(), -1, &smallerFont, line2Rect, &bottomFormat, &homeTextBrush);
			} else {
				// Regular team name or resolved placeholder - single line, normal font
				context->graphics->DrawString(homeTeam.c_str(), -1, &gameFont, homeTextRect, &leftFormat, &homeTextBrush);
			}
			
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
			
			// Check if team name starts with "Winner:" or "Loser:" (unresolved placeholder)
			std::string awayStr = game.away_team;
			bool awayIsUnresolved = (awayStr.find("Winner:") == 0 || awayStr.find("Loser:") == 0);
			
			if (awayIsUnresolved) {
				// Unresolved placeholder - split into two lines with smaller font
				size_t colonPos = awayStr.find(':');
				std::string prefix = awayStr.substr(0, colonPos);
				std::string matchup = awayStr.substr(colonPos + 2); // Skip ": "
				
				// Use smaller font for two-line display
				Gdiplus::Font smallerFont(L"Segoe UI", 22, FontStyleBold, UnitPixel);
				
				// Draw prefix on first line
				std::wstring prefixW = std::wstring(prefix.begin(), prefix.end());
				RectF line1Rect(awayTextRect.X, awayTextRect.Y, awayTextRect.Width, awayTextRect.Height / 2);
				StringFormat topFormat;
				topFormat.SetAlignment(StringAlignmentFar);
				topFormat.SetLineAlignment(StringAlignmentFar);
				topFormat.SetTrimming(StringTrimmingEllipsisCharacter);
				context->graphics->DrawString(prefixW.c_str(), -1, &smallerFont, line1Rect, &topFormat, &awayTextBrush);
				
				// Draw matchup on second line
				std::wstring matchupW = std::wstring(matchup.begin(), matchup.end());
				RectF line2Rect(awayTextRect.X, awayTextRect.Y + awayTextRect.Height / 2, awayTextRect.Width, awayTextRect.Height / 2);
				StringFormat bottomFormat;
				bottomFormat.SetAlignment(StringAlignmentFar);
				bottomFormat.SetLineAlignment(StringAlignmentNear);
				bottomFormat.SetTrimming(StringTrimmingEllipsisCharacter);
				context->graphics->DrawString(matchupW.c_str(), -1, &smallerFont, line2Rect, &bottomFormat, &awayTextBrush);
			} else {
				// Regular team name or resolved placeholder - single line, normal font
				context->graphics->DrawString(awayTeam.c_str(), -1, &gameFont, awayTextRect, &rightFormat, &awayTextBrush);
			}
			
			currentY += gameHeight + 10.0f;
			gameCount++;
		}
	}
}
#endif

#ifndef _WIN32
// ── macOS/Linux Qt rendering ──────────────────────────────────────────────────

static QColor team_color(const std::string &team_name, bool as_home)
{
	if (g_schedule_data) {
		auto it = g_schedule_data->teams.find(team_name);
		if (it != g_schedule_data->teams.end()) {
			uint32_t c = as_home ? it->second.home_bg : it->second.away_bg;
			return QColor((c >> 16) & 0xFF, (c >> 8) & 0xFF, c & 0xFF, 200);
		}
	}
	return as_home ? QColor(40, 80, 160, 180) : QColor(160, 40, 40, 180);
}

static QString friendly_date(const std::string &date_str)
{
	std::tm tm = {};
	std::istringstream ss(date_str);
	ss >> std::get_time(&tm, "%Y-%m-%d");
	if (ss.fail()) return QString::fromStdString(date_str);
	std::mktime(&tm);
	char buf[64];
	std::strftime(buf, sizeof(buf), "%A, %B %d", &tm);
	return QString(buf);
}

static void draw_logo_qt(QPainter &painter, const std::string &logo_path,
                         const std::string &config_dir, QRect rect)
{
	if (logo_path.empty()) return;
	std::string full = config_dir + "/" + logo_path;
	QString qp = QString::fromStdString(full);
	if (qp.endsWith(".svg", Qt::CaseInsensitive)) {
		QSvgRenderer r(qp);
		if (r.isValid()) { r.render(&painter, rect); return; }
	}
	QImage img(qp);
	if (!img.isNull()) {
		QImage scaled = img.scaled(rect.width(), rect.height(),
		                           Qt::KeepAspectRatio, Qt::SmoothTransformation);
		int ox = rect.x() + (rect.width()  - scaled.width())  / 2;
		int oy = rect.y() + (rect.height() - scaled.height()) / 2;
		painter.drawImage(QRect(ox, oy, scaled.width(), scaled.height()), scaled);
	}
}

static void render_schedule_qt(schedule_source_context *context, const std::string &date_str)
{
	int W = (int)context->width;

	const int mg   = 12;
	const int hdrH = 58;
	const int rowH = 96;
	const int gap  = 7;
	const int logoSz = 60;

	auto games = get_games_for_day(date_str);
	int gameCount = (int)games.size();

	// Compute full content height
	int fullH = mg + hdrH + gap
	          + std::max(1, gameCount) * (rowH + gap)
	          + mg;
	context->content_height = (uint32_t)fullH;

	// Viewport height is capped; texture matches the viewport (not full content)
	int vpH = (int)std::min((uint32_t)fullH, context->max_height);
	context->height = (uint32_t)vpH;

	// Reset scroll when content fits
	if ((uint32_t)fullH <= context->max_height) {
		context->scroll_y = 0.0f;
		context->scroll_dir_down = true;
		context->scroll_pause_timer = context->scroll_pause;
	}

	// QImage is viewport-sized; painter translate scrolls the content
	QImage image(W, vpH, QImage::Format_ARGB32);
	image.fill(Qt::transparent);
	QPainter p(&image);
	p.setRenderHint(QPainter::Antialiasing);
	p.setRenderHint(QPainter::TextAntialiasing);

	// Scroll: translate painter up so the correct window is visible
	p.translate(0, -(int)context->scroll_y);
	// Clip to the full content so items outside the viewport are hidden
	p.setClipRect(0, (int)context->scroll_y, W, vpH);

	// Background panel — fill the full content height
	QPainterPath bgPath;
	bgPath.addRoundedRect(0, 0, W, fullH, 8, 8);
	p.fillPath(bgPath, QColor(18, 18, 24, 248));

	// Header
	{
		QPainterPath hp;
		hp.addRoundedRect(mg, mg, W - 2*mg, hdrH, 6, 6);
		QLinearGradient hg(mg, mg, mg, mg + hdrH);
		hg.setColorAt(0, QColor(40, 40, 56, 255));
		hg.setColorAt(1, QColor(26, 26, 36, 255));
		p.fillPath(hp, hg);
		p.setPen(Qt::white);
		p.setFont(QFont("Arial", 22, QFont::Bold));
		p.drawText(QRect(mg, mg, W - 2*mg, hdrH), Qt::AlignCenter,
		           friendly_date(date_str));
	}

	if (games.empty()) {
		p.setPen(QColor(160, 160, 180));
		p.setFont(QFont("Arial", 18));
		p.drawText(QRect(mg, mg + hdrH + gap, W - 2*mg, rowH),
		           Qt::AlignCenter, "No games scheduled");
	} else {
		int y = mg + hdrH + gap;
		for (const auto &game : games) {
			if (y + rowH > fullH - mg) break;

			QColor homeC = team_color(game.home_team, true);
			QColor awayC = team_color(game.away_team, false);
			QRect rowRect(mg, y, W - 2*mg, rowH);

			// Row gradient: home left → dark center → away right
			QPainterPath rowPath;
			rowPath.addRoundedRect(rowRect, 8, 8);
			QLinearGradient rg(rowRect.left(), 0, rowRect.right(), 0);
			rg.setColorAt(0.00, homeC);
			rg.setColorAt(0.38, QColor(26, 26, 36, 230));
			rg.setColorAt(0.62, QColor(26, 26, 36, 230));
			rg.setColorAt(1.00, awayC);
			p.fillPath(rowPath, rg);
			p.setPen(QPen(QColor(255, 255, 255, 28), 1));
			p.drawPath(rowPath);

			// Center column: time + VS
			int cW = 110;
			int cX = rowRect.left() + (rowRect.width() - cW) / 2;
			int sideW = cX - rowRect.left() - 8;

			p.setPen(QColor(200, 200, 212));
			p.setFont(QFont("Arial", 14));
			p.drawText(QRect(cX, y, cW, rowH / 2), Qt::AlignCenter,
			           QString::fromStdString(game.time));
			p.setPen(QColor(230, 230, 240));
			p.setFont(QFont("Arial", 17, QFont::Bold));
			p.drawText(QRect(cX, y + rowH / 2, cW, rowH / 2),
			           Qt::AlignCenter, "VS");

			// Home side (logo + name)
			int lx = rowRect.left() + 8;
			draw_logo_qt(p, g_schedule_data && g_schedule_data->teams.count(game.home_team)
			               ? g_schedule_data->teams.at(game.home_team).logo_path : "",
			             context->config_dir,
			             QRect(lx, y + (rowH - logoSz) / 2, logoSz, logoSz));
			p.setPen(Qt::white);
			p.setFont(QFont("Arial", 19, QFont::Bold));
			p.drawText(QRect(lx + logoSz + 6, y, sideW - logoSz - 6, rowH),
			           Qt::AlignLeft | Qt::AlignVCenter,
			           QString::fromStdString(game.home_team));

			// Away side (name + logo)
			int rx = cX + cW + 8;
			int awayNameW = sideW - logoSz - 6;
			p.setPen(Qt::white);
			p.setFont(QFont("Arial", 19, QFont::Bold));
			p.drawText(QRect(rx, y, awayNameW, rowH),
			           Qt::AlignRight | Qt::AlignVCenter,
			           QString::fromStdString(game.away_team));
			int awayLogoX = rx + awayNameW + 6;
			draw_logo_qt(p, g_schedule_data && g_schedule_data->teams.count(game.away_team)
			               ? g_schedule_data->teams.at(game.away_team).logo_path : "",
			             context->config_dir,
			             QRect(awayLogoX, y + (rowH - logoSz) / 2, logoSz, logoSz));

			y += rowH + gap;
		}
	}

	p.end();

	// Upload viewport-sized texture
	if (context->texture) {
		gs_texture_destroy(context->texture);
		context->texture = nullptr;
	}
	const uint8_t *bits = image.constBits();
	context->texture = gs_texture_create(W, vpH, GS_BGRA, 1, &bits, 0);
}
#endif // !_WIN32

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

	if (context->texture) {
		obs_enter_graphics();
		gs_texture_destroy(context->texture);
		obs_leave_graphics();
		context->texture = nullptr;
	}

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
	
	// Check if global schedule data has been updated externally (e.g., from control panel)
	if (g_schedule_data && g_schedule_data->last_update > context->last_schedule_update) {
		blog(LOG_INFO, "[Schedule] Detected external schedule update, refreshing source");
		context->last_schedule_update = g_schedule_data->last_update;
		
		// Clear graphics to force re-render
		#ifdef _WIN32
		if (context->render_target) {
			delete context->render_target;
			context->render_target = nullptr;
		}
		if (context->graphics) {
			delete context->graphics;
			context->graphics = nullptr;
		}
		
		// Recreate graphics resources
		context->render_target = new Bitmap(context->width, context->height, PixelFormat32bppARGB);
		context->graphics = new Graphics(context->render_target);
		context->graphics->SetSmoothingMode(SmoothingModeAntiAlias);
		context->graphics->SetTextRenderingHint(TextRenderingHintAntiAlias);
		
		// Clear logo cache to reload
		for (auto& pair : context->team_logos) {
			if (pair.second) {
				delete pair.second;
			}
		}
		context->team_logos.clear();
		#endif
	}
	
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
	
	int new_width = (int)obs_data_get_int(settings, "source_width");
	if (new_width > 0 && (uint32_t)new_width != context->width) {
		context->width = (uint32_t)new_width;
		context->needs_update = true;
	}
	int new_max_h = (int)obs_data_get_int(settings, "max_height");
	if (new_max_h > 0) {
		context->max_height = (uint32_t)new_max_h;
		context->needs_update = true;
	}

	context->rotation_seconds = (int)obs_data_get_int(settings, "rotation_seconds");
	
	// Visual settings
	context->background_color = (uint32_t)obs_data_get_int(settings, "background_color");
	context->text_color = (uint32_t)obs_data_get_int(settings, "text_color");
	context->accent_color = (uint32_t)obs_data_get_int(settings, "accent_color");
	context->font_size = (int)obs_data_get_int(settings, "font_size");
	
	// Update active days
	update_active_days(context);
	context->needs_update = true;

	blog(LOG_INFO, "[Schedule] Settings updated - Active dates: %zu, Rotation: %ds",
		 context->active_days.size(), context->rotation_seconds);
}

static obs_properties_t *schedule_source_get_properties(void *data)
{
	UNUSED_PARAMETER(data);
	
	obs_properties_t *props = obs_properties_create();
	
	obs_properties_add_int_slider(props, "source_width",  "Width (px)",      400, 1920, 10);
	obs_properties_add_int_slider(props, "max_height",    "Max Height (px)", 200, 1080, 10);

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
	
	obs_data_set_default_int(settings, "source_width", 900);
	obs_data_set_default_int(settings, "max_height",   600);

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

	// If no dates are selected, fall back to showing all dates from global data
	if (context->active_days.empty() && g_schedule_data) {
		auto all_dates = get_schedule_dates();
		if (!all_dates.empty())
			context->active_days = all_dates;
	}

	if (context->active_days.empty())
		return;

	// Detect external schedule update
	if (g_schedule_data && g_schedule_data->last_update > context->last_schedule_update) {
		context->last_schedule_update = g_schedule_data->last_update;
		context->needs_update = true;
		// Sync config_dir from global data
		if (!g_schedule_data->config_dir.empty())
			context->config_dir = g_schedule_data->config_dir;
	}

	// Handle rotation
	auto now = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
		now - context->last_rotation).count();

	if (context->active_days.size() > 1 && elapsed >= context->rotation_seconds) {
		context->current_day_index =
			(context->current_day_index + 1) % context->active_days.size();
		context->last_rotation = now;
	}

	// Detect day index change → need redraw
	if (context->last_day_index != context->current_day_index) {
		context->last_day_index = context->current_day_index;
		context->needs_update = true;
	}

	if (context->current_day_index >= (int)context->active_days.size())
		context->current_day_index = 0;

	std::string current_day = context->active_days[context->current_day_index];

	if (context->needs_update) {
		context->needs_update = false;

#ifdef _WIN32
		render_day_schedule(context, current_day);
		if (context->render_target) {
			BitmapData bitmapData;
			Rect rect(0, 0, context->width, context->height);
			if (context->render_target->LockBits(&rect, ImageLockModeRead,
			    PixelFormat32bppARGB, &bitmapData) == Ok) {
				if (context->texture) {
					gs_texture_destroy(context->texture);
					context->texture = nullptr;
				}
				context->texture = gs_texture_create(
					context->width, context->height, GS_BGRA, 1,
					(const uint8_t **)&bitmapData.Scan0, 0);
				context->render_target->UnlockBits(&bitmapData);
			}
		}
#else
		render_schedule_qt(context, current_day);
#endif
	}

	if (context->texture)
		obs_source_draw(context->texture, 0, 0, context->width, context->height, false);

	UNUSED_PARAMETER(effect);
}

static void schedule_source_tick(void *data, float seconds)
{
	auto *ctx = static_cast<schedule_source_context *>(data);
	if (ctx->content_height <= ctx->max_height) return; // nothing to scroll

	float maxScroll = (float)((int)ctx->content_height - (int)ctx->max_height);

	if (ctx->scroll_pause_timer > 0.0f) {
		ctx->scroll_pause_timer -= seconds;
		return;
	}

	if (ctx->scroll_dir_down) {
		ctx->scroll_y += ctx->scroll_speed * seconds;
		if (ctx->scroll_y >= maxScroll) {
			ctx->scroll_y = maxScroll;
			ctx->scroll_dir_down = false;
			ctx->scroll_pause_timer = ctx->scroll_pause;
		}
	} else {
		ctx->scroll_y -= ctx->scroll_speed * seconds;
		if (ctx->scroll_y <= 0.0f) {
			ctx->scroll_y = 0.0f;
			ctx->scroll_dir_down = true;
			ctx->scroll_pause_timer = ctx->scroll_pause;
		}
	}
	ctx->needs_update = true; // redraw with new scroll position
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
	info.video_tick   = schedule_source_tick;
	info.get_width = schedule_source_get_width;
	info.get_height = schedule_source_get_height;
	
	obs_register_source(&info);
	
	blog(LOG_INFO, "Registered water_polo_schedule source");
}