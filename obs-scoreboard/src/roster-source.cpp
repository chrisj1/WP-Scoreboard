#include <obs-module.h>
#include <graphics/vec3.h>
#include <graphics/matrix4.h>
#include <util/platform.h>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <cctype>
#include <iomanip>
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

// Forward declaration of scoreboard structure
struct scoreboard_source {
	std::string home_team;
	std::string away_team;
	std::string home_logo_path;
	std::string away_logo_path;
	std::string config_dir;
};

// External function to get global scoreboard
extern struct scoreboard_source *get_global_scoreboard();

// Get saved config directory from control panel settings
std::string get_saved_config_dir_roster() {
	QSettings settings("WaterPoloScoreboard", "ControlPanel");
	QString configDir = settings.value("configDir", "").toString();
	return configDir.toUtf8().constData();
}

// Forward declarations and structures
struct Player {
	std::string first_name;
	std::string last_name;
	std::string cap_number;
};

struct RosterData {
	std::string team_name;
	std::vector<Player> players;
	uint32_t home_bg;
	uint32_t home_text;
	uint32_t away_bg;
	uint32_t away_text;
};

// Convert hex string to color value
uint32_t hex_to_color_roster(const std::string& hex) {
	std::string clean_hex = hex;
	if (clean_hex.front() == '#') {
		clean_hex = clean_hex.substr(1);
	}
	
	std::stringstream ss;
	ss << std::hex << clean_hex;
	uint32_t result;
	ss >> result;
	
	if (clean_hex.length() == 6) {
		result |= 0xFF000000;
	}
	
	return result;
}

// Load roster from CSV file
RosterData load_roster(const std::string& team_name, const std::string& config_dir) {
	RosterData roster;
	roster.team_name = team_name;
	roster.home_bg = 0xFF0080FF;
	roster.home_text = 0xFFFFFFFF;
	roster.away_bg = 0xFFFF8000;
	roster.away_text = 0xFFFFFFFF;
	
	// Check if team name is still a placeholder (can't load roster for unresolved teams)
	if (team_name.find("Winner Game ") != std::string::npos || 
	    team_name.find("Loser Game ") != std::string::npos ||
	    team_name.find(" vs ") != std::string::npos ||
	    team_name.find("Winner: ") != std::string::npos ||
	    team_name.find("Loser: ") != std::string::npos) {
		blog(LOG_INFO, "[Roster] Skipping roster load for unresolved team: %s", team_name.c_str());
		return roster;
	}
	
	// Get team colors from global schedule data
	if (g_schedule_data && g_schedule_data->teams.find(team_name) != g_schedule_data->teams.end()) {
		const auto& team = g_schedule_data->teams.at(team_name);
		roster.home_bg = team.home_bg;
		roster.home_text = team.home_text;
		roster.away_bg = team.away_bg;
		roster.away_text = team.away_text;
	}
	
	// Create filename from team name (lowercase, no spaces)
	std::string filename = team_name;
	std::transform(filename.begin(), filename.end(), filename.begin(), ::tolower);
	filename.erase(std::remove(filename.begin(), filename.end(), ' '), filename.end());
	
	std::string roster_path;
	if (!config_dir.empty()) {
		roster_path = config_dir + "/players/" + filename + ".csv";
	} else {
		roster_path = "config/players/" + filename + ".csv";
	}
	
	std::ifstream file(roster_path);
	if (!file.is_open()) {
		blog(LOG_WARNING, "[Roster] Could not open roster file: %s", roster_path.c_str());
		return roster;
	}
	
	std::string line;
	bool first_line = true;
	
	while (std::getline(file, line)) {
		if (first_line) {
			first_line = false;
			continue; // Skip header: First Name,Last Name,Cap Number
		}
		
		if (line.empty()) continue;
		
		std::stringstream ss(line);
		std::string first_name, last_name, cap_number;
		
		if (std::getline(ss, first_name, ',') &&
			std::getline(ss, last_name, ',') &&
			std::getline(ss, cap_number, ',')) {
			
			// Remove quotes if present
			auto remove_quotes = [](std::string& str) {
				if (!str.empty() && str.front() == '"' && str.back() == '"') {
					str = str.substr(1, str.length() - 2);
				}
			};
			
			remove_quotes(first_name);
			remove_quotes(last_name);
			remove_quotes(cap_number);
			
			Player player;
			player.first_name = first_name;
			player.last_name = last_name;
			player.cap_number = cap_number;
			
			roster.players.push_back(player);
		}
	}
	
	blog(LOG_INFO, "[Roster] Loaded %zu players for %s from %s", 
		 roster.players.size(), team_name.c_str(), roster_path.c_str());
	
	return roster;
}

// Get current game info (home and away teams)
struct CurrentGame {
	std::string home_team;
	std::string away_team;
	std::string date;
	std::string time;
	bool found;
};

CurrentGame get_current_game() {
	CurrentGame game;
	game.found = false;
	
	// First, check if we have a game selected in the scoreboard (via control panel)
	struct scoreboard_source *scoreboard = get_global_scoreboard();
	if (scoreboard && !scoreboard->home_team.empty() && !scoreboard->away_team.empty()) {
		game.home_team = scoreboard->home_team;
		game.away_team = scoreboard->away_team;
		game.found = true;
		blog(LOG_INFO, "[Roster] Using scoreboard game: %s vs %s", 
		     game.home_team.c_str(), game.away_team.c_str());
		return game;
	}
	
	// Fallback: use schedule to find next game
	if (!g_schedule_data || g_schedule_data->schedule.empty()) {
		return game;
	}
	
	auto now = std::chrono::system_clock::now();
	
	// Find the next upcoming game or current game
	for (const auto& sched_game : g_schedule_data->schedule) {
		if (sched_game.start_time >= now) {
			// Resolve placeholders to actual team names (without display prefix)
			game.home_team = resolve_team_placeholder(sched_game.home_team, false);
			game.away_team = resolve_team_placeholder(sched_game.away_team, false);
			game.date = sched_game.date;
			game.time = sched_game.time;
			game.found = true;
			blog(LOG_INFO, "[Roster] Current game resolved from schedule: %s vs %s", 
			     game.home_team.c_str(), game.away_team.c_str());
			break;
		}
	}
	
	// If no future game found, use the last game
	if (!game.found && !g_schedule_data->schedule.empty()) {
		const auto& sched_game = g_schedule_data->schedule.back();
		// Resolve placeholders to actual team names (without display prefix)
		game.home_team = resolve_team_placeholder(sched_game.home_team, false);
		game.away_team = resolve_team_placeholder(sched_game.away_team, false);
		game.date = sched_game.date;
		game.time = sched_game.time;
		game.found = true;
		blog(LOG_INFO, "[Roster] Last game resolved from schedule: %s vs %s", 
		     game.home_team.c_str(), game.away_team.c_str());
	}
	
	return game;
}

// Roster source context
struct roster_source_context {
	obs_source_t *source;
	
	// Rendering
	uint32_t width;
	uint32_t height;
	
	// Settings
	std::string config_dir;
	std::string team_mode; // "home", "away", "both", or specific team name
	std::string specific_team; // If team_mode is a specific team
	bool show_both_teams; // Show home and away side by side
	
	// Visual settings
	uint32_t background_color;
	uint32_t text_color;
	uint32_t accent_color;
	int font_size;
	
	// Data
	RosterData current_roster;
	RosterData away_roster; // For dual display mode
	std::chrono::system_clock::time_point last_update;
	
	// Scrolling
	float scroll_offset;
	std::chrono::steady_clock::time_point last_scroll_time;
	float scroll_speed; // pixels per second
	
#ifdef _WIN32
	// GDI+ resources
	Graphics *graphics;
	Bitmap *render_target;
#endif
	
	roster_source_context() : source(nullptr), width(1920), height(1080),
		team_mode("home"), specific_team(""), show_both_teams(false),
		background_color(0x001A1A1A), text_color(0xFFFFFFFF), accent_color(0xFF0080FF),
		font_size(36), scroll_offset(0.0f), scroll_speed(50.0f)
#ifdef _WIN32
		, graphics(nullptr), render_target(nullptr)
#endif
	{
		last_update = (std::chrono::system_clock::time_point::min)();
		last_scroll_time = std::chrono::steady_clock::now();
	}
};

// Forward declarations for OBS callbacks
static const char *roster_source_get_name(void *unused);
static void *roster_source_create(obs_data_t *settings, obs_source_t *source);
static void roster_source_destroy(void *data);
static void roster_source_update(void *data, obs_data_t *settings);
static obs_properties_t *roster_source_get_properties(void *data);
static void roster_source_get_defaults(obs_data_t *settings);
static void roster_source_render(void *data, gs_effect_t *effect);
static uint32_t roster_source_get_width(void *data);
static uint32_t roster_source_get_height(void *data);

#ifdef _WIN32
// Create rounded rectangle path
void add_rounded_rectangle_roster(GraphicsPath* path, float x, float y, float width, float height, float radius) {
	if (radius <= 0) {
		path->AddRectangle(RectF(x, y, width, height));
		return;
	}
	
	float diameter = radius * 2;
	path->AddArc(x, y, diameter, diameter, 180, 90);
	path->AddArc(x + width - diameter, y, diameter, diameter, 270, 90);
	path->AddArc(x + width - diameter, y + height - diameter, diameter, diameter, 0, 90);
	path->AddArc(x, y + height - diameter, diameter, diameter, 90, 90);
	path->CloseFigure();
}

// Helper function to render a single team's roster
void render_single_team_roster(Graphics* graphics, const RosterData& roster, const std::string& config_dir,
								float x, float y, float width, float height, float scroll_offset, 
								int font_size, bool is_home) {
	FontFamily fontFamily(L"Segoe UI");
	
	// Determine colors
	uint32_t bg_color = is_home ? roster.home_bg : roster.away_bg;
	uint32_t txt_color = is_home ? roster.home_text : roster.away_text;
	
	Color teamColor1(200, (bg_color >> 16) & 0xFF, (bg_color >> 8) & 0xFF, bg_color & 0xFF);
	Color teamColor2(100, (bg_color >> 16) & 0xFF, (bg_color >> 8) & 0xFF, bg_color & 0xFF);
	Color teamTextColor((txt_color >> 24) & 0xFF, (txt_color >> 16) & 0xFF, (txt_color >> 8) & 0xFF, txt_color & 0xFF);
	SolidBrush teamTextBrush(teamTextColor);
	
	float titleHeight = 80.0f;
	float margin = 15.0f;
	
	// Title
	std::wstring title = std::wstring(roster.team_name.begin(), roster.team_name.end());
	RectF titleRect(x, y, width, titleHeight);
	
	GraphicsPath titlePath;
	add_rounded_rectangle_roster(&titlePath, titleRect.X, titleRect.Y, titleRect.Width, titleRect.Height, 12.0f);
	
	LinearGradientBrush titleGradient(
		PointF(titleRect.X, titleRect.Y),
		PointF(titleRect.X, titleRect.Y + titleRect.Height),
		teamColor1,
		teamColor2
	);
	
	graphics->FillPath(&titleGradient, &titlePath);
	
	// Load and draw team logo
	Image* teamLogo = nullptr;
	if (g_schedule_data && g_schedule_data->teams.find(roster.team_name) != g_schedule_data->teams.end()) {
		const auto& team = g_schedule_data->teams.at(roster.team_name);
		std::string base_logo_path = team.logo_path;
		
		std::vector<std::string> paths_to_try;
		std::string png_path = base_logo_path;
		if (png_path.length() >= 4 && png_path.substr(png_path.length() - 4) == ".svg") {
			png_path = png_path.substr(0, png_path.length() - 4) + ".png";
		}
		
		if (!config_dir.empty()) {
			paths_to_try.push_back(config_dir + "/" + base_logo_path);
			paths_to_try.push_back(config_dir + "/" + png_path);
		} else {
			paths_to_try.push_back("config/" + base_logo_path);
			paths_to_try.push_back("config/" + png_path);
		}
		
		for (const auto& full_path : paths_to_try) {
			int size_needed = MultiByteToWideChar(CP_UTF8, 0, full_path.c_str(), -1, NULL, 0);
			std::wstring wide_path(size_needed, 0);
			MultiByteToWideChar(CP_UTF8, 0, full_path.c_str(), -1, &wide_path[0], size_needed);
			
			teamLogo = Image::FromFile(wide_path.c_str());
			if (teamLogo && teamLogo->GetLastStatus() == Ok) {
				break;
			} else {
				if (teamLogo) {
					delete teamLogo;
					teamLogo = nullptr;
				}
			}
		}
	}
	
	float logoSize = titleHeight - 20.0f;
	if (teamLogo) {
		graphics->DrawImage(teamLogo, x + 10.0f, y + 10.0f, logoSize, logoSize);
		delete teamLogo;
	}
	
	StringFormat centerFormat;
	centerFormat.SetAlignment(StringAlignmentCenter);
	centerFormat.SetLineAlignment(StringAlignmentCenter);
	
	Gdiplus::Font titleFont(&fontFamily, (REAL)(font_size * 1.4), FontStyleBold, UnitPixel);
	graphics->DrawString(title.c_str(), -1, &titleFont, titleRect, &centerFormat, &teamTextBrush);
	
	// Players list
	float contentStartY = y + titleHeight + 10.0f;
	float visibleHeight = height - titleHeight - 10.0f;
	float playerRowHeight = 120.0f;
	float playerSpacing = 15.0f;
	
	if (!roster.players.empty()) {
		// Set up clipping
		Region clipRegion(RectF(x, contentStartY, width, visibleHeight));
		graphics->SetClip(&clipRegion);
		
		int playerIndex = 0;
		for (const auto& player : roster.players) {
			float playerY = contentStartY + playerIndex * (playerRowHeight + playerSpacing) - scroll_offset;
			
			if (playerY + playerRowHeight >= contentStartY - playerRowHeight && playerY <= y + height + playerRowHeight) {
				// Narrower card
				float cardWidth = width - 2 * margin;
				RectF playerRect(x + margin, playerY, cardWidth, playerRowHeight);
				
				GraphicsPath playerPath;
				add_rounded_rectangle_roster(&playerPath, playerRect.X, playerRect.Y, playerRect.Width, playerRect.Height, 12.0f);
				
				LinearGradientBrush playerGradient(
					PointF(playerRect.X, playerRect.Y),
					PointF(playerRect.X + playerRect.Width, playerRect.Y),
					teamColor2,
					teamColor1
				);
				
				graphics->FillPath(&playerGradient, &playerPath);
				
				Pen borderPen(Color(150, 255, 255, 255), 2.0f);
				graphics->DrawPath(&borderPen, &playerPath);
				
				// Cap number
				float capWidth = 100.0f;
				RectF capRect(playerRect.X + 15, playerRect.Y, capWidth, playerRowHeight);
				std::wstring capStr = std::wstring(player.cap_number.begin(), player.cap_number.end());
				
				Gdiplus::Font capFontLarge(&fontFamily, (REAL)(font_size * 2.2), FontStyleBold, UnitPixel);
				StringFormat capFormat;
				capFormat.SetAlignment(StringAlignmentCenter);
				capFormat.SetLineAlignment(StringAlignmentCenter);
				graphics->DrawString(capStr.c_str(), -1, &capFontLarge, capRect, &capFormat, &teamTextBrush);
				
				// Player name (left-aligned, narrower)
				RectF nameRect(playerRect.X + capWidth + 20, playerRect.Y + 10, cardWidth - capWidth - 35, playerRowHeight - 20);
				std::wstring nameStr = std::wstring(player.first_name.begin(), player.first_name.end()) + L" " +
									   std::wstring(player.last_name.begin(), player.last_name.end());
				
				Gdiplus::Font nameFontLarge(&fontFamily, (REAL)(font_size * 1.6), FontStyleBold, UnitPixel);
				StringFormat nameFormat;
				nameFormat.SetAlignment(StringAlignmentNear);
				nameFormat.SetLineAlignment(StringAlignmentCenter);
				nameFormat.SetTrimming(StringTrimmingEllipsisCharacter);
				graphics->DrawString(nameStr.c_str(), -1, &nameFontLarge, nameRect, &nameFormat, &teamTextBrush);
			}
			
			playerIndex++;
		}
		
		graphics->ResetClip();
	}
}

// Render roster display
void render_roster(roster_source_context *context) {
	if (!context->graphics) return;
	
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
	
	// Update scroll offset
	auto now = std::chrono::steady_clock::now();
	float delta_time = std::chrono::duration<float>(now - context->last_scroll_time).count();
	context->last_scroll_time = now;
	
	float margin = 30.0f;
	
	if (context->show_both_teams) {
		// Dual display mode - show home and away side by side
		float columnWidth = (context->width - 3 * margin) / 2.0f;
		float contentHeight = context->height - 2 * margin;
		
		// Calculate scroll based on the longer roster
		size_t max_players = std::max(context->current_roster.players.size(), context->away_roster.players.size());
		float playerRowHeight = 120.0f;
		float playerSpacing = 15.0f;
		float totalContentHeight = max_players * (playerRowHeight + playerSpacing);
		float visibleHeight = contentHeight - 90.0f; // Subtract title height
		
		if (totalContentHeight > visibleHeight) {
			context->scroll_offset += context->scroll_speed * delta_time;
			float maxScroll = totalContentHeight + visibleHeight * 0.5f;
			if (context->scroll_offset > maxScroll) {
				context->scroll_offset = -visibleHeight * 0.5f;
			}
		} else {
			context->scroll_offset = 0.0f;
		}
		
		// Render home team (left side)
		render_single_team_roster(context->graphics, context->current_roster, context->config_dir,
								   margin, margin, columnWidth, contentHeight, context->scroll_offset,
								   context->font_size, true);
		
		// Render away team (right side)
		render_single_team_roster(context->graphics, context->away_roster, context->config_dir,
								   margin * 2 + columnWidth, margin, columnWidth, contentHeight, context->scroll_offset,
								   context->font_size, false);
	} else {
		// Single team display mode
		float contentWidth = context->width - 2 * margin;
		float contentHeight = context->height - 2 * margin;
		
		bool is_home = (context->team_mode != "away");
		
		float playerRowHeight = 120.0f;
		float playerSpacing = 15.0f;
		float totalContentHeight = context->current_roster.players.size() * (playerRowHeight + playerSpacing);
		float visibleHeight = contentHeight - 90.0f;
		
		if (totalContentHeight > visibleHeight) {
			context->scroll_offset += context->scroll_speed * delta_time;
			float maxScroll = totalContentHeight + visibleHeight * 0.5f;
			if (context->scroll_offset > maxScroll) {
				context->scroll_offset = -visibleHeight * 0.5f;
			}
		} else {
			context->scroll_offset = 0.0f;
		}
		
		render_single_team_roster(context->graphics, context->current_roster, context->config_dir,
								   margin, margin, contentWidth, contentHeight, context->scroll_offset,
								   context->font_size, is_home);
	}
}
#endif

// OBS Source callbacks implementation
static const char *roster_source_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Water Polo Roster";
}

static void *roster_source_create(obs_data_t *settings, obs_source_t *source)
{
	auto *context = new roster_source_context();
	context->source = source;
	
	// Initialize global schedule data if needed
	init_global_schedule_data();

#ifdef _WIN32
	// Create render target
	context->render_target = new Bitmap(context->width, context->height, PixelFormat32bppARGB);
	context->graphics = new Graphics(context->render_target);
#endif
	
	// Update from settings
	roster_source_update(context, settings);
	
	blog(LOG_INFO, "[Roster] Source created");
	return context;
}

static void roster_source_destroy(void *data)
{
	auto *context = static_cast<roster_source_context*>(data);
	
#ifdef _WIN32
	if (context->graphics) {
		delete context->graphics;
	}
	if (context->render_target) {
		delete context->render_target;
	}
#endif
	
	delete context;
	blog(LOG_INFO, "[Roster] Source destroyed");
}

static void roster_source_update(void *data, obs_data_t *settings)
{
	auto *context = static_cast<roster_source_context*>(data);
	
	// Get config directory
	const char *config_directory = obs_data_get_string(settings, "config_directory");
	std::string new_config_dir = config_directory ? config_directory : "";
	
	if (new_config_dir.empty()) {
		new_config_dir = get_saved_config_dir_roster();
	}
	
	if (new_config_dir != context->config_dir) {
		context->config_dir = new_config_dir;
		
		// Update global schedule data to get team info
		if (!new_config_dir.empty()) {
			update_global_schedule_data(new_config_dir);
		}
	}
	
	// Get team mode
	const char *team_mode = obs_data_get_string(settings, "team_mode");
	context->team_mode = team_mode ? team_mode : "home";
	
	// Check if showing both teams
	context->show_both_teams = (context->team_mode == "both");
	
	// Get specific team if applicable
	const char *specific_team = obs_data_get_string(settings, "specific_team");
	context->specific_team = specific_team ? specific_team : "";
	
	// Visual settings
	context->background_color = (uint32_t)obs_data_get_int(settings, "background_color");
	context->text_color = (uint32_t)obs_data_get_int(settings, "text_color");
	context->accent_color = (uint32_t)obs_data_get_int(settings, "accent_color");
	context->font_size = (int)obs_data_get_int(settings, "font_size");
	
	if (context->show_both_teams) {
		// Load both home and away rosters
		CurrentGame game = get_current_game();
		if (game.found) {
			if (game.home_team != context->current_roster.team_name) {
				context->current_roster = load_roster(game.home_team, context->config_dir);
			}
			if (game.away_team != context->away_roster.team_name) {
				context->away_roster = load_roster(game.away_team, context->config_dir);
			}
			context->last_update = std::chrono::system_clock::now();
			
			blog(LOG_INFO, "[Roster] Both teams mode - Home: %s (%zu players), Away: %s (%zu players)", 
				 context->current_roster.team_name.c_str(), context->current_roster.players.size(),
				 context->away_roster.team_name.c_str(), context->away_roster.players.size());
		}
	} else {
		// Determine which team to display
		std::string team_to_display;
		
		if (context->team_mode == "specific" && !context->specific_team.empty()) {
			team_to_display = context->specific_team;
		} else {
			CurrentGame game = get_current_game();
			if (game.found) {
				team_to_display = (context->team_mode == "away") ? game.away_team : game.home_team;
			}
		}
		
		// Load roster if team changed
		if (!team_to_display.empty() && team_to_display != context->current_roster.team_name) {
			context->current_roster = load_roster(team_to_display, context->config_dir);
			context->last_update = std::chrono::system_clock::now();
		}
		
		blog(LOG_INFO, "[Roster] Settings updated - Mode: %s, Team: %s, Players: %zu", 
			 context->team_mode.c_str(), context->current_roster.team_name.c_str(), 
			 context->current_roster.players.size());
	}
}

static obs_properties_t *roster_source_get_properties(void *data)
{
	UNUSED_PARAMETER(data);
	
	obs_properties_t *props = obs_properties_create();
	
	// Config directory
	obs_properties_add_path(props, "config_directory", "Config Directory",
		OBS_PATH_DIRECTORY, nullptr, nullptr);
	
	// Team mode selection
	obs_property_t *mode_list = obs_properties_add_list(props, "team_mode", "Display Team",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_property_list_add_string(mode_list, "Home Team (from current game)", "home");
	obs_property_list_add_string(mode_list, "Away Team (from current game)", "away");
	obs_property_list_add_string(mode_list, "Both Teams (side by side)", "both");
	obs_property_list_add_string(mode_list, "Specific Team", "specific");
	
	// Specific team selection (only shown when mode is "specific")
	obs_property_t *team_list = obs_properties_add_list(props, "specific_team", "Team",
		OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	
	// Populate with available teams
	if (g_schedule_data) {
		for (const auto& team_pair : g_schedule_data->teams) {
			obs_property_list_add_string(team_list, team_pair.first.c_str(), team_pair.first.c_str());
		}
	}
	
	// Visual settings
	obs_properties_add_color(props, "background_color", "Background Color");
	obs_properties_add_color(props, "text_color", "Text Color");
	obs_properties_add_color(props, "accent_color", "Accent Color");
	obs_properties_add_int_slider(props, "font_size", "Font Size", 12, 48, 2);
	
	return props;
}

static void roster_source_get_defaults(obs_data_t *settings)
{
	// Set default config directory from saved settings
	std::string saved_config_dir = get_saved_config_dir_roster();
	if (!saved_config_dir.empty()) {
		obs_data_set_default_string(settings, "config_directory", saved_config_dir.c_str());
	}
	
	obs_data_set_default_string(settings, "team_mode", "home");
	obs_data_set_default_string(settings, "specific_team", "");
	
	// Default colors
	obs_data_set_default_int(settings, "background_color", 0x001A1A1A);
	obs_data_set_default_int(settings, "text_color", 0xFFFFFFFF);
	obs_data_set_default_int(settings, "accent_color", 0xFF0080FF);
	obs_data_set_default_int(settings, "font_size", 32);
}

static void roster_source_render(void *data, gs_effect_t *effect)
{
	auto *context = static_cast<roster_source_context*>(data);
	
	// Check if we need to update roster (game might have changed)
	auto now = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - context->last_update).count();
	
	// Update every 5 seconds to check for game changes
	if (elapsed >= 5 && (context->team_mode == "home" || context->team_mode == "away")) {
		CurrentGame game = get_current_game();
		if (game.found) {
			std::string team_to_display = (context->team_mode == "away") ? game.away_team : game.home_team;
			if (team_to_display != context->current_roster.team_name && !team_to_display.empty()) {
				context->current_roster = load_roster(team_to_display, context->config_dir);
				context->last_update = now;
				blog(LOG_INFO, "[Roster] Auto-updated to %s team: %s", 
					 context->team_mode.c_str(), team_to_display.c_str());
			}
		}
	}
	
	// Also check for both teams mode
	if (elapsed >= 5 && context->show_both_teams) {
		CurrentGame game = get_current_game();
		if (game.found) {
			if (game.home_team != context->current_roster.team_name && !game.home_team.empty()) {
				context->current_roster = load_roster(game.home_team, context->config_dir);
			}
			if (game.away_team != context->away_roster.team_name && !game.away_team.empty()) {
				context->away_roster = load_roster(game.away_team, context->config_dir);
			}
			if (game.home_team != context->current_roster.team_name || game.away_team != context->away_roster.team_name) {
				context->last_update = now;
				blog(LOG_INFO, "[Roster] Auto-updated both teams: %s vs %s", 
					 game.home_team.c_str(), game.away_team.c_str());
			}
		}
	}
	
#ifdef _WIN32
	// Render the roster
	render_roster(context);
	
	// Convert to OBS texture
	if (context->render_target) {
		BitmapData bitmapData;
		Rect rect(0, 0, context->width, context->height);
		
		if (context->render_target->LockBits(&rect, ImageLockModeRead, PixelFormat32bppARGB, &bitmapData) == Ok) {
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

static uint32_t roster_source_get_width(void *data)
{
	auto *context = static_cast<roster_source_context*>(data);
	return context->width;
}

static uint32_t roster_source_get_height(void *data)
{
	auto *context = static_cast<roster_source_context*>(data);
	return context->height;
}

// Register the roster source
void register_roster_source()
{
	struct obs_source_info info = {};
	info.id = "water_polo_roster";
	info.type = OBS_SOURCE_TYPE_INPUT;
	info.output_flags = OBS_SOURCE_VIDEO;
	info.get_name = roster_source_get_name;
	info.create = roster_source_create;
	info.destroy = roster_source_destroy;
	info.update = roster_source_update;
	info.get_properties = roster_source_get_properties;
	info.get_defaults = roster_source_get_defaults;
	info.video_render = roster_source_render;
	info.get_width = roster_source_get_width;
	info.get_height = roster_source_get_height;
	
	obs_register_source(&info);
	
	blog(LOG_INFO, "Registered water_polo_roster source");
}
