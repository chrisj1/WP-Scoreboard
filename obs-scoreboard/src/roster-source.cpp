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

// Global schedule data reference (from schedule source)
extern struct GlobalScheduleData *g_schedule_data;
extern void init_global_schedule_data();
extern void update_global_schedule_data(const std::string& config_dir);

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
	
	if (!g_schedule_data || g_schedule_data->schedule.empty()) {
		return game;
	}
	
	auto now = std::chrono::system_clock::now();
	
	// Find the next upcoming game or current game
	for (const auto& sched_game : g_schedule_data->schedule) {
		if (sched_game.start_time >= now) {
			game.home_team = sched_game.home_team;
			game.away_team = sched_game.away_team;
			game.date = sched_game.date;
			game.time = sched_game.time;
			game.found = true;
			break;
		}
	}
	
	// If no future game found, use the last game
	if (!game.found && !g_schedule_data->schedule.empty()) {
		const auto& sched_game = g_schedule_data->schedule.back();
		game.home_team = sched_game.home_team;
		game.away_team = sched_game.away_team;
		game.date = sched_game.date;
		game.time = sched_game.time;
		game.found = true;
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
	std::string team_mode; // "home", "away", or specific team name
	std::string specific_team; // If team_mode is a specific team
	
	// Visual settings
	uint32_t background_color;
	uint32_t text_color;
	uint32_t accent_color;
	int font_size;
	
	// Data
	RosterData current_roster;
	std::chrono::system_clock::time_point last_update;
	
#ifdef _WIN32
	// GDI+ resources
	Graphics *graphics;
	Bitmap *render_target;
#endif
	
	roster_source_context() : source(nullptr), width(1920), height(1080),
		team_mode("home"), specific_team(""),
		background_color(0x001A1A1A), text_color(0xFFFFFFFF), accent_color(0xFF0080FF),
		font_size(36)
#ifdef _WIN32
		, graphics(nullptr), render_target(nullptr)
#endif
	{
		last_update = (std::chrono::system_clock::time_point::min)();
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
	
	// Fonts
	FontFamily fontFamily(L"Segoe UI");
	Gdiplus::Font titleFont(&fontFamily, (REAL)(context->font_size * 1.8), FontStyleBold, UnitPixel);
	Gdiplus::Font playerFont(&fontFamily, (REAL)(context->font_size * 0.9), FontStyleRegular, UnitPixel);
	Gdiplus::Font capFont(&fontFamily, (REAL)(context->font_size * 1.2), FontStyleBold, UnitPixel);
	
	// Colors
	SolidBrush textBrush(Color(
		(context->text_color >> 24) & 0xFF,
		(context->text_color >> 16) & 0xFF,
		(context->text_color >> 8) & 0xFF,
		context->text_color & 0xFF
	));
	
	// Determine which colors to use (home or away)
	uint32_t bg_color = context->current_roster.home_bg;
	uint32_t txt_color = context->current_roster.home_text;
	
	if (context->team_mode == "away") {
		bg_color = context->current_roster.away_bg;
		txt_color = context->current_roster.away_text;
	}
	
	Color teamColor1(200, (bg_color >> 16) & 0xFF, (bg_color >> 8) & 0xFF, bg_color & 0xFF);
	Color teamColor2(100, (bg_color >> 16) & 0xFF, (bg_color >> 8) & 0xFF, bg_color & 0xFF);
	Color teamTextColor((txt_color >> 24) & 0xFF, (txt_color >> 16) & 0xFF, (txt_color >> 8) & 0xFF, txt_color & 0xFF);
	
	SolidBrush teamTextBrush(teamTextColor);
	
	// Layout
	float margin = 30.0f;
	float titleHeight = 100.0f;
	float playerHeight = 60.0f;
	
	// Title
	std::wstring title = L"Roster - " + std::wstring(context->current_roster.team_name.begin(), 
													  context->current_roster.team_name.end());
	RectF titleRect(margin, margin, (REAL)(context->width - 2 * margin), titleHeight);
	
	GraphicsPath titlePath;
	add_rounded_rectangle_roster(&titlePath, titleRect.X, titleRect.Y, titleRect.Width, titleRect.Height, 15.0f);
	
	LinearGradientBrush titleGradient(
		PointF(titleRect.X, titleRect.Y),
		PointF(titleRect.X, titleRect.Y + titleRect.Height),
		teamColor1,
		teamColor2
	);
	
	context->graphics->FillPath(&titleGradient, &titlePath);
	
	StringFormat centerFormat;
	centerFormat.SetAlignment(StringAlignmentCenter);
	centerFormat.SetLineAlignment(StringAlignmentCenter);
	
	context->graphics->DrawString(title.c_str(), -1, &titleFont, titleRect, &centerFormat, &teamTextBrush);
	
	// Players grid
	float currentY = margin + titleHeight + 20.0f;
	float availableHeight = context->height - currentY - margin;
	
	// Calculate columns - fit as many as possible
	int columns = 2;
	float columnWidth = (context->width - 2 * margin - (columns - 1) * 15.0f) / columns;
	
	if (context->current_roster.players.empty()) {
		std::wstring noPlayers = L"No roster data available";
		RectF messageRect(margin, currentY, (REAL)(context->width - 2 * margin), playerHeight);
		context->graphics->DrawString(noPlayers.c_str(), -1, &playerFont, messageRect, &centerFormat, &textBrush);
	} else {
		int playerIndex = 0;
		int row = 0;
		
		for (const auto& player : context->current_roster.players) {
			int col = playerIndex % columns;
			row = playerIndex / columns;
			
			float x = margin + col * (columnWidth + 15.0f);
			float y = currentY + row * (playerHeight + 10.0f);
			
			// Check if we're out of vertical space
			if (y + playerHeight > context->height - margin) {
				break;
			}
			
			// Player card
			RectF playerRect(x, y, columnWidth, playerHeight);
			
			GraphicsPath playerPath;
			add_rounded_rectangle_roster(&playerPath, playerRect.X, playerRect.Y, playerRect.Width, playerRect.Height, 8.0f);
			
			// Card background with team color
			LinearGradientBrush playerGradient(
				PointF(playerRect.X, playerRect.Y),
				PointF(playerRect.X + playerRect.Width, playerRect.Y),
				teamColor2,
				teamColor1
			);
			
			context->graphics->FillPath(&playerGradient, &playerPath);
			
			// Border
			Pen borderPen(Color(100, 255, 255, 255), 1.5f);
			context->graphics->DrawPath(&borderPen, &playerPath);
			
			// Cap number on left side
			float capWidth = 80.0f;
			RectF capRect(x + 10, y, capWidth, playerHeight);
			std::wstring capStr = std::wstring(player.cap_number.begin(), player.cap_number.end());
			
			StringFormat capFormat;
			capFormat.SetAlignment(StringAlignmentCenter);
			capFormat.SetLineAlignment(StringAlignmentCenter);
			
			context->graphics->DrawString(capStr.c_str(), -1, &capFont, capRect, &capFormat, &teamTextBrush);
			
			// Player name on right side
			RectF nameRect(x + capWidth + 10, y + 10, columnWidth - capWidth - 20, playerHeight - 20);
			std::wstring nameStr = std::wstring(player.first_name.begin(), player.first_name.end()) + L" " +
								   std::wstring(player.last_name.begin(), player.last_name.end());
			
			StringFormat nameFormat;
			nameFormat.SetAlignment(StringAlignmentNear);
			nameFormat.SetLineAlignment(StringAlignmentCenter);
			nameFormat.SetTrimming(StringTrimmingEllipsisCharacter);
			
			context->graphics->DrawString(nameStr.c_str(), -1, &playerFont, nameRect, &nameFormat, &teamTextBrush);
			
			playerIndex++;
		}
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
	
	// Get specific team if applicable
	const char *specific_team = obs_data_get_string(settings, "specific_team");
	context->specific_team = specific_team ? specific_team : "";
	
	// Visual settings
	context->background_color = (uint32_t)obs_data_get_int(settings, "background_color");
	context->text_color = (uint32_t)obs_data_get_int(settings, "text_color");
	context->accent_color = (uint32_t)obs_data_get_int(settings, "accent_color");
	context->font_size = (int)obs_data_get_int(settings, "font_size");
	
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
	
	// Update every 60 seconds to check for game changes
	if (elapsed >= 60 && (context->team_mode == "home" || context->team_mode == "away")) {
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
