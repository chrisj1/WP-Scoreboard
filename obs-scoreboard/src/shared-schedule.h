#pragma once

#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cstdint>

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

// Global schedule data (shared between control panel and schedule source)
struct GlobalScheduleData {
	std::vector<Game> schedule;
	std::map<std::string, Team> teams;
	std::string config_dir;
	std::chrono::system_clock::time_point last_update;
	
	GlobalScheduleData();
};

// Global schedule data instance
extern GlobalScheduleData *g_schedule_data;

// Shared functions for schedule data management
void init_global_schedule_data();
void cleanup_global_schedule_data();
void update_global_schedule_data(const std::string& config_dir);
void notify_schedule_data_updated();

// Resolve placeholder team names
// If for_display=true, returns "Winner: Team" or "Loser: Team"
// If for_display=false, returns just "Team" (for roster loading)
std::string resolve_team_placeholder(const std::string& team_name, bool for_display = true);