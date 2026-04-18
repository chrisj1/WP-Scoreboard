#pragma once

#include <obs-module.h>
#include <string>
#include <cstdint>

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

	// Team records (wins-losses-ties)
	int home_wins;
	int home_losses;
	int home_ties;
	int away_wins;
	int away_losses;
	int away_ties;

	// Next game preview
	std::string next_home_team;
	std::string next_away_team;
	std::string next_home_logo_path;
	std::string next_away_logo_path;

	// Water polo specific
	int period;
	std::string period_text;
	int home_exclusions;
	int away_exclusions;
	int home_timeouts;
	int away_timeouts;

	// Man-up indicators
	bool home_manup;
	uint64_t home_manup_start_ns;
	uint64_t away_manup_start_ns;
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
	bool show_records;
};

struct scoreboard_source *get_global_scoreboard();
