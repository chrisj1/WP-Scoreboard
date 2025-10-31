#pragma once

#include <string>

// Shared functions for schedule data management
void init_global_schedule_data();
void cleanup_global_schedule_data();
void update_global_schedule_data(const std::string& config_dir);
void notify_schedule_data_updated();