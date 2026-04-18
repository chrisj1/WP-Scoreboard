#include <obs-module.h>
#include <obs-frontend-api.h>
#include <util/platform.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "replay-shared.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-scoreboard", "en-US")

// Forward declarations
void register_scoreboard_source();
void register_schedule_source();
void register_roster_source();
void init_global_schedule_data();
void cleanup_global_schedule_data();
void init_control_panel();
void shutdown_control_panel();
void init_websocket_server();
void shutdown_websocket_server();

#ifdef _WIN32
extern void init_gdiplus_simple();
extern void shutdown_gdiplus_simple();
#endif

bool obs_module_load(void)
{
	blog(LOG_INFO, "Water Polo Scoreboard Plugin loading (version %s)", PLUGIN_VERSION);

#ifdef _WIN32
	// Probe CNN dependency DLLs so missing ones appear in the log early.
	auto probe_dll = [](const char *name) {
		HMODULE h = LoadLibraryA(name);
		if (h) { blog(LOG_INFO,  "[deps] %s OK", name); FreeLibrary(h); }
		else    { blog(LOG_WARNING, "[deps] %s not found (error %lu)", name, GetLastError()); }
	};
	probe_dll("torch_cpu.dll");
	probe_dll("torch_cuda.dll");
	probe_dll("opencv_world4100.dll");

	init_gdiplus_simple();
#endif

	init_global_schedule_data();
	register_scoreboard_source();
	register_schedule_source();
	register_roster_source();
	register_replay_filter();
	register_replay_source();
	init_websocket_server();

	try {
		init_control_panel();
	} catch (const std::exception& e) {
		blog(LOG_ERROR, "Control panel init failed: %s — plugin continues without it", e.what());
	}

	blog(LOG_INFO, "Water Polo Scoreboard Plugin loaded");
	return true;
}

void obs_module_unload(void)
{
	blog(LOG_INFO, "Water Polo Scoreboard Plugin unloaded");
	
	// Cleanup
	shutdown_control_panel();
	shutdown_websocket_server();
	cleanup_global_schedule_data();
	
#ifdef _WIN32
	shutdown_gdiplus_simple();
#endif
}

const char *obs_module_name(void)
{
	return "Water Polo Scoreboard";
}

const char *obs_module_description(void)
{
	return "Water polo scoreboard overlay with shot clock and game clock";
}
