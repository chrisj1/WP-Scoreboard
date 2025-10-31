#include <obs-module.h>
#include <obs-frontend-api.h>
#include <util/platform.h>
#include <QMainWindow>
#include <QAction>
#include <QMessageBox>

// Diagnostic logging for DLL loading
#ifdef _WIN32
#include <windows.h>
#endif

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
	blog(LOG_INFO, "========================================");
	blog(LOG_INFO, "Water Polo Scoreboard Plugin - Starting Load");
	blog(LOG_INFO, "Version: %s", PLUGIN_VERSION);
	blog(LOG_INFO, "========================================");
	
#ifdef _WIN32
	// Log DLL loading diagnostics
	blog(LOG_INFO, "[Diagnostics] Checking critical dependencies...");
	
	// Check if LibTorch DLLs are accessible
	HMODULE torchCpu = LoadLibraryA("torch_cpu.dll");
	if (torchCpu) {
		blog(LOG_INFO, "[Diagnostics] torch_cpu.dll loaded successfully");
		FreeLibrary(torchCpu);
	} else {
		DWORD err = GetLastError();
		blog(LOG_ERROR, "[Diagnostics] Failed to load torch_cpu.dll - Error: %lu", err);
	}
	
	HMODULE torchCuda = LoadLibraryA("torch_cuda.dll");
	if (torchCuda) {
		blog(LOG_INFO, "[Diagnostics] torch_cuda.dll loaded successfully");
		FreeLibrary(torchCuda);
	} else {
		DWORD err = GetLastError();
		blog(LOG_ERROR, "[Diagnostics] Failed to load torch_cuda.dll - Error: %lu", err);
	}
	
	HMODULE opencv = LoadLibraryA("opencv_world4100.dll");
	if (opencv) {
		blog(LOG_INFO, "[Diagnostics] opencv_world4100.dll loaded successfully");
		FreeLibrary(opencv);
	} else {
		DWORD err = GetLastError();
		blog(LOG_ERROR, "[Diagnostics] Failed to load opencv_world4100.dll - Error: %lu", err);
	}
	
	blog(LOG_INFO, "[Diagnostics] Initializing GDI+...");
	init_gdiplus_simple();
	blog(LOG_INFO, "[Diagnostics] GDI+ initialized");
#else
	blog(LOG_INFO, "Water Polo Scoreboard Plugin loaded successfully (version %s)", 
	     PLUGIN_VERSION);
#endif
	
	// Initialize global schedule data
	blog(LOG_INFO, "[Diagnostics] Initializing global schedule data...");
	init_global_schedule_data();
	blog(LOG_INFO, "[Diagnostics] Global schedule data initialized");
	
	// Register the scoreboard source
	blog(LOG_INFO, "[Diagnostics] Registering scoreboard source...");
	register_scoreboard_source();
	blog(LOG_INFO, "[Diagnostics] Scoreboard source registered");
	
	// Register the schedule source
	blog(LOG_INFO, "[Diagnostics] Registering schedule source...");
	register_schedule_source();
	blog(LOG_INFO, "[Diagnostics] Schedule source registered");
	
	// Register the roster source
	blog(LOG_INFO, "[Diagnostics] Registering roster source...");
	register_roster_source();
	blog(LOG_INFO, "[Diagnostics] Roster source registered");
	
	// Initialize WebSocket server
	blog(LOG_INFO, "[Diagnostics] Initializing WebSocket server...");
	init_websocket_server();
	blog(LOG_INFO, "[Diagnostics] WebSocket server initialized");
	
	// Initialize control panel GUI
	blog(LOG_INFO, "[Diagnostics] Initializing control panel...");
	init_control_panel();
	blog(LOG_INFO, "[Diagnostics] Control panel initialized");
	
	blog(LOG_INFO, "========================================");
	blog(LOG_INFO, "Water Polo Scoreboard Plugin LOADED SUCCESSFULLY");
	blog(LOG_INFO, "========================================");
	
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
