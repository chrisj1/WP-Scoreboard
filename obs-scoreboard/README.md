# Water Polo Scoreboard - OBS Plugin

A fully-featured water polo scoreboard plugin for OBS Studio with GUI controls and WebSocket API.

## Features

- **Water Polo Specific**
  - 30-second shot clock
  - 8-minute game clock (per period)
  - 4 periods
  - Team scores (home/away)
  - Exclusion tracking (up to 6 per team)
  - Timeout tracking (3 per team)
  - Period display

- **Control Options**
  - GUI control panel (Tools → Water Polo Scoreboard Control)
  - WebSocket API for automation (port 8765)
  - Live clock countdown with pause/resume

- **Customization**
  - Team names (editable in settings)
  - Colors (background, text, accent configurable)
  - Display formatting

## Installation

### Prerequisites
- OBS Studio 32.0.1 (or compatible version)
- Qt 6.9.3 MSVC 2022 64-bit (for building from source)
- Visual Studio 2022 Community with C++ tools (for building from source)

### Installing the Plugin

1. Run the install script with administrator privileges:
   ```powershell
   .\install-plugin.ps1
   ```

2. The plugin DLL will be copied to:
   ```
   C:\Program Files\obs-studio\obs-plugins\64bit\obs-scoreboard.dll
   ```

3. Restart OBS Studio if it's running

## Usage

### Adding the Scoreboard Source

1. In OBS Studio, add a new source
2. Select **"Water Polo Scoreboard"** from the source list
3. Configure colors and team names in the source properties
4. Resize and position the scoreboard in your scene

### Using the GUI Control Panel

1. Go to **Tools → Water Polo Scoreboard Control**
2. Control panel provides:
   - Score adjustments (+1, +2 buttons)
   - Period selection (1-4)
   - Exclusions (add/remove per team)
   - Timeouts (track used timeouts)
   - Game clock (start, pause, reset)
   - Shot clock (start, pause, reset)

### Using the WebSocket API

The plugin listens on **port 8765** for WebSocket connections.

#### Connection
```python
import websocket
import json

ws = websocket.create_connection("ws://localhost:8765")
```

#### Message Format
```json
{
  "command": "update",
  "data": {
    "home_score": 5,
    "away_score": 3,
    "game_clock": "07:23",
    "shot_clock": 25,
    "period": 2,
    "home_exclusions": 1,
    "away_exclusions": 0,
    "home_timeouts": 1,
    "away_timeouts": 2
  }
}
```

#### Example (Python)
```python
message = {
    "command": "update",
    "data": {
        "home_score": 10,
        "away_score": 8,
        "period": 3
    }
}
ws.send(json.dumps(message))
```

Test the WebSocket with:
```bash
python test_websocket.py
```

## Building from Source

### 1. Install Prerequisites

- **Visual Studio 2022 Community** with "Desktop development with C++"
- **Qt 6.9.3** with MSVC 2022 64-bit component
- **OBS Studio 32.0.1** installed
- **OBS Studio 32.0.1 source code** downloaded

### 2. Build the Plugin

```powershell
.\build-direct.ps1
```

The build script will:
1. Generate import libraries from OBS DLLs
2. Configure CMake with MSVC
3. Compile all source files
4. Link into `obs-scoreboard.dll`

### 3. Install

```powershell
.\install-plugin.ps1
```

## Project Structure

```
obs-scoreboard/
├── src/
│   ├── plugin-main.cpp           # Plugin entry point
│   ├── scoreboard-source.cpp     # OBS source implementation
│   ├── control-panel.cpp         # GUI control panel
│   └── websocket-server.cpp      # WebSocket API server
├── build/                        # Build output directory
├── CMakeLists.txt                # CMake configuration
├── build-direct.ps1              # Build script
├── install-plugin.ps1            # Installation script
├── generate-libs.ps1             # Import library generator
├── test_websocket.py             # WebSocket test script
└── README.md                     # This file
```

## Water Polo Rules Implemented

- **Shot Clock**: 30 seconds (resets on possession change)
- **Game Clock**: 8 minutes per period (stops during timeouts/exclusions)
- **Periods**: 4 periods per game
- **Exclusions**: Major fouls (20-second player exclusion)
- **Timeouts**: 3 per team per game

## Troubleshooting

### Plugin doesn't appear in OBS
- Ensure OBS Studio version is compatible (32.0.1 tested)
- Check that Qt DLLs are available (MSVC 2022 64-bit runtime)
- Look for errors in OBS logs: `%APPDATA%\obs-studio\logs`

### Build errors
- Verify Visual Studio 2022 with C++ tools installed
- Check Qt installation has MSVC 2022 64-bit component
- Ensure OBS source code downloaded to correct path

### WebSocket connection fails
- Check port 8765 is not blocked by firewall
- Verify plugin is loaded (check Tools menu for control panel)
- Try connecting with: `Test-NetConnection localhost -Port 8765`

## Development

### Code Structure

**plugin-main.cpp**: Module registration, initialization
- `obs_module_load()`: Registers scoreboard source
- `obs_module_unload()`: Cleanup

**scoreboard-source.cpp**: OBS source implementation
- Renders scoreboard texture
- Manages scoreboard data
- Provides properties UI
- Global `g_active_scoreboard` pointer for external updates

**control-panel.cpp**: Qt GUI control panel
- QWidget-based interface
- QTimer for clock countdown
- Calls `update_scoreboard_data()` to modify source

**websocket-server.cpp**: WebSocket API
- Qt WebSocket server on port 8765
- JSON message parsing
- Calls `update_scoreboard_data()` to modify source

### Key Functions

- `update_scoreboard_data(obs_data_t* data)`: Updates the active scoreboard
- `register_scoreboard_source()`: Registers source with OBS
- `init_websocket_server()`: Starts WebSocket server
- `init_control_panel()`: Creates GUI control panel

## License

This plugin is provided as-is for water polo streaming and production use.

## Credits

Built for water polo game streaming with OBS Studio integration.
