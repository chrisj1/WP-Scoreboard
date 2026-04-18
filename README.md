# WP Scoreboard

A real-time water polo scoreboard system for OBS Studio live streaming. Includes a control GUI, WebSocket API, StreamDock button panel integration, iOS remote control, and optional CNN-based automatic clock detection.

---

## Components

| Component | Description |
|---|---|
| `obs-scoreboard/` | OBS Studio plugin — renders the scoreboard overlay and exposes a WebSocket API |
| `vsd-scoreboard-plugin/` | StreamDock (VSDinside) plugin — physical button panel control |
| `ios-controller/` | iOS Swift app — wireless remote control |

---

## Quick Start (macOS)

### 1. Build and install the OBS plugin

```bash
cd obs-scoreboard

# Prerequisites (first time only)
brew install cmake ninja simde
pip3 install aqtinstall
python3 -m aqt install-qt mac desktop 6.8.3 clang_64 --outputdir ~/Qt6.8.3 --modules qtwebsockets

# Download OBS 32.0.1 source (first time only)
curl -L https://github.com/obsproject/obs-studio/archive/refs/tags/32.0.1.tar.gz -o /tmp/obs.tar.gz
tar -xzf /tmp/obs.tar.gz -C ~/Downloads

# Build
./build-mac.sh

# Install
sudo cp -R build-mac/obs-scoreboard.plugin /Applications/OBS.app/Contents/PlugIns/
```

Restart OBS after installing.

### 2. Build and install the StreamDock plugin

```bash
cd vsd-scoreboard-plugin
./build.sh
# Restart StreamDock after installation
```

### 3. Set up OBS scenes

In OBS:
1. Add a **"Water Polo Scoreboard"** source — main scoreboard overlay
2. Add a **"Schedule"** source — rotating display of today's games
3. Add a **"Roster"** source — team player cards (optional)
4. Add an **"Instant Replay Camera"** source — slow-motion replay playback
   - First attach a **"Replay Buffer"** filter to your camera source
   - In the replay source properties, select that camera

### 4. Configure teams and schedule

Edit `teams.csv` and `schedule.csv`, then point the Schedule source to the folder containing them.

---

## OBS Plugin Features

- **Scoreboard overlay** — team names, scores, period, game clock, shot clock, exclusions, timeouts, man-up indicators, logos and colors
- **Schedule display** — rotating game schedule with team branding
- **Roster display** — player name cards with team colors
- **Instant replay** — frame-accurate slow-motion playback from a rolling camera buffer
- **GUI control panel** — Tools → Water Polo Scoreboard Control
- **WebSocket API** on port `8766`
- **CNN clock detection** (optional) — reads clock digits from camera feed using a trained PyTorch model

---

## WebSocket API

The plugin listens on `ws://localhost:8766`. All messages are JSON.

> **Note:** Port `4455` is the separate OBS WebSocket plugin — do not confuse them.

### Get current state

```json
{ "type": "get_state" }
```

Response:
```json
{
  "type": "state",
  "data": {
    "home_score": 5, "away_score": 3,
    "period": 2, "period_text": "",
    "game_clock_minutes": 6, "game_clock_seconds": 42,
    "shot_clock": 18,
    "home_team": "Cornell", "away_team": "Rochester",
    "home_exclusions": 1, "away_exclusions": 0,
    "home_timeouts": 2, "away_timeouts": 3,
    "home_manup": false, "away_manup": false
  }
}
```

### Update scoreboard

```json
{
  "type": "update",
  "data": { "home_score": 6, "period": 3, "shot_clock": 30 }
}
```

Only include fields you want to change — unset fields are preserved.

### Other messages

| `type` | Description |
|---|---|
| `next_game` / `prev_game` | Advance or go back in the schedule |
| `trigger_replay` | Snapshot camera buffer; switch to replay scene to play |
| `get_schedule` | Returns full schedule as JSON |
| `get_rois` | Returns clock OCR region coordinates |
| `set_settings` | Apply global plugin settings |
| `ping` | Responds with `pong` |

---

## Configuration

### `teams.csv`

```csv
name,home_bg,home_text,away_bg,away_text
Cornell,#B31B1B,#FFFFFF,#D46060,#000000
Rochester,#003087,#FFFFFF,#4070B0,#FFFFFF
```

### `schedule.csv`

```csv
start_time,home,away
2026-04-13 10:00,Cornell,Rochester
2026-04-13 12:00,RPI,Colgate
```

`start_time` format: `YYYY-MM-DD HH:MM` (24-hour). Team names must match `teams.csv`.

### Team logos

Place SVG or PNG files in your config `logos/` folder, named as the team name in lowercase with no spaces: `Cornell` → `cornell.svg`, `Coast Guard` → `coastguard.svg`.

---

## CNN Clock Detection (Optional)

Automatically reads the shot clock and game clock from camera, eliminating manual clock entry.

### Prerequisites

```bash
brew install opencv
# Download LibTorch for Apple Silicon from https://pytorch.org
# (Select: Stable, macOS, LibTorch, C++/Java, Default)
# Extract to ~/Downloads/libtorch
```

### Build with CNN support

```bash
export LIBTORCH_PATH=~/Downloads/libtorch
./build-mac.sh
# Confirm "CNN clock detection enabled" in cmake output
```

### Setup in OBS

1. Open **Tools → Water Polo Scoreboard Control → CNN Clock** tab
2. Select the camera source for shot clock and game clock
3. Click **Select ROI** and draw a box around the clock digits
4. Click **Start Detection**

### Training your own models

```bash
python label_dataset.py              # label shot clock frames
python label_dataset_game_clock.py   # label game clock frames
python train_shot_clock_cnn.py
python train_game_clock_cnn.py
```

Pre-trained models: `shot_clock_model.pt`, `game_clock_model.pt`

---

## StreamDock Plugin

Connects to `ws://localhost:8766` automatically on startup. Provides buttons for:

| Category | Actions |
|---|---|
| Score | Home/Away ±1 |
| Period | Next, Previous, Set Final, Set Shootout |
| Shot clock | Reset to 30s, Reset to 35s |
| Exclusions | Add / Clear (per team) |
| Timeouts | Use / Restore (per team) |
| Man-up | Toggle (per team) |
| Game | Next game, Previous game, Reset game |
| Replay | Trigger instant replay |
| Info displays | Live score, clock, team names |

---

## Project Structure

```
WP-Scoreboard/
├── obs-scoreboard/
│   ├── src/
│   │   ├── plugin-main.cpp               # Plugin entry point
│   │   ├── scoreboard-source.cpp         # Scoreboard overlay rendering
│   │   ├── schedule-source.cpp           # Game schedule display
│   │   ├── roster-source.cpp             # Team roster display
│   │   ├── control-panel.cpp             # Qt GUI control panel
│   │   ├── websocket-server.cpp          # WebSocket API (port 8766)
│   │   ├── replay-source.cpp             # Instant replay playback source
│   │   ├── replay-filter.cpp             # Rolling frame buffer filter
│   │   ├── replay-shared.h               # Shared replay types
│   │   ├── clock-ocr-engine.cpp/h        # CNN clock detection (optional)
│   │   ├── roi-selector-widget.cpp/h     # ROI selector UI (optional)
│   │   ├── histogram-viz-source.cpp/h    # Debug: histogram overlay
│   │   ├── averaged-frame-viz-source.cpp/h  # Debug: averaged frame overlay
│   │   ├── scoreboard-source.h           # Scoreboard state struct (shared)
│   │   └── shared-schedule.h             # Schedule types (shared)
│   ├── config/                           # Default logos and team data
│   ├── CMakeLists.txt
│   └── build-mac.sh
│
├── vsd-scoreboard-plugin/
│   ├── src/
│   │   ├── scoreboard_bridge.py          # Singleton WebSocket bridge
│   │   ├── core/
│   │   │   ├── action.py                 # Base VSDinside action class
│   │   │   ├── action_factory.py         # Auto-discovers and registers actions
│   │   │   ├── plugin.py                 # VSDinside SDK connection
│   │   │   └── timer.py
│   │   └── actions/
│   │       ├── base_action.py            # ScoreboardAction base + button rendering
│   │       ├── score_actions.py
│   │       ├── period_actions.py
│   │       ├── exclusion_actions.py
│   │       ├── timeout_actions.py
│   │       ├── manup_actions.py
│   │       ├── game_actions.py
│   │       ├── info_actions.py
│   │       └── replay_actions.py
│   ├── main.py
│   ├── main.spec                         # PyInstaller bundle configuration
│   ├── build.sh
│   └── generate_icons.py
│
├── ios-controller/                       # iOS Swift remote control app
├── teams.csv                             # Active team configuration
├── schedule.csv                          # Active game schedule
├── shot_clock_model.pt                   # Pre-trained shot clock CNN
├── game_clock_model.pt                   # Pre-trained game clock CNN
└── README.md
```

---

## Troubleshooting

**Plugin not visible in OBS**
Copy to `OBS.app/Contents/PlugIns/` (capital I). Check OBS logs: Help → Log Files → View Current Log, search for `obs-scoreboard`.

**StreamDock buttons not responding**
The OBS plugin must be running first. Check `~/Library/Application Support/HotSpot/StreamDock/plugins/com.wps.scoreboard.sdPlugin/plugin.log`.

**Replay crashes OBS**
The "Replay Buffer" filter must be attached to the camera source before clicking the replay button. Restart OBS after adding the filter if issues persist.

**CNN detection disabled**
Both OpenCV and LibTorch are required. Run cmake and look for "CNN clock detection enabled". Ensure `LIBTORCH_PATH` is exported before running `./build-mac.sh`.

**Next game area is blank**
Expected — the area hides automatically when no next game is in the schedule.

**Scores revert after pressing StreamDock buttons**
Restart OBS to load the latest plugin build. The fix requires the new plugin to be running.
