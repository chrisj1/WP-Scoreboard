# obs-scoreboard

OBS Studio plugin for live streaming water polo. See the [root README](../README.md) for full documentation.

## Build (macOS)

```bash
# Optional: enable CNN clock detection
export LIBTORCH_PATH=~/Downloads/libtorch

./build-mac.sh

sudo cp -R build-mac/obs-scoreboard.plugin /Applications/OBS.app/Contents/PlugIns/
```

## Sources registered by this plugin

| Source ID | Display Name | Purpose |
|---|---|---|
| `water_polo_scoreboard` | Water Polo Scoreboard | Main scoreboard overlay |
| `water_polo_schedule` | Schedule | Today's game schedule display |
| `water_polo_roster` | Roster | Team player cards |
| `instant_replay_camera` | Instant Replay Camera | Slow-motion replay playback |

### Filters

| Filter ID | Display Name | Attach to |
|---|---|---|
| `replay_camera_filter` | Replay Buffer | Any camera/video source |

## WebSocket API — port 8766

See [root README → WebSocket API](../README.md#websocket-api) for message reference.

## CNN clock detection

Requires OpenCV and LibTorch. Pass `LIBTORCH_PATH` to the build script. If dependencies are missing, the build succeeds without CNN support.

See [root README → CNN Clock Detection](../README.md#cnn-clock-detection-optional).
