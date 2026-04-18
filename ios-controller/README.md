# WP Scoreboard iOS Controller

Live scoreboard controller for iPhone/iPad. Connects to the OBS plugin over Wi-Fi.

## Setup

1. Open Xcode → **File → New → Project** → iOS App
2. Name: `WPScoreboard`, Interface: SwiftUI, Language: Swift
3. Delete the generated `ContentView.swift`
4. Drag all `.swift` files from this folder into the project
5. Replace `Info.plist` with the one in this folder (or merge the keys manually)
6. Run on device or simulator

## Usage

1. Make sure the OBS scoreboard plugin is running (WebSocket on port 8766)
2. Open the app, enter the IP address of the OBS computer
3. The app connects and starts showing live state at 4 updates/second

## Tabs

| Tab | Controls |
|-----|----------|
| **Score** | Home/Away +1/−1 with team colors |
| **Clocks** | Period advance/back, shot clock reset, game clock display |
| **Extras** | Timeouts, exclusions, man-up toggles |
| **Game** | Next/prev game in schedule, reset game |

## Requirements

- iOS 16+
- OBS computer and iPhone/iPad on the same Wi-Fi network
- OBS scoreboard plugin running
