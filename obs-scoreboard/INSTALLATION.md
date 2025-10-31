# Water Polo Scoreboard - Installation Guide

## Installation Steps

1. **Close OBS Studio** completely before installation

2. **Run the build script** to build and install the plugin:
   ```powershell
   .\build-direct.ps1
   ```
   When prompted, type `y` to install the plugin to OBS.

3. **Open OBS Studio**

## Configuration

### Setting Up the Config Directory

The plugin requires a config directory containing:
- `teams.csv` - Team information (name, colors)
- `schedule.csv` - Game schedule
- `logos/` folder - Team logo images (.svg or .png files)

**To configure the path:**

1. In OBS, add a new **Schedule Source** to your scene
2. In the source properties, find **"Config Directory"**
3. Click **Browse** and select your folder containing teams.csv and schedule.csv
4. Click **OK** to save

### File Structure

Your config directory should look like this:
```
C:\YourConfigFolder\
├── teams.csv
├── schedule.csv
└── logos\
    ├── rpi.svg (or .png)
    ├── syracuse.svg (or .png)
    └── ... (other team logos)
```

**Note:** Logo filenames should match team names from teams.csv (lowercase, no spaces).
- Example: "RPI" → `rpi.svg` or `rpi.png`
- Example: "Coast Guard" → `coastguard.svg` or `coastguard.png`

### teams.csv Format

```csv
name,home_bg,home_text,away_bg,away_text
RPI,#800000,#FFFFFF,#FF6666,#000000
Syracuse,#F76900,#FFFFFF,#FFB366,#000000
```

Columns:
- `name` - Team name
- `home_bg` - Home background color (hex)
- `home_text` - Home text color (hex)
- `away_bg` - Away background color (hex)
- `away_text` - Away text color (hex)

### schedule.csv Format

```csv
start_time,home,away
2025-09-20 13:00,RPI,Syracuse
2025-09-20 15:00,Cornell,Columbia
```

Columns:
- `start_time` - Game start time (YYYY-MM-DD HH:MM format, 24-hour)
- `home` - Home team name (must match teams.csv)
- `away` - Away team name (must match teams.csv)

## Troubleshooting

### Logos Not Showing
- Check that logos are in a `logos/` subfolder within your config directory
- Verify filenames match team names (lowercase, no spaces)
- Ensure files are .svg or .png format

### No Schedule Data
- Verify the config directory path is set in source properties
- Check that teams.csv and schedule.csv exist in the directory
- Review OBS logs for any error messages

### Changes Not Appearing
- Right-click the schedule source and select **Properties**
- Make sure to select which dates to display using the checkboxes
- Adjust rotation time if needed (how long each date is shown)

## Customization

In the source properties, you can customize:
- **Selected Dates** - Check which dates to display
- **Rotation Time** - Seconds to show each date (2-30 seconds)
- **Background Color** - Overall background color
- **Text Color** - General text color
- **Accent Color** - Accent elements color
- **Font Size** - Size of text (12-48)

## For Developers

See `README.md` for build requirements and development setup.
