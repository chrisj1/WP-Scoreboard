#!/usr/bin/env bash
# Build the WP Scoreboard VSDinside plugin with PyInstaller.
# Output: dist/main  (single executable, no console window)
#
# Usage:
#   chmod +x build.sh
#   ./build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==> Installing / verifying dependencies..."
pip install -r requirements.txt pyinstaller --quiet

echo "==> Running PyInstaller..."
pyinstaller --noconfirm main.spec

PLUGIN_NAME="com.wps.scoreboard.sdPlugin"
PLUGIN_DIR="$SCRIPT_DIR/$PLUGIN_NAME"
INSTALL_DIR="$HOME/Library/Application Support/HotSpot/StreamDock/plugins/$PLUGIN_NAME"

echo "==> Assembling plugin bundle: $PLUGIN_DIR/"
rm -rf "$PLUGIN_DIR"
mkdir -p "$PLUGIN_DIR/images/actions"

cp dist/main     "$PLUGIN_DIR/main"
cp manifest.json "$PLUGIN_DIR/manifest.json"
chmod +x         "$PLUGIN_DIR/main"

echo "==> Generating icons..."
python generate_icons.py

echo "==> Installing to StreamDock plugins folder..."
rm -rf "$INSTALL_DIR"
cp -R  "$PLUGIN_DIR" "$INSTALL_DIR"

echo ""
echo "==> Done. Restart StreamDock to load the plugin."
