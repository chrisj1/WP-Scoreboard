#!/usr/bin/env zsh
# Mac build script for obs-scoreboard plugin
# Replaces build-direct.ps1 (Windows PowerShell script)
#
# Prerequisites:
#   brew install cmake ninja qt simde
#   Download OBS source: https://github.com/obsproject/obs-studio/archive/refs/tags/32.0.1.tar.gz
#   Extract to ~/Downloads/obs-studio-32.0.1/
#
# Optional (for CNN clock detection):
#   brew install opencv
#   Download LibTorch for macOS from https://pytorch.org (select: PyTorch, macOS, ARM)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-mac"
OBS_APP_DIR="/Applications/OBS.app/Contents"
OBS_SOURCE_DIR="${HOME}/Downloads/obs-studio-32.0.1"

# Qt 6.8.3 — must match the Qt version bundled with OBS.app.
# Installed via: python3 -m aqt install-qt mac desktop 6.8.3 clang_64 --outputdir ~/Qt6.8.3 --modules qtwebsockets
QT_PREFIX="${HOME}/Qt6.8.3/6.8.3/macos"
QT6_CMAKE_DIR="${QT_PREFIX}/lib/cmake/Qt6"
if [[ -d "${QT6_CMAKE_DIR}" ]]; then
    echo "[build] Using Qt 6.8.3 at: ${QT_PREFIX}"
else
    echo "ERROR: Qt 6.8.3 not found at ${QT_PREFIX}"
    echo "Install with:"
    echo "  pip3 install aqtinstall"
    echo "  python3 -m aqt install-qt mac desktop 6.8.3 clang_64 --outputdir ~/Qt6.8.3 --modules qtwebsockets"
    exit 1
fi

# LibTorch (optional) — default to ~/Downloads/libtorch if not set
LIBTORCH_PATH="${LIBTORCH_PATH:-${HOME}/Downloads/libtorch}"
if [[ -f "${LIBTORCH_PATH}/share/cmake/Torch/TorchConfig.cmake" ]]; then
    export LIBTORCH_PATH
    echo "[build] LibTorch found at: ${LIBTORCH_PATH}"
else
    echo "[build] LibTorch not found at ${LIBTORCH_PATH} - CNN clock detection will be disabled"
    LIBTORCH_PATH=""
fi

# Verify OBS source exists
if [[ ! -d "${OBS_SOURCE_DIR}" ]]; then
    echo ""
    echo "ERROR: OBS source not found at: ${OBS_SOURCE_DIR}"
    echo ""
    echo "Please download it:"
    echo "  curl -L https://github.com/obsproject/obs-studio/archive/refs/tags/32.0.1.tar.gz -o /tmp/obs.tar.gz"
    echo "  tar -xzf /tmp/obs.tar.gz -C ~/Downloads"
    echo ""
    exit 1
fi

# Verify cmake is installed
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install with: brew install cmake"
    exit 1
fi

echo ""
echo "========================================"
echo " Building obs-scoreboard (macOS)"
echo "========================================"
echo " OBS App:    ${OBS_APP_DIR}"
echo " OBS Source: ${OBS_SOURCE_DIR}"
echo " Build dir:  ${BUILD_DIR}"
echo "========================================"
echo ""

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Always reconfigure from a clean cache (avoids stale Qt/OBS paths when switching setups)
rm -f "${BUILD_DIR}/CMakeCache.txt"
rm -rf "${BUILD_DIR}/CMakeFiles"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DOBS_APP_DIR="${OBS_APP_DIR}" \
    -DOBS_SOURCE_DIR="${OBS_SOURCE_DIR}" \
    -DQt6_DIR="${QT6_CMAKE_DIR}" \
    -DCMAKE_PREFIX_PATH="${QT_PREFIX}" \
    -DLIBTORCH_PATH="${LIBTORCH_PATH}" \
    -G Ninja

cmake --build . --config RelWithDebInfo

echo ""
echo "========================================"
echo " Build complete! Packaging plugin..."
echo "========================================"

# OBS on macOS loads plugins as .plugin bundles from OBS.app/Contents/PlugIns/
# The binary inside the bundle must match CFBundleExecutable (no extension, no lib prefix)
BUILT_SO="${BUILD_DIR}/libobs-scoreboard.so"
BUNDLE_DIR="${BUILD_DIR}/obs-scoreboard.plugin"
BUNDLE_MACOS="${BUNDLE_DIR}/Contents/MacOS"

mkdir -p "${BUNDLE_MACOS}"

# Copy binary into bundle (rename: strip lib prefix and .so extension)
cp "${BUILT_SO}" "${BUNDLE_MACOS}/obs-scoreboard"

# ── Remap Qt framework links to OBS's bundled Qt ──────────────────────────────
# OBS bundles its own Qt (currently 6.8.x). If we compiled against a newer
# Homebrew Qt the dylib references will point to the wrong version, causing a
# "Symbol not found" crash when OBS loads the plugin.  Use install_name_tool to
# redirect each framework path to the copy inside OBS.app.
echo "[build] Remapping Qt frameworks to OBS's bundled Qt..."
OBS_FW="/Applications/OBS.app/Contents/Frameworks"
PLUGIN_BIN="${BUNDLE_MACOS}/obs-scoreboard"
for qt in QtCore QtGui QtNetwork QtWidgets QtSvg; do
    current=$(otool -L "${PLUGIN_BIN}" 2>/dev/null | grep "${qt}.framework" | awk '{print $1}')
    if [[ -n "${current}" ]]; then
        new="${OBS_FW}/${qt}.framework/Versions/A/${qt}"
        install_name_tool -change "${current}" "${new}" "${PLUGIN_BIN}" 2>/dev/null || true
        echo "  ${qt}: remapped → OBS bundle"
    fi
done

# Write Info.plist
cat > "${BUNDLE_DIR}/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>obs-scoreboard</string>
    <key>CFBundleExecutable</key>
    <string>obs-scoreboard</string>
    <key>CFBundleIdentifier</key>
    <string>com.obsproject.obs-scoreboard</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>obs-scoreboard</string>
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>12.0</string>
</dict>
</plist>
PLIST

# Re-sign with ad-hoc signature (install_name_tool invalidates the original signature,
# and macOS will SIGKILL the process if it loads a binary with an invalid signature).
echo "[build] Re-signing plugin bundle (ad-hoc)..."
xattr -cr "${BUNDLE_DIR}"
codesign --force --deep --sign - "${BUNDLE_DIR}"
echo "[build] Plugin signed OK"

echo "[build] Plugin bundle: ${BUNDLE_DIR}"
echo ""
echo "========================================"
echo " Installing to OBS..."
echo "========================================"

INSTALL_DIR="${OBS_APP_DIR}/PlugIns"

echo ""
echo "========================================"
echo " Install Instructions"
echo "========================================"
echo ""
echo "macOS requires authentication to modify app bundles."
echo "Run this command in your terminal:"
echo ""
echo "  sudo cp -R \"${BUNDLE_DIR}\" \"${INSTALL_DIR}/\""
echo ""
echo "Then restart OBS."
echo ""
echo "(Or open Finder → Go → Applications → right-click OBS → Show Package Contents"
echo " → Contents/PlugIns → drag obs-scoreboard.plugin from:"
echo " ${BUNDLE_DIR})"
