cmake_minimum_required(VERSION 3.16...3.25)

project(obs-scoreboard VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OBS paths
if(WIN32)
    set(OBS_INSTALL_DIR "C:/Program Files/obs-studio" CACHE PATH "OBS installation directory")
    set(OBS_SOURCE_DIR "$ENV{USERPROFILE}/Downloads/obs-studio-32.0.1" CACHE PATH "OBS source directory")

    # Use OBS's Qt installation to ensure version compatibility
    set(Qt6_DIR "${OBS_INSTALL_DIR}/bin/64bit" CACHE PATH "Qt6 directory from OBS")
    set(CMAKE_PREFIX_PATH "${OBS_INSTALL_DIR}/bin/64bit" CACHE PATH "Search path for Qt6")
else()
    # On macOS/Linux these are typically provided by the build scripts/toolchain.
    set(OBS_SOURCE_DIR "" CACHE PATH "OBS source directory")
endif()

# Try to find OBS package first
find_package(libobs QUIET)
find_package(obs-frontend-api QUIET)

# If not found, set up manually
if (NOT libobs_FOUND)
    message(STATUS "Setting up OBS from source: ${OBS_SOURCE_DIR}")
    
    # Include directories from source
    set(OBS_INCLUDE_DIRS
        "${OBS_SOURCE_DIR}/libobs"
        "${OBS_SOURCE_DIR}/frontend/api"
    )
    
    # For Windows, we'll generate import libraries from the DLLs
    set(OBS_BIN_DIR "${OBS_INSTALL_DIR}/bin/64bit")
    
    # Create dummy imported targets (plugins don't need to link against OBS at build time on Windows)
    add_library(OBS::libobs INTERFACE IMPORTED)
    set_target_properties(OBS::libobs PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OBS_INCLUDE_DIRS}"
    )
    
    add_library(OBS::obs-frontend-api INTERFACE IMPORTED)
    set_target_properties(OBS::obs-frontend-api PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OBS_INCLUDE_DIRS}"
    )
    
    message(STATUS "OBS headers: ${OBS_INCLUDE_DIRS}")
    message(STATUS "OBS binaries: ${OBS_BIN_DIR}")

    # OBS headers include "obsconfig.h" which is generated at configure-time when building OBS.
    # For plugin-only builds, generate it from the OBS template so compilation succeeds.
    if(NOT EXISTS "${OBS_SOURCE_DIR}/libobs/obsconfig.h.in")
        message(FATAL_ERROR "OBS_SOURCE_DIR does not look like an OBS source tree (missing libobs/obsconfig.h.in): ${OBS_SOURCE_DIR}")
    endif()

    if(NOT DEFINED OBS_RELEASE_CANDIDATE)
        set(OBS_RELEASE_CANDIDATE 0)
    endif()
    if(NOT DEFINED OBS_BETA)
        set(OBS_BETA 0)
    endif()

    configure_file(
        "${OBS_SOURCE_DIR}/libobs/obsconfig.h.in"
        "${CMAKE_BINARY_DIR}/obsconfig.h"
        @ONLY
    )
endif()

# Find Qt6
# Note: QJson* lives in QtCore (there is no separate Qt6Json component).
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Network Svg)


# Find OpenCV
find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    set(USE_OPENCV TRUE)
else()
    message(WARNING "OpenCV not found - CNN clock detection will be disabled")
    set(USE_OPENCV FALSE)
endif()

# Find LibTorch (PyTorch C++)
set(CMAKE_PREFIX_PATH "$ENV{LIBTORCH_PATH}" "${CMAKE_PREFIX_PATH}")
find_package(Torch QUIET)
if(Torch_FOUND)
    message(STATUS "Found LibTorch: ${TORCH_VERSION}")
    set(USE_LIBTORCH TRUE)
    # Fix for missing NVTX (NVIDIA Tools Extension) - not critical for inference
    if(TARGET torch::nvtoolsext)
        set_target_properties(torch::nvtoolsext PROPERTIES INTERFACE_LINK_LIBRARIES "")
    endif()
    # AGL framework was removed in macOS 14 (Sonoma); strip it from LibTorch targets
    if(APPLE)
        foreach(_tgt torch torch_cpu c10)
            if(TARGET "${_tgt}")
                get_target_property(_iface_libs "${_tgt}" INTERFACE_LINK_LIBRARIES)
                if(_iface_libs)
                    list(REMOVE_ITEM _iface_libs "-framework AGL")
                    set_target_properties("${_tgt}" PROPERTIES INTERFACE_LINK_LIBRARIES "${_iface_libs}")
                endif()
            endif()
        endforeach()
    endif()
else()
    message(WARNING "LibTorch not found - CNN clock detection will be disabled")
    set(USE_LIBTORCH FALSE)
endif()

# Enable Qt MOC (Meta-Object Compiler) for Q_OBJECT classes
set(CMAKE_AUTOMOC ON)

# Plugin sources
set(obs-scoreboard_SOURCES
    src/plugin-main.cpp
    src/scoreboard-source.cpp
    src/schedule-source.cpp
    src/roster-source.cpp
    src/control-panel.cpp
    src/websocket-server.cpp
    src/replay-filter.cpp
    src/replay-source.cpp
)

# OBS headers on Apple Silicon depend on SIMDe for SSE emulation.
if(APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    find_path(SIMDE_INCLUDE_DIR
        NAMES simde/x86/sse2.h
        PATHS /opt/homebrew/opt/simde/include /usr/local/opt/simde/include
    )
    if(NOT SIMDE_INCLUDE_DIR)
        message(FATAL_ERROR "SIMDe headers not found (simde/x86/sse2.h). Install with: brew install simde")
    endif()
endif()

# Add CNN sources if dependencies are available
if(USE_OPENCV AND USE_LIBTORCH)
    list(APPEND obs-scoreboard_SOURCES
        src/clock-ocr-engine.cpp
        src/roi-selector-widget.cpp
        src/histogram-viz-source.cpp
        src/averaged-frame-viz-source.cpp
    )
    message(STATUS "CNN clock detection enabled")
else()
    message(STATUS "CNN clock detection disabled (missing OpenCV or LibTorch)")
endif()

# Create plugin
add_library(obs-scoreboard MODULE ${obs-scoreboard_SOURCES})

# Ensure generated OBS headers (e.g. obsconfig.h) are discoverable.
target_include_directories(obs-scoreboard SYSTEM PRIVATE "${CMAKE_BINARY_DIR}")

if(SIMDE_INCLUDE_DIR)
    target_include_directories(obs-scoreboard SYSTEM PRIVATE "${SIMDE_INCLUDE_DIR}")
endif()

# Qt6::Gui always adds -framework AGL even on macOS 14+ where AGL was removed.
# Create a stub dylib inside AGL.framework so ld can resolve the reference without error.
if(APPLE)
    set(_agl_fw "${CMAKE_BINARY_DIR}/AGL.framework")
    if(NOT EXISTS "${_agl_fw}/AGL")
        file(MAKE_DIRECTORY "${_agl_fw}")
        # Write a minimal C file — /dev/null is empty and rejected by clang
        set(_agl_stub_src "${CMAKE_BINARY_DIR}/agl_stub.c")
        file(WRITE "${_agl_stub_src}" "/* AGL stub: framework removed in macOS 14 */\n")
        execute_process(
            COMMAND ${CMAKE_C_COMPILER}
                    -arch ${CMAKE_OSX_ARCHITECTURES}
                    -dynamiclib
                    -install_name "/System/Library/Frameworks/AGL.framework/AGL"
                    -o "${_agl_fw}/AGL"
                    "${_agl_stub_src}"
            RESULT_VARIABLE _agl_result
        )
        if(NOT _agl_result EQUAL 0)
            message(WARNING "Could not create AGL stub dylib; linker may fail with 'framework AGL not found'")
        endif()
    endif()
    target_link_options(obs-scoreboard PRIVATE "-F${CMAKE_BINARY_DIR}")
endif()

target_link_libraries(obs-scoreboard
    OBS::libobs
    OBS::obs-frontend-api
    "${CMAKE_BINARY_DIR}/libobs.dylib"
    "${CMAKE_BINARY_DIR}/libobs-frontend-api.dylib"
    Qt6::Core
    Qt6::Widgets
    Qt6::Network
    Qt6::Svg
)


# Add CNN dependencies if available
if(USE_OPENCV AND USE_LIBTORCH)
    target_link_libraries(obs-scoreboard
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
    )
    target_include_directories(obs-scoreboard PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${TORCH_INCLUDE_DIRS}
    )
    target_compile_definitions(obs-scoreboard PRIVATE USE_CNN_OCR)
endif()

# Set plugin properties
set_target_properties(obs-scoreboard PROPERTIES
    FOLDER "plugins"
)

# Compile definitions
target_compile_definitions(obs-scoreboard PRIVATE
    PLUGIN_VERSION="${PROJECT_VERSION}"
)

# Installation
if(WIN32)
    install(TARGETS obs-scoreboard
        RUNTIME DESTINATION "obs-plugins/64bit"
        LIBRARY DESTINATION "obs-plugins/64bit"
    )
elseif(APPLE)
    install(TARGETS obs-scoreboard
        LIBRARY DESTINATION "$ENV{HOME}/Library/Application Support/obs-studio/plugins"
    )
else()
    install(TARGETS obs-scoreboard
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/obs-plugins"
    )
endif()
