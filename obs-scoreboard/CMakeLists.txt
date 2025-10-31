cmake_minimum_required(VERSION 3.16...3.25)

project(obs-scoreboard VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OBS paths
set(OBS_INSTALL_DIR "C:/Program Files/obs-studio" CACHE PATH "OBS installation directory")
set(OBS_SOURCE_DIR "$ENV{USERPROFILE}/Downloads/obs-studio-32.0.1" CACHE PATH "OBS source directory")

# Use OBS's Qt installation to ensure version compatibility
set(Qt6_DIR "${OBS_INSTALL_DIR}/bin/64bit" CACHE PATH "Qt6 directory from OBS")
set(CMAKE_PREFIX_PATH "${OBS_INSTALL_DIR}/bin/64bit" CACHE PATH "Search path for Qt6")

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
endif()

# Find Qt6
find_package(Qt6 REQUIRED COMPONENTS Core Widgets WebSockets)

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
else()
    message(WARNING "LibTorch not found - CNN clock detection will be disabled")
    set(USE_LIBTORCH FALSE)
endif()

# Enable Qt MOC (Meta-Object Compiler) for Q_OBJECT classes
set(CMAKE_AUTOMOC ON)

# Plugin sources
set(obs-scoreboard_SOURCES
    src/plugin-main.cpp
    src/scoreboard-source-simple.cpp
    src/schedule-source.cpp
    src/roster-source.cpp
    src/control-panel.cpp
    src/websocket-server.cpp
)

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

target_link_libraries(obs-scoreboard
    OBS::libobs
    OBS::obs-frontend-api
    "C:/Users/Chris Jerrett/Desktop/stream-deck-stuff/obs-scoreboard/build/obs.lib"
    "C:/Users/Chris Jerrett/Desktop/stream-deck-stuff/obs-scoreboard/build/obs-frontend-api.lib"
    Qt6::Core
    Qt6::Widgets
    Qt6::WebSockets
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
    VERSION ${PROJECT_VERSION}
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
