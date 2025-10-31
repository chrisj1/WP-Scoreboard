# Build script that directly sets compiler paths
# Bypasses the broken VsDevCmd.bat

param(
    [string]$QtPath = "C:\Qt\6.8.3\msvc2022_64",
    [string]$ObsSourcePath = "$env:USERPROFILE\Downloads\obs-studio-32.0.1",
    [string]$LibTorchPath = "C:\libtorch-cpu",
    [switch]$Clean
)

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Water Polo Scoreboard - Direct Build" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Find MSVC compiler
$vs2022Path = "C:\Program Files\Microsoft Visual Studio\2022\Community"
$msvcBasePath = "$vs2022Path\VC\Tools\MSVC"

if (Test-Path $msvcBasePath) {
    $msvcVersion = (Get-ChildItem $msvcBasePath | Sort-Object Name -Descending | Select-Object -First 1).Name
    $msvcPath = "$msvcBasePath\$msvcVersion"
    Write-Host "✅ Found MSVC: $msvcVersion" -ForegroundColor Green
} else {
    Write-Host "❌ MSVC not found" -ForegroundColor Red
    exit 1
}

# Find Windows SDK
$sdkPath = "C:\Program Files (x86)\Windows Kits\10"
if (Test-Path $sdkPath) {
    $sdkVersion = (Get-ChildItem "$sdkPath\Include" | Sort-Object Name -Descending | Select-Object -First 1).Name
    Write-Host "✅ Found Windows SDK: $sdkVersion" -ForegroundColor Green
} else {
    Write-Host "⚠️  Windows SDK not found" -ForegroundColor Yellow
}

# Check LibTorch
if (Test-Path $LibTorchPath) {
    Write-Host "✅ Found LibTorch: $LibTorchPath" -ForegroundColor Green
} else {
    Write-Host "⚠️  LibTorch not found at: $LibTorchPath" -ForegroundColor Yellow
    Write-Host "   CNN clock detection will be disabled" -ForegroundColor Yellow
}

# Setup build directory (only clean if forced)
if ($Clean -and (Test-Path "build")) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "build"
}

if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

Write-Host ""
Write-Host "Checking import libraries..." -ForegroundColor Yellow

# Generate .lib files from DLLs (only if they don't exist)
$dumpbin = "$msvcPath\bin\Hostx64\x64\dumpbin.exe"
$lib = "$msvcPath\bin\Hostx64\x64\lib.exe"
$obsBinDir = "C:\Program Files\obs-studio\bin\64bit"

function Generate-LibFromDll {
    param($dllName)
    
    $dllPath = "$obsBinDir\$dllName"
    $defFile = ".\build\$($dllName -replace '\.dll$', '.def')"
    $libFile = ".\build\$($dllName -replace '\.dll$', '.lib')"
    
    # Skip if .lib already exists
    if (Test-Path $libFile) {
        Write-Host "  ✅ $($dllName -replace '\.dll$', '.lib') (cached)" -ForegroundColor Green
        return $true
    }
    
    # Export symbols from DLL
    $exports = & $dumpbin /EXPORTS $dllPath | Select-String "^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)"
    
    if ($exports.Count -eq 0) {
        Write-Host "  ❌ No exports found in $dllName!" -ForegroundColor Red
        return $false
    }
    
    # Create .def file
    $defContent = @"
LIBRARY $dllName
EXPORTS
"@
    
    foreach ($match in $exports) {
        $symbolName = $match.Matches.Groups[1].Value
        $defContent += "`n    $symbolName"
    }
    
    Set-Content -Path $defFile -Value $defContent
    
    # Generate .lib file
    & $lib /DEF:$defFile /OUT:$libFile /MACHINE:X64 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✅ Generated $($dllName -replace '\.dll$', '.lib')" -ForegroundColor Green
        return $true
    } else {
        Write-Host "  ❌ Failed to generate .lib for $dllName" -ForegroundColor Red
        return $false
    }
}

# Generate libs for obs.dll and obs-frontend-api.dll
Generate-LibFromDll "obs.dll" | Out-Null
Generate-LibFromDll "obs-frontend-api.dll" | Out-Null

Write-Host ""
Write-Host "Configuring with CMake..." -ForegroundColor Yellow

# Set environment for CMake (use forward slashes for CMake)
$env:CC = ($msvcPath + "\bin\Hostx64\x64\cl.exe").Replace('\', '/')
$env:CXX = ($msvcPath + "\bin\Hostx64\x64\cl.exe").Replace('\', '/')
$compilerPath = ($msvcPath + "\bin\Hostx64\x64").Replace('\', '/')
$sdkBinPath = ($sdkPath + "\bin\$sdkVersion\x64").Replace('\', '/')
$env:PATH = "$msvcPath\bin\Hostx64\x64;$sdkPath\bin\$sdkVersion\x64;$env:PATH"
$env:INCLUDE = "$msvcPath\include;$sdkPath\Include\$sdkVersion\ucrt;$sdkPath\Include\$sdkVersion\um;$sdkPath\Include\$sdkVersion\shared"
$env:LIB = "$msvcPath\lib\x64;$sdkPath\Lib\$sdkVersion\ucrt\x64;$sdkPath\Lib\$sdkVersion\um\x64"

Set-Location build

# Run CMake (skip if CMakeCache.txt exists and not cleaning)
if (-not (Test-Path "CMakeCache.txt") -or $Clean) {
    Write-Host ""
    Write-Host "Configuring with CMake (using Ninja for faster builds)..." -ForegroundColor Yellow
    
    cmake .. `
        -G "Ninja" `
        -DCMAKE_BUILD_TYPE=Release `
        -DCMAKE_PREFIX_PATH="$QtPath;$ObsSourcePath;$LibTorchPath" `
        -DTorch_DIR="$LibTorchPath\share\cmake\Torch" `
        -DQT_VERSION=6 `
        -DCMAKE_C_COMPILER="$($env:CC)" `
        -DCMAKE_CXX_COMPILER="$($env:CXX)"

    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "❌ CMake configuration failed!" -ForegroundColor Red
        Set-Location ..
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "Using cached CMake configuration (use -Clean to reconfigure)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Building with Ninja (parallel build)..." -ForegroundColor Yellow

# Use parallel build with /MP flag (set in environment)
$env:CL = "/MP"
ninja

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host ""
    
    if (Test-Path "obs-scoreboard.dll") {
        Write-Host "Plugin built: $PWD\obs-scoreboard.dll" -ForegroundColor White
        
        # Try to install
        $obsPluginPath = "C:\Program Files\obs-studio\obs-plugins\64bit"
        if (Test-Path $obsPluginPath) {
            Write-Host ""
            $response = Read-Host "Install plugin to OBS? (y/n)"
            if ($response -eq "y") {
                try {
                    Copy-Item "obs-scoreboard.dll" -Destination "$obsPluginPath\obs-scoreboard.dll" -Force
                    Write-Host "✅ Plugin installed to OBS!" -ForegroundColor Green
                    Write-Host ""
                } catch {
                    Write-Host "❌ Failed to copy. Make sure OBS is closed." -ForegroundColor Red
                }
            }
        }
    }
} else {
    Write-Host ""
    Write-Host "❌ Build failed!" -ForegroundColor Red
}