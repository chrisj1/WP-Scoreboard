# Download and extract CPU-only LibTorch 2.5.1
# This script downloads the correct CPU-only version without CUDA dependencies

$downloadUrl = "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.5.1%2Bcpu.zip"
$zipFile = "C:\libtorch-cpu-2.5.1.zip"
$extractPath = "C:\"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LibTorch CPU-only Downloader" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if already exists
if (Test-Path "C:\libtorch-cpu-2.5.1") {
    Write-Host "⚠️  LibTorch CPU already exists at C:\libtorch-cpu-2.5.1" -ForegroundColor Yellow
    $response = Read-Host "Do you want to re-download? (y/n)"
    if ($response -ne "y") {
        Write-Host "✅ Using existing LibTorch installation" -ForegroundColor Green
        exit 0
    }
    Remove-Item "C:\libtorch-cpu-2.5.1" -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host "📥 Downloading LibTorch 2.5.1 CPU-only (this may take several minutes)..." -ForegroundColor Cyan
Write-Host "   URL: $downloadUrl" -ForegroundColor Gray

try {
    # Download with progress
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $downloadUrl -OutFile $zipFile -UseBasicParsing
    $ProgressPreference = 'Continue'
    
    Write-Host "✅ Download complete!" -ForegroundColor Green
    Write-Host ""
    
    # Get file size
    $fileSize = (Get-Item $zipFile).Length / 1MB
    Write-Host "📦 Downloaded file size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Cyan
    Write-Host ""
    
    Write-Host "📂 Extracting to $extractPath..." -ForegroundColor Cyan
    Expand-Archive -Path $zipFile -DestinationPath $extractPath -Force
    
    # Rename from 'libtorch' to 'libtorch-cpu-2.5.1'
    if (Test-Path "C:\libtorch") {
        if (Test-Path "C:\libtorch-cpu-2.5.1") {
            Remove-Item "C:\libtorch-cpu-2.5.1" -Recurse -Force
        }
        Rename-Item "C:\libtorch" "libtorch-cpu-2.5.1"
    }
    
    Write-Host "✅ Extraction complete!" -ForegroundColor Green
    Write-Host ""
    
    # Clean up zip file
    Remove-Item $zipFile -Force -ErrorAction SilentlyContinue
    Write-Host "🗑️  Cleaned up temporary files" -ForegroundColor Gray
    Write-Host ""
    
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✅ LibTorch CPU-only installed successfully!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Installation location: C:\libtorch-cpu-2.5.1" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Run: .\build-direct.ps1 -Clean" -ForegroundColor White
    Write-Host "2. The plugin will be rebuilt with CNN features enabled" -ForegroundColor White
    Write-Host ""
    
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
