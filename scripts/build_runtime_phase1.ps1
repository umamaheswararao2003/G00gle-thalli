# Phase 1 Runtime Build Script
# This script builds the GoogleRuntime library and test

param(
    [string]$LLVMDir = "C:\Users\Asus\Desktop\llvm-project\build\lib\cmake\llvm",
    [string]$MLIRDir = "C:\Users\Asus\Desktop\llvm-project\build\lib\cmake\mlir",
    [switch]$Clean
)

Write-Host "=== Google MLIR Runtime - Phase 1 Build ===" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = $PSScriptRoot

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build directory..." -ForegroundColor Yellow
    if (Test-Path "$ProjectRoot\build") {
        Remove-Item -Recurse -Force "$ProjectRoot\build"
    }
}

# Configure CMake
Write-Host "Configuring CMake..." -ForegroundColor Green
cmake -B "$ProjectRoot\build" `
    -DCMAKE_BUILD_TYPE=Release `
    -DLLVM_DIR="$LLVMDir" `
    -DMLIR_DIR="$MLIRDir"

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# Build GoogleRuntime library
Write-Host ""
Write-Host "Building GoogleRuntime library..." -ForegroundColor Green
cmake --build "$ProjectRoot\build" --target GoogleRuntime

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "GoogleRuntime library built successfully!" -ForegroundColor Green
Write-Host "Location: build\lib\GoogleRuntime.lib" -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Compile MLIR kernel: google-opt test\test_matmul_l3_tiling.mlir --google-extreme-l3-full -o test\matmul.ll"
Write-Host "  2. Build test executable (after linking with compiled kernel)"
Write-Host "  3. Run: build\bin\test_runtime_phase1.exe"
