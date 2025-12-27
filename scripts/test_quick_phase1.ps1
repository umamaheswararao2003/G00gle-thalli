# Quick Test Build Script for Phase 1
# Builds a standalone test without requiring full MLIR compilation

param(
    [string]$LLVMDir = "C:\Users\Asus\Desktop\llvm-project\build\lib\cmake\llvm",
    [string]$MLIRDir = "C:\Users\Asus\Desktop\llvm-project\build\lib\cmake\mlir"
)

Write-Host "=== Building Quick Tests for Phase 1 ===" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = Split-Path -Parent $PSScriptRoot

# Simple standalone build
Write-Host "Compiling GoogleRuntime.cpp..." -ForegroundColor Green
clang++ -std=c++17 `
    -I"$ProjectRoot\include" `
    -c "$ProjectRoot\lib\Google\Runtime\GoogleRuntime.cpp" `
    -o "$ProjectRoot\build\GoogleRuntime.o"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation of GoogleRuntime failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Compiling test_quick_phase1.cpp..." -ForegroundColor Green
clang++ -std=c++17 `
    -I"$ProjectRoot\include" `
    -c "$ProjectRoot\test\test_quick_phase1.cpp" `
    -o "$ProjectRoot\build\test_quick_phase1.o"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation of test failed!" -ForegroundColor Red
    exit 1
}

Write-Host "Linking test executable..." -ForegroundColor Green
clang++ `
    "$ProjectRoot\build\GoogleRuntime.o" `
    "$ProjectRoot\build\test_quick_phase1.o" `
    -o "$ProjectRoot\build\test_quick_phase1.exe"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Linking failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "Running tests..." -ForegroundColor Yellow
Write-Host ""

& "$ProjectRoot\build\test_quick_phase1.exe"

$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "=== All Tests Passed! ===" -ForegroundColor Green
} else {
    Write-Host "=== Some Tests Failed ===" -ForegroundColor Red
}

exit $exitCode
