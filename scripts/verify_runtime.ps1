# Quick Verification Script
# Builds and runs tests to verify runtime engine is working

Write-Host "=== Runtime Engine Verification ===" -ForegroundColor Cyan
Write-Host ""

$ProjectRoot = "c:\Users\Asus\Desktop\google"
$BuildDir = "$ProjectRoot\build"

# Step 1: Build Runtime
Write-Host "Step 1: Building GoogleRuntime library..." -ForegroundColor Green
cmake --build "$BuildDir" --target GoogleRuntime 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "X Build failed! Runtime engine is NOT working." -ForegroundColor Red
    Write-Host "Run: cmake --build build --target GoogleRuntime" -ForegroundColor Yellow
    exit 1
}
Write-Host "  OK GoogleRuntime library built successfully" -ForegroundColor Gray

# Step 2: Build Tests
Write-Host ""
Write-Host "Step 2: Building test executable..." -ForegroundColor Green
cmake --build "$BuildDir" --target test_quick_phase1 2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Host "X Test build failed!" -ForegroundColor Red
    Write-Host "Run: cmake --build build --target test_quick_phase1" -ForegroundColor Yellow
    exit 1
}
Write-Host "  OK Test executable built successfully" -ForegroundColor Gray

# Step 3: Run Tests
Write-Host ""
Write-Host "Step 3: Running tests..." -ForegroundColor Green
Write-Host ""

& "$BuildDir\bin\test_quick_phase1.exe"

$testResult = $LASTEXITCODE

Write-Host ""
if ($testResult -eq 0) {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "OK RUNTIME ENGINE IS WORKING!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "All 16 tests passed successfully." -ForegroundColor Gray
    Write-Host "The runtime engine is fully functional." -ForegroundColor Gray
} else {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "X RUNTIME ENGINE HAS ISSUES" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Some tests failed. Check output above." -ForegroundColor Yellow
}

exit $testResult
