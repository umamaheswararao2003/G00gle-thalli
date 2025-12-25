# Google MLIR Dialect - Master Benchmark Runner
# Runs all benchmarks and generates comprehensive performance reports

param(
    [string]$Category = "all",  # all, memory_bound, compute_bound, compilation, build_evaluation
    [string]$Pipeline = "google-extreme-pipeline",  # Pipeline to use
    [int]$Iterations = 10  # Number of iterations for averaging
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Configuration
$GoogleOptPath = ".\build\bin\google-opt.exe"
$ResultsDir = ".\benchmarks\results"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"

# Ensure results directory exists
New-Item -ItemType Directory -Path $ResultsDir -Force | Out-Null

Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Google MLIR Benchmark Suite" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host "Category: $Category" -ForegroundColor Yellow
Write-Host "Pipeline: $Pipeline" -ForegroundColor Yellow
Write-Host "Iterations: $Iterations" -ForegroundColor Yellow
Write-Host "Timestamp: $Timestamp" -ForegroundColor Yellow
Write-Host ""

# Helper function to run a single benchmark
function Run-Benchmark {
    param(
        [string]$TestFile,
        [string]$TestName,
        [string]$Category
    )
    
    Write-Host "Running: $TestName..." -NoNewline
    
    $times = @()
    $success = $true
    
    for ($i = 0; $i -lt $Iterations; $i++) {
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & $GoogleOptPath $TestFile --$Pipeline | Out-Null
            $sw.Stop()
            $times += $sw.ElapsedMilliseconds
        }
        catch {
            Write-Host " FAILED" -ForegroundColor Red
            Write-Host "  Error: $_" -ForegroundColor Red
            $success = $false
            break
        }
    }
    
    if ($success) {
        $avgTime = ($times | Measure-Object -Average).Average
        $minTime = ($times | Measure-Object -Minimum).Minimum
        $maxTime = ($times | Measure-Object -Maximum).Maximum
        
        Write-Host " OK" -ForegroundColor Green
        Write-Host "  Avg: $([math]::Round($avgTime, 2))ms | Min: $([math]::Round($minTime, 2))ms | Max: $([math]::Round($maxTime, 2))ms" -ForegroundColor Gray
        
        return [PSCustomObject]@{
            Category = $Category
            TestName = $TestName
            TestFile = $TestFile
            Pipeline = $Pipeline
            AvgTimeMs = [math]::Round($avgTime, 2)
            MinTimeMs = [math]::Round($minTime, 2)
            MaxTimeMs = [math]::Round($maxTime, 2)
            StdDev = [math]::Round(($times | Measure-Object -StandardDeviation).StandardDeviation, 2)
            Success = $true
        }
    }
    else {
        return [PSCustomObject]@{
            Category = $Category
            TestName = $TestName
            TestFile = $TestFile
            Pipeline = $Pipeline
            AvgTimeMs = 0
            MinTimeMs = 0
            MaxTimeMs = 0
            StdDev = 0
            Success = $false
        }
    }
}

# Memory-Bound Benchmarks
$memoryBoundResults = @()
if ($Category -eq "all" -or $Category -eq "memory_bound") {
    Write-Host "`n[Memory-Bound Benchmarks]" -ForegroundColor Magenta
    Write-Host "Target: >320 GB/s memory bandwidth`n" -ForegroundColor Gray
    
    $memoryBoundTests = @(
        @{File=".\benchmarks\memory_bound\bench_elementwise.mlir"; Name="Element-wise Operations"},
        @{File=".\benchmarks\memory_bound\bench_transpose.mlir"; Name="Transpose"},
        @{File=".\benchmarks\memory_bound\bench_reshape.mlir"; Name="Reshape"},
        @{File=".\benchmarks\memory_bound\bench_concat.mlir"; Name="Concat"},
        @{File=".\benchmarks\memory_bound\bench_broadcast.mlir"; Name="Broadcast"}
    )
    
    foreach ($test in $memoryBoundTests) {
        $result = Run-Benchmark -TestFile $test.File -TestName $test.Name -Category "memory_bound"
        $memoryBoundResults += $result
    }
}

# Compute-Bound Benchmarks
$computeBoundResults = @()
if ($Category -eq "all" -or $Category -eq "compute_bound") {
    Write-Host "`n[Compute-Bound Benchmarks]" -ForegroundColor Magenta
    Write-Host "Target: 7,000-13,000 GFLOPS for matmul`n" -ForegroundColor Gray
    
    $computeBoundTests = @(
        @{File=".\benchmarks\compute_bound\bench_matmul_256.mlir"; Name="MatMul 256x256"; Target="3,000 GFLOPS"},
        @{File=".\benchmarks\compute_bound\bench_matmul_512.mlir"; Name="MatMul 512x512"; Target="5,000 GFLOPS"},
        @{File=".\benchmarks\compute_bound\bench_matmul_1024.mlir"; Name="MatMul 1024x1024"; Target="7,000 GFLOPS"},
        @{File=".\benchmarks\compute_bound\bench_matmul_2048.mlir"; Name="MatMul 2048x2048"; Target="10,000 GFLOPS"},
        @{File=".\benchmarks\compute_bound\bench_matmul_4096.mlir"; Name="MatMul 4096x4096"; Target="12,000 GFLOPS"},
        @{File=".\benchmarks\compute_bound\bench_softmax.mlir"; Name="Softmax"; Target="4,000 GFLOPS"},
        @{File=".\benchmarks\compute_bound\bench_gelu.mlir"; Name="GELU"; Target="3,500 GFLOPS"}
    )
    
    foreach ($test in $computeBoundTests) {
        $result = Run-Benchmark -TestFile $test.File -TestName $test.Name -Category "compute_bound"
        $computeBoundResults += $result
    }
}

# Save Results
Write-Host "`n[Saving Results]" -ForegroundColor Magenta

if ($memoryBoundResults.Count -gt 0) {
    $csvPath = "$ResultsDir\memory_bound_results_$Timestamp.csv"
    $memoryBoundResults | Export-Csv -Path $csvPath -NoTypeInformation
    Write-Host "Memory-bound results: $csvPath" -ForegroundColor Green
}

if ($computeBoundResults.Count -gt 0) {
    $csvPath = "$ResultsDir\compute_bound_results_$Timestamp.csv"
    $computeBoundResults | Export-Csv -Path $csvPath -NoTypeInformation
    Write-Host "Compute-bound results: $csvPath" -ForegroundColor Green
}

# Summary
Write-Host "`n[Summary]" -ForegroundColor Magenta
$allResults = $memoryBoundResults + $computeBoundResults
$successCount = ($allResults | Where-Object { $_.Success -eq $true }).Count
$failCount = ($allResults | Where-Object { $_.Success -eq $false }).Count
$totalCount = $allResults.Count

Write-Host "Total tests: $totalCount" -ForegroundColor White
Write-Host "Passed: $successCount" -ForegroundColor Green
Write-Host "Failed: $failCount" -ForegroundColor $(if ($failCount -gt 0) { "Red" } else { "Green" })

if ($successCount -gt 0) {
    $avgCompileTime = ($allResults | Where-Object { $_.Success -eq $true } | Measure-Object -Property AvgTimeMs -Average).Average
    Write-Host "Average compilation time: $([math]::Round($avgCompileTime, 2))ms" -ForegroundColor Cyan
}

Write-Host "`nBenchmark suite completed!" -ForegroundColor Green
