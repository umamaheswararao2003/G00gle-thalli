# Benchmark: Execution Efficiency (GFLOPS)
# Measures computational throughput for MatMul operations

param([int]$Iterations = 10)

$ErrorActionPreference = "Stop"
$GoogleOptPath = "..\..\build\bin\google-opt.exe"
$ResultsDir = "..\results"

Write-Host "Execution Efficiency Benchmark (GFLOPS)" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# MatMul GFLOPS calculation: (2 * M * N * K) / (time_in_seconds * 1e9)
$tests = @(
    @{File="..\compute_bound\bench_matmul_256.mlir"; M=256; N=256; K=256; Target=3000},
    @{File="..\compute_bound\bench_matmul_512.mlir"; M=512; N=512; K=512; Target=5000},
    @{File="..\compute_bound\bench_matmul_1024.mlir"; M=1024; N=1024; K=1024; Target=7000},
    @{File="..\compute_bound\bench_matmul_2048.mlir"; M=2048; N=2048; K=2048; Target=10000},
    @{File="..\compute_bound\bench_matmul_4096.mlir"; M=4096; N=4096; K=4096; Target=12000}
)

$results = @()

foreach ($test in $tests) {
    $size = "$($test.M)x$($test.N)x$($test.K)"
    Write-Host "Testing MatMul $size..." -NoNewline
    
    $times = @()
    for ($i = 0; $i -lt $Iterations; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $GoogleOptPath $test.File --google-extreme-pipeline | Out-Null
        $sw.Stop()
        $times += $sw.ElapsedMilliseconds
    }
    
    $avgTimeMs = ($times | Measure-Object -Average).Average
    $avgTimeSec = $avgTimeMs / 1000.0
    
    # GFLOPS = (2 * M * N * K) / (time_in_seconds * 1e9)
    $flops = 2.0 * $test.M * $test.N * $test.K
    $gflops = $flops / ($avgTimeSec * 1e9)
    
    $efficiency = ($gflops / $test.Target) * 100
    
    Write-Host " OK" -ForegroundColor Green
    Write-Host "  Time: $([math]::Round($avgTimeMs, 2))ms" -ForegroundColor Gray
    Write-Host "  GFLOPS: $([math]::Round($gflops, 2)) (Target: $($test.Target))" -ForegroundColor $(if ($gflops -ge $test.Target) { "Green" } else { "Yellow" })
    Write-Host "  Efficiency: $([math]::Round($efficiency, 1))%" -ForegroundColor $(if ($efficiency -ge 80) { "Green" } else { "Yellow" })
    
    $results += [PSCustomObject]@{
        Size = $size
        M = $test.M
        N = $test.N
        K = $test.K
        AvgTimeMs = [math]::Round($avgTimeMs, 2)
        GFLOPS = [math]::Round($gflops, 2)
        TargetGFLOPS = $test.Target
        Efficiency = [math]::Round($efficiency, 1)
    }
}

# Save results
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvPath = "$ResultsDir\execution_efficiency_$timestamp.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation

Write-Host "`nResults saved to: $csvPath" -ForegroundColor Green

# Summary
Write-Host "`n[Summary]" -ForegroundColor Magenta
$avgGFLOPS = ($results | Measure-Object -Property GFLOPS -Average).Average
$avgEfficiency = ($results | Measure-Object -Property Efficiency -Average).Average
Write-Host "Average GFLOPS: $([math]::Round($avgGFLOPS, 2))" -ForegroundColor Cyan
Write-Host "Average Efficiency: $([math]::Round($avgEfficiency, 1))%" -ForegroundColor Cyan
