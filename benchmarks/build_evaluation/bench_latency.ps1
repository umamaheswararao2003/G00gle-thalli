# Benchmark: Build Latency (Compilation Time)
# Measures end-to-end compilation time for different workload sizes

param([int]$Iterations = 10)

$ErrorActionPreference = "Stop"
$GoogleOptPath = "..\..\build\bin\google-opt.exe"
$ResultsDir = "..\results"

Write-Host "Build Latency Benchmark (Compilation Time)" -ForegroundColor Cyan
Write-Host "===========================================`n" -ForegroundColor Cyan

$workloads = @(
    @{File="..\compute_bound\bench_matmul_256.mlir"; Name="Small (256x256)"; Target=200},
    @{File="..\compute_bound\bench_matmul_1024.mlir"; Name="Medium (1024x1024)"; Target=500},
    @{File="..\compute_bound\bench_matmul_4096.mlir"; Name="Large (4096x4096)"; Target=1000}
)

$results = @()

foreach ($workload in $workloads) {
    Write-Host "Testing $($workload.Name)..." -NoNewline
    
    $times = @()
    for ($i = 0; $i -lt $Iterations; $i++) {
        $sw = [System.Diagnostics.Stopwatch]::StartNew()
        & $GoogleOptPath $workload.File --google-extreme-pipeline 2>&1 | Out-Null
        $sw.Stop()
        $times += $sw.ElapsedMilliseconds
    }
    
    $avgTimeMs = ($times | Measure-Object -Average).Average
    $minTimeMs = ($times | Measure-Object -Minimum).Minimum
    $maxTimeMs = ($times | Measure-Object -Maximum).Maximum
    
    $withinTarget = $avgTimeMs -le $workload.Target
    
    Write-Host " OK" -ForegroundColor Green
    Write-Host "  Avg: $([math]::Round($avgTimeMs, 2))ms (Target: $($workload.Target)ms)" -ForegroundColor $(if ($withinTarget) { "Green" } else { "Yellow" })
    Write-Host "  Min: $([math]::Round($minTimeMs, 2))ms | Max: $([math]::Round($maxTimeMs, 2))ms" -ForegroundColor Gray
    
    $results += [PSCustomObject]@{
        Workload = $workload.Name
        AvgTimeMs = [math]::Round($avgTimeMs, 2)
        MinTimeMs = [math]::Round($minTimeMs, 2)
        MaxTimeMs = [math]::Round($maxTimeMs, 2)
        TargetMs = $workload.Target
        WithinTarget = $withinTarget
    }
}

# Save results
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvPath = "$ResultsDir\build_latency_$timestamp.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation

Write-Host "`nResults saved to: $csvPath" -ForegroundColor Green

# Summary
Write-Host "`n[Summary]" -ForegroundColor Magenta
$avgLatency = ($results | Measure-Object -Property AvgTimeMs -Average).Average
$allWithinTarget = ($results | Where-Object { $_.WithinTarget -eq $false }).Count -eq 0

Write-Host "Average Latency: $([math]::Round($avgLatency, 2))ms" -ForegroundColor Cyan
Write-Host "All Within Target: $(if ($allWithinTarget) { 'YES' } else { 'NO' })" -ForegroundColor $(if ($allWithinTarget) { "Green" } else { "Yellow" })
