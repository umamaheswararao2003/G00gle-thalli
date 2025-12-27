# Benchmark: Optimization Gain (Speedup Ratios)
# Compares different optimization levels

param([int]$Iterations = 10)

$ErrorActionPreference = "Stop"
$GoogleOptPath = "..\..\build\bin\google-opt.exe"
$ResultsDir = "..\results"
$TestFile = "..\compute_bound\bench_matmul_1024.mlir"

Write-Host "Optimization Gain Benchmark (Speedup Ratios)" -ForegroundColor Cyan
Write-Host "=============================================`n" -ForegroundColor Cyan

$pipelines = @(
    @{Name="Basic"; Flag="--google-basic-pipeline"},
    @{Name="Optimized"; Flag="--google-optimized-pipeline"},
    @{Name="Extreme L1"; Flag="--google-extreme-l1"},
    @{Name="Extreme L2"; Flag="--google-extreme-l2"},
    @{Name="Extreme L3"; Flag="--google-extreme-l3"},
    @{Name="Extreme L3 Full"; Flag="--google-extreme-l3-full"},
    @{Name="Extreme Pipeline"; Flag="--google-extreme-pipeline"}
)

$results = @()
$baselineTime = 0

foreach ($pipeline in $pipelines) {
    Write-Host "Testing $($pipeline.Name)..." -NoNewline
    
    $times = @()
    $success = $true
    
    for ($i = 0; $i -lt $Iterations; $i++) {
        try {
            $sw = [System.Diagnostics.Stopwatch]::StartNew()
            & $GoogleOptPath $TestFile $pipeline.Flag 2>&1 | Out-Null
            $sw.Stop()
            $times += $sw.ElapsedMilliseconds
        }
        catch {
            Write-Host " FAILED" -ForegroundColor Red
            $success = $false
            break
        }
    }
    
    if ($success) {
        $avgTimeMs = ($times | Measure-Object -Average).Average
        
        if ($pipeline.Name -eq "Basic") {
            $baselineTime = $avgTimeMs
            $speedup = 1.0
        }
        else {
            $speedup = $baselineTime / $avgTimeMs
        }
        
        Write-Host " OK" -ForegroundColor Green
        Write-Host "  Time: $([math]::Round($avgTimeMs, 2))ms" -ForegroundColor Gray
        Write-Host "  Speedup: $([math]::Round($speedup, 2))x" -ForegroundColor $(if ($speedup -ge 10) { "Green" } elseif ($speedup -ge 5) { "Yellow" } else { "White" })
        
        $results += [PSCustomObject]@{
            Pipeline = $pipeline.Name
            AvgTimeMs = [math]::Round($avgTimeMs, 2)
            Speedup = [math]::Round($speedup, 2)
            Success = $true
        }
    }
    else {
        $results += [PSCustomObject]@{
            Pipeline = $pipeline.Name
            AvgTimeMs = 0
            Speedup = 0
            Success = $false
        }
    }
}

# Save results
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$csvPath = "$ResultsDir\optimization_gain_$timestamp.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation

Write-Host "`nResults saved to: $csvPath" -ForegroundColor Green

# Summary
Write-Host "`n[Summary]" -ForegroundColor Magenta
$maxSpeedup = ($results | Where-Object { $_.Success -eq $true } | Measure-Object -Property Speedup -Maximum).Maximum
Write-Host "Baseline (Basic): $([math]::Round($baselineTime, 2))ms" -ForegroundColor Cyan
Write-Host "Maximum Speedup: $([math]::Round($maxSpeedup, 2))x" -ForegroundColor Cyan

$targetAchieved = $maxSpeedup -ge 10
Write-Host "Target (10x): $(if ($targetAchieved) { 'ACHIEVED' } else { 'NOT ACHIEVED' })" -ForegroundColor $(if ($targetAchieved) { "Green" } else { "Yellow" })
