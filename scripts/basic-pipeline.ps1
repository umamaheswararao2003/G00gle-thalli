param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    [string]$OutputFile = "output_basic.mlir"
)

Write-Host "Running Basic Pipeline on $InputFile..." -ForegroundColor Cyan

& .\build\bin\google-opt.exe $InputFile `
  --convert-google-to-linalg `
  --linalg-fuse-elementwise-ops `
  --one-shot-bufferize `
  --convert-linalg-to-loops `
  --convert-scf-to-cf `
  --convert-vector-to-llvm `
  --convert-func-to-llvm `
  --convert-arith-to-llvm `
  --finalize-memref-to-llvm `
  --reconcile-unrealized-casts `
  -o $OutputFile

if ($LASTEXITCODE -eq 0) {
    Write-Host "Success! Output written to $OutputFile" -ForegroundColor Green
} else {
    Write-Host "Pipeline failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
