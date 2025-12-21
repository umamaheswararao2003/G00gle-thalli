param(
    [Parameter(Mandatory=$true)]
    [string]$InputFile,
    [string]$OutputFile = "output_optimized.mlir"
)

Write-Host "Running Optimized Pipeline on $InputFile..." -ForegroundColor Cyan

& .\build\bin\google-opt.exe $InputFile `
  --convert-google-to-linalg `
  --linalg-fuse-elementwise-ops `
  --linalg-generalize-named-ops `
  --one-shot-bufferize `
  --convert-linalg-to-affine-loops `
  --affine-loop-fusion `
  --affine-loop-coalescing `
  --lower-affine `
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
