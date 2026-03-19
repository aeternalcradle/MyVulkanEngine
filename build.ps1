# 快速构建脚本
# 用法:
#   ./build.ps1              # Debug 构建
#   ./build.ps1 -Config Release

param(
    [string]$Config = "Debug"
)

$BuildDir = "build/$Config"

Write-Host "==> cmake configure ..." -ForegroundColor Cyan
cmake -S . -B $BuildDir -DCMAKE_BUILD_TYPE=$Config
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> cmake build ($Config) ..." -ForegroundColor Cyan
cmake --build $BuildDir --config $Config --parallel
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "==> 构建成功！可执行文件位于 $BuildDir\bin\" -ForegroundColor Green
