# 中文功能说明：React 前端开发启动脚本，从项目根目录进入 apps/frontend 并启动 Vite 服务。

param(
    [int]$Port = 5174,
    [string]$HostName = "127.0.0.1",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$frontend = Join-Path $root "apps\frontend"
Set-Location $frontend

if (-not $SkipInstall -and -not (Test-Path (Join-Path $frontend "node_modules"))) {
    npm install
}

npm run dev -- --host $HostName --port $Port
