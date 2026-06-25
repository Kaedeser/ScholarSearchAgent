# 中文功能说明：后端开发启动脚本，从项目根目录启动 ScholarSearch API 服务。

param(
    [string]$Backend = "auto",
    [string]$HostName = "127.0.0.1",
    [int]$Port = 8765,
    [switch]$DisableModelServices
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$argsList = @("-m", "apps.backend.scholar_api.cli", "--backend", $Backend)
if ($DisableModelServices) {
    $argsList += "--disable-model-services"
}
$argsList += @("serve", "--host", $HostName, "--port", [string]$Port)
python @argsList
