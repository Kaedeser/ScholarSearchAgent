# 中文功能说明：项目测试脚本，运行默认 pytest 回归测试。

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m pytest -q
