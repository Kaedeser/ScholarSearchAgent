$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:PYTHONPATH = "$Root\src;$Root\framework\sentence-transformers;$env:PYTHONPATH"

python -m selector_reranker.data_builder `
  --pasa-data-dir "..\..\..\数据集\pasa\data" `
  --output-dir "data\processed"
