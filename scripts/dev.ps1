# CrossMind - Windows 开发脚本（等价于 Makefile）
# 用法：pwsh -File scripts/dev.ps1 <install|run|test|eval|load|smoke>

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("install", "run", "test", "eval", "load", "smoke")]
    [string]$Task
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "src"
$python = if ($env:XM_PYTHON) { $env:XM_PYTHON } else { "python" }

switch ($Task) {
    "install" {
        & $python -m pip install -r requirements.txt
        & $python -m pip install -e ".[dev]"
    }
    "run" { & $python -m uvicorn crossmind.api.app:app --host 127.0.0.1 --port 8000 --reload }
    "test" { & $python -m pytest tests -q }
    "eval" { & $python scripts/run_eval.py }
    "load" { & $python scripts/load_test.py --requests 300 --concurrency 20 }
    "smoke" { & $python -m crossmind.cli smoke }
}
