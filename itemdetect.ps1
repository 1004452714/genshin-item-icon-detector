[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$RepoRoot = $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src"

if (-not $Python) {
    if ($env:ITEMDETECT_PYTHON) {
        $Python = $env:ITEMDETECT_PYTHON
    } else {
        $Python = "python"
    }
}

& $Python (Join-Path $RepoRoot "itemdetect.py") @Arguments
exit $LASTEXITCODE
