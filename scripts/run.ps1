param([switch]$Replay, [int]$Camera = 0)
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$taskArgs = @('app.py','--camera',"$Camera")
if ($Replay) { $taskArgs += '--replay' }
& ./.venv/Scripts/python.exe @taskArgs
