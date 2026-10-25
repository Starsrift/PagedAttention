# Keep the interpreter, packages, download cache and temporary files in this checkout.
$taskUvArguments = $args
$ErrorActionPreference = 'Stop'
$taskSettings = @{
    UV_PYTHON_INSTALL_DIR = Join-Path $PSScriptRoot '.local\python'
    UV_CACHE_DIR = Join-Path $PSScriptRoot '.local\uv-cache'
    UV_PROJECT_ENVIRONMENT = Join-Path $PSScriptRoot '.venv'
    UV_PYTHON_PREFERENCE = 'only-managed'
    HF_HOME = Join-Path $PSScriptRoot '.local\huggingface'
    HF_HUB_CACHE = Join-Path $PSScriptRoot '.local\huggingface\hub'
    HF_HUB_DISABLE_XET = '1'
    HF_HUB_DISABLE_TELEMETRY = '1'
    TEMP = Join-Path $PSScriptRoot '.local\tmp'
    TMP = Join-Path $PSScriptRoot '.local\tmp'
}
New-Item -ItemType Directory -Force -Path $taskSettings.TEMP | Out-Null
$taskPrevious = @{}
foreach ($taskKey in $taskSettings.Keys) {
    $taskPrevious[$taskKey] = [Environment]::GetEnvironmentVariable($taskKey, 'Process')
    [Environment]::SetEnvironmentVariable($taskKey, $taskSettings[$taskKey], 'Process')
}
$taskExitCode = 1
Push-Location $PSScriptRoot
try {
    & uv.exe @taskUvArguments
    $taskExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
    foreach ($taskKey in $taskSettings.Keys) {
        [Environment]::SetEnvironmentVariable($taskKey, $taskPrevious[$taskKey], 'Process')
    }
}
exit $taskExitCode
