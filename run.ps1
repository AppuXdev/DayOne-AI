param(
    [switch]$SkipIngest,
    [switch]$SkipInstall,
    [switch]$NoStreamlit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Step,
        [switch]$AllowFailure
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        if ($AllowFailure) {
            Write-Warning "$Step failed with exit code $LASTEXITCODE. Continuing startup."
            return
        }

        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

$venvDir = Join-Path $projectRoot '.venv'
$pythonExe = Join-Path $venvDir 'Scripts\python.exe'

if (-not (Test-Path $pythonExe)) {
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        throw 'Python is not installed or not on PATH. Install Python 3.10+ and re-run .\run.ps1.'
    }

    Write-Host 'Creating virtual environment (.venv)...' -ForegroundColor Cyan
    & $pythonCmd.Source -m venv $venvDir
}

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found at $pythonExe after virtual environment creation."
}

# Some Windows shells leak global Python vars that cause '<prefix>' startup warnings.
if (Test-Path Env:PYTHONHOME) { Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue }
if (Test-Path Env:PYTHONPATH) { Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue }

if (-not $SkipInstall) {
    Write-Host 'Installing Python dependencies...' -ForegroundColor Cyan
    Invoke-ExternalCommand -Step 'pip upgrade' -Command { & $pythonExe -m pip install --upgrade pip }
    Invoke-ExternalCommand -Step 'pip install requirements' -Command { & $pythonExe -m pip install -r (Join-Path $projectRoot 'requirements.txt') }

    $npmCmd = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCmd) {
        $npmCmd = Get-Command npm -ErrorAction SilentlyContinue
    }
    if (-not $npmCmd) {
        throw 'npm is not installed or not on PATH. Install Node.js 18+ and re-run .\run.ps1.'
    }

    Write-Host 'Installing Node.js dependencies...' -ForegroundColor Cyan
    Invoke-ExternalCommand -Step 'npm install' -Command { & $npmCmd.Source install }
}

if (-not $SkipIngest) {
    Write-Host 'Running initial ingest...' -ForegroundColor Cyan
    Invoke-ExternalCommand -Step 'initial ingest' -AllowFailure -Command { & $pythonExe ingest.py }
}

Write-Host 'Starting auto-ingest watcher in a new terminal...' -ForegroundColor Cyan
$watcherCommand = "Set-Location '$projectRoot'; & '$pythonExe' auto_ingest.py"
Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $watcherCommand | Out-Null

Write-Host 'Starting FastAPI backend in a new terminal...' -ForegroundColor Cyan
$apiCommand = "Set-Location '$projectRoot'; & '$pythonExe' -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload"
Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $apiCommand | Out-Null

Write-Host 'Starting Next.js frontend in a new terminal...' -ForegroundColor Cyan
$frontendCommand = "Set-Location '$projectRoot'; `$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8000'; npm.cmd run dev"
Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $frontendCommand | Out-Null

if ($NoStreamlit) {
    Write-Host 'Startup complete.' -ForegroundColor Green
    Write-Host 'Frontend: http://localhost:3000' -ForegroundColor Green
    Write-Host 'Backend : http://127.0.0.1:8000' -ForegroundColor Green
    Write-Host 'Watcher : running in separate terminal' -ForegroundColor Green
    return
}

Write-Host 'Starting Streamlit app (current terminal)...' -ForegroundColor Green
Write-Host 'Streamlit: http://localhost:8501' -ForegroundColor Green
& $pythonExe -m streamlit run app.py
