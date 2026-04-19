param(
    [string]$Org = "org_acme",
    [string]$ComposeFile = "infra/docker-compose.yml",
    [string]$DatabaseUrl = "postgresql+psycopg://dayone:dayone@postgres:5432/dayone",
    [string]$BaselineFile = "scripts/legacy_benchmark/faiss_baseline_org_acme.json",
    [double]$HitRateThreshold = 0.03,
    [double]$P1Threshold = 0.05,
    [double]$ConfidenceThreshold = 0.05,
    [double]$LatencyRatioThreshold = 1.15,
    [switch]$SkipEval,
    [switch]$NoAssertTenantIsolation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Command,
        [Parameter(Mandatory = $true)][string]$Step
    )

    Write-Host "[step] $Step" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

function Get-Summary {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string]$Mode = "reranker_on"
    )

    if (-not (Test-Path $FilePath)) {
        throw "Missing eval output: $FilePath"
    }

    $json = Get-Content -Path $FilePath -Raw | ConvertFrom-Json
    $summary = $json.summaries | Where-Object { $_.mode -eq $Mode } | Select-Object -First 1
    if (-not $summary) {
        throw "No summary for mode '$Mode' in $FilePath"
    }
    return $summary
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot

try {
    $env:DAYONE_ASSERT_TENANT_ISOLATION = if ($NoAssertTenantIsolation) { "0" } else { "1" }

    if (-not $SkipEval) {
        Invoke-Step -Step "Start postgres" -Command {
            docker compose -f $ComposeFile up -d postgres
        }

        Invoke-Step -Step "Run pgvector evaluation" -Command {
            docker compose -f $ComposeFile run --rm --no-deps `
                -e DATABASE_URL=$DatabaseUrl `
                -e DAYONE_ASSERT_TENANT_ISOLATION=$env:DAYONE_ASSERT_TENANT_ISOLATION `
                backend python eval.py --org $Org --output eval_pgvector.json
        }
    }

    Invoke-Step -Step "Summarize latency/TTFT percentiles" -Command {
        .\.venv\Scripts\python.exe scripts\latency_percentiles.py
    }

    $pg = Get-Summary -FilePath "eval_pgvector.json" -Mode "reranker_on"
    if (-not (Test-Path $BaselineFile)) {
        throw "Missing baseline file: $BaselineFile"
    }
    $baseline = Get-Content -Path $BaselineFile -Raw | ConvertFrom-Json

    $hitDelta = [math]::Abs([double]$pg.positive_hit_rate - [double]$baseline.positive_hit_rate)
    $p1Delta = [math]::Abs([double]$pg.precision_at_1 - [double]$baseline.precision_at_1)
    $confidenceDelta = [math]::Abs([double]$pg.avg_confidence - [double]$baseline.avg_confidence)
    $latencyRatio = if ([double]$baseline.avg_latency_ms -le 0) { 999.0 } else { [double]$pg.avg_latency_ms / [double]$baseline.avg_latency_ms }

    $hitPass = $hitDelta -le $HitRateThreshold
    $p1Pass = $p1Delta -le $P1Threshold
    $confidencePass = $confidenceDelta -le $ConfidenceThreshold
    $latencyPass = $latencyRatio -le $LatencyRatioThreshold

    Write-Host ""
    Write-Host "=== Stabilization Gate (reranker_on) ===" -ForegroundColor Yellow
    Write-Host ("Hit rate:     pgvector={0:P1} baseline={1:P1} delta={2:P2} threshold={3:P0} => {4}" -f $pg.positive_hit_rate, $baseline.positive_hit_rate, $hitDelta, $HitRateThreshold, ($(if ($hitPass) { 'PASS' } else { 'FAIL' })))
    Write-Host ("P@1:          pgvector={0:P1} baseline={1:P1} delta={2:P2} threshold={3:P0} => {4}" -f $pg.precision_at_1, $baseline.precision_at_1, $p1Delta, $P1Threshold, ($(if ($p1Pass) { 'PASS' } else { 'FAIL' })))
    Write-Host ("Confidence:   pgvector={0:N3} baseline={1:N3} delta={2:N3} threshold={3:N3} => {4}" -f $pg.avg_confidence, $baseline.avg_confidence, $confidenceDelta, $ConfidenceThreshold, ($(if ($confidencePass) { 'PASS' } else { 'FAIL' })))
    Write-Host ("Avg latency:  pgvector={0:N1}ms baseline={1:N1}ms ratio={2:N3} threshold={3:N3} => {4}" -f $pg.avg_latency_ms, $baseline.avg_latency_ms, $latencyRatio, $LatencyRatioThreshold, ($(if ($latencyPass) { 'PASS' } else { 'FAIL' })))

    $allPass = $hitPass -and $p1Pass -and $confidencePass -and $latencyPass
    Write-Host ""
    if ($allPass) {
        Write-Host "STABILIZATION GATE: PASS" -ForegroundColor Green
        exit 0
    }

    Write-Host "STABILIZATION GATE: FAIL" -ForegroundColor Red
    exit 1
}
finally {
    Pop-Location
}
