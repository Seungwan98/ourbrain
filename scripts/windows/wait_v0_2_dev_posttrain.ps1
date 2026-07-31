$ErrorActionPreference = 'Stop'

$project = 'D:\ourbrain'
$taskName = 'OurBrain-V02-Dev-Posttrain'
$runDir = Join-Path $project 'runs\v0.2-dev-positive-only-ab'
$trainingComplete = Join-Path $runDir 'training_complete.json'
$trainingError = Join-Path $runDir 'orchestration_error.txt'
$benchmarkComplete = Join-Path (
    $project
) 'runs\v0.2-dev-benchmark\benchmark_complete.json'
$benchmarkRunner = Join-Path (
    $project
) 'scripts\windows\run_v0_2_dev_benchmark.ps1'
$startedAt = [DateTime]::UtcNow

function Write-AtomicText {
    param([string]$Path, [string]$Value)
    $temporary = "$Path.tmp"
    Set-Content -Path $temporary -Value $Value -Encoding utf8
    Move-Item -Path $temporary -Destination $Path -Force
}

try {
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    Remove-Item (Join-Path $runDir 'posttrain_wait_error.txt') `
        -Force -ErrorAction SilentlyContinue
    Write-AtomicText -Path (Join-Path $runDir 'posttrain_wait_started_at.txt') `
        -Value ($startedAt.ToString('o'))

    while ($true) {
        if (Test-Path $benchmarkComplete -PathType Leaf) {
            Write-AtomicText -Path (
                Join-Path $runDir 'posttrain_wait_status.txt'
            ) -Value 'benchmark-already-complete'
            exit 0
        }
        if (Test-Path $trainingError -PathType Leaf) {
            throw (
                'Training orchestration failed before post-training benchmark: ' +
                (Get-Content $trainingError -Raw)
            )
        }
        if (Test-Path $trainingComplete -PathType Leaf) {
            Write-AtomicText -Path (
                Join-Path $runDir 'posttrain_wait_status.txt'
            ) -Value 'running-benchmark'
            & powershell -NoProfile -ExecutionPolicy Bypass `
                -File $benchmarkRunner
            if ($LASTEXITCODE -ne 0) {
                throw "Post-training benchmark returned exit $LASTEXITCODE."
            }
            if (-not (Test-Path $benchmarkComplete -PathType Leaf)) {
                throw "Post-training benchmark returned without completion."
            }
            Write-AtomicText -Path (
                Join-Path $runDir 'posttrain_wait_status.txt'
            ) -Value 'benchmark-complete'
            exit 0
        }
        if (([DateTime]::UtcNow - $startedAt).TotalHours -ge 12) {
            throw "Timed out waiting 12 hours for v0.2 development training."
        }
        Write-AtomicText -Path (
            Join-Path $runDir 'posttrain_wait_status.txt'
        ) -Value 'waiting-for-training'
        Start-Sleep -Seconds 30
    }
}
catch {
    Write-AtomicText -Path (
        Join-Path $runDir 'posttrain_wait_error.txt'
    ) -Value ($_ | Out-String)
    throw
}
finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false `
        -ErrorAction SilentlyContinue
}
