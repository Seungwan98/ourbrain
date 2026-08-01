param([switch]$Once)

$ErrorActionPreference = 'SilentlyContinue'
$project = 'D:\ourbrain'
$orchestrationDir = Join-Path $project 'runs\v0.3-model-sweep'
$host.UI.RawUI.WindowTitle = 'OurBrain V0.3 Model Sweep'
$experiments = @(
    [ordered]@{
        name = 'A UPerNet'
        id = 'v0.3-a-upernet-swin-tiny-positive-only'
        run_dir = Join-Path $project 'runs\v0.3-a-upernet-swin-tiny-positive-only'
    },
    [ordered]@{
        name = 'B SegFormer-B1'
        id = 'v0.3-b-segformer-b1-positive-only'
        run_dir = Join-Path $project 'runs\v0.3-b-segformer-b1-positive-only'
    },
    [ordered]@{
        name = 'C SegFormer-B2'
        id = 'v0.3-c-segformer-b2-positive-only'
        run_dir = Join-Path $project 'runs\v0.3-c-segformer-b2-positive-only'
    }
)

function Get-ExperimentStatus {
    param([System.Collections.IDictionary]$Experiment)

    $smoke = Join-Path (
        $orchestrationDir
    ) "smoke\$($Experiment.id)\completion.json"
    $completion = Join-Path $Experiment.run_dir 'completion.json'
    $stderr = Join-Path $Experiment.run_dir 'stderr.log'
    $status = [ordered]@{ name = $Experiment.name; state = 'WAITING'; progress = '-' }
    if (Test-Path $completion -PathType Leaf) {
        $payload = Get-Content $completion -Raw | ConvertFrom-Json
        $status.state = 'COMPLETED'
        $status.progress = (
            "best epoch $($payload.best_epoch) | Dice " +
            ([math]::Round([double]$payload.best_val_crack_dice, 4))
        )
        return $status
    }
    if (Test-Path $smoke -PathType Leaf) {
        $payload = Get-Content $smoke -Raw | ConvertFrom-Json
        if ($payload.full_training_eligible -ne $true) {
            $status.state = 'SKIPPED'
            $status.progress = $payload.reason
            return $status
        }
        $status.state = 'SMOKE PASSED'
        $status.progress = "peak VRAM $($payload.peak_cuda_memory_gib) GiB"
    }
    if (Test-Path (Join-Path $Experiment.run_dir 'started_at.txt')) {
        $status.state = 'TRAINING'
    }
    if (Test-Path $stderr -PathType Leaf) {
        $line = Get-Content $stderr -Tail 400 |
            Where-Object { $_ -match 'epoch\s+\d+/\d+' } |
            Select-Object -Last 1
        if ($line -match 'epoch\s+(\d+)/(\d+):\s+(\d+)%.*?\|\s*(\d+)/(\d+)') {
            $epoch = [int]$Matches[1]
            $epochs = [int]$Matches[2]
            $percent = [int]$Matches[3]
            $step = [int]$Matches[4]
            $steps = [int]$Matches[5]
            $overall = [math]::Round(
                ((($epoch - 1) + ($step / $steps)) / $epochs) * 100,
                1
            )
            $status.progress = (
                "epoch $epoch/$epochs | step $step/$steps | " +
                "$percent% | total $overall%"
            )
        }
    }
    return $status
}

do {
    if (-not $Once) {
        Clear-Host
    }
    Write-Host '============================================================' `
        -ForegroundColor DarkCyan
    Write-Host '            OURBRAIN V0.3 MODEL SWEEP' -ForegroundColor Cyan
    Write-Host '============================================================' `
        -ForegroundColor DarkCyan
    Write-Host 'DEVELOPMENT ONLY: held-out test remains unopened.' `
        -ForegroundColor Yellow

    foreach ($experiment in $experiments) {
        $status = Get-ExperimentStatus $experiment
        $color = switch -Wildcard ($status.state) {
            'COMPLETED' { 'Green' }
            'TRAINING' { 'Cyan' }
            'SKIPPED' { 'Yellow' }
            'SMOKE*' { 'DarkCyan' }
            default { 'DarkGray' }
        }
        Write-Host ("{0,-18}: {1}" -f $status.name, $status.state) `
            -ForegroundColor $color
        Write-Host ("  progress : {0}" -f $status.progress)
    }

    $benchmark = Join-Path $orchestrationDir 'benchmark\benchmark_complete.json'
    $errorFile = Join-Path $orchestrationDir 'orchestration_error.txt'
    if (Test-Path $benchmark -PathType Leaf) {
        $payload = Get-Content $benchmark -Raw | ConvertFrom-Json
        Write-Host ''
        Write-Host "RESULT: retained model = $($payload.retained_model)" `
            -ForegroundColor Green
    }
    elseif (Test-Path $errorFile -PathType Leaf) {
        Write-Host ''
        Write-Host 'ORCHESTRATION FAILED:' -ForegroundColor Red
        Get-Content $errorFile -Tail 8 | ForEach-Object { Write-Host $_ }
    }

    $gpu = & nvidia-smi `
        --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw `
        --format=csv,noheader,nounits
    $parts = $gpu -split ',\s*'
    Write-Host ''
    Write-Host (
        "GPU: {0}% | VRAM {1}/{2} MiB | {3} C | {4} W" -f `
            $parts[0], $parts[1], $parts[2], $parts[3], $parts[4]
    ) -ForegroundColor Magenta
    Write-Host ('Updated: ' + (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))
    Write-Host 'This monitor can be closed without stopping training.' `
        -ForegroundColor DarkGray
    if ($Once) {
        break
    }
    Start-Sleep -Seconds 5
} while ($true)
