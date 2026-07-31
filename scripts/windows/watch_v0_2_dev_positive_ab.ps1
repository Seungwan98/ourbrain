param([switch]$Once)

$ErrorActionPreference = 'SilentlyContinue'
$project = 'D:\ourbrain'
$host.UI.RawUI.WindowTitle = 'OurBrain V0.2 Dev Positive-Only A/B'
$experiments = @(
    [ordered]@{
        name = 'A augmentation'
        run_dir = Join-Path $project (
            'runs\v0.2-dev-a-augmentation-positive-only'
        )
    },
    [ordered]@{
        name = 'B aug+recall'
        run_dir = Join-Path $project (
            'runs\v0.2-dev-b-augmentation-recall-positive-only'
        )
    }
)

function Get-ExperimentStatus {
    param([System.Collections.IDictionary]$Experiment)

    $runDir = [string]$Experiment.run_dir
    $exitFile = Join-Path $runDir 'exit_code.txt'
    $stderr = Join-Path $runDir 'stderr.log'
    $status = [ordered]@{
        name = $Experiment.name
        state = 'WAITING'
        progress = '-'
    }
    if (Test-Path $exitFile -PathType Leaf) {
        $code = [int](Get-Content $exitFile)
        $status.state = if ($code -eq 0) {
            'COMPLETED'
        }
        else {
            "FAILED ($code)"
        }
    }
    elseif (Test-Path (Join-Path $runDir 'started_at.txt')) {
        $status.state = 'TRAINING'
    }
    if (Test-Path $stderr -PathType Leaf) {
        $line = Get-Content $stderr -Tail 300 |
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
    Write-Host '       OURBRAIN V0.2 DEV - POSITIVE-ONLY A/B' `
        -ForegroundColor Cyan
    Write-Host '============================================================' `
        -ForegroundColor DarkCyan
    Write-Host 'DEVELOPMENT ONLY: normal-image specificity is unavailable.' `
        -ForegroundColor Yellow

    $statuses = @(
        $experiments | ForEach-Object { Get-ExperimentStatus $_ }
    )
    foreach ($status in $statuses) {
        $color = switch -Wildcard ($status.state) {
            'COMPLETED' { 'Green' }
            'TRAINING' { 'Cyan' }
            'FAILED*' { 'Red' }
            default { 'DarkGray' }
        }
        Write-Host ("{0,-15}: {1}" -f $status.name, $status.state) `
            -ForegroundColor $color
        Write-Host ("  progress : {0}" -f $status.progress)
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
