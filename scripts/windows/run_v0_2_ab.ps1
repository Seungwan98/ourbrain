param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PIP_CACHE_DIR = 'D:\ourbrain-cache\pip'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$manifest = Join-Path $project 'artifacts\manifest_with_negatives.csv'
$orchestrationDir = Join-Path $project 'runs\v0.2-ab'
$taskName = 'OurBrain-V02-AB'

$experiments = @(
    [ordered]@{
        id = 'v0.2-a-baseline-with-negatives'
        config = Join-Path $project 'configs\v0_2_a_baseline_with_negatives_cuda.yaml'
        run_dir = Join-Path $project 'runs\v0.2-a-baseline-with-negatives'
    },
    [ordered]@{
        id = 'v0.2-b-recall-with-negatives'
        config = Join-Path $project 'configs\v0_2_b_recall_with_negatives_cuda.yaml'
        run_dir = Join-Path $project 'runs\v0.2-b-recall-with-negatives'
    }
)

function Write-AtomicText {
    param([string]$Path, [string]$Value)
    $temporary = "$Path.tmp"
    Set-Content -Path $temporary -Value $Value -Encoding utf8
    Move-Item -Path $temporary -Destination $Path -Force
}

function Invoke-Preflight {
    if (-not (Test-Path $python)) {
        throw "Python environment does not exist: $python"
    }
    foreach ($experiment in $experiments) {
        if (-not (Test-Path $experiment.config)) {
            throw "Experiment config does not exist: $($experiment.config)"
        }
    }
    $results = @()
    foreach ($experiment in $experiments) {
        $arguments = @(
            '-m', 'ourbrain_cv.cli', 'training-preflight',
            '--config', $experiment.config,
            '--manifest', $manifest,
            '--require-local-checkpoint',
            '--device', 'cuda'
        )
        if ($results.Count -eq 0) {
            $arguments += '--verify-files'
        }
        $result = & $python @arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw ($result | Out-String)
        }
        $results += ($result | Out-String | ConvertFrom-Json)
    }
    New-Item -ItemType Directory -Force -Path $orchestrationDir | Out-Null
    Remove-Item (Join-Path $orchestrationDir 'orchestration_error.txt') `
        -Force -ErrorAction SilentlyContinue
    Write-AtomicText -Path (Join-Path $orchestrationDir 'preflight.json') `
        -Value ($results | ConvertTo-Json -Depth 10)
    return $results
}

function Invoke-Experiment {
    param([System.Collections.IDictionary]$Experiment)

    $runDir = [string]$Experiment.run_dir
    $exitFile = Join-Path $runDir 'exit_code.txt'
    $checkpoint = Join-Path $runDir 'checkpoint\model.safetensors'
    $lastCheckpoint = Join-Path $runDir 'checkpoint\last\model.safetensors'
    $completionPath = Join-Path $runDir 'completion.json'
    $historyPath = Join-Path $runDir 'checkpoint\history.json'
    $trainingConfigPath = Join-Path $runDir 'checkpoint\training_config.json'
    if (Test-Path $exitFile) {
        $existingCode = [int](Get-Content $exitFile)
        if (
            $existingCode -eq 0 -and
            (Test-Path $checkpoint) -and
            (Test-Path $lastCheckpoint) -and
            (Test-Path $completionPath)
        ) {
            $existingCompletion = Get-Content $completionPath -Raw |
                ConvertFrom-Json
            if (
                $existingCompletion.schema_version -ne 1 -or
                $existingCompletion.status -ne 'training-complete' -or
                $existingCompletion.run_id -ne $Experiment.id
            ) {
                throw "Completed experiment record is invalid: $completionPath"
            }
            $expectedHashes = [ordered]@{
                best_checkpoint_sha256 = (
                    Get-FileHash -Algorithm SHA256 $checkpoint
                ).Hash.ToLowerInvariant()
                last_checkpoint_sha256 = (
                    Get-FileHash -Algorithm SHA256 $lastCheckpoint
                ).Hash.ToLowerInvariant()
                history_sha256 = (
                    Get-FileHash -Algorithm SHA256 $historyPath
                ).Hash.ToLowerInvariant()
                training_config_sha256 = (
                    Get-FileHash -Algorithm SHA256 $trainingConfigPath
                ).Hash.ToLowerInvariant()
                config_sha256 = (
                    Get-FileHash -Algorithm SHA256 $Experiment.config
                ).Hash.ToLowerInvariant()
                manifest_sha256 = (
                    Get-FileHash -Algorithm SHA256 $manifest
                ).Hash.ToLowerInvariant()
            }
            foreach ($field in $expectedHashes.Keys) {
                if ($existingCompletion.$field -ne $expectedHashes[$field]) {
                    throw (
                        "Completed experiment artifact changed: " +
                        "$($Experiment.id) $field"
                    )
                }
            }
            Write-Host "Skipping completed experiment: $($Experiment.id)"
            return $existingCompletion
        }
        throw "Existing experiment is not safely resumable: $runDir (exit $existingCode)"
    }
    if (Test-Path $checkpoint) {
        throw "Checkpoint exists without a completion record; refusing to overwrite: $checkpoint"
    }

    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    $stdout = Join-Path $runDir 'stdout.log'
    $stderr = Join-Path $runDir 'stderr.log'
    $startedAt = [DateTime]::UtcNow.ToString('o')
    Write-AtomicText -Path (Join-Path $runDir 'started_at.txt') -Value $startedAt

    $metadata = [ordered]@{
        run_id = $Experiment.id
        mode = 'reviewed-negative-controlled-ab'
        production_eligible = $false
        config = $Experiment.config
        config_sha256 = (Get-FileHash -Algorithm SHA256 $Experiment.config).Hash.ToLower()
        manifest = $manifest
        manifest_sha256 = (Get-FileHash -Algorithm SHA256 $manifest).Hash.ToLower()
        source_checkpoint = 'D:\ourbrain\runs\v0-positive-only\checkpoint'
        source_checkpoint_sha256 = (
            Get-FileHash -Algorithm SHA256 `
                'D:\ourbrain\runs\v0-positive-only\checkpoint\model.safetensors'
        ).Hash.ToLower()
        device = 'cuda'
        started_at_utc = $startedAt
    }
    Write-AtomicText -Path (Join-Path $runDir 'run_metadata.json') `
        -Value ($metadata | ConvertTo-Json -Depth 5)

    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'train',
        '--config', $Experiment.config,
        '--manifest', $manifest,
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $project -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -NoNewWindow -Wait -PassThru
    $code = $process.ExitCode
    Write-AtomicText -Path (Join-Path $runDir 'finished_at.txt') `
        -Value ([DateTime]::UtcNow.ToString('o'))
    if ($code -ne 0) {
        Write-AtomicText -Path $exitFile -Value ([string]$code)
        throw "Experiment failed with exit ${code}: $($Experiment.id)"
    }
    $requiredOutputs = @(
        $checkpoint,
        $lastCheckpoint,
        $historyPath,
        $trainingConfigPath
    )
    $missingOutputs = @(
        $requiredOutputs | Where-Object { -not (Test-Path $_ -PathType Leaf) }
    )
    if ($missingOutputs.Count -gt 0) {
        Write-AtomicText -Path $exitFile -Value '2'
        throw (
            "Experiment returned success without required artifact(s): " +
            ($missingOutputs -join ', ')
        )
    }
    $historyPayload = Get-Content $historyPath -Raw | ConvertFrom-Json
    $history = @()
    foreach ($entry in $historyPayload) {
        $history += $entry
    }
    if ($history.Count -eq 0) {
        Write-AtomicText -Path $exitFile -Value '2'
        throw "Experiment history is empty: $($Experiment.id)"
    }
    $best = $history |
        Sort-Object -Property @{
            Expression = { [double]$_.val_crack_dice }
            Descending = $true
        } |
        Select-Object -First 1
    $completion = [ordered]@{
        schema_version = 1
        run_id = $Experiment.id
        status = 'training-complete'
        exit_code = 0
        config = $Experiment.config
        config_sha256 = (
            Get-FileHash -Algorithm SHA256 $Experiment.config
        ).Hash.ToLowerInvariant()
        manifest = $manifest
        manifest_sha256 = (
            Get-FileHash -Algorithm SHA256 $manifest
        ).Hash.ToLowerInvariant()
        maximum_epochs = 15
        completed_epochs = $history.Count
        early_stopped = ($history.Count -lt 15)
        best_epoch = [int]$best.epoch
        best_val_crack_dice = [double]$best.val_crack_dice
        best_checkpoint = $checkpoint
        best_checkpoint_sha256 = (
            Get-FileHash -Algorithm SHA256 $checkpoint
        ).Hash.ToLowerInvariant()
        last_checkpoint = $lastCheckpoint
        last_checkpoint_sha256 = (
            Get-FileHash -Algorithm SHA256 $lastCheckpoint
        ).Hash.ToLowerInvariant()
        history = $historyPath
        history_sha256 = (
            Get-FileHash -Algorithm SHA256 $historyPath
        ).Hash.ToLowerInvariant()
        training_config = $trainingConfigPath
        training_config_sha256 = (
            Get-FileHash -Algorithm SHA256 $trainingConfigPath
        ).Hash.ToLowerInvariant()
        finished_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 5)
    Write-AtomicText -Path $exitFile -Value '0'
    return [pscustomobject]$completion
}

try {
    $preflight = Invoke-Preflight
    if ($PreflightOnly) {
        $preflight
        exit 0
    }
    Write-AtomicText -Path (Join-Path $orchestrationDir 'started_at.txt') `
        -Value ([DateTime]::UtcNow.ToString('o'))
    $completionRecords = @()
    foreach ($experiment in $experiments) {
        $completionRecords += Invoke-Experiment -Experiment $experiment
    }
    $summary = [ordered]@{
        schema_version = 1
        status = 'training-complete'
        experiments = $completionRecords
        manifest = $manifest
        manifest_sha256 = (
            Get-FileHash -Algorithm SHA256 $manifest
        ).Hash.ToLowerInvariant()
        next_step = 'calibrate both validation checkpoints, select one, then evaluate test once'
        finished_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path (Join-Path $orchestrationDir 'training_complete.json') `
        -Value ($summary | ConvertTo-Json -Depth 4)
}
catch {
    New-Item -ItemType Directory -Force -Path $orchestrationDir | Out-Null
    Write-AtomicText -Path (Join-Path $orchestrationDir 'orchestration_error.txt') `
        -Value ($_ | Out-String)
    throw
}
finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false `
        -ErrorAction SilentlyContinue
}
