param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PIP_CACHE_DIR = 'D:\ourbrain-cache\pip'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$manifest = Join-Path $project 'artifacts\manifest.csv'
$orchestrationDir = Join-Path $project 'runs\v0.2-dev-positive-only-ab'
$taskName = 'OurBrain-V02-Dev-Positive-AB'
$sourceCheckpoint = Join-Path $project 'runs\v0-positive-only\checkpoint'

$experiments = @(
    [ordered]@{
        id = 'v0.2-dev-a-augmentation-positive-only'
        config = Join-Path $project (
            'configs\v0_2_dev_a_augmentation_positive_only_cuda.yaml'
        )
        run_dir = Join-Path $project (
            'runs\v0.2-dev-a-augmentation-positive-only'
        )
    },
    [ordered]@{
        id = 'v0.2-dev-b-augmentation-recall-positive-only'
        config = Join-Path $project (
            'configs\v0_2_dev_b_augmentation_recall_positive_only_cuda.yaml'
        )
        run_dir = Join-Path $project (
            'runs\v0.2-dev-b-augmentation-recall-positive-only'
        )
    }
)

function Write-AtomicText {
    param([string]$Path, [string]$Value)
    $temporary = "$Path.tmp"
    Set-Content -Path $temporary -Value $Value -Encoding utf8
    Move-Item -Path $temporary -Destination $Path -Force
}

function Get-Hash {
    param([string]$Path)
    return (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLowerInvariant()
}

function Invoke-Preflight {
    foreach ($path in @(
        $python,
        $manifest,
        (Join-Path $sourceCheckpoint 'model.safetensors')
    )) {
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "Development A/B prerequisite is missing: $path"
        }
    }
    foreach ($experiment in $experiments) {
        if (-not (Test-Path $experiment.config -PathType Leaf)) {
            throw "Development experiment config is missing: $($experiment.config)"
        }
    }

    $results = @()
    foreach ($experiment in $experiments) {
        $arguments = @(
            '-m', 'ourbrain_cv.cli', 'training-preflight',
            '--config', $experiment.config,
            '--manifest', $manifest,
            '--allow-positive-only',
            '--require-local-checkpoint',
            '--device', 'cuda'
        )
        if ($results.Count -eq 0) {
            $arguments += '--verify-files'
        }
        $output = & $python @arguments 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw ($output | Out-String)
        }
        $result = $output | Out-String | ConvertFrom-Json
        if (
            $result.status -ne 'ready' -or
            $result.manifest.positive_only_override -ne $true -or
            $result.manifest.group_leakage_count -ne 0
        ) {
            throw "Development positive-only preflight contract failed."
        }
        $negativeCounts = $result.manifest.reviewed_negative_counts
        if (
            $negativeCounts.train -ne 0 -or
            $negativeCounts.val -ne 0 -or
            $negativeCounts.test -ne 0
        ) {
            throw (
                'Development A/B is only for the pre-negative dataset; ' +
                'reviewed negatives are already present.'
            )
        }
        $results += $result
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
    $checkpointDir = Join-Path $runDir 'checkpoint'
    $checkpoint = Join-Path $checkpointDir 'model.safetensors'
    $lastCheckpoint = Join-Path $checkpointDir 'last\model.safetensors'
    $historyPath = Join-Path $checkpointDir 'history.json'
    $trainingConfigPath = Join-Path $checkpointDir 'training_config.json'
    $finishedAtPath = Join-Path $runDir 'finished_at.txt'
    $exitFile = Join-Path $runDir 'exit_code.txt'
    $completionPath = Join-Path $runDir 'completion.json'

    if (Test-Path $exitFile -PathType Leaf) {
        $existingCode = [int](Get-Content $exitFile)
        if (
            $existingCode -eq 0 -and
            (Test-Path $checkpoint -PathType Leaf) -and
            (Test-Path $lastCheckpoint -PathType Leaf) -and
            (Test-Path $completionPath -PathType Leaf)
        ) {
            $existing = Get-Content $completionPath -Raw | ConvertFrom-Json
            if (
                $existing.schema_version -ne 1 -or
                $existing.status -ne 'development-training-complete' -or
                $existing.run_id -ne $Experiment.id -or
                $existing.production_eligible -ne $false
            ) {
                throw "Completed development record is invalid: $completionPath"
            }
            $expectedHashes = [ordered]@{
                best_checkpoint_sha256 = Get-Hash $checkpoint
                last_checkpoint_sha256 = Get-Hash $lastCheckpoint
                history_sha256 = Get-Hash $historyPath
                training_config_sha256 = Get-Hash $trainingConfigPath
                config_sha256 = Get-Hash $Experiment.config
                manifest_sha256 = Get-Hash $manifest
            }
            foreach ($field in $expectedHashes.Keys) {
                if ($existing.$field -ne $expectedHashes[$field]) {
                    throw "Completed development artifact changed: $($Experiment.id) $field"
                }
            }
            Write-Host "Skipping completed development experiment: $($Experiment.id)"
            return $existing
        }
        throw "Development experiment is not safely resumable: $runDir"
    }
    $requiredOutputs = @(
        $checkpoint,
        $lastCheckpoint,
        $historyPath,
        $trainingConfigPath
    )
    $recoverable = (
        (Test-Path $runDir) -and
        (Test-Path $finishedAtPath -PathType Leaf) -and
        @(
            $requiredOutputs |
                Where-Object { Test-Path $_ -PathType Leaf }
        ).Count -eq $requiredOutputs.Count
    )
    if (Test-Path $runDir) {
        if (-not $recoverable) {
            throw "Incomplete development run exists; refusing overwrite: $runDir"
        }
        Write-Host (
            'Recovering completion metadata without retraining: ' +
            $Experiment.id
        )
    }
    else {
        New-Item -ItemType Directory -Force -Path $runDir | Out-Null
        $stdout = Join-Path $runDir 'stdout.log'
        $stderr = Join-Path $runDir 'stderr.log'
        $startedAt = [DateTime]::UtcNow.ToString('o')
        Write-AtomicText -Path (Join-Path $runDir 'started_at.txt') `
            -Value $startedAt

        $metadata = [ordered]@{
            schema_version = 1
            run_id = $Experiment.id
            mode = 'development-positive-only-augmentation-ab'
            development_only = $true
            production_eligible = $false
            held_out_test_opened = $false
            config = $Experiment.config
            config_sha256 = Get-Hash $Experiment.config
            manifest = $manifest
            manifest_sha256 = Get-Hash $manifest
            reviewed_negative_rows = 0
            source_checkpoint = $sourceCheckpoint
            source_checkpoint_model_sha256 = Get-Hash (
                Join-Path $sourceCheckpoint 'model.safetensors'
            )
            device = 'cuda'
            started_at_utc = $startedAt
        }
        Write-AtomicText -Path (Join-Path $runDir 'run_metadata.json') `
            -Value ($metadata | ConvertTo-Json -Depth 5)

        $arguments = @(
            '-m', 'ourbrain_cv.cli', 'train',
            '--config', $Experiment.config,
            '--manifest', $manifest,
            '--allow-positive-only',
            '--device', 'cuda'
        )
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $project `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -NoNewWindow -Wait -PassThru
        $code = $process.ExitCode
        Write-AtomicText -Path $finishedAtPath `
            -Value ([DateTime]::UtcNow.ToString('o'))
        if ($code -ne 0) {
            Write-AtomicText -Path $exitFile -Value ([string]$code)
            throw (
                "Development experiment failed with exit ${code}: " +
                $Experiment.id
            )
        }
        $missingOutputs = @(
            $requiredOutputs |
                Where-Object { -not (Test-Path $_ -PathType Leaf) }
        )
        if ($missingOutputs.Count -gt 0) {
            Write-AtomicText -Path $exitFile -Value '2'
            throw (
                'Development experiment returned success without artifact(s): ' +
                ($missingOutputs -join ', ')
            )
        }
    }

    $historyPayload = Get-Content $historyPath -Raw | ConvertFrom-Json
    $history = @()
    foreach ($entry in $historyPayload) {
        $history += $entry
    }
    if ($history.Count -eq 0) {
        Write-AtomicText -Path $exitFile -Value '2'
        throw "Development experiment history is empty: $($Experiment.id)"
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
        status = 'development-training-complete'
        development_only = $true
        production_eligible = $false
        positive_only = $true
        held_out_test_opened = $false
        exit_code = 0
        config = $Experiment.config
        config_sha256 = Get-Hash $Experiment.config
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        reviewed_negative_rows = 0
        maximum_epochs = 15
        completed_epochs = $history.Count
        early_stopped = ($history.Count -lt 15)
        best_epoch = [int]$best.epoch
        best_val_crack_dice = [double]$best.val_crack_dice
        best_val_recall = [double]$best.val_recall
        best_val_precision = [double]$best.val_precision
        best_val_boundary_f1 = [double]$best.val_boundary_f1
        best_checkpoint = $checkpoint
        best_checkpoint_sha256 = Get-Hash $checkpoint
        last_checkpoint = $lastCheckpoint
        last_checkpoint_sha256 = Get-Hash $lastCheckpoint
        history = $historyPath
        history_sha256 = Get-Hash $historyPath
        training_config = $trainingConfigPath
        training_config_sha256 = Get-Hash $trainingConfigPath
        limitation = (
            'No human-reviewed normal images were used. This run cannot ' +
            'estimate operational false-positive performance.'
        )
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
        status = 'development-training-complete'
        development_only = $true
        production_eligible = $false
        positive_only = $true
        held_out_test_opened = $false
        experiments = $completionRecords
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        next_step = (
            'Compare both checkpoints on validation only. Do not open the ' +
            'held-out test before reviewed negatives are available.'
        )
        finished_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path (
        Join-Path $orchestrationDir 'training_complete.json'
    ) -Value ($summary | ConvertTo-Json -Depth 6)
}
catch {
    New-Item -ItemType Directory -Force -Path $orchestrationDir | Out-Null
    Write-AtomicText -Path (
        Join-Path $orchestrationDir 'orchestration_error.txt'
    ) -Value ($_ | Out-String)
    throw
}
finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false `
        -ErrorAction SilentlyContinue
}
