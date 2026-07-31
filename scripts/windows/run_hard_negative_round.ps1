param(
    [Parameter(Mandatory = $true)]
    [string]$RoundId,
    [switch]$PreflightOnly
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PYTHONUNBUFFERED = '1'

if ($RoundId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') {
    throw 'RoundId must contain only letters, numbers, dot, underscore, or hyphen.'
}

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$roundRoot = Join-Path $project "artifacts\hard-negative-rounds\$RoundId"
$importRecordPath = Join-Path $roundRoot 'import_complete.json'
$manifest = Join-Path $roundRoot 'manifest.csv'
$audit = Join-Path $roundRoot 'manifest.review.json'
$runDir = Join-Path $project "runs\hard-negative-$RoundId"
$checkpointDir = Join-Path $runDir 'checkpoint'
$completionPath = Join-Path $runDir 'completion.json'
$exitFile = Join-Path $runDir 'exit_code.txt'
$taskName = "OurBrain-HN-$RoundId"

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

function Assert-ImportReady {
    foreach ($path in @($python, $importRecordPath, $manifest, $audit)) {
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "Hard-negative round import artifact is missing: $path"
        }
    }
    $record = Get-Content $importRecordPath -Raw | ConvertFrom-Json
    if (
        $record.schema_version -ne 1 -or
        $record.status -ne 'hard-negative-import-complete' -or
        $record.round_id -ne $RoundId -or
        $record.output_manifest -ne $manifest -or
        $record.output_manifest_sha256 -ne (Get-Hash $manifest) -or
        $record.review_audit -ne $audit -or
        $record.review_audit_sha256 -ne (Get-Hash $audit) -or
        -not (Test-Path $record.review_csv -PathType Leaf) -or
        $record.review_csv_sha256 -ne (Get-Hash $record.review_csv) -or
        -not (Test-Path $record.crop_metadata -PathType Leaf) -or
        $record.crop_metadata_sha256 -ne (Get-Hash $record.crop_metadata) -or
        -not (Test-Path $record.base_manifest -PathType Leaf) -or
        $record.base_manifest_sha256 -ne (Get-Hash $record.base_manifest) -or
        -not (Test-Path $record.source_model_selection -PathType Leaf) -or
        $record.source_model_selection_sha256 -ne (
            Get-Hash $record.source_model_selection
        ) -or
        -not (Test-Path $record.training_config -PathType Leaf) -or
        $record.training_config_sha256 -ne (
            Get-Hash $record.training_config
        ) -or
        -not (Test-Path $record.source_checkpoint) -or
        -not (Test-Path $record.source_checkpoint_model -PathType Leaf) -or
        $record.source_checkpoint_model_sha256 -ne (
            Get-Hash $record.source_checkpoint_model
        ) -or
        -not (Test-Path $record.source_calibration -PathType Leaf) -or
        $record.source_calibration_sha256 -ne (
            Get-Hash $record.source_calibration
        )
    ) {
        throw 'Hard-negative round import provenance is invalid or changed.'
    }
    $metadata = Get-Content $record.crop_metadata -Raw | ConvertFrom-Json
    foreach ($property in $metadata.crop_sha256.PSObject.Properties) {
        $cropPath = Join-Path (
            Split-Path $record.review_csv -Parent
        ) $property.Name
        if (
            -not (Test-Path $cropPath -PathType Leaf) -or
            (Get-Hash $cropPath) -ne [string]$property.Value
        ) {
            throw "Hard-negative crop changed after import: $cropPath"
        }
    }
    return $record
}

function Invoke-Preflight {
    param($Record)
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'training-preflight',
        '--config', $Record.training_config,
        '--manifest', $manifest,
        '--model-checkpoint', $Record.source_checkpoint,
        '--output-dir', $checkpointDir,
        '--require-local-checkpoint',
        '--verify-files',
        '--device', 'cuda'
    )
    $result = & $python @arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ($result | Out-String)
    }
    return ($result | Out-String)
}

try {
    $record = Assert-ImportReady
    $preflight = Invoke-Preflight -Record $record
    if ($PreflightOnly) {
        Write-Output $preflight
        exit 0
    }
    if (Test-Path $runDir) {
        throw "Hard-negative training run already exists; refusing overwrite: $runDir"
    }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null
    Write-AtomicText -Path (Join-Path $runDir 'preflight.json') `
        -Value $preflight
    Write-AtomicText -Path (Join-Path $runDir 'started_at.txt') `
        -Value ([DateTime]::UtcNow.ToString('o'))
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'train',
        '--config', $record.training_config,
        '--manifest', $manifest,
        '--model-checkpoint', $record.source_checkpoint,
        '--output-dir', $checkpointDir,
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $project `
        -RedirectStandardOutput (Join-Path $runDir 'stdout.log') `
        -RedirectStandardError (Join-Path $runDir 'stderr.log') `
        -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        Write-AtomicText -Path $exitFile -Value ([string]$process.ExitCode)
        throw "Hard-negative training failed with exit $($process.ExitCode)."
    }
    $best = Join-Path $checkpointDir 'model.safetensors'
    $last = Join-Path $checkpointDir 'last\model.safetensors'
    $historyPath = Join-Path $checkpointDir 'history.json'
    $trainingConfigPath = Join-Path $checkpointDir 'training_config.json'
    foreach ($path in @($best, $last, $historyPath, $trainingConfigPath)) {
        if (-not (Test-Path $path -PathType Leaf)) {
            Write-AtomicText -Path $exitFile -Value '2'
            throw "Training returned success without required artifact: $path"
        }
    }
    $historyPayload = Get-Content $historyPath -Raw | ConvertFrom-Json
    $history = @()
    foreach ($entry in $historyPayload) {
        $history += $entry
    }
    if ($history.Count -eq 0) {
        Write-AtomicText -Path $exitFile -Value '2'
        throw 'Hard-negative training history is empty.'
    }
    $bestEpoch = $history |
        Sort-Object -Property @{
            Expression = { [double]$_.val_crack_dice }
            Descending = $true
        } |
        Select-Object -First 1
    $completion = [ordered]@{
        schema_version = 1
        status = 'training-complete'
        round_id = $RoundId
        exit_code = 0
        import_record = $importRecordPath
        import_record_sha256 = Get-Hash $importRecordPath
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        review_audit = $audit
        review_audit_sha256 = Get-Hash $audit
        config = $record.training_config
        config_sha256 = Get-Hash $record.training_config
        source_model_id = $record.source_model_id
        source_checkpoint = $record.source_checkpoint
        source_checkpoint_model_sha256 = (
            $record.source_checkpoint_model_sha256
        )
        output_checkpoint = $checkpointDir
        best_checkpoint = $best
        best_checkpoint_sha256 = Get-Hash $best
        last_checkpoint = $last
        last_checkpoint_sha256 = Get-Hash $last
        history = $historyPath
        history_sha256 = Get-Hash $historyPath
        training_config = $trainingConfigPath
        training_config_sha256 = Get-Hash $trainingConfigPath
        completed_epochs = $history.Count
        best_epoch = [int]$bestEpoch.epoch
        best_val_crack_dice = [double]$bestEpoch.val_crack_dice
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 7)
    Write-AtomicText -Path $exitFile -Value '0'
    Write-Output "Hard-negative training completed: $completionPath"
}
catch {
    if (Test-Path $runDir) {
        Write-AtomicText -Path (Join-Path $runDir 'training_error.txt') `
            -Value ($_ | Out-String)
    }
    throw
}
finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false `
        -ErrorAction SilentlyContinue
}
