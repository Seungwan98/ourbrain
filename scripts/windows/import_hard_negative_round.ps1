param(
    [Parameter(Mandatory = $true)]
    [string]$RoundId,
    [string]$ReviewCsv,
    [string]$BaseManifest
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUNBUFFERED = '1'

if ($RoundId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') {
    throw 'RoundId must contain only letters, numbers, dot, underscore, or hyphen.'
}

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$roundRoot = Join-Path $project "artifacts\hard-negative-rounds\$RoundId"
$manifest = Join-Path $roundRoot 'manifest.csv'
$audit = Join-Path $roundRoot 'manifest.review.json'
$recordPath = Join-Path $roundRoot 'import_complete.json'
$selectionPath = Join-Path $project 'runs\v0.2-ab\model_selection.json'
$evaluationCompletePath = Join-Path $project 'runs\v0.2-ab\evaluation_complete.json'
$trainingOutput = Join-Path $project "runs\hard-negative-$RoundId\checkpoint"
if (-not $ReviewCsv) {
    $ReviewCsv = Join-Path $project (
        "data\pilot_hard_negatives\$RoundId\hard_negative_review.csv"
    )
}
if (-not $BaseManifest) {
    $BaseManifest = Join-Path $project 'artifacts\manifest_with_negatives.csv'
}
$ReviewCsv = [System.IO.Path]::GetFullPath($ReviewCsv)
$BaseManifest = [System.IO.Path]::GetFullPath($BaseManifest)
$metadataPath = Join-Path (Split-Path $ReviewCsv -Parent) 'metadata.json'

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

function Invoke-OurBrain {
    param([string[]]$Arguments)
    Push-Location $project
    try {
        $result = & $python @Arguments 2>&1
        $code = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    if ($code -ne 0) {
        throw ($result | Out-String)
    }
    return ($result | Out-String)
}

if (-not (Test-Path $python -PathType Leaf)) {
    throw "Python environment does not exist: $python"
}
if (Test-Path $roundRoot) {
    throw "Round import directory already exists; refusing to overwrite: $roundRoot"
}
foreach ($path in @(
    $ReviewCsv,
    $BaseManifest,
    $metadataPath,
    $selectionPath,
    $evaluationCompletePath
)) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Required hard-negative round artifact is missing: $path"
    }
}

$metadata = Get-Content $metadataPath -Raw | ConvertFrom-Json
$reviewRows = @(Import-Csv -Path $ReviewCsv)
if (
    $metadata.schema_version -ne 1 -or
    $metadata.status -notin @(
        'human_re_review_required',
        'no_false_positive_crops'
    ) -or
    [int]$metadata.crop_count -ne $reviewRows.Count
) {
    throw "Hard-negative metadata does not match its review CSV: $metadataPath"
}
if ($reviewRows.Count -eq 0) {
    throw 'No false-positive crops exist, so a retraining round is unnecessary.'
}
$blankRows = @($reviewRows | Where-Object { -not $_.review_label.Trim() })
if ($blankRows.Count -gt 0) {
    throw "Second human review still has $($blankRows.Count) blank decision(s)."
}
foreach ($property in $metadata.crop_sha256.PSObject.Properties) {
    $cropPath = Join-Path (Split-Path $ReviewCsv -Parent) $property.Name
    if (
        -not (Test-Path $cropPath -PathType Leaf) -or
        (Get-Hash $cropPath) -ne [string]$property.Value
    ) {
        throw "Hard-negative crop provenance mismatch: $cropPath"
    }
}

$selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
$evaluationComplete = (
    Get-Content $evaluationCompletePath -Raw | ConvertFrom-Json
)
$sourceCalibration = $selection.winner.calibration
$sourceCheckpointModel = Join-Path $selection.winner.checkpoint 'model.safetensors'
if (
    $selection.schema_version -ne 1 -or
    -not $selection.winner -or
    $evaluationComplete.schema_version -ne 1 -or
    $evaluationComplete.status -ne 'evaluation-complete' -or
    $evaluationComplete.artifacts.model_selection_sha256 -ne (
        Get-Hash $selectionPath
    ) -or
    -not (Test-Path $sourceCalibration -PathType Leaf) -or
    $selection.winner.calibration_sha256 -ne (Get-Hash $sourceCalibration) -or
    -not (Test-Path $sourceCheckpointModel -PathType Leaf)
) {
    throw 'P4 model selection/evaluation provenance is incomplete or changed.'
}

try {
    New-Item -ItemType Directory -Force -Path $roundRoot | Out-Null
    $importOutput = Invoke-OurBrain -Arguments @(
        '-m', 'ourbrain_cv.cli', 'import-negatives',
        '--review', $ReviewCsv,
        '--manifest', $BaseManifest,
        '--output', $manifest
    )
    foreach ($path in @($manifest, $audit)) {
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "Hard-negative import returned success without artifact: $path"
        }
    }
    $preflightOutput = Invoke-OurBrain -Arguments @(
        '-m', 'ourbrain_cv.cli', 'training-preflight',
        '--config', $selection.winner.config,
        '--manifest', $manifest,
        '--model-checkpoint', $selection.winner.checkpoint,
        '--output-dir', $trainingOutput,
        '--require-local-checkpoint',
        '--verify-files',
        '--device', 'cuda'
    )
    $record = [ordered]@{
        schema_version = 1
        status = 'hard-negative-import-complete'
        round_id = $RoundId
        review_csv = $ReviewCsv
        review_csv_sha256 = Get-Hash $ReviewCsv
        crop_metadata = $metadataPath
        crop_metadata_sha256 = Get-Hash $metadataPath
        base_manifest = $BaseManifest
        base_manifest_sha256 = Get-Hash $BaseManifest
        output_manifest = $manifest
        output_manifest_sha256 = Get-Hash $manifest
        review_audit = $audit
        review_audit_sha256 = Get-Hash $audit
        source_model_selection = $selectionPath
        source_model_selection_sha256 = Get-Hash $selectionPath
        source_model_id = $selection.winner.id
        source_checkpoint = $selection.winner.checkpoint
        source_checkpoint_model = $sourceCheckpointModel
        source_checkpoint_model_sha256 = Get-Hash $sourceCheckpointModel
        source_calibration = $sourceCalibration
        source_calibration_sha256 = Get-Hash $sourceCalibration
        training_config = $selection.winner.config
        training_config_sha256 = Get-Hash $selection.winner.config
        training_output = $trainingOutput
        crop_count = $reviewRows.Count
        imported_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $recordPath `
        -Value ($record | ConvertTo-Json -Depth 7)
    Write-Output $importOutput
    Write-Output $preflightOutput
    Write-Output "Hard-negative round import passed: $recordPath"
}
catch {
    if (Test-Path $roundRoot) {
        Remove-Item $roundRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    throw
}
