param([switch]$LaunchTraining)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$reviewDir = Join-Path $project 'data\negative_review'
$reviewCsv = Join-Path $reviewDir 'negative_review_reviewed.csv'
$baseManifest = Join-Path $project 'artifacts\manifest.csv'
$outputManifest = Join-Path $project 'artifacts\manifest_with_negatives.csv'
$auditPath = Join-Path $project 'artifacts\manifest_with_negatives.review.json'
$importRecord = Join-Path $project 'artifacts\v0.2_review_import.json'
$launcher = Join-Path $project 'scripts\windows\launch_v0_2_ab.ps1'
$configs = @(
    (Join-Path $project 'configs\v0_2_a_baseline_with_negatives_cuda.yaml'),
    (Join-Path $project 'configs\v0_2_b_recall_with_negatives_cuda.yaml')
)

function Write-AtomicText {
    param([string]$Path, [string]$Value)
    $temporary = "$Path.tmp"
    Set-Content -Path $temporary -Value $Value -Encoding utf8
    Move-Item -Path $temporary -Destination $Path -Force
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

if (-not (Test-Path $python)) {
    throw "Python environment does not exist: $python"
}
foreach ($path in @($reviewCsv, $baseManifest)) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Required import file is missing: $path"
    }
}
foreach ($config in $configs) {
    if (-not (Test-Path $config -PathType Leaf)) {
        throw "CUDA experiment config is missing: $config"
    }
}

$reviewRows = @(Import-Csv -Path $reviewCsv)
if ($reviewRows.Count -ne 200) {
    throw "Reviewed CSV must contain exactly 200 candidates; found $($reviewRows.Count)."
}
$unreviewed = @($reviewRows | Where-Object { -not $_.review_label.Trim() })
if ($unreviewed.Count -gt 0) {
    throw "Reviewed CSV still contains $($unreviewed.Count) blank decision(s)."
}
$candidateFiles = @(
    Get-ChildItem -Path $reviewDir -Filter '*_neg_*.png' -File
)
if ($candidateFiles.Count -ne 200) {
    throw "Windows review directory must contain 200 candidate PNG files; found $($candidateFiles.Count)."
}

$importOutput = Invoke-OurBrain -Arguments @(
    '-m', 'ourbrain_cv.cli', 'import-negatives',
    '--review', $reviewCsv,
    '--manifest', $baseManifest,
    '--output', $outputManifest
)
foreach ($path in @($outputManifest, $auditPath)) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Strict import returned success without artifact: $path"
    }
}

$preflights = @()
foreach ($config in $configs) {
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'training-preflight',
        '--config', $config,
        '--manifest', $outputManifest,
        '--require-local-checkpoint',
        '--device', 'cuda'
    )
    if ($preflights.Count -eq 0) {
        $arguments += '--verify-files'
    }
    $preflights += Invoke-OurBrain -Arguments $arguments
}

$record = [ordered]@{
    schema_version = 1
    status = 'strict-import-complete'
    review_csv = $reviewCsv
    review_csv_sha256 = (
        Get-FileHash -Algorithm SHA256 $reviewCsv
    ).Hash.ToLowerInvariant()
    base_manifest = $baseManifest
    base_manifest_sha256 = (
        Get-FileHash -Algorithm SHA256 $baseManifest
    ).Hash.ToLowerInvariant()
    output_manifest = $outputManifest
    output_manifest_sha256 = (
        Get-FileHash -Algorithm SHA256 $outputManifest
    ).Hash.ToLowerInvariant()
    review_audit = $auditPath
    review_audit_sha256 = (
        Get-FileHash -Algorithm SHA256 $auditPath
    ).Hash.ToLowerInvariant()
    candidate_png_count = $candidateFiles.Count
    imported_at_utc = [DateTime]::UtcNow.ToString('o')
}
Write-AtomicText -Path $importRecord -Value ($record | ConvertTo-Json -Depth 4)

Write-Output $importOutput
$preflights | Write-Output
Write-Output "Strict Windows import passed: $outputManifest"

if ($LaunchTraining) {
    if (-not (Test-Path $launcher)) {
        throw "A/B launcher is missing: $launcher"
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $launcher
    if ($LASTEXITCODE -ne 0) {
        throw "A/B launcher returned exit $LASTEXITCODE."
    }
}
