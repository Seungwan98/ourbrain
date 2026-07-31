$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$manifest = Join-Path $project 'artifacts\manifest.csv'
$outputDir = Join-Path $project 'runs\development-benchmark'
$completionPath = Join-Path $outputDir 'benchmark_complete.json'

$models = @(
    [ordered]@{
        id = 'v0-positive-only'
        config = Join-Path $project 'configs\v0_positive_only_to_30_cuda.yaml'
        checkpoint = Join-Path $project 'runs\v0-positive-only\checkpoint'
        metrics = Join-Path $outputDir 'v0-validation-metrics.json'
        stdout = Join-Path $outputDir 'v0.stdout.log'
        stderr = Join-Path $outputDir 'v0.stderr.log'
    },
    [ordered]@{
        id = 'v0.1-sampling-tversky-cldice'
        config = Join-Path $project 'configs\v0_1_sampling_tversky_cldice_cuda.yaml'
        checkpoint = Join-Path $project (
            'runs\v0.1-sampling-tversky-cldice\checkpoint'
        )
        metrics = Join-Path $outputDir 'v0.1-validation-metrics.json'
        stdout = Join-Path $outputDir 'v0.1.stdout.log'
        stderr = Join-Path $outputDir 'v0.1.stderr.log'
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

if (Test-Path $completionPath -PathType Leaf) {
    Write-Output "Development benchmark already complete: $completionPath"
    exit 0
}
if (Test-Path $outputDir) {
    throw "Incomplete benchmark output exists; refusing overwrite: $outputDir"
}
foreach ($path in @($python, $manifest)) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Benchmark prerequisite is missing: $path"
    }
}
foreach ($model in $models) {
    if (-not (Test-Path $model.config -PathType Leaf)) {
        throw "Benchmark config is missing: $($model.config)"
    }
    if (-not (Test-Path (Join-Path $model.checkpoint 'model.safetensors'))) {
        throw "Benchmark checkpoint is missing: $($model.checkpoint)"
    }
}

try {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $records = @()
    foreach ($model in $models) {
        Write-Output "Evaluating $($model.id) on validation..."
        $arguments = @(
            '-m', 'ourbrain_cv.cli', 'evaluate',
            '--config', $model.config,
            '--manifest', $manifest,
            '--checkpoint', $model.checkpoint,
            '--split', 'val',
            '--threshold', '0.5',
            '--boundary-tolerance', '2',
            '--output', $model.metrics,
            '--device', 'cuda'
        )
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $project `
            -RedirectStandardOutput $model.stdout `
            -RedirectStandardError $model.stderr `
            -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "Benchmark failed with exit $($process.ExitCode): $($model.id)"
        }
        if (-not (Test-Path $model.metrics -PathType Leaf)) {
            throw "Benchmark returned success without metrics: $($model.id)"
        }
        $metrics = Get-Content $model.metrics -Raw | ConvertFrom-Json
        if (
            $metrics.schema_version -ne 1 -or
            $metrics.evaluation_split -ne 'val' -or
            [double]$metrics.threshold -ne 0.5 -or
            $metrics.provenance.manifest -ne $manifest -or
            $metrics.provenance.manifest_sha256 -ne (Get-Hash $manifest)
        ) {
            throw "Benchmark metrics provenance is invalid: $($model.id)"
        }
        $records += [pscustomobject][ordered]@{
            id = $model.id
            config = $model.config
            config_sha256 = Get-Hash $model.config
            checkpoint = $model.checkpoint
            checkpoint_sha256 = $metrics.provenance.checkpoint_sha256
            metrics_json = $model.metrics
            metrics_sha256 = Get-Hash $model.metrics
            samples = [int]$metrics.samples
            crack_dice = [double]$metrics.crack_dice
            crack_iou = [double]$metrics.crack_iou
            precision = [double]$metrics.precision
            recall = [double]$metrics.recall
            specificity = [double]$metrics.specificity
            boundary_f1 = [double]$metrics.boundary_f1
            image_level_recall = [double]$metrics.image_level_recall
            image_level_specificity = [double]$metrics.image_level_specificity
            image_false_negatives = [int]$metrics.image_fn
            image_false_positives = [int]$metrics.image_fp
            error_case_count = [int]$metrics.error_case_count
            elapsed_seconds = [double]$metrics.elapsed_seconds
            samples_per_second = [double]$metrics.samples_per_second
        }
    }
    $winner = $records |
        Sort-Object -Property @{
            Expression = { [double]$_.crack_dice }
            Descending = $true
        }, @{
            Expression = { [double]$_.boundary_f1 }
            Descending = $true
        } |
        Select-Object -First 1
    $completion = [ordered]@{
        schema_version = 1
        status = 'development-validation-benchmark-complete'
        development_only = $true
        production_eligible = $false
        evaluation_split = 'val'
        threshold = 0.5
        boundary_tolerance = 2
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        reviewed_negative_rows = 0
        selected_by_crack_dice = $winner.id
        models = $records
        limitations = @(
            'The manifest contains no human-reviewed normal/hard-negative images.',
            'Image specificity is not an operational false-positive estimate.',
            'The held-out test split was not opened by this benchmark.',
            'Final model selection remains blocked until reviewed negatives exist.'
        )
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 8)
    Write-Output "Development benchmark completed: $completionPath"
}
catch {
    if (Test-Path $outputDir) {
        Write-AtomicText -Path (Join-Path $outputDir 'benchmark_error.txt') `
            -Value ($_ | Out-String)
    }
    throw
}
