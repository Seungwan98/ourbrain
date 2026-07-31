param(
    [string]$InputImage = (
        'D:\ourbrain-data\performance-benchmark\Tube_009_1.bmp'
    )
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$manifest = Join-Path $project 'artifacts\manifest.csv'
$validationComplete = Join-Path (
    $project
) 'runs\development-benchmark\benchmark_complete.json'
$outputRoot = Join-Path (
    $project
) 'runs\development-benchmark\large-bmp'
$completionPath = Join-Path $outputRoot 'large_bmp_complete.json'

$models = @(
    [ordered]@{
        id = 'v0-positive-only'
        config = Join-Path $project 'configs\v0_positive_only_to_30_cuda.yaml'
        checkpoint = Join-Path $project 'runs\v0-positive-only\checkpoint'
        output = Join-Path $outputRoot 'v0'
    },
    [ordered]@{
        id = 'v0.1-sampling-tversky-cldice'
        config = Join-Path $project 'configs\v0_1_sampling_tversky_cldice_cuda.yaml'
        checkpoint = Join-Path $project (
            'runs\v0.1-sampling-tversky-cldice\checkpoint'
        )
        output = Join-Path $outputRoot 'v0.1'
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
    Write-Output "Large BMP benchmark already complete: $completionPath"
    exit 0
}
if (Test-Path $outputRoot) {
    throw "Incomplete large BMP benchmark exists; refusing overwrite: $outputRoot"
}
foreach ($path in @(
    $python,
    $manifest,
    $inputImage,
    $validationComplete
)) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "Large BMP benchmark prerequisite is missing: $path"
    }
}

try {
    New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
    $records = @()
    foreach ($model in $models) {
        New-Item -ItemType Directory -Force -Path $model.output | Out-Null
        $stdout = Join-Path $model.output 'stdout.log'
        $stderr = Join-Path $model.output 'stderr.log'
        $memmap = Join-Path $model.output '.memmap'
        $arguments = @(
            '-m', 'ourbrain_cv.cli', 'infer',
            '--config', $model.config,
            '--manifest', $manifest,
            '--checkpoint', $model.checkpoint,
            '--input', $inputImage,
            '--output', $model.output,
            '--threshold', '0.5',
            '--device', 'cuda',
            '--memmap-dir', $memmap
        )
        Write-Output "Running large BMP inference: $($model.id)"
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $project `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -NoNewWindow -Wait -PassThru
        $stopwatch.Stop()
        if ($process.ExitCode -ne 0) {
            throw "Large BMP inference failed with exit $($process.ExitCode): $($model.id)"
        }
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($inputImage)
        $summaryPath = Join-Path $model.output "${stem}_summary.json"
        $probabilityPath = Join-Path $model.output "${stem}_probability.npy"
        $maskPath = Join-Path $model.output "${stem}_mask.png"
        $overlayPath = Join-Path $model.output "${stem}_overlay.png"
        foreach ($path in @(
            $summaryPath,
            $probabilityPath,
            $maskPath,
            $overlayPath
        )) {
            if (-not (Test-Path $path -PathType Leaf)) {
                throw "Large BMP inference artifact is missing: $path"
            }
        }
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        if (
            $summary.image_path -ne $inputImage -or
            [double]$summary.threshold -ne 0.5
        ) {
            throw "Large BMP summary provenance is invalid: $($model.id)"
        }
        $records += [pscustomobject][ordered]@{
            id = $model.id
            config = $model.config
            config_sha256 = Get-Hash $model.config
            checkpoint = $model.checkpoint
            checkpoint_model_sha256 = Get-Hash (
                Join-Path $model.checkpoint 'model.safetensors'
            )
            elapsed_seconds = [math]::Round(
                $stopwatch.Elapsed.TotalSeconds,
                3
            )
            width = [int]$summary.image_size.width
            height = [int]$summary.image_size.height
            megapixels = [math]::Round(
                (
                    [double]$summary.image_size.width *
                    [double]$summary.image_size.height /
                    1000000.0
                ),
                3
            )
            megapixels_per_second = [math]::Round(
                (
                    [double]$summary.image_size.width *
                    [double]$summary.image_size.height /
                    1000000.0 /
                    $stopwatch.Elapsed.TotalSeconds
                ),
                3
            )
            raw_positive_ratio = [double]$summary.raw_positive_ratio
            crack_ratio = [double]$summary.crack_ratio
            presence = $summary.presence
            quality_gate_passed = $summary.quality_gate.passed
            postprocessing_applied = $summary.postprocessing_applied
            component_count = $summary.components.count
            summary = $summaryPath
            summary_sha256 = Get-Hash $summaryPath
            probability = $probabilityPath
            probability_sha256 = Get-Hash $probabilityPath
            mask = $maskPath
            mask_sha256 = Get-Hash $maskPath
            overlay = $overlayPath
            overlay_sha256 = Get-Hash $overlayPath
        }
    }
    $completion = [ordered]@{
        schema_version = 1
        status = 'development-large-bmp-benchmark-complete'
        development_only = $true
        production_eligible = $false
        input = $inputImage
        input_sha256 = Get-Hash $inputImage
        input_bytes = (Get-Item $inputImage).Length
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        threshold = 0.5
        models = $records
        limitation = (
            'The BMP has no trusted ground-truth mask; prediction ratios and ' +
            'runtime are smoke diagnostics, not accuracy measurements.'
        )
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 8)
    Write-Output "Large BMP benchmark completed: $completionPath"
}
catch {
    if (Test-Path $outputRoot) {
        Write-AtomicText -Path (Join-Path $outputRoot 'benchmark_error.txt') `
            -Value ($_ | Out-String)
    }
    throw
}
