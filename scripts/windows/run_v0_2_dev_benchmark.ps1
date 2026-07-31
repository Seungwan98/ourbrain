$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$manifest = Join-Path $project 'artifacts\manifest.csv'
$trainingComplete = Join-Path (
    $project
) 'runs\v0.2-dev-positive-only-ab\training_complete.json'
$baselineBenchmark = Join-Path (
    $project
) 'runs\development-benchmark\benchmark_complete.json'
$baselineLargeBmp = Join-Path (
    $project
) 'runs\development-benchmark\large-bmp\large_bmp_complete.json'
$inputImage = 'D:\ourbrain-data\performance-benchmark\Tube_009_1.bmp'
$outputDir = Join-Path $project 'runs\v0.2-dev-benchmark'
$completionPath = Join-Path $outputDir 'benchmark_complete.json'

$experiments = @(
    [ordered]@{
        id = 'v0.2-dev-a-augmentation-positive-only'
        config = Join-Path $project (
            'configs\v0_2_dev_a_augmentation_positive_only_cuda.yaml'
        )
        checkpoint = Join-Path $project (
            'runs\v0.2-dev-a-augmentation-positive-only\checkpoint'
        )
        metrics = Join-Path $outputDir 'dev-a-validation-metrics.json'
        inference = Join-Path $outputDir 'large-bmp\dev-a'
    },
    [ordered]@{
        id = 'v0.2-dev-b-augmentation-recall-positive-only'
        config = Join-Path $project (
            'configs\v0_2_dev_b_augmentation_recall_positive_only_cuda.yaml'
        )
        checkpoint = Join-Path $project (
            'runs\v0.2-dev-b-augmentation-recall-positive-only\checkpoint'
        )
        metrics = Join-Path $outputDir 'dev-b-validation-metrics.json'
        inference = Join-Path $outputDir 'large-bmp\dev-b'
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
    Write-Output "v0.2 development benchmark already complete: $completionPath"
    exit 0
}
if (Test-Path $outputDir) {
    throw "Incomplete v0.2 development benchmark exists: $outputDir"
}
foreach ($path in @(
    $python,
    $manifest,
    $trainingComplete,
    $baselineBenchmark,
    $baselineLargeBmp,
    $inputImage
)) {
    if (-not (Test-Path $path -PathType Leaf)) {
        throw "v0.2 development benchmark prerequisite is missing: $path"
    }
}

$training = Get-Content $trainingComplete -Raw | ConvertFrom-Json
if (
    $training.status -ne 'development-training-complete' -or
    $training.development_only -ne $true -or
    $training.positive_only -ne $true -or
    $training.held_out_test_opened -ne $false
) {
    throw "v0.2 development training completion contract is invalid."
}
$baseline = Get-Content $baselineBenchmark -Raw | ConvertFrom-Json
$baselineLarge = Get-Content $baselineLargeBmp -Raw | ConvertFrom-Json
if (
    $baseline.evaluation_split -ne 'val' -or
    [double]$baseline.threshold -ne 0.5 -or
    $baseline.manifest_sha256 -ne (Get-Hash $manifest) -or
    $baselineLarge.input_sha256 -ne (Get-Hash $inputImage)
) {
    throw "Baseline benchmark provenance does not match this comparison."
}
foreach ($experiment in $experiments) {
    foreach ($path in @(
        $experiment.config,
        (Join-Path $experiment.checkpoint 'model.safetensors')
    )) {
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "v0.2 development model artifact is missing: $path"
        }
    }
}

try {
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $validationRecords = @()
    $largeRecords = @()

    foreach ($experiment in $experiments) {
        $stdout = Join-Path $outputDir "$($experiment.id).evaluate.stdout.log"
        $stderr = Join-Path $outputDir "$($experiment.id).evaluate.stderr.log"
        $arguments = @(
            '-m', 'ourbrain_cv.cli', 'evaluate',
            '--config', $experiment.config,
            '--manifest', $manifest,
            '--checkpoint', $experiment.checkpoint,
            '--split', 'val',
            '--threshold', '0.5',
            '--boundary-tolerance', '2',
            '--output', $experiment.metrics,
            '--device', 'cuda'
        )
        $process = Start-Process -FilePath $python -ArgumentList $arguments `
            -WorkingDirectory $project `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -NoNewWindow -Wait -PassThru
        if ($process.ExitCode -ne 0) {
            throw "Validation benchmark failed: $($experiment.id)"
        }
        $metrics = Get-Content $experiment.metrics -Raw | ConvertFrom-Json
        if (
            $metrics.evaluation_split -ne 'val' -or
            [double]$metrics.threshold -ne 0.5 -or
            $metrics.provenance.manifest_sha256 -ne (Get-Hash $manifest)
        ) {
            throw "Validation metrics provenance is invalid: $($experiment.id)"
        }
        $validationRecords += [pscustomobject][ordered]@{
            id = $experiment.id
            config = $experiment.config
            config_sha256 = Get-Hash $experiment.config
            checkpoint = $experiment.checkpoint
            checkpoint_model_sha256 = Get-Hash (
                Join-Path $experiment.checkpoint 'model.safetensors'
            )
            metrics = $experiment.metrics
            metrics_sha256 = Get-Hash $experiment.metrics
            samples = [int]$metrics.samples
            crack_dice = [double]$metrics.crack_dice
            crack_iou = [double]$metrics.crack_iou
            precision = [double]$metrics.precision
            recall = [double]$metrics.recall
            specificity = [double]$metrics.specificity
            boundary_f1 = [double]$metrics.boundary_f1
            elapsed_seconds = [double]$metrics.elapsed_seconds
            samples_per_second = [double]$metrics.samples_per_second
        }

        New-Item -ItemType Directory -Force -Path $experiment.inference |
            Out-Null
        $inferStdout = Join-Path $experiment.inference 'stdout.log'
        $inferStderr = Join-Path $experiment.inference 'stderr.log'
        $memmap = Join-Path $experiment.inference '.memmap'
        $inferArguments = @(
            '-m', 'ourbrain_cv.cli', 'infer',
            '--config', $experiment.config,
            '--manifest', $manifest,
            '--checkpoint', $experiment.checkpoint,
            '--input', $inputImage,
            '--output', $experiment.inference,
            '--threshold', '0.5',
            '--device', 'cuda',
            '--memmap-dir', $memmap
        )
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $inferProcess = Start-Process -FilePath $python `
            -ArgumentList $inferArguments `
            -WorkingDirectory $project `
            -RedirectStandardOutput $inferStdout `
            -RedirectStandardError $inferStderr `
            -NoNewWindow -Wait -PassThru
        $stopwatch.Stop()
        if ($inferProcess.ExitCode -ne 0) {
            throw "Large BMP inference failed: $($experiment.id)"
        }
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($inputImage)
        $summaryPath = Join-Path (
            $experiment.inference
        ) "${stem}_summary.json"
        $maskPath = Join-Path $experiment.inference "${stem}_mask.png"
        $overlayPath = Join-Path $experiment.inference "${stem}_overlay.png"
        $probabilityPath = Join-Path (
            $experiment.inference
        ) "${stem}_probability.npy"
        foreach ($path in @(
            $summaryPath,
            $maskPath,
            $overlayPath,
            $probabilityPath
        )) {
            if (-not (Test-Path $path -PathType Leaf)) {
                throw "Large BMP artifact is missing: $path"
            }
        }
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $megapixels = (
            [double]$summary.image_size.width *
            [double]$summary.image_size.height /
            1000000.0
        )
        $largeRecords += [pscustomobject][ordered]@{
            id = $experiment.id
            elapsed_seconds = [math]::Round(
                $stopwatch.Elapsed.TotalSeconds,
                3
            )
            megapixels = [math]::Round($megapixels, 3)
            megapixels_per_second = [math]::Round(
                $megapixels / $stopwatch.Elapsed.TotalSeconds,
                3
            )
            crack_ratio = [double]$summary.crack_ratio
            presence = $summary.presence
            quality_gate_passed = $summary.quality_gate.passed
            component_count = [int]$summary.components.count
            summary = $summaryPath
            summary_sha256 = Get-Hash $summaryPath
            probability_sha256 = Get-Hash $probabilityPath
            mask_sha256 = Get-Hash $maskPath
            overlay = $overlayPath
            overlay_sha256 = Get-Hash $overlayPath
        }
    }

    $allValidation = @($baseline.models) + @($validationRecords)
    $winner = $allValidation |
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
        status = 'v0.2-development-benchmark-complete'
        development_only = $true
        production_eligible = $false
        positive_only = $true
        held_out_test_opened = $false
        evaluation_split = 'val'
        threshold = 0.5
        boundary_tolerance = 2
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        reviewed_negative_rows = 0
        source_training_complete = $trainingComplete
        source_training_complete_sha256 = Get-Hash $trainingComplete
        baseline_benchmark = $baselineBenchmark
        baseline_benchmark_sha256 = Get-Hash $baselineBenchmark
        selected_by_positive_validation_crack_dice = $winner.id
        validation_models = $allValidation
        large_bmp = [ordered]@{
            input = $inputImage
            input_sha256 = Get-Hash $inputImage
            baseline = $baselineLarge.models
            development_models = $largeRecords
        }
        limitations = @(
            'No human-reviewed normal images were used.',
            'Image specificity is not an operational false-positive estimate.',
            'The held-out test split was not opened.',
            'This benchmark cannot promote a model to production.'
        )
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 10)
    Write-Output "v0.2 development benchmark completed: $completionPath"
}
catch {
    if (Test-Path $outputDir) {
        Write-AtomicText -Path (Join-Path $outputDir 'benchmark_error.txt') `
            -Value ($_ | Out-String)
    }
    throw
}
