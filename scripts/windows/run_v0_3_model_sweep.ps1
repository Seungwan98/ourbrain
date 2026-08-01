param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PIP_CACHE_DIR = 'D:\ourbrain-cache\pip'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$dataRoot = 'D:\ourbrain-data\train'
$manifest = Join-Path $project 'artifacts\manifest_windows.csv'
$manifestAudit = Join-Path $project 'artifacts\data_audit_windows.json'
$baselineMetrics = Join-Path (
    $project
) 'runs\development-benchmark\v0-validation-metrics.json'
$inputImage = 'D:\ourbrain-data\performance-benchmark\Tube_009_1.bmp'
$orchestrationDir = Join-Path $project 'runs\v0.3-model-sweep'
$smokeRoot = Join-Path $orchestrationDir 'smoke'
$benchmarkDir = Join-Path $orchestrationDir 'benchmark'
$taskName = 'OurBrain-V03-Model-Sweep'
$b2MemoryLimitBytes = [int64](3.6 * 1024 * 1024 * 1024)

$experiments = @(
    [ordered]@{
        id = 'v0.3-a-upernet-swin-tiny-positive-only'
        name = 'UPerNet Swin-Tiny'
        architecture = 'upernet'
        checkpoint_source = 'openmmlab/upernet-swin-tiny'
        optional = $false
        config = Join-Path $project (
            'configs\v0_3_a_upernet_swin_tiny_positive_only_cuda.yaml'
        )
        run_dir = Join-Path $project (
            'runs\v0.3-a-upernet-swin-tiny-positive-only'
        )
    },
    [ordered]@{
        id = 'v0.3-b-segformer-b1-positive-only'
        name = 'SegFormer-B1'
        architecture = 'segformer'
        checkpoint_source = 'nvidia/segformer-b1-finetuned-ade-512-512'
        optional = $false
        config = Join-Path $project (
            'configs\v0_3_b_segformer_b1_positive_only_cuda.yaml'
        )
        run_dir = Join-Path $project (
            'runs\v0.3-b-segformer-b1-positive-only'
        )
    },
    [ordered]@{
        id = 'v0.3-c-segformer-b2-positive-only'
        name = 'SegFormer-B2'
        architecture = 'segformer'
        checkpoint_source = 'nvidia/segformer-b2-finetuned-ade-512-512'
        optional = $true
        config = Join-Path $project (
            'configs\v0_3_c_segformer_b2_positive_only_cuda.yaml'
        )
        run_dir = Join-Path $project (
            'runs\v0.3-c-segformer-b2-positive-only'
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

function Start-LoggedProcess {
    param(
        [string[]]$Arguments,
        [string]$Stdout,
        [string]$Stderr
    )
    return Start-Process -FilePath $python -ArgumentList $Arguments `
        -WorkingDirectory $project `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -NoNewWindow -Wait -PassThru
}

function Invoke-Preflight {
    foreach ($path in @($python, $baselineMetrics, $inputImage)) {
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "v0.3 prerequisite is missing: $path"
        }
    }
    if (-not (Test-Path $dataRoot -PathType Container)) {
        throw "v0.3 Windows data root is missing: $dataRoot"
    }
    if (-not (Test-Path $manifest -PathType Leaf)) {
        $prepareOutput = Join-Path $orchestrationDir 'prepare_manifest.stdout.log'
        $prepareError = Join-Path $orchestrationDir 'prepare_manifest.stderr.log'
        New-Item -ItemType Directory -Force -Path $orchestrationDir | Out-Null
        $prepareArguments = @(
            '-m', 'ourbrain_cv.cli', 'prepare',
            '--data-root', $dataRoot,
            '--manifest', $manifest,
            '--audit', $manifestAudit,
            '--train-ratio', '0.70',
            '--val-ratio', '0.15',
            '--test-ratio', '0.15',
            '--seed', '42',
            '--mask-threshold', '127'
        )
        $prepareProcess = Start-LoggedProcess -Arguments $prepareArguments `
            -Stdout $prepareOutput -Stderr $prepareError
        if ($prepareProcess.ExitCode -ne 0) {
            throw 'Failed to build the Windows v0.3 manifest.'
        }
    }
    if (
        -not (Test-Path $manifest -PathType Leaf) -or
        -not (Test-Path $manifestAudit -PathType Leaf)
    ) {
        throw 'Windows v0.3 manifest or audit is missing after preparation.'
    }
    $audit = Get-Content $manifestAudit -Raw | ConvertFrom-Json
    if ($audit.paired -ne 1223 -or $audit.groups -ne 89) {
        throw (
            'Windows v0.3 manifest dataset identity changed: paired=' +
            $audit.paired + ', groups=' + $audit.groups
        )
    }
    $baseline = Get-Content $baselineMetrics -Raw | ConvertFrom-Json
    if ($baseline.provenance.manifest_sha256 -ne (Get-Hash $manifest)) {
        throw (
            'Windows manifest does not match the manifest used for the v0 ' +
            'validation baseline.'
        )
    }
    foreach ($experiment in $experiments) {
        if (-not (Test-Path $experiment.config -PathType Leaf)) {
            throw "v0.3 config is missing: $($experiment.config)"
        }
    }

    $results = @()
    foreach ($experiment in $experiments) {
        $arguments = @(
            '-m', 'ourbrain_cv.cli', 'training-preflight',
            '--config', $experiment.config,
            '--manifest', $manifest,
            '--allow-positive-only',
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
            throw "v0.3 positive-only preflight contract failed."
        }
        $negativeCounts = $result.manifest.reviewed_negative_counts
        if (
            $negativeCounts.train -ne 0 -or
            $negativeCounts.val -ne 0 -or
            $negativeCounts.test -ne 0
        ) {
            throw 'v0.3 development sweep must not consume reviewed negatives.'
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

function Invoke-Smoke {
    param([System.Collections.IDictionary]$Experiment)

    $runDir = Join-Path $smokeRoot $Experiment.id
    $completionPath = Join-Path $runDir 'completion.json'
    if (Test-Path $completionPath -PathType Leaf) {
        $existing = Get-Content $completionPath -Raw | ConvertFrom-Json
        if (
            $existing.schema_version -ne 1 -or
            $existing.config_sha256 -ne (Get-Hash $Experiment.config)
        ) {
            throw "v0.3 smoke provenance mismatch: $($Experiment.id)"
        }
        return $existing
    }
    if (Test-Path $runDir) {
        throw "Incomplete v0.3 smoke run exists: $runDir"
    }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $checkpointDir = Join-Path $runDir 'checkpoint'
    $stdout = Join-Path $runDir 'stdout.log'
    $stderr = Join-Path $runDir 'stderr.log'
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'train',
        '--config', $Experiment.config,
        '--manifest', $manifest,
        '--output-dir', $checkpointDir,
        '--epochs', '1',
        '--freeze-backbone-epochs', '0',
        '--max-train-samples', '10',
        '--max-val-samples', '4',
        '--allow-positive-only',
        '--device', 'cuda'
    )
    $startedAt = [DateTime]::UtcNow.ToString('o')
    Write-AtomicText -Path (Join-Path $runDir 'started_at.txt') -Value $startedAt
    $process = Start-LoggedProcess -Arguments $arguments `
        -Stdout $stdout -Stderr $stderr

    if ($process.ExitCode -ne 0) {
        $errorText = Get-Content $stderr -Raw -ErrorAction SilentlyContinue
        if ($Experiment.optional -and $errorText -match '(?i)out of memory') {
            $completion = [ordered]@{
                schema_version = 1
                id = $Experiment.id
                status = 'skipped-vram'
                full_training_eligible = $false
                reason = 'CUDA out of memory during unfrozen 10-step smoke run'
                config = $Experiment.config
                config_sha256 = Get-Hash $Experiment.config
                started_at_utc = $startedAt
                finished_at_utc = [DateTime]::UtcNow.ToString('o')
            }
            Write-AtomicText -Path $completionPath `
                -Value ($completion | ConvertTo-Json -Depth 6)
            return [pscustomobject]$completion
        }
        throw "v0.3 smoke run failed: $($Experiment.id)"
    }

    $result = Get-Content $stdout -Raw | ConvertFrom-Json
    $peakMemory = [int64]$result.peak_cuda_memory_bytes
    $eligible = $true
    $status = 'smoke-passed'
    $reason = $null
    if ($Experiment.optional -and $peakMemory -gt $b2MemoryLimitBytes) {
        $eligible = $false
        $status = 'skipped-vram'
        $reason = 'Peak allocated CUDA memory exceeded the 3.6 GiB B2 gate'
    }
    $modelWeight = Join-Path $checkpointDir 'model.safetensors'
    if (-not (Test-Path $modelWeight -PathType Leaf)) {
        throw "v0.3 smoke checkpoint is missing: $modelWeight"
    }
    $completion = [ordered]@{
        schema_version = 1
        id = $Experiment.id
        status = $status
        full_training_eligible = $eligible
        reason = $reason
        architecture = $Experiment.architecture
        checkpoint_source = $Experiment.checkpoint_source
        config = $Experiment.config
        config_sha256 = Get-Hash $Experiment.config
        samples = [ordered]@{ train = 10; validation = 4 }
        epochs = 1
        backbone_frozen = $false
        peak_cuda_memory_bytes = $peakMemory
        peak_cuda_memory_gib = [math]::Round(
            $peakMemory / 1GB,
            3
        )
        checkpoint_model_sha256 = Get-Hash $modelWeight
        started_at_utc = $startedAt
        finished_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 8)
    return [pscustomobject]$completion
}

function Invoke-FullTraining {
    param([System.Collections.IDictionary]$Experiment)

    $runDir = [string]$Experiment.run_dir
    $completionPath = Join-Path $runDir 'completion.json'
    $checkpointDir = Join-Path $runDir 'checkpoint'
    $modelWeight = Join-Path $checkpointDir 'model.safetensors'
    $lastWeight = Join-Path $checkpointDir 'last\model.safetensors'
    $historyPath = Join-Path $checkpointDir 'history.json'
    $trainingConfigPath = Join-Path $checkpointDir 'training_config.json'
    $modelConfigPath = Join-Path $checkpointDir 'model_config.json'

    if (Test-Path $completionPath -PathType Leaf) {
        $existing = Get-Content $completionPath -Raw | ConvertFrom-Json
        if (
            $existing.status -ne 'development-training-complete' -or
            $existing.config_sha256 -ne (Get-Hash $Experiment.config) -or
            $existing.checkpoint_model_sha256 -ne (Get-Hash $modelWeight)
        ) {
            throw "Completed v0.3 training provenance mismatch: $($Experiment.id)"
        }
        return $existing
    }
    if (Test-Path $runDir) {
        throw "Incomplete v0.3 full training run exists: $runDir"
    }
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $stdout = Join-Path $runDir 'stdout.log'
    $stderr = Join-Path $runDir 'stderr.log'
    $startedAt = [DateTime]::UtcNow.ToString('o')
    Write-AtomicText -Path (Join-Path $runDir 'started_at.txt') -Value $startedAt
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'train',
        '--config', $Experiment.config,
        '--manifest', $manifest,
        '--allow-positive-only',
        '--device', 'cuda'
    )
    $process = Start-LoggedProcess -Arguments $arguments `
        -Stdout $stdout -Stderr $stderr
    Write-AtomicText -Path (Join-Path $runDir 'exit_code.txt') `
        -Value ([string]$process.ExitCode)
    if ($process.ExitCode -ne 0) {
        throw "v0.3 full training failed: $($Experiment.id)"
    }
    foreach ($path in @(
        $modelWeight,
        $lastWeight,
        $historyPath,
        $trainingConfigPath,
        $modelConfigPath
    )) {
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "v0.3 training artifact is missing: $path"
        }
    }

    $result = Get-Content $stdout -Raw | ConvertFrom-Json
    $history = @(Get-Content $historyPath -Raw | ConvertFrom-Json)
    $best = $history | Sort-Object -Property @{
        Expression = { [double]$_.val_crack_dice }
        Descending = $true
    } | Select-Object -First 1
    $completion = [ordered]@{
        schema_version = 1
        id = $Experiment.id
        status = 'development-training-complete'
        development_only = $true
        production_eligible = $false
        positive_only = $true
        held_out_test_opened = $false
        architecture = $Experiment.architecture
        checkpoint_source = $Experiment.checkpoint_source
        config = $Experiment.config
        config_sha256 = Get-Hash $Experiment.config
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        manifest_audit = $manifestAudit
        manifest_audit_sha256 = Get-Hash $manifestAudit
        maximum_epochs = 30
        completed_epochs = $history.Count
        early_stopped = ($history.Count -lt 30)
        best_epoch = [int]$best.epoch
        best_val_crack_dice = [double]$best.val_crack_dice
        best_val_recall = [double]$best.val_recall
        best_val_boundary_f1 = [double]$best.val_boundary_f1
        peak_cuda_memory_bytes = [int64]$result.peak_cuda_memory_bytes
        checkpoint = $checkpointDir
        checkpoint_model_sha256 = Get-Hash $modelWeight
        last_checkpoint_model_sha256 = Get-Hash $lastWeight
        history_sha256 = Get-Hash $historyPath
        training_config_sha256 = Get-Hash $trainingConfigPath
        model_config_sha256 = Get-Hash $modelConfigPath
        started_at_utc = $startedAt
        finished_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 8)
    return [pscustomobject]$completion
}

function Invoke-Benchmark {
    param([object[]]$TrainingRecords)

    $completionPath = Join-Path $benchmarkDir 'benchmark_complete.json'
    if (Test-Path $completionPath -PathType Leaf) {
        return Get-Content $completionPath -Raw | ConvertFrom-Json
    }
    if (Test-Path $benchmarkDir) {
        throw "Incomplete v0.3 benchmark exists: $benchmarkDir"
    }
    New-Item -ItemType Directory -Force -Path $benchmarkDir | Out-Null

    $validationRecords = @()
    $largeRecords = @()
    foreach ($training in $TrainingRecords) {
        $experiment = $experiments | Where-Object { $_.id -eq $training.id } |
            Select-Object -First 1
        $metricsPath = Join-Path $benchmarkDir "$($experiment.id)-validation.json"
        $evalStdout = Join-Path $benchmarkDir "$($experiment.id)-evaluate.stdout.log"
        $evalStderr = Join-Path $benchmarkDir "$($experiment.id)-evaluate.stderr.log"
        $evalArguments = @(
            '-m', 'ourbrain_cv.cli', 'evaluate',
            '--config', $experiment.config,
            '--manifest', $manifest,
            '--checkpoint', $training.checkpoint,
            '--split', 'val',
            '--threshold', '0.5',
            '--boundary-tolerance', '2',
            '--output', $metricsPath,
            '--device', 'cuda'
        )
        $evalProcess = Start-LoggedProcess -Arguments $evalArguments `
            -Stdout $evalStdout -Stderr $evalStderr
        if ($evalProcess.ExitCode -ne 0) {
            throw "v0.3 validation failed: $($experiment.id)"
        }
        $metrics = Get-Content $metricsPath -Raw | ConvertFrom-Json
        if (
            $metrics.evaluation_split -ne 'val' -or
            [double]$metrics.threshold -ne 0.5 -or
            $metrics.provenance.manifest_sha256 -ne (Get-Hash $manifest)
        ) {
            throw "v0.3 validation provenance mismatch: $($experiment.id)"
        }
        $validationRecords += [pscustomobject][ordered]@{
            id = $experiment.id
            architecture = $experiment.architecture
            checkpoint_source = $experiment.checkpoint_source
            checkpoint = $training.checkpoint
            checkpoint_model_sha256 = $training.checkpoint_model_sha256
            config = $experiment.config
            config_sha256 = Get-Hash $experiment.config
            metrics = $metricsPath
            metrics_sha256 = Get-Hash $metricsPath
            samples = [int]$metrics.samples
            crack_dice = [double]$metrics.crack_dice
            crack_iou = [double]$metrics.crack_iou
            precision = [double]$metrics.precision
            recall = [double]$metrics.recall
            boundary_f1 = [double]$metrics.boundary_f1
            samples_per_second = [double]$metrics.samples_per_second
            peak_training_cuda_memory_bytes = [int64]$training.peak_cuda_memory_bytes
        }

        $inferenceDir = Join-Path $benchmarkDir "large-bmp\$($experiment.id)"
        $inferStdout = Join-Path $benchmarkDir "$($experiment.id)-infer.stdout.log"
        $inferStderr = Join-Path $benchmarkDir "$($experiment.id)-infer.stderr.log"
        $inferArguments = @(
            '-m', 'ourbrain_cv.cli', 'infer',
            '--config', $experiment.config,
            '--manifest', $manifest,
            '--checkpoint', $training.checkpoint,
            '--input', $inputImage,
            '--output', $inferenceDir,
            '--threshold', '0.5',
            '--device', 'cuda',
            '--memmap-dir', (Join-Path $inferenceDir '.memmap')
        )
        $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $inferProcess = Start-LoggedProcess -Arguments $inferArguments `
            -Stdout $inferStdout -Stderr $inferStderr
        $stopwatch.Stop()
        if ($inferProcess.ExitCode -ne 0) {
            throw "v0.3 large BMP inference failed: $($experiment.id)"
        }
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($inputImage)
        $summaryPath = Join-Path $inferenceDir "${stem}_summary.json"
        $maskPath = Join-Path $inferenceDir "${stem}_mask.png"
        $overlayPath = Join-Path $inferenceDir "${stem}_overlay.png"
        $probabilityPath = Join-Path $inferenceDir "${stem}_probability.npy"
        foreach ($path in @($summaryPath, $maskPath, $overlayPath, $probabilityPath)) {
            if (-not (Test-Path $path -PathType Leaf)) {
                throw "v0.3 large BMP artifact is missing: $path"
            }
        }
        $summary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        $megapixels = (
            [double]$summary.image_size.width *
            [double]$summary.image_size.height / 1000000.0
        )
        $largeRecords += [pscustomobject][ordered]@{
            id = $experiment.id
            elapsed_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
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

    $selectionPath = Join-Path $benchmarkDir 'development_selection.json'
    $selectionStdout = Join-Path $benchmarkDir 'selection.stdout.log'
    $selectionStderr = Join-Path $benchmarkDir 'selection.stderr.log'
    $selectionArguments = @(
        '-m', 'ourbrain_cv.development_selection',
        '--baseline', $baselineMetrics,
        '--output', $selectionPath,
        '--maximum-recall-drop', '0.02',
        '--iterations', '10000',
        '--seed', '42'
    )
    foreach ($validation in $validationRecords) {
        $selectionArguments += @(
            '--candidate', "$($validation.id)=$($validation.metrics)"
        )
    }
    $selectionProcess = Start-LoggedProcess -Arguments $selectionArguments `
        -Stdout $selectionStdout -Stderr $selectionStderr
    if ($selectionProcess.ExitCode -ne 0) {
        throw 'v0.3 paired development selection failed.'
    }
    $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
    if (
        $selection.production_eligible -ne $false -or
        $selection.held_out_test_opened -ne $false
    ) {
        throw 'v0.3 selection violated the development-only contract.'
    }

    $completion = [ordered]@{
        schema_version = 1
        status = 'v0.3-development-benchmark-complete'
        development_only = $true
        production_eligible = $false
        positive_only = $true
        held_out_test_opened = $false
        manifest = $manifest
        manifest_sha256 = Get-Hash $manifest
        baseline_metrics = $baselineMetrics
        baseline_metrics_sha256 = Get-Hash $baselineMetrics
        validation_models = $validationRecords
        development_selection = $selectionPath
        development_selection_sha256 = Get-Hash $selectionPath
        retained_model = $selection.retained_model
        large_bmp = [ordered]@{
            input = $inputImage
            input_sha256 = Get-Hash $inputImage
            models = $largeRecords
        }
        limitations = $selection.limitations
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $completionPath `
        -Value ($completion | ConvertTo-Json -Depth 12)
    return [pscustomobject]$completion
}

try {
    $preflight = Invoke-Preflight
    if ($PreflightOnly) {
        $preflight
        exit 0
    }

    New-Item -ItemType Directory -Force -Path $smokeRoot | Out-Null
    Write-AtomicText -Path (Join-Path $orchestrationDir 'started_at.txt') `
        -Value ([DateTime]::UtcNow.ToString('o'))
    $smokeRecords = @()
    $eligibleExperiments = @()
    foreach ($experiment in $experiments) {
        $smoke = Invoke-Smoke -Experiment $experiment
        $smokeRecords += $smoke
        if ($smoke.full_training_eligible -eq $true) {
            $eligibleExperiments += $experiment
        }
    }
    Write-AtomicText -Path (Join-Path $orchestrationDir 'smoke_complete.json') `
        -Value ($smokeRecords | ConvertTo-Json -Depth 10)

    $requiredIds = @(
        'v0.3-a-upernet-swin-tiny-positive-only',
        'v0.3-b-segformer-b1-positive-only'
    )
    foreach ($requiredId in $requiredIds) {
        if ($requiredId -notin @($eligibleExperiments | ForEach-Object { $_.id })) {
            throw "Required v0.3 candidate failed smoke eligibility: $requiredId"
        }
    }

    $trainingRecords = @()
    foreach ($experiment in $eligibleExperiments) {
        $trainingRecords += Invoke-FullTraining -Experiment $experiment
    }
    Write-AtomicText -Path (Join-Path $orchestrationDir 'training_complete.json') `
        -Value ($trainingRecords | ConvertTo-Json -Depth 10)
    $benchmark = Invoke-Benchmark -TrainingRecords $trainingRecords
    Write-AtomicText -Path (Join-Path $orchestrationDir 'completed_at.txt') `
        -Value ([DateTime]::UtcNow.ToString('o'))
    Write-Output ($benchmark | ConvertTo-Json -Depth 12)
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
