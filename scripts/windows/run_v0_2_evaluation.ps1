param([switch]$PreflightOnly)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
$orchestrationDir = Join-Path $project 'runs\v0.2-ab'
$trainingComplete = Join-Path $orchestrationDir 'training_complete.json'
$selectionInputPath = Join-Path $orchestrationDir 'selection_candidates.json'
$selectionPath = Join-Path $orchestrationDir 'model_selection.json'
$testMetricsPath = Join-Path $orchestrationDir 'held_out_test_metrics.json'
$evaluationCompletePath = Join-Path $orchestrationDir 'evaluation_complete.json'
$taskName = 'OurBrain-V02-Evaluation'
$thresholdGrid = @(
    1..19 | ForEach-Object {
        ([double]($_ * 0.05)).ToString(
            '0.00',
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
) -join ','

$candidates = @(
    [ordered]@{
        id = 'v0.2-a-baseline-with-negatives'
        config = Join-Path $project 'configs\v0_2_a_baseline_with_negatives_cuda.yaml'
        checkpoint = Join-Path $project 'runs\v0.2-a-baseline-with-negatives\checkpoint'
        calibration = Join-Path $orchestrationDir 'calibration-a.json'
        log = Join-Path $orchestrationDir 'calibration-a.log'
    },
    [ordered]@{
        id = 'v0.2-b-recall-with-negatives'
        config = Join-Path $project 'configs\v0_2_b_recall_with_negatives_cuda.yaml'
        checkpoint = Join-Path $project 'runs\v0.2-b-recall-with-negatives\checkpoint'
        calibration = Join-Path $orchestrationDir 'calibration-b.json'
        log = Join-Path $orchestrationDir 'calibration-b.log'
    }
)

function Write-AtomicText {
    param([string]$Path, [string]$Value)
    $temporary = "$Path.tmp"
    Set-Content -Path $temporary -Value $Value -Encoding utf8
    Move-Item -Path $temporary -Destination $Path -Force
}

function Assert-Ready {
    if (-not (Test-Path $trainingComplete)) {
        throw "A/B training is not complete: $trainingComplete"
    }
    if (Test-Path $evaluationCompletePath) {
        throw (
            "Evaluation is already marked complete. Refusing to run it twice: " +
            $evaluationCompletePath
        )
    }
    $testResultNeedsCompletionRecovery = Test-Path $testMetricsPath
    $training = Get-Content $trainingComplete -Raw | ConvertFrom-Json
    if (
        $training.schema_version -ne 1 -or
        $training.status -ne 'training-complete'
    ) {
        throw "A/B training completion record is invalid: $trainingComplete"
    }
    $manifest = Join-Path $project 'artifacts\manifest_with_negatives.csv'
    $manifestHash = (
        Get-FileHash -Algorithm SHA256 $manifest
    ).Hash.ToLowerInvariant()
    if (
        $training.manifest -ne $manifest -or
        $training.manifest_sha256 -ne $manifestHash
    ) {
        throw "A/B training manifest provenance no longer matches."
    }
    foreach ($candidate in $candidates) {
        if (-not (Test-Path (Join-Path $candidate.checkpoint 'model.safetensors'))) {
            throw "Best checkpoint is missing: $($candidate.checkpoint)"
        }
        if (-not (Test-Path $candidate.config)) {
            throw "Config is missing: $($candidate.config)"
        }
        $records = @(
            $training.experiments |
                Where-Object { $_.run_id -eq $candidate.id }
        )
        if ($records.Count -ne 1) {
            throw "A/B completion record is missing or duplicated: $($candidate.id)"
        }
        $record = $records[0]
        $checkpointFile = Join-Path $candidate.checkpoint 'model.safetensors'
        $lastCheckpointFile = Join-Path $candidate.checkpoint 'last\model.safetensors'
        if (-not (Test-Path $lastCheckpointFile)) {
            throw "Last checkpoint is missing: $lastCheckpointFile"
        }
        $checkpointHash = (
            Get-FileHash -Algorithm SHA256 $checkpointFile
        ).Hash.ToLowerInvariant()
        $lastCheckpointHash = (
            Get-FileHash -Algorithm SHA256 $lastCheckpointFile
        ).Hash.ToLowerInvariant()
        $configHash = (
            Get-FileHash -Algorithm SHA256 $candidate.config
        ).Hash.ToLowerInvariant()
        if (
            $record.best_checkpoint_sha256 -ne $checkpointHash -or
            $record.last_checkpoint_sha256 -ne $lastCheckpointHash -or
            $record.config_sha256 -ne $configHash -or
            $record.manifest_sha256 -ne $manifestHash
        ) {
            throw "A/B artifact provenance no longer matches: $($candidate.id)"
        }
    }
    return $testResultNeedsCompletionRecovery
}

function Invoke-Calibration {
    param([System.Collections.IDictionary]$Candidate)
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'calibrate',
        '--config', $Candidate.config,
        '--checkpoint', $Candidate.checkpoint,
        '--minimum-image-recall', '0.95',
        '--thresholds', $thresholdGrid,
        '--output', $Candidate.calibration,
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $project -RedirectStandardOutput $Candidate.log `
        -RedirectStandardError "$($Candidate.log).stderr" `
        -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Calibration failed with exit $($process.ExitCode): $($Candidate.id)"
    }
    if (-not (Test-Path $Candidate.calibration)) {
        throw "Calibration returned success without JSON: $($Candidate.id)"
    }
}

function Select-Candidate {
    $descriptor = [ordered]@{
        schema_version = 1
        candidates = @(
            $candidates | ForEach-Object {
                [ordered]@{
                    id = $_.id
                    config = $_.config
                    checkpoint = $_.checkpoint
                    calibration = $_.calibration
                }
            }
        )
    }
    Write-AtomicText -Path $selectionInputPath `
        -Value ($descriptor | ConvertTo-Json -Depth 5)
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'select-calibrated-model',
        '--candidates', $selectionInputPath,
        '--minimum-image-recall', '0.95',
        '--output', $selectionPath
    )
    Push-Location $project
    try {
        $result = & $python @arguments 2>&1
        $code = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    Write-AtomicText -Path (Join-Path $orchestrationDir 'model_selection.log') `
        -Value ($result | Out-String)
    if ($code -ne 0) {
        throw "Validation model selection failed with exit $code."
    }
    if (-not (Test-Path $selectionPath)) {
        throw "Model selection returned success without JSON: $selectionPath"
    }
    $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
    if (-not $selection.winner) {
        throw "Model selection JSON has no winner: $selectionPath"
    }
    return $selection.winner
}

function Invoke-HeldOutTest {
    param($Winner)
    $stdout = Join-Path $orchestrationDir 'held_out_test.stdout.log'
    $stderr = Join-Path $orchestrationDir 'held_out_test.stderr.log'
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'evaluate',
        '--config', $Winner.config,
        '--checkpoint', $Winner.checkpoint,
        '--calibration', $Winner.calibration,
        '--split', 'test',
        '--output', $testMetricsPath,
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $project -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Held-out test failed with exit $($process.ExitCode)."
    }
    if (-not (Test-Path $testMetricsPath)) {
        throw 'Held-out test returned success without a metrics JSON.'
    }
}

function Write-EvaluationCompletion {
    param($Winner)

    $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
    $testMetrics = Get-Content $testMetricsPath -Raw | ConvertFrom-Json
    $calibration = Get-Content $Winner.calibration -Raw | ConvertFrom-Json
    $calibrationHash = (
        Get-FileHash -Algorithm SHA256 $Winner.calibration
    ).Hash.ToLowerInvariant()
    $configHash = (
        Get-FileHash -Algorithm SHA256 $Winner.config
    ).Hash.ToLowerInvariant()
    $bestCheckpoint = Join-Path $Winner.checkpoint 'model.safetensors'
    $bestCheckpointHash = (
        Get-FileHash -Algorithm SHA256 $bestCheckpoint
    ).Hash.ToLowerInvariant()
    if (
        $selection.schema_version -ne 1 -or
        $selection.selection_split -ne 'val' -or
        $selection.winner.id -ne $Winner.id
    ) {
        throw "Model selection provenance is invalid: $selectionPath"
    }
    if (
        $selection.winner.calibration_sha256 -ne $calibrationHash -or
        $calibration.schema_version -ne 1 -or
        $calibration.recall_constraint_met -ne $true -or
        [double]$calibration.selected_threshold -ne [double]$Winner.threshold -or
        $testMetrics.schema_version -ne 1 -or
        $testMetrics.evaluation_split -ne 'test' -or
        $testMetrics.provenance.config -ne $Winner.config -or
        $testMetrics.provenance.config_sha256 -ne $configHash -or
        $testMetrics.provenance.checkpoint -ne $Winner.checkpoint -or
        $testMetrics.provenance.checkpoint_sha256 -ne (
            $calibration.provenance.checkpoint_sha256
        ) -or
        $testMetrics.provenance.calibration -ne $Winner.calibration -or
        $testMetrics.provenance.calibration_sha256 -ne $calibrationHash -or
        $testMetrics.provenance.manifest -ne $calibration.provenance.manifest -or
        $testMetrics.provenance.manifest_sha256 -ne (
            $calibration.provenance.manifest_sha256
        ) -or
        $testMetrics.provenance.review_audit -ne (
            $calibration.provenance.review_audit
        ) -or
        $testMetrics.provenance.review_audit_sha256 -ne (
            $calibration.provenance.review_audit_sha256
        ) -or
        [double]$testMetrics.threshold -ne [double]$Winner.threshold
    ) {
        throw "Held-out test provenance does not match the selected model."
    }
    $completion = [ordered]@{
        schema_version = 1
        status = 'evaluation-complete'
        selection_split = 'val'
        evaluation_split = 'test'
        winner_id = $Winner.id
        fixed_threshold = [double]$Winner.threshold
        artifacts = [ordered]@{
            model_selection = $selectionPath
            model_selection_sha256 = (
                Get-FileHash -Algorithm SHA256 $selectionPath
            ).Hash.ToLowerInvariant()
            calibration = $Winner.calibration
            calibration_sha256 = $calibrationHash
            held_out_test_metrics = $testMetricsPath
            held_out_test_metrics_sha256 = (
                Get-FileHash -Algorithm SHA256 $testMetricsPath
            ).Hash.ToLowerInvariant()
            config = $Winner.config
            config_sha256 = $configHash
            best_checkpoint = $bestCheckpoint
            best_checkpoint_sha256 = $bestCheckpointHash
            manifest = $testMetrics.provenance.manifest
            manifest_sha256 = $testMetrics.provenance.manifest_sha256
            review_audit = $testMetrics.provenance.review_audit
            review_audit_sha256 = $testMetrics.provenance.review_audit_sha256
        }
        metrics = [ordered]@{
            samples = $testMetrics.samples
            crack_dice = $testMetrics.crack_dice
            boundary_f1 = $testMetrics.boundary_f1
            image_level_recall = $testMetrics.image_level_recall
            image_level_specificity = $testMetrics.image_level_specificity
            error_case_count = $testMetrics.error_case_count
        }
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $evaluationCompletePath `
        -Value ($completion | ConvertTo-Json -Depth 6)
}

try {
    $testResultNeedsCompletionRecovery = Assert-Ready
    Remove-Item (Join-Path $orchestrationDir 'evaluation_error.txt') `
        -Force -ErrorAction SilentlyContinue
    if ($PreflightOnly) {
        if ($testResultNeedsCompletionRecovery) {
            Write-Output (
                'V0.2 evaluation preflight passed: held-out test exists and ' +
                'requires completion-record recovery without reevaluation.'
            )
        }
        else {
            Write-Output 'V0.2 evaluation preflight passed.'
        }
        exit 0
    }
    if ($testResultNeedsCompletionRecovery) {
        if (-not (Test-Path $selectionPath)) {
            throw (
                "Held-out test exists but model selection is missing; refusing " +
                "reevaluation: $selectionPath"
            )
        }
        $winner = (Get-Content $selectionPath -Raw | ConvertFrom-Json).winner
        if (-not $winner) {
            throw "Model selection JSON has no winner: $selectionPath"
        }
        Write-EvaluationCompletion -Winner $winner
        Write-Output (
            'Recovered evaluation_complete.json from the existing held-out test; ' +
            'the test split was not evaluated again.'
        )
        exit 0
    }
    foreach ($candidate in $candidates) {
        Invoke-Calibration -Candidate $candidate
    }
    $winner = Select-Candidate
    Invoke-HeldOutTest -Winner $winner
    Write-EvaluationCompletion -Winner $winner
}
catch {
    New-Item -ItemType Directory -Force -Path $orchestrationDir | Out-Null
    Write-AtomicText -Path (Join-Path $orchestrationDir 'evaluation_error.txt') `
        -Value ($_ | Out-String)
    throw
}
finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false `
        -ErrorAction SilentlyContinue
}
