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
$artifactRoot = Join-Path $project "artifacts\hard-negative-rounds\$RoundId"
$manifest = Join-Path $artifactRoot 'manifest.csv'
$audit = Join-Path $artifactRoot 'manifest.review.json'
$runDir = Join-Path $project "runs\hard-negative-$RoundId"
$trainingCompletePath = Join-Path $runDir 'completion.json'
$evaluationDir = Join-Path $runDir 'evaluation'
$selectionInputPath = Join-Path $evaluationDir 'selection_candidates.json'
$selectionPath = Join-Path $evaluationDir 'model_selection.json'
$testMetricsPath = Join-Path $evaluationDir 'held_out_test_metrics.json'
$evaluationCompletePath = Join-Path $evaluationDir 'evaluation_complete.json'
$sourceSelectionPath = Join-Path $project 'runs\v0.2-ab\model_selection.json'
$sourceEvaluationPath = Join-Path $project 'runs\v0.2-ab\evaluation_complete.json'
$taskName = "OurBrain-HN-Eval-$RoundId"
$thresholdGrid = @(
    1..19 | ForEach-Object {
        ([double]($_ * 0.05)).ToString(
            '0.00',
            [System.Globalization.CultureInfo]::InvariantCulture
        )
    }
) -join ','

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

function Assert-Ready {
    foreach ($path in @(
        $python,
        $manifest,
        $audit,
        $trainingCompletePath,
        $sourceSelectionPath,
        $sourceEvaluationPath
    )) {
        if (-not (Test-Path $path -PathType Leaf)) {
            throw "Hard-negative evaluation artifact is missing: $path"
        }
    }
    if (Test-Path $evaluationCompletePath) {
        throw "Hard-negative evaluation is already complete: $evaluationCompletePath"
    }
    $training = Get-Content $trainingCompletePath -Raw | ConvertFrom-Json
    $sourceSelection = Get-Content $sourceSelectionPath -Raw | ConvertFrom-Json
    $sourceEvaluation = Get-Content $sourceEvaluationPath -Raw | ConvertFrom-Json
    if (
        $training.schema_version -ne 1 -or
        $training.status -ne 'training-complete' -or
        $training.round_id -ne $RoundId -or
        $training.manifest -ne $manifest -or
        $training.manifest_sha256 -ne (Get-Hash $manifest) -or
        $training.review_audit_sha256 -ne (Get-Hash $audit) -or
        $training.config_sha256 -ne (Get-Hash $training.config) -or
        $training.best_checkpoint_sha256 -ne (
            Get-Hash $training.best_checkpoint
        ) -or
        $sourceSelection.schema_version -ne 1 -or
        -not $sourceSelection.winner -or
        $sourceEvaluation.schema_version -ne 1 -or
        $sourceEvaluation.status -ne 'evaluation-complete' -or
        $sourceEvaluation.artifacts.model_selection_sha256 -ne (
            Get-Hash $sourceSelectionPath
        )
    ) {
        throw 'Hard-negative training or baseline provenance is invalid or changed.'
    }
    return [ordered]@{
        training = $training
        source_selection = $sourceSelection
        recover_completion = (Test-Path $testMetricsPath)
    }
}

function Invoke-Calibration {
    param([System.Collections.IDictionary]$Candidate)
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'calibrate',
        '--config', $Candidate.config,
        '--manifest', $manifest,
        '--checkpoint', $Candidate.checkpoint,
        '--minimum-image-recall', '0.95',
        '--thresholds', $thresholdGrid,
        '--boundary-tolerance', '2',
        '--output', $Candidate.calibration,
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $project `
        -RedirectStandardOutput $Candidate.log `
        -RedirectStandardError "$($Candidate.log).stderr" `
        -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Calibration failed with exit $($process.ExitCode): $($Candidate.id)"
    }
    if (-not (Test-Path $Candidate.calibration -PathType Leaf)) {
        throw "Calibration returned success without JSON: $($Candidate.id)"
    }
}

function Select-Candidate {
    param([array]$Candidates)
    $descriptor = [ordered]@{
        schema_version = 1
        candidates = @(
            $Candidates | ForEach-Object {
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
        -Value ($descriptor | ConvertTo-Json -Depth 6)
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'select-calibrated-model',
        '--candidates', $selectionInputPath,
        '--minimum-image-recall', '0.95',
        '--output', $selectionPath
    )
    $result = & $python @arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw ($result | Out-String)
    }
    $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
    if (-not $selection.winner) {
        throw "Hard-negative model selection has no winner: $selectionPath"
    }
    return $selection.winner
}

function Invoke-HeldOutTest {
    param($Winner)
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'evaluate',
        '--config', $Winner.config,
        '--manifest', $manifest,
        '--checkpoint', $Winner.checkpoint,
        '--calibration', $Winner.calibration,
        '--split', 'test',
        '--boundary-tolerance', '2',
        '--output', $testMetricsPath,
        '--device', 'cuda'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments `
        -WorkingDirectory $project `
        -RedirectStandardOutput (Join-Path $evaluationDir 'test.stdout.log') `
        -RedirectStandardError (Join-Path $evaluationDir 'test.stderr.log') `
        -NoNewWindow -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Hard-negative held-out test failed with exit $($process.ExitCode)."
    }
    if (-not (Test-Path $testMetricsPath -PathType Leaf)) {
        throw 'Held-out test returned success without metrics JSON.'
    }
}

function Write-EvaluationCompletion {
    param($Winner)
    $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
    $test = Get-Content $testMetricsPath -Raw | ConvertFrom-Json
    $calibration = Get-Content $Winner.calibration -Raw | ConvertFrom-Json
    $calibrationHash = Get-Hash $Winner.calibration
    if (
        $selection.schema_version -ne 1 -or
        $selection.selection_split -ne 'val' -or
        $selection.winner.id -ne $Winner.id -or
        $selection.winner.calibration_sha256 -ne $calibrationHash -or
        $calibration.schema_version -ne 1 -or
        $calibration.recall_constraint_met -ne $true -or
        $test.schema_version -ne 1 -or
        $test.evaluation_split -ne 'test' -or
        $test.provenance.config -ne $Winner.config -or
        $test.provenance.checkpoint -ne $Winner.checkpoint -or
        $test.provenance.calibration -ne $Winner.calibration -or
        $test.provenance.calibration_sha256 -ne $calibrationHash -or
        $test.provenance.manifest -ne $manifest -or
        $test.provenance.manifest_sha256 -ne (Get-Hash $manifest) -or
        $test.provenance.review_audit -ne $audit -or
        $test.provenance.review_audit_sha256 -ne (Get-Hash $audit) -or
        [double]$test.threshold -ne [double]$Winner.threshold
    ) {
        throw 'Hard-negative held-out test provenance does not match selection.'
    }
    $completion = [ordered]@{
        schema_version = 1
        status = 'evaluation-complete'
        round_id = $RoundId
        selection_split = 'val'
        evaluation_split = 'test'
        winner_id = $Winner.id
        retrained_model_selected = (
            $Winner.id -eq "hard-negative-$RoundId"
        )
        fixed_threshold = [double]$Winner.threshold
        artifacts = [ordered]@{
            training_completion = $trainingCompletePath
            training_completion_sha256 = Get-Hash $trainingCompletePath
            model_selection = $selectionPath
            model_selection_sha256 = Get-Hash $selectionPath
            calibration = $Winner.calibration
            calibration_sha256 = $calibrationHash
            held_out_test_metrics = $testMetricsPath
            held_out_test_metrics_sha256 = Get-Hash $testMetricsPath
            manifest = $manifest
            manifest_sha256 = Get-Hash $manifest
            review_audit = $audit
            review_audit_sha256 = Get-Hash $audit
        }
        metrics = [ordered]@{
            samples = $test.samples
            crack_dice = $test.crack_dice
            boundary_f1 = $test.boundary_f1
            image_level_recall = $test.image_level_recall
            image_level_specificity = $test.image_level_specificity
            error_case_count = $test.error_case_count
        }
        completed_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $evaluationCompletePath `
        -Value ($completion | ConvertTo-Json -Depth 7)
}

try {
    $context = Assert-Ready
    if ($PreflightOnly) {
        if ($context.recover_completion) {
            Write-Output (
                'Hard-negative evaluation preflight passed: completion-record ' +
                'recovery is required without test reevaluation.'
            )
        }
        else {
            Write-Output 'Hard-negative evaluation preflight passed.'
        }
        exit 0
    }
    if ($context.recover_completion) {
        if (-not (Test-Path $selectionPath -PathType Leaf)) {
            throw 'Test metrics exist without model selection; refusing reevaluation.'
        }
        $winner = (Get-Content $selectionPath -Raw | ConvertFrom-Json).winner
        Write-EvaluationCompletion -Winner $winner
        Write-Output (
            'Recovered hard-negative evaluation completion without test reevaluation.'
        )
        exit 0
    }
    if (Test-Path $evaluationDir) {
        throw "Evaluation directory already exists; refusing overwrite: $evaluationDir"
    }
    New-Item -ItemType Directory -Force -Path $evaluationDir | Out-Null
    $config = $context.training.config
    $candidates = @(
        [ordered]@{
            id = "$($context.source_selection.winner.id)-baseline"
            config = $config
            checkpoint = $context.source_selection.winner.checkpoint
            calibration = Join-Path $evaluationDir 'calibration-baseline.json'
            log = Join-Path $evaluationDir 'calibration-baseline.log'
        },
        [ordered]@{
            id = "hard-negative-$RoundId"
            config = $config
            checkpoint = $context.training.output_checkpoint
            calibration = Join-Path $evaluationDir 'calibration-round.json'
            log = Join-Path $evaluationDir 'calibration-round.log'
        }
    )
    foreach ($candidate in $candidates) {
        Invoke-Calibration -Candidate $candidate
    }
    $winner = Select-Candidate -Candidates $candidates
    Invoke-HeldOutTest -Winner $winner
    Write-EvaluationCompletion -Winner $winner
    Write-Output "Hard-negative evaluation completed: $evaluationCompletePath"
}
catch {
    if (Test-Path $evaluationDir) {
        Write-AtomicText -Path (Join-Path $evaluationDir 'evaluation_error.txt') `
            -Value ($_ | Out-String)
    }
    throw
}
finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false `
        -ErrorAction SilentlyContinue
}
