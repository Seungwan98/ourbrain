param(
    [switch]$PreflightOnly,
    [string]$RoundId
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:HF_HOME = 'D:\ourbrain-cache\hf'
$env:HUGGINGFACE_HUB_CACHE = 'D:\ourbrain-cache\hf\hub'
$env:PYTHONUNBUFFERED = '1'

$project = 'D:\ourbrain'
$python = Join-Path $project '.venv\Scripts\python.exe'
if ($RoundId -and $RoundId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') {
    throw 'RoundId must contain only letters, numbers, dot, underscore, or hyphen.'
}
if ($RoundId) {
    $orchestrationDir = Join-Path $project (
        "runs\hard-negative-$RoundId\evaluation"
    )
    $pilotDir = Join-Path $project "runs\hard-negative-$RoundId\pilot"
    $taskName = "OurBrain-HN-Pilot-$RoundId"
}
else {
    $orchestrationDir = Join-Path $project 'runs\v0.2-ab'
    $pilotDir = Join-Path $project 'runs\v0.2-pilot'
    $taskName = 'OurBrain-V02-Pilot'
}
$selectionPath = Join-Path $orchestrationDir 'model_selection.json'
$testMetricsPath = Join-Path $orchestrationDir 'held_out_test_metrics.json'
$evaluationCompletePath = Join-Path $orchestrationDir 'evaluation_complete.json'
$inputListPath = Join-Path $project 'artifacts\pilot_inputs.txt'
$summaryPath = Join-Path $pilotDir 'pilot_summary.json'
$reviewPath = Join-Path $pilotDir 'pilot_review.csv'

function Write-AtomicText {
    param([string]$Path, [string]$Value)
    $temporary = "$Path.tmp"
    Set-Content -Path $temporary -Value $Value -Encoding utf8
    Move-Item -Path $temporary -Destination $Path -Force
}

function Get-PilotInputs {
    if (-not (Test-Path $inputListPath)) {
        throw (
            "Pilot input list is missing: $inputListPath. Copy " +
            "configs\pilot_inputs.example.txt there and replace every example " +
            "with a representative absolute BMP path."
        )
    }
    $inputs = @(
        Get-Content $inputListPath |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and -not $_.StartsWith('#') }
    )
    if ($inputs.Count -eq 0) {
        throw "Pilot input list has no image paths: $inputListPath"
    }
    $duplicates = @(
        $inputs |
            Group-Object |
            Where-Object { $_.Count -gt 1 } |
            ForEach-Object { $_.Name }
    )
    if ($duplicates.Count -gt 0) {
        throw "Pilot input list contains duplicate paths: $($duplicates -join ', ')"
    }
    foreach ($inputPath in $inputs) {
        if (-not [System.IO.Path]::IsPathRooted($inputPath)) {
            throw "Pilot image path must be absolute: $inputPath"
        }
        if ([System.IO.Path]::GetExtension($inputPath).ToLowerInvariant() -ne '.bmp') {
            throw "Pilot input must be a BMP file: $inputPath"
        }
        if (-not (Test-Path -LiteralPath $inputPath -PathType Leaf)) {
            throw "Pilot image does not exist: $inputPath"
        }
    }
    return $inputs
}

function Get-PilotContext {
    if (-not (Test-Path $python)) {
        throw "Python environment does not exist: $python"
    }
    if (-not (Test-Path $selectionPath)) {
        throw "Validation model selection is not complete: $selectionPath"
    }
    if (-not (Test-Path $testMetricsPath)) {
        throw (
            "Held-out test has not been evaluated. The pilot cannot run before " +
            "the one-time test result exists: $testMetricsPath"
        )
    }
    if (-not (Test-Path $evaluationCompletePath)) {
        throw (
            "Evaluation completion record is missing. The pilot cannot run from " +
            "partial evaluation artifacts: $evaluationCompletePath"
        )
    }
    $selection = Get-Content $selectionPath -Raw | ConvertFrom-Json
    if ($selection.schema_version -ne 1 -or -not $selection.winner) {
        throw "Unsupported or incomplete model selection JSON: $selectionPath"
    }
    $winner = $selection.winner
    foreach ($path in @($winner.config, $winner.checkpoint, $winner.calibration)) {
        if (-not (Test-Path $path)) {
            throw "Selected model artifact is missing: $path"
        }
    }
    $calibration = Get-Content $winner.calibration -Raw | ConvertFrom-Json
    if ($calibration.recall_constraint_met -ne $true) {
        throw "Selected model calibration does not meet the recall constraint."
    }
    if ([double]$calibration.selected_threshold -ne [double]$winner.threshold) {
        throw "Selection and calibration thresholds do not match."
    }
    $testMetrics = Get-Content $testMetricsPath -Raw | ConvertFrom-Json
    if (-not $testMetrics) {
        throw "Held-out test metrics JSON is empty: $testMetricsPath"
    }
    $evaluationComplete = (
        Get-Content $evaluationCompletePath -Raw | ConvertFrom-Json
    )
    $selectionHash = (
        Get-FileHash -Algorithm SHA256 $selectionPath
    ).Hash.ToLowerInvariant()
    $testMetricsHash = (
        Get-FileHash -Algorithm SHA256 $testMetricsPath
    ).Hash.ToLowerInvariant()
    if (
        $evaluationComplete.schema_version -ne 1 -or
        $evaluationComplete.status -ne 'evaluation-complete' -or
        $evaluationComplete.selection_split -ne 'val' -or
        $evaluationComplete.evaluation_split -ne 'test' -or
        $evaluationComplete.winner_id -ne $winner.id -or
        [double]$evaluationComplete.fixed_threshold -ne [double]$winner.threshold -or
        $evaluationComplete.artifacts.model_selection -ne $selectionPath -or
        $evaluationComplete.artifacts.model_selection_sha256 -ne $selectionHash -or
        $evaluationComplete.artifacts.held_out_test_metrics -ne $testMetricsPath -or
        $evaluationComplete.artifacts.held_out_test_metrics_sha256 -ne $testMetricsHash
    ) {
        throw "Evaluation completion provenance no longer matches its artifacts."
    }
    return [ordered]@{
        selection = $selection
        winner = $winner
        calibration = $calibration
        evaluation_complete = $evaluationComplete
        inputs = @(Get-PilotInputs)
    }
}

function Get-StableImageDirectory {
    param([int]$Index, [string]$ImagePath)

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($ImagePath)
    $safeStem = $stem -replace '[^A-Za-z0-9._-]', '_'
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($ImagePath.ToLowerInvariant())
        $hash = ([System.BitConverter]::ToString($sha.ComputeHash($bytes))).Replace('-', '')
        $shortHash = $hash.Substring(0, 12).ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
    return Join-Path $pilotDir ('{0:D3}-{1}-{2}' -f $Index, $safeStem, $shortHash)
}

function Invoke-PilotImage {
    param(
        [int]$Index,
        [string]$ImagePath,
        $Winner
    )

    $outputDir = Get-StableImageDirectory -Index $Index -ImagePath $ImagePath
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($ImagePath)
    $imageSummaryPath = Join-Path $outputDir "${stem}_summary.json"
    $requiredOutputs = @(
        $imageSummaryPath,
        (Join-Path $outputDir "${stem}_probability.npy"),
        (Join-Path $outputDir "${stem}_mask.png"),
        (Join-Path $outputDir "${stem}_overlay.png")
    )
    if (($requiredOutputs | Where-Object { -not (Test-Path -LiteralPath $_) }).Count -eq 0) {
        $existing = Get-Content -LiteralPath $imageSummaryPath -Raw | ConvertFrom-Json
        return [ordered]@{
            index = $Index
            input = $ImagePath
            status = 'already-complete'
            duration_seconds = $null
            summary_path = $imageSummaryPath
            presence = $existing.presence
            crack_ratio = $existing.crack_ratio
            quality_gate = $existing.quality_gate
            overlay = $existing.outputs.overlay
        }
    }
    if ((Test-Path $outputDir) -and (Get-ChildItem $outputDir -Force | Select-Object -First 1)) {
        throw "Incomplete pilot output exists; refusing to overwrite: $outputDir"
    }

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    $stdout = Join-Path $outputDir 'stdout.log'
    $stderr = Join-Path $outputDir 'stderr.log'
    $memmap = Join-Path $outputDir '.memmap'
    $arguments = @(
        '-m', 'ourbrain_cv.cli', 'infer',
        '--config', $Winner.config,
        '--checkpoint', $Winner.checkpoint,
        '--input', $ImagePath,
        '--output', $outputDir,
        '--calibration', $Winner.calibration,
        '--device', 'cuda',
        '--memmap-dir', $memmap
    )
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    Push-Location $project
    try {
        & $python @arguments 1> $stdout 2> $stderr
        $exitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    $stopwatch.Stop()
    if ($exitCode -ne 0) {
        throw "Pilot inference failed with exit ${exitCode}: $ImagePath"
    }
    foreach ($path in $requiredOutputs) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Pilot inference returned success without artifact: $path"
        }
    }
    $result = Get-Content -LiteralPath $imageSummaryPath -Raw | ConvertFrom-Json
    return [ordered]@{
        index = $Index
        input = $ImagePath
        status = 'completed'
        duration_seconds = [math]::Round($stopwatch.Elapsed.TotalSeconds, 3)
        summary_path = $imageSummaryPath
        presence = $result.presence
        crack_ratio = $result.crack_ratio
        quality_gate = $result.quality_gate
        overlay = $result.outputs.overlay
    }
}

try {
    $context = Get-PilotContext
    Remove-Item (Join-Path $pilotDir 'pilot_error.txt') `
        -Force -ErrorAction SilentlyContinue
    $inputListHash = (
        Get-FileHash -Algorithm SHA256 $inputListPath
    ).Hash.ToLowerInvariant()
    if (Test-Path $summaryPath) {
        $existingSummary = Get-Content $summaryPath -Raw | ConvertFrom-Json
        if (
            $existingSummary.provenance.input_list_sha256 -and
            $existingSummary.provenance.input_list_sha256 -ne $inputListHash
        ) {
            throw (
                "Pilot input list changed after outputs were created. Archive the " +
                "existing pilot directory before starting a new pilot."
            )
        }
    }
    if ($PreflightOnly) {
        Write-Output (
            "V0.2 pilot preflight passed: {0} BMP image(s), model {1}, fixed threshold {2}" -f `
                $context.inputs.Count,
                $context.winner.id,
                $context.winner.threshold
        )
        exit 0
    }

    New-Item -ItemType Directory -Force -Path $pilotDir | Out-Null
    $records = @()
    for ($offset = 0; $offset -lt $context.inputs.Count; $offset++) {
        $index = $offset + 1
        $imagePath = $context.inputs[$offset]
        Write-Output "[$index/$($context.inputs.Count)] $imagePath"
        $records += Invoke-PilotImage -Index $index -ImagePath $imagePath `
            -Winner $context.winner
    }
    if (Test-Path $reviewPath) {
        $existingReview = @(Import-Csv -Path $reviewPath)
        if ($existingReview.Count -ne $records.Count) {
            throw (
                "Existing pilot review row count does not match current inputs: " +
                "$reviewPath"
            )
        }
        for ($offset = 0; $offset -lt $records.Count; $offset++) {
            if ($existingReview[$offset].input -ne $records[$offset].input) {
                throw "Existing pilot review inputs do not match at row $($offset + 2)."
            }
        }
    }
    else {
        $reviewRows = @(
            $records | ForEach-Object {
                [pscustomobject][ordered]@{
                    input = $_.input
                    overlay = $_.overlay
                    model_presence = $_.presence
                    quality_gate_passed = $_.quality_gate.passed
                    review_label = ''
                    left = ''
                    top = ''
                    right = ''
                    bottom = ''
                    error_category = ''
                    note = ''
                }
            }
        )
        $reviewTemporary = "$reviewPath.tmp"
        $reviewRows | Export-Csv -Path $reviewTemporary -NoTypeInformation `
            -Encoding UTF8
        Move-Item -Path $reviewTemporary -Destination $reviewPath -Force
    }
    $pilotSummary = [ordered]@{
        schema_version = 1
        status = 'inference-complete-human-review-required'
        round_id = $RoundId
        model = [ordered]@{
            id = $context.winner.id
            config = $context.winner.config
            checkpoint = $context.winner.checkpoint
            calibration = $context.winner.calibration
            fixed_threshold = $context.winner.threshold
        }
        provenance = [ordered]@{
            model_selection = $selectionPath
            model_selection_sha256 = (
                Get-FileHash -Algorithm SHA256 $selectionPath
            ).Hash.ToLowerInvariant()
            held_out_test_metrics = $testMetricsPath
            held_out_test_metrics_sha256 = (
                Get-FileHash -Algorithm SHA256 $testMetricsPath
            ).Hash.ToLowerInvariant()
            evaluation_complete = $evaluationCompletePath
            evaluation_complete_sha256 = (
                Get-FileHash -Algorithm SHA256 $evaluationCompletePath
            ).Hash.ToLowerInvariant()
            input_list = $inputListPath
            input_list_sha256 = $inputListHash
        }
        image_count = $records.Count
        quality_gate_failures = @(
            $records | Where-Object { $_.quality_gate.passed -ne $true }
        ).Count
        images = $records
        human_review_csv = $reviewPath
        allowed_review_labels = @(
            'correct_crack',
            'correct_normal',
            'false_positive',
            'false_negative',
            'uncertain'
        )
        next_step = (
            'human review of every overlay; false positives and false negatives ' +
            'must include coordinates when a local region can be identified'
        )
        finished_at_utc = [DateTime]::UtcNow.ToString('o')
    }
    Write-AtomicText -Path $summaryPath `
        -Value ($pilotSummary | ConvertTo-Json -Depth 8)
}
catch {
    New-Item -ItemType Directory -Force -Path $pilotDir | Out-Null
    Write-AtomicText -Path (Join-Path $pilotDir 'pilot_error.txt') `
        -Value ($_ | Out-String)
    throw
}
finally {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false `
        -ErrorAction SilentlyContinue
}
