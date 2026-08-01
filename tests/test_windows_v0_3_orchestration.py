from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "windows" / "run_v0_3_model_sweep.ps1"


def _section(script: str, start_marker: str, end_marker: str) -> str:
    start = script.index(start_marker)
    end = script.index(end_marker, start)
    return script[start:end]


def test_recovery_progress_does_not_pollute_training_records() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    body = _section(
        script,
        "function Invoke-FullTraining",
        "function Invoke-Benchmark",
    )

    assert 'Write-Host "Recovering completed v0.3 artifacts:' in body
    assert 'Write-Output "Recovering completed v0.3 artifacts:' not in body


def test_incomplete_benchmark_is_archived_before_retry() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    body = _section(script, "function Invoke-Benchmark", "try {")

    assert "Move-Item -Path $benchmarkDir -Destination $archivePath" in body
    assert "Incomplete v0.3 benchmark exists" not in body


def test_logged_process_rejects_empty_arguments_with_clear_error() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    body = _section(
        script,
        "function Start-LoggedProcess",
        "function Invoke-Preflight",
    )

    assert "$invalidArguments" in body
    assert "Refusing to start a process with an empty argument." in body
