param([string]$RoundId)

$ErrorActionPreference = 'Stop'

if ($RoundId -and $RoundId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') {
    throw 'RoundId must contain only letters, numbers, dot, underscore, or hyphen.'
}
if ($RoundId) {
    $taskName = "OurBrain-HN-Pilot-$RoundId"
}
else {
    $taskName = 'OurBrain-V02-Pilot'
}
$runner = 'D:\ourbrain\scripts\windows\run_v0_2_pilot.ps1'
if (-not (Test-Path $runner)) {
    throw "Runner does not exist: $runner"
}

$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and $existing.State -eq 'Running') {
    Write-Output "$taskName is already running."
    exit 0
}
if ($existing) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

$runnerArguments = "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""
if ($RoundId) {
    $runnerArguments += " -RoundId `"$RoundId`""
}
$action = New-ScheduledTaskAction -Execute 'powershell.exe' `
    -Argument $runnerArguments
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddYears(1)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 12)
$userId = (& whoami).Trim()
$principal = New-ScheduledTaskPrincipal -UserId $userId `
    -LogonType Interactive -RunLevel Highest

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal -ErrorAction Stop | Out-Null
if (-not (Get-ScheduledTask -TaskName $taskName -ErrorAction Stop)) {
    throw "Scheduled task registration could not be verified: $taskName"
}
Start-ScheduledTask -TaskName $taskName -ErrorAction Stop
Write-Output (
    "Started $taskName. It will run the selected model with the fixed validation " +
    "threshold over the representative BMP list."
)
