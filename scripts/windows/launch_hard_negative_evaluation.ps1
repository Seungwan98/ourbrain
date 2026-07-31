param(
    [Parameter(Mandatory = $true)]
    [string]$RoundId
)

$ErrorActionPreference = 'Stop'

if ($RoundId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') {
    throw 'RoundId must contain only letters, numbers, dot, underscore, or hyphen.'
}

$taskName = "OurBrain-HN-Eval-$RoundId"
$runner = 'D:\ourbrain\scripts\windows\run_hard_negative_evaluation.ps1'
if (-not (Test-Path $runner -PathType Leaf)) {
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

$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument (
    "-NoProfile -ExecutionPolicy Bypass -File `"$runner`" -RoundId `"$RoundId`""
)
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
Write-Output "Started hard-negative evaluation round: $RoundId"
