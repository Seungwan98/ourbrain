$ErrorActionPreference = 'Stop'

$taskName = 'OurBrain-V02-Dev-Positive-AB'
$runner = 'D:\ourbrain\scripts\windows\run_v0_2_dev_positive_ab.ps1'
if (-not (Test-Path $runner -PathType Leaf)) {
    throw "Development runner does not exist: $runner"
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
    "-NoProfile -ExecutionPolicy Bypass -File `"$runner`""
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
Write-Output (
    'Started development-only v0.2 positive A/B. ' +
    'It is not production eligible and will not open the held-out test.'
)
