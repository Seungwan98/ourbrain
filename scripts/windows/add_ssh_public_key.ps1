param(
    [Parameter(Mandatory = $true)]
    [string]$PublicKeyFile,

    [string]$Destination = 'C:\ProgramData\ssh\administrators_authorized_keys',

    [switch]$SkipAcl
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $PublicKeyFile -PathType Leaf)) {
    throw "Public key file not found: $PublicKeyFile"
}

$keyLines = @(
    Get-Content -LiteralPath $PublicKeyFile |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -ne '' }
)

if ($keyLines.Count -ne 1) {
    throw 'The public key file must contain exactly one non-empty line.'
}

$keyLine = $keyLines[0]
if ($keyLine -match 'PRIVATE KEY') {
    throw 'A private key was provided. Only a .pub public key is accepted.'
}

$parts = @($keyLine -split '\s+', 3)
$allowedTypes = @(
    'ssh-ed25519',
    'ecdsa-sha2-nistp256',
    'ssh-rsa'
)

if ($parts.Count -lt 2 -or $allowedTypes -notcontains $parts[0]) {
    throw 'Unsupported or malformed SSH public key.'
}

try {
    [void][Convert]::FromBase64String($parts[1])
}
catch {
    throw 'The SSH public key payload is not valid base64.'
}

$destinationDirectory = Split-Path -Parent $Destination
New-Item -ItemType Directory -Force -Path $destinationDirectory | Out-Null

$existingLines = @()
if (Test-Path -LiteralPath $Destination -PathType Leaf) {
    $existingLines = @(
        Get-Content -LiteralPath $Destination |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -ne '' }
    )
}

$keyIdentity = "$($parts[0]) $($parts[1])"
$alreadyPresent = $false
foreach ($existingLine in $existingLines) {
    $existingParts = @($existingLine -split '\s+', 3)
    if ($existingParts.Count -ge 2 -and
        "$($existingParts[0]) $($existingParts[1])" -eq $keyIdentity) {
        $alreadyPresent = $true
        break
    }
}

if (-not $alreadyPresent) {
    if (Test-Path -LiteralPath $Destination -PathType Leaf) {
        $backup = "$Destination.bak-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Copy-Item -LiteralPath $Destination -Destination $backup -Force
        Write-Output "BACKUP=$backup"
    }

    $temporary = "$Destination.tmp-$PID"
    @($existingLines + $keyLine) | Set-Content -LiteralPath $temporary -Encoding Ascii
    Move-Item -LiteralPath $temporary -Destination $Destination -Force
    Write-Output 'KEY_ADDED=true'
}
else {
    Write-Output 'KEY_ADDED=false'
}

if (-not $SkipAcl) {
    & icacls.exe $Destination /inheritance:r | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to disable inherited ACLs on $Destination"
    }

    & icacls.exe $Destination /grant:r '*S-1-5-32-544:(F)' '*S-1-5-18:(F)' | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to grant Administrators and SYSTEM access to $Destination"
    }
}

$fingerprint = (& ssh-keygen.exe -lf $PublicKeyFile 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw 'ssh-keygen could not validate the public key.'
}

Write-Output "DESTINATION=$Destination"
Write-Output "FINGERPRINT=$fingerprint"
