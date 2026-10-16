# Run from the directory containing all parts and SHA256SUMS, on Windows PowerShell.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$lines = Get-Content -LiteralPath 'SHA256SUMS'
$names = @()
foreach ($line in $lines) {
    if ($line -notmatch '^([0-9a-f]{64})  (CBM-offline-full\.zip\.part[0-9]{4})$') {
        throw "Invalid checksum record: $line"
    }
    $expected = $Matches[1]
    $name = $Matches[2]
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $name).Hash.ToLowerInvariant() -ne $expected) {
        throw "Checksum mismatch: $name"
    }
    $names += $name
}
if ($names.Count -eq 0) { throw 'No archive parts in SHA256SUMS' }
$target = Join-Path $PSScriptRoot 'CBM-offline-full.zip'
$output = [System.IO.File]::Open($target, [System.IO.FileMode]::CreateNew)
try {
    foreach ($name in $names) {
        $inputStream = [System.IO.File]::OpenRead((Join-Path $PSScriptRoot $name))
        try { $inputStream.CopyTo($output) } finally { $inputStream.Dispose() }
    }
} finally { $output.Dispose() }
Write-Host 'Checksums passed. Extract CBM-offline-full.zip with 7-Zip (ZIP64 support).'
