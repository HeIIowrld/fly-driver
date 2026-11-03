$ErrorActionPreference = 'Stop'
$source = [IO.Path]::GetFullPath('C:\Users\jeff4\.cache\ecl\terra866-teleport-development-r1')
$sourceParent = [IO.Path]::GetFullPath('C:\Users\jeff4\.cache\ecl')
$backup = [IO.Path]::GetFullPath('C:\Users\jeff4\.cache\ecl\terra866-teleport-development-r1__verified_move_20261101')
$destinationParent = [IO.Path]::GetFullPath('D:\CodexResearchRuns\ecl-relocated-20261101')
$destination = [IO.Path]::GetFullPath('D:\CodexResearchRuns\ecl-relocated-20261101\terra866-teleport-development-r1')
if ([IO.Path]::GetDirectoryName($source) -ne $sourceParent -or
    [IO.Path]::GetDirectoryName($backup) -ne $sourceParent -or
    [IO.Path]::GetDirectoryName($destination) -ne $destinationParent) {
    throw 'Explicit source, backup, and destination paths do not match their intended parents.'
}
if (-not (Test-Path -LiteralPath $source -PathType Container) -or
    (Test-Path -LiteralPath $backup) -or (Test-Path -LiteralPath $destination)) {
    throw 'Expected untouched source and unused backup/destination required.'
}
$audit = Get-Content -LiteralPath (Join-Path $source 'ledger-audit.json') -Raw | ConvertFrom-Json
if ($audit.active_policies -ne 0 -or $audit.queued_policies -ne 0 -or
    @($audit.slots | Where-Object { $_.state -ne 'closed' }).Count -ne 0) {
    throw 'Original ledger does not show closed, inactive development slots.'
}
function Assert-NoNamedProcess {
    $active = @(Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID -and $_.CommandLine -and
        $_.CommandLine.Contains('terra866-teleport-development-r1')
    })
    if ($active.Count -ne 0) { throw 'A process command line names this run; preserve the original.' }
}
function Assert-LockFree {
    $lock = [IO.File]::Open((Join-Path $source 'master.lock'), [IO.FileMode]::Open,
        [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $lock.Dispose()
}
Assert-NoNamedProcess
Assert-LockFree
if (@(Get-ChildItem -LiteralPath $source -Recurse -Force -Attributes ReparsePoint).Count -ne 0) {
    throw 'Nested links are not included in this relocation.'
}
New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
New-Item -ItemType Directory -Path $destination | Out-Null
# Restrict D before copying; preserve the source directory's effective access rules.
$acl = Get-Acl -LiteralPath $source
$acl.SetAccessRuleProtection($true, $true)
Set-Acl -LiteralPath $destination -AclObject $acl
$afterAcl = Get-Acl -LiteralPath $destination
if (@($afterAcl.Access | Where-Object { $_.IdentityReference.Value -match 'Authenticated Users' }).Count -ne 0) {
    throw 'Destination is more broadly accessible than the source.'
}
$files = @(Get-ChildItem -LiteralPath $source -File -Recurse -Force)
$initial = @{}
foreach ($file in $files) {
    $relative = $file.FullName.Substring($source.Length).TrimStart('\')
    $initial[$relative] = @{ length = $file.Length; writeUtc = $file.LastWriteTimeUtc.Ticks }
}
Write-Output ("Copying {0} files to D; original remains intact." -f $files.Count)
Get-ChildItem -LiteralPath $source -Force | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $destination -Recurse -Force
}
$copied = @(Get-ChildItem -LiteralPath $destination -File -Recurse -Force)
if ($copied.Count -ne $files.Count -or
    @(Get-ChildItem -LiteralPath $destination -Recurse -Force -Attributes ReparsePoint).Count -ne 0) {
    throw 'Destination inventory differs or contains links; original retained.'
}
$records = [Collections.Generic.List[object]]::new()
$n = 0
foreach ($file in $files) {
    $relative = $file.FullName.Substring($source.Length).TrimStart('\')
    $other = Join-Path $destination $relative
    $current = Get-Item -LiteralPath $file.FullName -Force
    $destItem = Get-Item -LiteralPath $other -Force
    if ($current.Length -ne $initial[$relative].length -or
        $current.LastWriteTimeUtc.Ticks -ne $initial[$relative].writeUtc -or
        $destItem.Length -ne $current.Length) { throw "File changed or copy differs: $relative" }
    $srcHash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash
    $dstHash = (Get-FileHash -LiteralPath $other -Algorithm SHA256).Hash
    if ($srcHash -ne $dstHash) { throw "SHA256 mismatch: $relative" }
    $records.Add([pscustomobject]@{ path=$relative; bytes=$current.Length; sha256=$srcHash;
        source_write_utc_ticks=$current.LastWriteTimeUtc.Ticks })
    $n++
    if ($n % 500 -eq 0) { Write-Output ("Verified {0}/{1} files." -f $n, $files.Count) }
}
Assert-NoNamedProcess
Assert-LockFree
$finalFiles = @(Get-ChildItem -LiteralPath $source -File -Recurse -Force)
if ($finalFiles.Count -ne $files.Count) { throw 'Source inventory changed; original retained.' }
foreach ($f in $finalFiles) {
    $relative = $f.FullName.Substring($source.Length).TrimStart('\')
    if (-not $initial.ContainsKey($relative) -or $f.Length -ne $initial[$relative].length -or
        $f.LastWriteTimeUtc.Ticks -ne $initial[$relative].writeUtc) { throw 'Source changed; original retained.' }
}
$receipt = Join-Path $destinationParent 'terra866-verified-files.json'
@{source=$source; destination=$destination; files=$records; count=$records.Count;
  bytes=(($records | Measure-Object -Property bytes -Sum).Sum); verified_utc=[DateTime]::UtcNow.ToString('o')} |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $receipt -Encoding utf8NoBOM
Write-Output 'Verified all file hashes; replacing original directory with a junction.'
Move-Item -LiteralPath $source -Destination $backup
try {
    New-Item -ItemType Junction -Path $source -Target $destination | Out-Null
    $junction = Get-Item -LiteralPath $source -Force
    if (-not ($junction.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
        $junction.Target -ne $destination) { throw 'Junction verification failed.' }
    $testFile = Join-Path $source 'ledger-audit.json'
    if ((Get-FileHash -LiteralPath $testFile).Hash -ne
        (Get-FileHash -LiteralPath (Join-Path $backup 'ledger-audit.json')).Hash) {
        throw 'Original path does not expose the verified data.'
    }
} catch {
    if (Test-Path -LiteralPath $source) { Remove-Item -LiteralPath $source -Force }
    Move-Item -LiteralPath $backup -Destination $source
    throw
}
$resolvedBackup = (Get-Item -LiteralPath $backup -Force).FullName
if ($resolvedBackup -ne $backup -or [IO.Path]::GetDirectoryName($resolvedBackup) -ne $sourceParent -or
    (Get-Item -LiteralPath $backup -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Backup deletion target is not the explicitly verified original directory.'
}
Remove-Item -LiteralPath $backup -Recurse -Force
@{status='completed'; source=$source; junction_target=$destination; receipt=$receipt;
  file_count=$records.Count; bytes=(($records | Measure-Object -Property bytes -Sum).Sum)} |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $destinationParent 'terra866-result.json') -Encoding utf8NoBOM
Write-Output 'Completed: verified original path now resolves to D.'
