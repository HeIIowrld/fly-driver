$ErrorActionPreference = 'Stop'
$taskSource = Join-Path $env:USERPROFILE '.codex/flybrain-stage-20261027'
$taskParent = Join-Path $env:USERPROFILE '.codex'
$taskBackup = Join-Path $taskParent 'flybrain-stage-20261027-relocation-backup-20261102'
$taskDestination = 'D:\CodexStorage\flybrain-stage-archive-20261102\flybrain-stage-20261027'
$taskReceipt = 'D:\CodexStorage\flybrain-stage-archive-20261102\receipt.json'
if ((Test-Path -LiteralPath $taskBackup) -or (Test-Path -LiteralPath $taskDestination)) { throw 'Destination or backup already exists' }
$sourceItem = Get-Item -LiteralPath $taskSource -Force
if ($sourceItem.LinkType -or $sourceItem.FullName -ne $taskSource) { throw 'Source must be the exact original directory' }
$running = Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like '*flybrain-stage-20261027*' }
if ($running) { throw 'A process still refers to the source stage' }
$entries = @(Get-ChildItem -LiteralPath $taskSource -Recurse -Force)
if (@($entries | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count -gt 0) { throw 'Reparse point inside source; refusing move' }
$files = @($entries | Where-Object { -not $_.PSIsContainer })
$inventory = foreach ($file in $files) {
 $relative = $file.FullName.Substring($taskSource.Length).TrimStart('\')
 [pscustomobject]@{path=$relative; bytes=$file.Length; modified=$file.LastWriteTimeUtc.Ticks; sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
}
New-Item -ItemType Directory -Path (Split-Path $taskDestination -Parent) -Force | Out-Null
New-Item -ItemType Directory -Path $taskDestination | Out-Null
$sourceAcl=Get-Acl -LiteralPath $taskSource; $sourceAcl.SetAccessRuleProtection($true,$true); Set-Acl -LiteralPath $taskDestination -AclObject $sourceAcl
Get-ChildItem -LiteralPath $taskSource -Force | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $taskDestination -Recurse -Force }
$destinationFiles=@(Get-ChildItem -LiteralPath $taskDestination -Recurse -Force -File)
if ($destinationFiles.Count -ne $inventory.Count) { throw 'Destination file count mismatch' }
foreach ($row in $inventory) {
 $original=Join-Path $taskSource $row.path; $copy=Join-Path $taskDestination $row.path
 $now=Get-Item -LiteralPath $original -Force; $dest=Get-Item -LiteralPath $copy -Force
 if ($now.Length -ne $row.bytes -or $now.LastWriteTimeUtc.Ticks -ne $row.modified -or $dest.Length -ne $row.bytes -or (Get-FileHash -LiteralPath $original -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256 -or (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw "Source or destination changed: $($row.path)" }
}
$again=@(Get-ChildItem -LiteralPath $taskSource -Recurse -Force -File)
if ($again.Count -ne $inventory.Count) { throw 'Source inventory changed' }
$running = Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $PID -and $_.CommandLine -like '*flybrain-stage-20261027*' }
if ($running) { throw 'A process started using source; no rename' }
$receipt=[ordered]@{status='copied_and_verified'; source=$taskSource; destination=$taskDestination; file_count=$inventory.Count; bytes=($inventory | Measure-Object -Property bytes -Sum).Sum; files=$inventory}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $taskReceipt -Encoding utf8
Rename-Item -LiteralPath $taskSource -NewName (Split-Path $taskBackup -Leaf)
try {
 New-Item -ItemType Junction -Path $taskSource -Target $taskDestination | Out-Null
 $junction=Get-Item -LiteralPath $taskSource -Force
 if ($junction.LinkType -ne 'Junction' -or $junction.Target.TrimEnd('\') -ne $taskDestination.TrimEnd('\')) { throw 'Junction target mismatch' }
 foreach ($row in $inventory) { if ((Get-FileHash -LiteralPath (Join-Path $taskSource $row.path) -Algorithm SHA256).Hash.ToLowerInvariant() -ne $row.sha256) { throw 'Junction read mismatch' } }
} catch {
 if (Test-Path -LiteralPath $taskSource) { (Get-Item -LiteralPath $taskSource -Force).Delete() }
 Rename-Item -LiteralPath $taskBackup -NewName (Split-Path $taskSource -Leaf)
 throw
}
$backupItem=Get-Item -LiteralPath $taskBackup -Force
if ($backupItem.FullName -ne $taskBackup -or $backupItem.LinkType -or $taskBackup -ne (Join-Path $taskParent 'flybrain-stage-20261027-relocation-backup-20261102')) { throw 'Backup removal target mismatch' }
Remove-Item -LiteralPath $taskBackup -Recurse -Force
$receipt.status='moved_verified_junction'; $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $taskReceipt -Encoding utf8
[pscustomobject]@{status=$receipt.status; files=$receipt.file_count; bytes=$receipt.bytes; source=$taskSource; destination=$taskDestination}
