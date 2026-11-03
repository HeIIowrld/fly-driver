param(
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [int]$MaximumSeconds = 120
)
$ErrorActionPreference = 'Stop'
if ($MaximumSeconds -lt 2 -or $MaximumSeconds -gt 120) { throw 'Bounded2..120s required' }
New-Item -ItemType Directory -Path $OutputDirectory -ErrorAction Stop | Out-Null
$resourceRoot = (Resolve-Path -LiteralPath $OutputDirectory).Path
$resourceStart = [Diagnostics.Stopwatch]::GetTimestamp() / [double][Diagnostics.Stopwatch]::Frequency
$resourceSamples = [Collections.Generic.List[object]]::new()
$resourceErrors = [Collections.Generic.List[string]]::new()
$resourceReady = $false
$resourceState = 'failed'
function ResourceClock {
    return [Diagnostics.Stopwatch]::GetTimestamp() / [double][Diagnostics.Stopwatch]::Frequency
}
try {
    while ((ResourceClock)-$resourceStart -lt $MaximumSeconds) {
        if (Test-Path -LiteralPath (Join-Path $resourceRoot 'stop')) { break }
        $queryStarted = ResourceClock
        $queryErrors = @()
        $query = Get-Counter '\GPU Engine(*)\Utilization Percentage' -SampleInterval 1 `
            -MaxSamples 1 -ErrorAction SilentlyContinue -ErrorVariable queryErrors
        $queryFinished = ResourceClock
        foreach ($counterError in $queryErrors) { $resourceErrors.Add($counterError.ToString()) }
        $engines = @($query.CounterSamples | ForEach-Object {
            $number = $_.CookedValue
            if ($number -gt .5 -or $_.Status -ne 0) {
                if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { $number = $null }
                [ordered]@{name=$_.InstanceName; percent=$number; status=$_.Status}
            }
        })
        $owners = @($engines | ForEach-Object {
            if ($_.name -match '^pid_(\d+)_') { [int]$Matches[1] }
        } | Sort-Object -Unique | ForEach-Object {
            $ownerId = $_
            $ownerProcess = $null
            try {
                $ownerProcess = [Diagnostics.Process]::GetProcessById($ownerId)
                [ordered]@{pid=$ownerId; name=$ownerProcess.ProcessName;
                    exited=$ownerProcess.HasExited; readable=$true}
            } catch {
                [ordered]@{pid=$ownerId; readable=$false; error=$_.Exception.GetType().Name}
            } finally {
                if ($null -ne $ownerProcess) { $ownerProcess.Dispose() }
            }
        })
        $resourceSamples.Add([ordered]@{query_started_perf_counter=$queryStarted;
            query_finished_perf_counter=$queryFinished; engines=$engines; owners=$owners})
        if (-not $resourceReady) {
            [ordered]@{pid=$PID; started_perf_counter=$resourceStart;
                first_query_finished_perf_counter=$queryFinished} | ConvertTo-Json |
                Set-Content -LiteralPath (Join-Path $resourceRoot 'ready.json') -Encoding utf8
            $resourceReady = $true
        }
        # The parent reads this only after the same owned process exits.
        @($resourceSamples) | ConvertTo-Json -Depth 8 |
            Set-Content -LiteralPath (Join-Path $resourceRoot 'samples.json') -Encoding utf8
    }
    $resourceState = 'completed_resource_window'
} finally {
    [ordered]@{status=$resourceState; pid=$PID; actuator_messages_sent=0;
        started_perf_counter=$resourceStart; finished_perf_counter=(ResourceClock);
        samples=$resourceSamples.Count; errors=@($resourceErrors);
        conservative_windows=$true; process_mutations=0} | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath (Join-Path $resourceRoot 'result.json') -Encoding utf8
}
