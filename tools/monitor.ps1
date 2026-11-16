<#
.SYNOPSIS
    Logs the machine's load every few seconds, to find what goes on before a hang.

.DESCRIPTION
    Every -Every seconds, one line of build\monitor\monitor_<start>.csv: the time; the CPU's
    use (all of it, and its busiest thread), its effective clock (% of its 3.6 GHz base), the
    ACPI thermal zones (a board sensor, not the CPU's own); memory free and committed; the disks'
    load; the graphics card's temperature, power, use and clocks (nvidia-smi); and the three
    processes using the most CPU, and the most memory. Each line is written through to the disk
    as it is made, so the log keeps its last lines when the machine hangs.

    The CPU's own temperature, power and voltages need a sensor driver Windows has not: run
    HWiNFO64 with its own CSV log beside this, and line the two up by their times.

.EXAMPLE
    powershell -File tools\monitor.ps1
    Logs until stopped (Ctrl+C), every 2 seconds.
#>
param(
    [double]$Every = 2,
    [string]$Out = (Join-Path (Split-Path $PSScriptRoot) 'build\monitor')
)
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$path = Join-Path $Out ('monitor_{0}.csv' -f (Get-Date -Format 'yyyy-MM-dd_HH-mm-ss'))
# written through to the disk, not left in the cache, so a hang loses nothing written
$fs = [IO.FileStream]::new($path, [IO.FileMode]::Append, [IO.FileAccess]::Write, [IO.FileShare]::Read, 4096, [IO.FileOptions]::WriteThrough)
$w = [IO.StreamWriter]::new($fs)
function Line($s) { $w.WriteLine($s); $w.Flush(); $fs.Flush($true) }

$counters = @(
    '\Processor Information(_Total)\% Processor Utility',
    '\Processor Information(_Total)\% Processor Performance',
    '\Processor Information(*)\% Processor Utility',
    '\Thermal Zone Information(*)\Temperature',
    '\Memory\Available MBytes',
    '\Memory\% Committed Bytes In Use',
    '\Memory\Committed Bytes',
    '\PhysicalDisk(_Total)\% Disk Time',
    '\PhysicalDisk(_Total)\Disk Bytes/sec'
)
$gpu = [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)
Line 'time,cpu_pct,cpu_busiest_thread_pct,cpu_clock_pct_of_base,thermal_zones_c,mem_free_gb,commit_pct,commit_gb,disk_pct,disk_mb_s,gpu_temp_c,gpu_power_w,gpu_pct,gpu_clock_mhz,top_cpu,top_mem'
Write-Host "logging to $path every $Every s; Ctrl+C to stop"
$cores = [Environment]::ProcessorCount
$prev = @{}
$prevTime = Get-Date
try {
    while ($true) {
        $c = (Get-Counter $counters -ErrorAction SilentlyContinue).CounterSamples
        $v = { param($p) ($c | Where-Object { $_.Path -like "*$p" } | Select-Object -First 1).CookedValue }
        $threads = $c | Where-Object { $_.Path -like '*% processor utility' -and $_.InstanceName -notmatch '_total' } | ForEach-Object CookedValue
        $zones = ($c | Where-Object { $_.Path -like '*thermal zone*' } | ForEach-Object { '{0:N1}' -f ($_.CookedValue - 273.15) }) -join ' '
        $g = ',,,'
        if ($gpu) {
            $q = nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks.gr --format=csv,noheader,nounits 2>$null
            if ($q) { $g = ($q -split ',\s*') -join ',' }
        }
        # the processes using the most CPU since the last line, and the most memory
        $now = Get-Date
        $span = ($now - $prevTime).TotalSeconds * $cores
        $procs = Get-Process | Where-Object { $_.Id -ne 0 }
        $use = foreach ($p in $procs) {
            $t = $p.TotalProcessorTime.TotalSeconds
            $key = $p.Id
            if ($prev.ContainsKey($key) -and $span -gt 0) {
                [pscustomobject]@{ Name = $p.ProcessName; Id = $p.Id; Cpu = ($t - $prev[$key]) / $span * 100; Mem = $p.WorkingSet64 }
            }
            $prev[$key] = $t
        }
        $prevTime = $now
        $topCpu = ($use | Sort-Object Cpu -Descending | Select-Object -First 3 | ForEach-Object { '{0}({1}) {2:N0}%' -f $_.Name, $_.Id, $_.Cpu }) -join '; '
        $topMem = ($procs | Sort-Object WorkingSet64 -Descending | Select-Object -First 3 | ForEach-Object { '{0}({1}) {2:N1}GB' -f $_.ProcessName, $_.Id, ($_.WorkingSet64 / 1GB) }) -join '; '
        Line ('{0},{1:N0},{2:N0},{3:N0},{4},{5:N1},{6:N0},{7:N1},{8:N0},{9:N1},{10},"{11}","{12}"' -f
            $now.ToString('yyyy-MM-dd HH:mm:ss'), (& $v '% processor utility'), ($threads | Measure-Object -Maximum).Maximum,
            (& $v '% processor performance'), $zones, ((& $v 'available mbytes') / 1024), (& $v '% committed bytes in use'),
            ((& $v 'committed bytes') / 1GB), (& $v '% disk time'), ((& $v 'disk bytes/sec') / 1MB), $g, $topCpu, $topMem)
        Start-Sleep -Seconds $Every
    }
} finally {
    $w.Dispose()
}
