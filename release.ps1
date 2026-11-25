<#
.SYNOPSIS
    Prepares a new release of the Techspeed FuelTech Velopark track.

.DESCRIPTION
    Updates the version of the track in its sources, source\track: in its README,
    the layout files and the Custom Shaders Patch config. Then builds the three games
    from the sources with the Top Race Editor, and packs and tests
    build\dist\techspeed_fueltech_velopark_ac.7z for Assetto Corsa,
    build\dist\techspeed_fueltech_velopark_beamng.zip for BeamNG.drive and
    build\dist\techspeed_fueltech_velopark_rf2.7z for rFactor 2. Their names are the
    same in every version, so a new archive replaces the old one.

    Add a "## <version>" section to CHANGELOG.md first: the release
    stops without one. Then commit, tag v<version> and push the tag: the Release
    action builds the same three archives from the sources and attaches them to a
    GitHub release, ready to upload to overtake.gg.

.EXAMPLE
    .\release.ps1 -Version 0.8.0
    Updates the version and builds the three archives.

.EXAMPLE
    .\release.ps1 -Version 0.8.0 -Install
    Also copies the track into Assetto Corsa.

.EXAMPLE
    .\release.ps1 -Version 0.8.0 -SkipBuild
    Only updates the version: the Release action builds the archives.

.EXAMPLE
    .\release.ps1 -Version 0.8.0 -DryRun
    Shows what would change and touches nothing.

#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,

    # Copy the track into Assetto Corsa after building.
    [switch]$Install,

    # Only update the version, build nothing.
    [switch]$SkipBuild,

    # Report what would happen and change nothing.
    [switch]$DryRun,

    [string]$GameTracks = 'D:\SteamLibrary\steamapps\common\assettocorsa\content\tracks',

    # The Top Race Editor that builds the track: its installed Top Race Editor.exe (or the
    # folder holding it), or its project's folder, built and run with npm. By default
    # TOP_RACE_EDITOR, else the installed one, else ..\top-race-editor.
    [string]$Editor
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$trackName = 'techspeed_fueltech_velopark'
$noBom = New-Object System.Text.UTF8Encoding($false)

function Update-File {
    param([string]$Path, [string]$Pattern, [string]$Replacement)
    $old = [IO.File]::ReadAllText($Path)
    $new = [regex]::Replace($old, $Pattern, $Replacement)
    if ($new -ceq $old) { return }
    Write-Host "  update $Path"
    if (-not $DryRun) { [IO.File]::WriteAllText($Path, $new, $noBom) }
}

# The editor to build with: its program, or its project (npm run build-track).
function Find-Editor {
    $places = @()
    if ($Editor) { $places += $Editor }
    elseif ($env:TOP_RACE_EDITOR) { $places += $env:TOP_RACE_EDITOR }
    else {
        $places += Join-Path $env:LOCALAPPDATA 'Programs\Top Race Editor'
        $places += Join-Path $env:LOCALAPPDATA 'Programs\top-race-editor'
        $places += Join-Path (Split-Path $root -Parent) 'top-race-editor'
    }
    foreach ($place in $places) {
        if (Test-Path $place -PathType Leaf) { return @{ Program = $place } }
        $exe = Join-Path $place 'Top Race Editor.exe'
        if (Test-Path $exe -PathType Leaf) { return @{ Program = $exe } }
        if (Test-Path (Join-Path $place 'desktop\launch.mjs') -PathType Leaf) { return @{ Project = $place } }
    }
    throw "The Top Race Editor was not found (looked in: $($places -join ', ')). Pass -Editor or set TOP_RACE_EDITOR."
}

# The track's files as they ship, besides its models: the build copies them.
$track = Join-Path $root 'source\track'
if (-not (Test-Path $track)) { throw "Track sources not found: $track" }

$config = Join-Path $track 'extension\ext_config.ini'
$oldVersion = [regex]::Match([IO.File]::ReadAllText($config), '(?m)^VERSION = (\d+\.\d+\.\d+)').Groups[1].Value
Write-Host "Current version: $oldVersion, new version: $Version"
if ($oldVersion -and [version]$Version -lt [version]$oldVersion) { throw "$Version is older than $oldVersion." }

# Nothing is changed until these checks pass.
$changelog = Join-Path $root 'CHANGELOG.md'
$heading = '(?m)^## ' + [regex]::Escape($Version) + '\s*$'
if ([IO.File]::ReadAllText($changelog) -notmatch $heading) {
    throw "CHANGELOG.md has no '## $Version' section. Add one first."
}

$build = Join-Path $root 'build'
$dist = Join-Path $build 'dist'
$archive = Join-Path $dist "${trackName}_ac.7z"
$beamng = Join-Path $dist "${trackName}_beamng.zip"
$rf2 = Join-Path $dist "${trackName}_rf2.7z"

if (-not $SkipBuild) {
    $sevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source
    if (-not $sevenZip) { $sevenZip = 'C:\Program Files\7-Zip\7z.exe' }
    if (-not (Test-Path $sevenZip)) { throw '7-Zip not found.' }
    $found = Find-Editor
    if ($found.Program) { Write-Host "Editor: $($found.Program)" } else { Write-Host "Editor: $($found.Project) (npm run build-track)" }

    if ($Install -and (Get-Process acs -ErrorAction SilentlyContinue)) {
        throw 'Assetto Corsa is running. Close it before using -Install.'
    }
}

$today = (Get-Date).ToString('d MMMM yyyy', [Globalization.CultureInfo]::InvariantCulture)
Update-File (Join-Path $track 'README.md') '(?m)^Version \d+\.\d+\.\d+,' "Version $Version,"
Update-File $config '(?m)^; Techspeed FuelTech Velopark \d+\.\d+\.\d+' "; Techspeed FuelTech Velopark $Version"
Update-File $config '(?m)^VERSION = \d+\.\d+\.\d+' "VERSION = $Version"
if ($Version -ne $oldVersion) {
    Update-File $config '(?m)^DATE_RELEASE = .*$' "DATE_RELEASE = $today"
}
Get-ChildItem -Path (Join-Path $track 'ui') -Recurse -Filter 'ui_track.json' | ForEach-Object {
    Update-File $_.FullName '"version":\s*"[^"]*"' ('"version": "' + $Version + '"')
}

if ($DryRun) {
    Write-Host 'Dry run: nothing was changed.'
    return
}
if ($SkipBuild) {
    Write-Host "Version set to $Version. Next: commit, then 'git tag v$Version' and 'git push origin main v$Version': the Release action builds the archives."
    return
}

# The three games from the sources, everything made again as the Release action does; the
# editor packs the Assetto Corsa and BeamNG.drive archives itself. -Install copies the track
# into the folder -GameTracks names.
$buildArgs = @('--build', '--project', $root, '--games', 'ac,beamng,rf2', '--all')
if ($Install) { $buildArgs += '--install' }
$env:AC_TRACKS = $GameTracks
if ($found.Program) {
    # (a windowed program: PowerShell waits for it, and has its exit code, only so)
    $quoted = $buildArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $run = Start-Process -FilePath $found.Program -ArgumentList $quoted -Wait -NoNewWindow -PassThru
    $code = $run.ExitCode
}
else {
    & npm.cmd --prefix $found.Project run build-track -- @($buildArgs | Select-Object -Skip 1)
    $code = $LASTEXITCODE
}
if ($code -ne 0) { throw "The build failed with exit code $code." }
if (-not (Test-Path $archive)) { throw "The build did not make $archive." }
if (-not (Test-Path $beamng)) { throw "The build did not make $beamng." }

# rFactor 2's location, built beside the final name, so a failed pack leaves the old archive intact.
$location = Join-Path $build "rf2\Locations\$trackName"
if (-not (Test-Path $location)) { throw "The build did not make $location." }
$temp = "$rf2.new"
if (Test-Path $temp) { Remove-Item -LiteralPath $temp -Confirm:$false }
Push-Location (Join-Path $build 'rf2\Locations')
try {
    & $sevenZip a -t7z -mx=9 $temp $trackName | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
& $sevenZip t $temp | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'The archive failed its test.' }
Move-Item -LiteralPath $temp -Destination $rf2 -Force

foreach ($a in @($archive, $beamng, $rf2)) {
    Write-Host ("Built {0} ({1:N1} MB)" -f $a, ((Get-Item $a).Length / 1MB))
}
Write-Host "Next: commit, then 'git tag v$Version' and 'git push origin main v$Version' to publish it."
