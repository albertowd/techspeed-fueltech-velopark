<#
.SYNOPSIS
    Prepares a new release of the Techspeed Velopark track.

.DESCRIPTION
    Updates the version of the track in its sources, source\track: in its README,
    the layout files and the Custom Shaders Patch config. Then builds both games
    from source\ with tools\build_tracks.py, makes content\ the same as the
    Assetto Corsa build, and packs and tests build\velopark-<version>.7z for
    Assetto Corsa and build\velopark-beamng-<version>.zip for BeamNG.drive.

    Add a "## <version>" section to source\track\CHANGELOG.md first: the release
    stops without one. Then commit, tag v<version> and push the tag: the Release
    action builds the same two archives from content\ and attaches them to a
    GitHub release, ready to upload to overtake.gg.

.EXAMPLE
    .\release.ps1 -Version 0.6.0
    Updates the version and builds both archives.

.EXAMPLE
    .\release.ps1 -Version 0.6.0 -Install
    Also copies the track into the game.

.EXAMPLE
    .\release.ps1 -Version 0.6.0 -DryRun
    Shows what would change and touches nothing.

.EXAMPLE
    .\release.ps1 -Version 0.5.1 -Force
    Rebuilds the archives of the current version.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,

    # Copy the track into the game after building.
    [switch]$Install,

    # Overwrite archives that already exist.
    [switch]$Force,

    # Report what would happen and change nothing.
    [switch]$DryRun,

    [string]$GameTracks = 'D:\SteamLibrary\steamapps\common\assettocorsa\content\tracks',

    # The Python with numpy and pillow (in tools\requirements.txt), for the build.
    # By default the one in .venv, else the one on the path.
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$trackName = 'techspeed_velopark'
$noBom = New-Object System.Text.UTF8Encoding($false)

function Update-File {
    param([string]$Path, [string]$Pattern, [string]$Replacement)
    $old = [IO.File]::ReadAllText($Path)
    $new = [regex]::Replace($old, $Pattern, $Replacement)
    if ($new -ceq $old) { return }
    Write-Host "  update $Path"
    if (-not $DryRun) { [IO.File]::WriteAllText($Path, $new, $noBom) }
}

# The track's files as they ship, besides its models: the build copies them.
$track = Join-Path $root 'source\track'
if (-not (Test-Path $track)) { throw "Track sources not found: $track" }

$config = Join-Path $track 'extension\ext_config.ini'
$oldVersion = [regex]::Match([IO.File]::ReadAllText($config), '(?m)^VERSION = (\d+\.\d+\.\d+)').Groups[1].Value
Write-Host "Current version: $oldVersion, new version: $Version"
if ($oldVersion -and [version]$Version -lt [version]$oldVersion) { throw "$Version is older than $oldVersion." }

# Nothing is changed until these checks pass.
$changelog = Join-Path $track 'CHANGELOG.md'
$heading = '(?m)^## ' + [regex]::Escape($Version) + '\s*$'
if ([IO.File]::ReadAllText($changelog) -notmatch $heading) {
    throw "source\track\CHANGELOG.md has no '## $Version' section. Add one first."
}

$build = Join-Path $root 'build'
$archive = Join-Path $build "velopark-$Version.7z"
$beamng = Join-Path $build "velopark-beamng-$Version.zip"
foreach ($a in $archive, $beamng) {
    if ((Test-Path $a) -and -not $Force) { throw "$a already exists. Use -Force to replace it." }
}

$sevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source
if (-not $sevenZip) { $sevenZip = 'C:\Program Files\7-Zip\7z.exe' }
if (-not (Test-Path $sevenZip)) { throw '7-Zip not found.' }

if (-not $Python) {
    $Python = Join-Path $root '.venv\Scripts\python.exe'
    if (-not (Test-Path $Python)) { $Python = 'python' }
}
# Its error output would stop the script with Python's traceback in place of the message below.
$ErrorActionPreference = 'Continue'
& $Python -c 'import numpy, PIL' 2>$null
$ErrorActionPreference = 'Stop'
if ($LASTEXITCODE -ne 0) {
    throw "$Python lacks the packages in tools\requirements.txt. Install them, or pass -Python."
}

if ($Install -and (Get-Process acs -ErrorAction SilentlyContinue)) {
    throw 'Assetto Corsa is running. Close it before using -Install.'
}

$today = (Get-Date).ToString('d MMMM yyyy', [Globalization.CultureInfo]::InvariantCulture)
Update-File (Join-Path $track 'README.md') '(?m)^Version \d+\.\d+\.\d+,' "Version $Version,"
Update-File $config '(?m)^; Techspeed Velopark \d+\.\d+\.\d+' "; Techspeed Velopark $Version"
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

# Both games from the sources, content\ made the same as the Assetto Corsa build, and the
# BeamNG archive packed. -Install copies the track into the folder -GameTracks names.
$buildArgs = @('--pack', '--content', '--check')
if ($Install) { $buildArgs += '--install' }
$env:AC_TRACKS = $GameTracks
& $Python (Join-Path $root 'tools\build_tracks.py') @buildArgs
if ($LASTEXITCODE -ne 0) { throw "The build failed with exit code $LASTEXITCODE." }
if (-not (Test-Path $beamng)) { throw "The build did not make $beamng." }

# Built beside the final name, so a failed build leaves the old archive intact.
New-Item -ItemType Directory -Force -Path $build | Out-Null
$temp = "$archive.new"
if (Test-Path $temp) { Remove-Item -LiteralPath $temp -Confirm:$false }
Push-Location (Join-Path $build 'ac')
try {
    & $sevenZip a -t7z -mx=9 $temp content | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "7-Zip failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
& $sevenZip t $temp | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'The archive failed its test.' }
Move-Item -LiteralPath $temp -Destination $archive -Force
Write-Host ("Built {0} ({1:N1} MB)" -f $archive, ((Get-Item $archive).Length / 1MB))

Write-Host ("Built {0} ({1:N1} MB)" -f $beamng, ((Get-Item $beamng).Length / 1MB))
Write-Host "Next: commit, then 'git tag v$Version' and 'git push origin v$Version' to publish it."
