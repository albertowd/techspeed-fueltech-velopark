<#
.SYNOPSIS
    Prepares a new release of the Techspeed Velopark track.

.DESCRIPTION
    Updates the version of the track in content\ in its README, the layout
    files and the Custom Shaders Patch config, then builds and tests
    build\velopark-<version>.7z to try out.

    Add a "## <version>" section to CHANGELOG.md first: the release stops
    without one. Then commit, tag v<version> and push the tag: the Release
    action builds the same archive and attaches it to a GitHub release, ready
    to upload to overtake.gg.

.EXAMPLE
    .\release.ps1 -Version 0.6.0
    Updates the version and builds the archive.

.EXAMPLE
    .\release.ps1 -Version 0.6.0 -Install
    Also copies the track into the game.

.EXAMPLE
    .\release.ps1 -Version 0.6.0 -DryRun
    Shows what would change and touches nothing.

.EXAMPLE
    .\release.ps1 -Version 0.5.1 -Force
    Rebuilds the archive of the current version.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d+\.\d+\.\d+$')]
    [string]$Version,

    # Copy the track into the game after building.
    [switch]$Install,

    # Overwrite an archive that already exists.
    [switch]$Force,

    # Report what would happen and change nothing.
    [switch]$DryRun,

    [string]$GameTracks = 'D:\SteamLibrary\steamapps\common\assettocorsa\content\tracks'
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

$track = Join-Path $root "content\tracks\$trackName"
if (-not (Test-Path $track)) { throw "Track folder not found: $track" }

$config = Join-Path $track 'extension\ext_config.ini'
$oldVersion = [regex]::Match([IO.File]::ReadAllText($config), '(?m)^VERSION = (\d+\.\d+\.\d+)').Groups[1].Value
Write-Host "Current version: $oldVersion, new version: $Version"
if ($oldVersion -and [version]$Version -lt [version]$oldVersion) { throw "$Version is older than $oldVersion." }

# Nothing is changed until these checks pass.
$changelog = Join-Path $track 'CHANGELOG.md'
$heading = '(?m)^## ' + [regex]::Escape($Version) + '\s*$'
if ([IO.File]::ReadAllText($changelog) -notmatch $heading) {
    throw "CHANGELOG.md has no '## $Version' section. Add one first."
}

$build = Join-Path $root 'build'
$archive = Join-Path $build "velopark-$Version.7z"
if ((Test-Path $archive) -and -not $Force) { throw "$archive already exists. Use -Force to replace it." }

$sevenZip = (Get-Command 7z -ErrorAction SilentlyContinue).Source
if (-not $sevenZip) { $sevenZip = 'C:\Program Files\7-Zip\7z.exe' }
if (-not (Test-Path $sevenZip)) { throw '7-Zip not found.' }

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

# Built beside the final name, so a failed build leaves the old archive intact.
New-Item -ItemType Directory -Force -Path $build | Out-Null
$temp = "$archive.new"
if (Test-Path $temp) { Remove-Item -LiteralPath $temp -Confirm:$false }
Push-Location $root
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

if ($Install) {
    $dest = Join-Path $GameTracks $trackName
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Copy-Item -Path (Join-Path $track '*') -Destination $dest -Recurse -Force
    Write-Host "Installed to $dest"
}
Write-Host "Next: commit, then 'git tag v$Version' and 'git push origin v$Version' to publish it."
