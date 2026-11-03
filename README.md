# Techspeed Velopark for Assetto Corsa

The Techspeed Velopark karting circuit in Nova Santa Rita, Brazil, for Assetto Corsa: eight layouts, from the 227 m kids track to the 2.5 km endurance layout. Released on overtake.gg.

The track itself, with its layouts and changelog, is in [content/tracks/techspeed_velopark](content/tracks/techspeed_velopark); its [README](content/tracks/techspeed_velopark/README.md) is the one that ships with it.

## Folders

| Folder | What is in it |
|---|---|
| `content/` | The track as it is released. Copy `content/tracks/techspeed_velopark` into the game's `content/tracks` to drive it. |
| `tools/` | Python scripts that made the 0.5.1 kerbs, light poles, start panel and road marks, and render previews of them. |
| `tools/reference/` | Photos of the real track and textures taken from the old one, that the tools work from. |
| `overtake/` | The overtake.gg description and its screenshots. |
| `.github/workflows/` | The release action. |

The `.kn5` models and `.dds` textures are stored with Git LFS, as some are over GitHub's 100 MB file limit. Install [Git LFS](https://git-lfs.com) before cloning.

## Making a release

1. Make the changes in `content/` and add a `## <version>` section at the top of `content/tracks/techspeed_velopark/CHANGELOG.md`.
2. Run `.\release.ps1 -Version <version>`. It sets the version in the track's README, layout files and Custom Shaders Patch config, and builds `build\velopark-<version>.7z` to try out. Add `-Install` to copy the track into the game, or `-DryRun` to see what it would change.
3. Commit, then tag and push the tag:

   ```
   git tag v<version>
   git push origin main v<version>
   ```

The Release action builds `velopark-<version>.7z` from `content/` and attaches it to a GitHub release, with the version's changelog section as its notes. Download it from there and upload it to overtake.gg.

## Tools

The tools need Python 3.13 and the packages in `tools/requirements.txt` (`bpy` is Blender as a Python module, used only for the preview renders):

```
pip install -r tools/requirements.txt
```

They also read files that are kept out of the repository:

| Path | What it is | How to get it back |
|---|---|---|
| `base/velopark-0.5.0/` | The 0.5.0 track, which `tools/build_051.py` builds 0.5.1 from. | Unpack the 0.5.0 release there. |
| `tools/textures/` | Kerb, floor kerb, pole and context textures. | `python tools/build_preview.py tex`, then `python tools/build_simple.py tex` |
| `tools/kn5/` | The lowered start panel models. | `python tools/build_simple.py gantry` |
| `build/` | Rebuilt tracks, logs and archives. | Made by the tools and `release.ps1`. |

`tools/build_051.py` copies `base/velopark-0.5.0` to `build/velopark-0.5.1` and rebuilds 0.5.1 there. Do not copy that over `content/` once `content/` has moved past 0.5.1, as it would undo the later changes. `tools/check_051.py` renders the corners of the track in `content/`.
