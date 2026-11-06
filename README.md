# Techspeed Velopark for Assetto Corsa

The Techspeed Velopark karting circuit in Nova Santa Rita, Brazil, for Assetto Corsa: eight layouts, from the 227 m kids track to the 2.5 km endurance layout. Released on overtake.gg.

The track itself, with its layouts and changelog, is in [content/tracks/techspeed_velopark](content/tracks/techspeed_velopark); its [README](content/tracks/techspeed_velopark/README.md) is the one that ships with it.

| | |
|---|---|
| ![The grid](docs/screens/1-grid.jpg) | ![The kids track](docs/screens/2-kids.jpg) |
| ![The start panel at night](docs/screens/3-gantry-night.jpg) | ![The track at night](docs/screens/4-night-overview.jpg) |

## Folders

| Folder | What is in it |
|---|---|
| `content/` | The track as it is released. Copy `content/tracks/techspeed_velopark` into the game's `content/tracks` to drive it. |
| `source/` | The track's sources from 0.6.0: models, textures and track files, that `tools/build_tracks.py` builds each game from. See [Sources](#sources). |
| `tools/` | Python scripts that made the 0.5.1 kerbs, light poles, start panel and road marks, and render previews of them. |
| `tools/reference/` | Photos of the real track and textures taken from the old one, that the tools work from. |
| `docs/` | The screenshots above, which the overtake.gg page uses too, and the track map below. |
| `overtake/` | The overtake.gg description. |
| `.github/workflows/` | The release action. |

The `.kn5` models and `.dds` textures are stored with Git LFS, as some are over GitHub's 100 MB file limit. Install [Git LFS](https://git-lfs.com) before cloning.

## Track map

Every road and kerb of the track by name, to point at one when talking about a change. Roads are the `1ROAD_` meshes of `velopark.kn5` (`R17` is `1ROAD_17`). Kerbs are the `1ROAD_KERB_` meshes of `velopark_kerbs.kn5`, named as in `tools/build_051.py`; the vp_1000's start with `K1_`. The kids track's roads are the `1ROAD_KIDS_` meshes of `velopark_kids.kn5`. Blue kerbs are concrete, red ones painted ramps, orange ones floor kerbs, and purple ones the old kerbs still there. The map shows the track as it lies on the ground: x runs to the left and z up.

[![Track map](docs/track_map.png)](docs/track_map.png)

After changing the track, `python tools/track_map.py` draws it again from `content/`.

## Making a release

1. Make the changes in `source/` and add a `## <version>` section at the top of `source/track/CHANGELOG.md`.
2. Run `.\release.ps1 -Version <version>`. It sets the version in the track's README, layout files and Custom Shaders Patch config in `source/track`, builds both games with `tools/build_tracks.py`, makes `content/` the same as the Assetto Corsa build, and packs `build\velopark-<version>.7z` for Assetto Corsa and `build\velopark-beamng-<version>.zip` for BeamNG.drive. The build runs the Python in `.venv`, or the one on the path; pass `-Python <path>` to use another. Add `-Install` to copy the track into the game, or `-DryRun` to see what it would change.
3. Commit, then tag and push the tag:

   ```
   git tag v<version>
   git push origin main v<version>
   ```

The Release action builds `velopark-<version>.7z` and `velopark-beamng-<version>.zip` from `content/` and attaches both to a GitHub release, with the version's changelog section as its notes. Download the 7z from there and upload it to overtake.gg; the zip is the BeamNG.drive download.

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
| `build/` | Rebuilt tracks, logs and archives. | Made by the tools, `tools/build_tracks.py` and `release.ps1`. |

## Sources

From 0.6.0 the track is made from its sources in `source/`, for every game, instead of being processed again from the 0.5.0 track:

| Path | What it is |
|---|---|
| `source/models/<name>.glb` | Each model of the track as a glTF binary, which Blender opens. Everything Assetto Corsa needs that glTF has no place for (shaders, material values, mesh flags, LOD distances) is kept in the glTF custom properties under `kn5`. |
| `source/textures/` | The textures, each once, which the models point at. A texture two models hold different versions of goes in a folder named after the model, as `velopark_paddock/TarmacDark.dds`. |
| `source/track/` | The rest of the track as it ships: layouts, AI lines, ui, the Custom Shaders Patch config, README and changelog. |

`python tools/build_tracks.py` builds both games from them:

- `build/ac/content/tracks/techspeed_velopark`, the Assetto Corsa track;
- `build/beamng/levels/techspeed_velopark`, one layout for BeamNG.drive (`--layout`, `vp_1500` by default), unpacked: copy `build/beamng` into BeamNG's `mods/unpacked/velopark` to try it. `--pack` packs the zip too.

Pass `ac` or `beamng` to build one game only. Only what changed since the last build is made again, so after editing a model a build takes a second or two; `--all` makes everything again. `--install` copies what changed into Assetto Corsa (the folder in `AC_TRACKS`, else the one `release.ps1` uses). `--check` compares the AC build with `content/` file for file.

To edit a model in Blender: File > Import > glTF 2.0 the `.glb`, then File > Export > glTF 2.0 over it, as glTF Binary, with Include > Custom Properties on and Images set to None. The build then puts back what Blender changes: names it made unique with `.001`, moved objects, missing tangents, textures it leaves out. Faces with no area are dropped. A new material without `kn5` properties becomes a plain `ksPerPixel` one with its base colour texture; copy the properties from another material to choose its shader.

`tools/extract_sources.py` made `source/` from `content/` once, and checked that every model comes back the same, byte for byte. The kn5 reading and writing are in `tools/kn5tree.py`, the glTF ones in `tools/gltf_source.py`.

`tools/export_beamng.py [layout]` converts the track in `content/` into a BeamNG.drive level, `build/velopark-beamng-<version>.zip`; `release.ps1` runs it every time. To try it: drop the zip in BeamNG's `mods` folder. It carries over the models, textures and surfaces of one layout (`vp_1500` by default), the lights, the tyre stacks' simpler copies for far away as their BeamNG detail levels, a start for every layout to pick from, with its picture, the layouts' racing lines as BeamNG's road network, which its map draws and its AI drives, and a timed race for each layout (`tools/beamng_races.py`): a marker at its start that offers it, with a countdown, checkpoints and lap times. The start gantry's board shows the best ten laps of the layout raced under it and the race clock, and its start lamps follow the countdown, as `tools/beamng_board.py` lays them out and the level's Lua (`tools/beamng/`) runs them. Road traffic is off. BeamNG has no lights that come on at night only, so they are lit by day too. As BeamNG grows no grass on meshes, `tools/beamng_grass.py` stands clumps of it where Assetto Corsa grows its own, near the racing line.

`tools/build_051.py` copies `base/velopark-0.5.0` to `build/velopark-0.5.1` and rebuilds 0.5.1 there. Do not copy that over `content/` once `content/` has moved past 0.5.1, as it would undo the later changes. `tools/check_051.py` renders the corners of the track in `content/`.
