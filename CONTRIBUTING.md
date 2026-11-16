# Contributing to Techspeed FuelTech Velopark

How the track is made, for anyone changing it. Playing it? See the [README](README.md).

The track is made from its sources in [source/](source), its layouts in [source/track](source/track); its [README](source/track/README.md) is the one that ships with it. What changed in each version of the repository is in the [CHANGELOG](CHANGELOG.md): note each change under the next version's section at its top, one line per change, written for the players: what they will see or feel, not how it was made. What is still to be done is in the [TODO](TODO.md), in the same format: move an item to the changelog when it is done. `tools/build_tracks.py` builds both games from the sources (see [Sources](#sources)), and `release.ps1` packs the release (see [Making a release](#making-a-release)).

## Folders

| Folder | What is in it |
|---|---|
| `source/` | The track's sources from 0.6.0: models, textures and track files, that `tools/build_tracks.py` builds each game from. See [Sources](#sources). |
| `tools/` | The build, the scripts that change the sources, and the scripts that made the 0.5.1 kerbs, light poles, start panel and road marks and render previews of them. |
| `tools/reference/` | Photos of the real track and textures taken from the old one, that the tools work from. |
| `tools/reference/elevation/` | Google's heights of the real ground, where the track lies on the Earth, and how its ground compares, from `tools/elevation.py`. See its [README](tools/reference/elevation/README.md). |
| `docs/` | The README's screenshots, of each game in `screens/ac` and `screens/beamng` (the overtake.gg page uses the Assetto Corsa ones too), and the track and objects maps below. |
| `overtake/` | The overtake.gg description. |
| `.github/workflows/` | The release action. |

The source models (`.glb`) and textures are stored with Git LFS, as some are over GitHub's 100 MB file limit. Install [Git LFS](https://git-lfs.com) before cloning.

## Track map

Every road and kerb of the track by name, to point at one when talking about a change. Roads are the `1ROAD_` meshes of `velopark.kn5` (`R17` is `1ROAD_17`). Kerbs are the `1ROAD_KERB_` meshes of `velopark_kerbs.kn5`, named as in `tools/build_051.py`, the round ones as in `tools/build_round_kerbs.py`; the vp_1000's start with `K1_`. The kids track's roads and kerbs are the `1ROAD_KIDS_` and `1ROAD_KERB_KIDS_` meshes of `velopark_kids.kn5`. Blue kerbs are concrete, red ones painted ramps, yellow ones round kerbs, and orange ones floor kerbs. The map shows the track as it lies on the ground: x runs to the left and z up.

[![Track map](docs/track_map.png)](docs/track_map.png)

After changing the track, `python tools/track_map.py` draws it again from the Assetto Corsa build in `build/ac`.

## Road editor

The roads and rumbles described in `tools/reference/roads.json` (see [ROADS.md](ROADS.md)), and the objects in `tools/reference/objects.json` (the grid, pit boxes, cones, light poles, tyre walls), are edited in the browser with the road editor. It is its own project, top-race-editor, beside this one; it reads this track's `tools/reference/track.json` and its tools (`roads.py`, `surface.py`, `build_kerbs.py`, `track_surface.py`, `read_road.py`), so what it shows is what the build makes. The build makes the models from what it saves.

## Objects map

What stands round the track, by name, to point at one when talking about a change: the buildings by what they are, the tyre walls (`TYRES_` meshes of `velopark_tyres.kn5`) and light poles (`POLE_`, of `velopark_poles.kn5`) by their mesh names, the fences (`velopark_outside.kn5`'s `FENCE_` meshes) by theirs. Grass, trees, the dirt roads and the pond are only background colours, the trees too many to draw each.

[![Objects map](docs/objects_map.png)](docs/objects_map.png)

After changing what stands round the track, `python tools/objects_map.py` draws it again from the Assetto Corsa build in `build/ac`.

## The terrain

The ground the track lies on is a folder of its own, `tools/reference/terrain/`, made first in the build (`tools/terrain.py`), before the roads and everything standing on the map. Its `terrain.json` says where the track lies, its satellite picture and ground photo, the files of the real ground's heights, the filters that take the tops of trees and buildings out of them (their rules, and areas drawn as ground or not ground), and the edits to the ground, in order (a raster of metres added, or a shape made flat, raised or lowered, or smoothed). The middle buildings' platform, the pits and the shed are its first edit, `paddock_platform.npz`. The road editor (top-race-editor) shows and edits it, and makes it for a new track.

`python tools/terrain.py build` writes the original heights (`original.png`), the filtered ones (`filtered.png`) and the ground with its edits (`ground.png`), 16 bit, their scales in `heights.json`, and every height with why a filter took it out (`points.json`). The roads' heights are taken from that ground, so an edit to the terrain records them again and runs every step after it.

## The ground and the surroundings (how it was made)


The track's heights and what stands round it are made from the real ground's heights, the ground photo, OpenStreetMap and screenshots of the real track, by tools run in turn (each finds its work done when run again):

1. `python tools/level_ground.py`: everything at the heights of the real ground, from Google's in `tools/reference/elevation/` (see its [README](tools/reference/elevation/README.md)).
2. `python tools/flatten_roads.py`: that levelling made the same across every road, so it twists none.
3. `python tools/fill_hollows.py`: the near ground's sunk points lifted to the ground round them.
4. `python tools/level_pavers.py`: R29's pavers laid on their plane, the grass brought up to them.
5. `python tools/restyle_paddock.py`: the paddock's buildings in grey block with salmon red pillars and darker roofs, without the race control tower, the fuel station, the buildings in the middle of the track and the garage annex, which `build_outside.py` makes again. Run again, it does only what is left to do.
6. `python tools/build_outside.py`: `source/models/velopark_outside.glb`, with the dirt roads, the fences, the buildings outside the track, the race control tower, the fuel station, the marshals' tents, the flagpoles and the banners; and the buildings in the middle of the track as the screenshots show them, with their rounded roofs, and the big tent by the pit boxes. Every building's windows are steel framed glass in holes in its walls (the race control tower's top lights opened out), with a plastered room behind them and ceiling lamps. It writes the lamps' night lights into the Custom Shaders Patch config as `LIGHT_SERIES_3`, and the fuel station's prices (`FUEL_PRICES`) as a display there, lit digits as the start gantry's board has; to change the prices, edit `FUEL_PRICES` and run it again. A Velopark logo saved as `tools/reference/velopark_logo.png` goes on the fuel station's sign in place of the drawn one. It makes the model again each run.
7. `python tools/plant_outside.py`: the trees round the track in `velopark_trees.glb` and BeamNG's forest, clear of what `build_outside.py` made and of the track's infields; `--prune` takes out any new one standing in them. `--strip` plants the strip between the vp_1000's north tyre wall and the north fence thick with eucalyptus. `--south` makes the grove south of the fuel station tall eucalyptus in place of its broad leaf trees.
8. `python tools/settle_objects.py`: everything that stands on the ground (tyre walls, poles, buildings, trees) set back on it where the ground's large triangles did not follow the levelling; then `build_outside.py` again, which builds on the ground as it is.
9. `python tools/split_tyres.py`: gaps in the tyre walls where something stands in them, as the vp_1000's flag marshal's tent and cones.
10. `python tools/settle_kids.py`: the kids track's roads and kerbs, and its AI lines, set back 1 cm over the ground's large triangles round it, which had come to lie over them; then `python tools/lift_paint.py`, to bring its paint back over them.
11. `python tools/smooth_south.py`: the vp_1500's south straight and the road beside it brought onto Google's heights, smooth: the levelling left a crest of the old track there, by the pit lane's junction. Everything there moves with the roads; then `build_outside.py` again.
12. `python tools/far_ground.py`: the far ground (`GRASS_1`, out to 1.5 km) given its own texture from the satellite pictures round the track in `tools/reference/surroundings` (`tools/surroundings.py photo`), its colours matched to the ground photo's, where it took the ground photo repeated.
13. `python tools/far_terrain.py`: the far ground made again at the real ground's heights, from every height in `tools/reference/elevation` and `tools/reference/surroundings` but the tops of trees and buildings, a point every 20 m out to the satellite pictures' edges, meeting the near ground's edge with no gap. Run again, it makes it again, the same.
14. `python tools/level_ground.py --again`: the whole track and its ground levelled again, onto the same real ground the far ground is made from (trees found in the satellite pictures and the ground photo both), even across every road; then `far_terrain.py` and `build_outside.py` again.
15. `python tools/drape_track.py`: the track draped over the real ground, as Race Track Builder made it. The real ground has the middle buildings' platform on it, a metre over the 1500's side (the heights' filter took it out with the buildings: there are no roofs among those heights, so in a box round it the raw ones, smoothed, raise it, by 0.7 m at most), the pit lanes ramping down from it to both tracks and the 1000's up to it from the storage building's end. The near ground is the real ground (tools/far_terrain.py's, a smooth surface between its 10 m points), every point of it, its faces cut by a grid of 3 m within 15 m of the roads; every road and the paving lie 2 cm (their own height) over it, their skirts as deep under their tops as they were; the kerbs, the paint and the start points move as the roads round them did, keeping their shapes; the rest standing on the ground, the trees, BeamNG's forest and the lights move with the ground under them, each piece smaller than 4 m across (a tyre stack, a post) as one, so none by a road's edge is sheared. Every edge of what is driven on (roads, paving, kerbs) has a ramp of grass down to the grass, 12 cm a metre (from 12 cm wide at a road to a metre behind a tall kerb), its outer end just under it (`1GRASS_EDGE_RAMP`): so the grass never comes over a road, and a tyre leaving the track rolls off onto the grass, which pulls a kart off, with no step to jump. Before it, the roads and the ground each followed the real ground on their own, up to a quarter metre off it, and where they parted the ground stood over an edge or fell under it (a hidden step a kart hit as a wall by the vp_1500's top hairpin's pavers, a drop by R30, a hump outside the S).
16. `python tools/build_outside.py` again, on the ground as it is now, then `python tools/place_karts.py`. (`tools/drape_track.py` makes the far ground again itself, round the whole near ground: `tools/far_terrain.py` had traced the edge of only the first of the near ground's meshes, so the two did not meet all round.)

## Making a release

1. Make the changes in `source/` and add a `## <version>` section at the top of [CHANGELOG.md](CHANGELOG.md), at the root of the repository; the release stops without one, and the Release action uses it as the release notes.
2. Run `.\release.ps1 -Version <version>`. It sets the version in the track's README, layout files and Custom Shaders Patch config in `source/track`, builds both games from the sources with `tools/build_tracks.py`, and packs `build\dist\techspeed_fueltech_velopark_ac.7z` for Assetto Corsa and `build\dist\techspeed_fueltech_velopark_beamng.zip` for BeamNG.drive. The build runs the Python in `.venv`, or the one on the path; pass `-Python <path>` to use another. Add `-Install` to copy the track into the game, or `-DryRun` to see what it would change.
3. Commit, then tag and push the tag:

   ```
   git tag v<version>
   git push origin main v<version>
   ```

The Release action builds `techspeed_fueltech_velopark_ac.7z` and `techspeed_fueltech_velopark_beamng.zip` from `source/`, as `release.ps1` does, and attaches both to a GitHub release, with the version's changelog section as its notes. Download the 7z from there and upload it to overtake.gg; the zip is the BeamNG.drive download.

The archives are named as the Techspeed karts' are: the track's id, `techspeed_fueltech_velopark`, which is also its Assetto Corsa folder and its BeamNG.drive level (and would be an Assetto Corsa EVO version's), then the game, with no version, in letters, digits and `_` only, as BeamNG.drive asks of mod zips. So a new version replaces the old one in the mods folder.

## Tools

The tools need Python 3.13 and the packages in `tools/requirements.txt` (`bpy` is Blender as a Python module, used for the preview renders and the replay cameras' ray casting):

```
pip install -r tools/requirements.txt
```

They also read files that are kept out of the repository:

| Path | What it is | How to get it back |
|---|---|---|
| `base/velopark-0.5.0/` | The 0.5.0 track, which `tools/build_051.py` builds 0.5.1 from. | Unpack the 0.5.0 release there. |
| `tools/textures/` | Kerb, floor kerb, pole and context textures. | `python tools/build_preview.py tex`, then `python tools/build_simple.py tex` |
| `tools/kn5/` | The lowered start panel models. | `python tools/build_simple.py gantry` |
| `build/` | Each game's build in `ac/` and `beamng/`, and the archives to release in `dist/`, as `techspeed_fueltech_velopark_ac.7z` and `techspeed_fueltech_velopark_beamng.zip`. | Made by the tools, `tools/build_tracks.py` and `release.ps1`. |

## Sources

From 0.6.0 the track is made from its sources in `source/`, for every game, instead of being processed again from the 0.5.0 track:

| Path | What it is |
|---|---|
| `source/models/<name>.glb` | Each model of the track as a glTF binary, which Blender opens. Everything Assetto Corsa needs that glTF has no place for (shaders, material values, mesh flags, LOD distances) is kept in the glTF custom properties under `kn5`. |
| `source/textures/` | The textures, each once, which the models point at. A texture two models hold different versions of goes in a folder named after the model, as `velopark_paddock/TarmacDark.dds`. |
| `source/track/` | The rest of the track as it ships: layouts, AI lines, ui, the Custom Shaders Patch config and README. |

The models, each holding one kind of thing (`tools/organize_models.py` put them in order):

| Model | What it holds |
|---|---|
| `velopark.glb` | The ground (near and far), the track's roads and pavers, the edges easing down to the grass, the invisible walls along the track, the cones every layout has. |
| `velopark_kerbs.glb` | The kerbs, with their beds and skirts. |
| `velopark_runoffs.glb` | The asphalt, pavers and grass laid round the track's edges: run-offs, patches, islands, the asphalt round the vp_1000's corners. |
| `velopark_lines.glb` | Every line painted on the track: edge lines, junction lines, grids and their numbers, pit boxes, the car park's bays. |
| `velopark_tyres.glb` | The tyre walls, each with its wall to hit. |
| `velopark_poles.glb` | The light poles, their tyre rings and walls. |
| `velopark_paddock.glb` | The paddock's asphalt (the pits, the car park, the middle) and the infield paths; the old buildings not made again. |
| `velopark_outside.glb` | What `tools/build_outside.py` makes: the buildings, fences, stands, dirt roads and the rest round the track. |
| `velopark_trees.glb` | The trees; BeamNG's forest of them in `velopark_trees.json`. |
| `velopark_kids.glb` | The kids track's roads, kerbs and gate. |
| `velopark_karts.glb` | The parked rental karts. |
| `velopark_gantry_*.glb` | The start gantries, working (`velopark_gantry_1000`, `velopark_gantry_1500`) and still (`_still`) for the layouts that do not start under them. |
| `vp_<layout>.glb` | A layout's own: its cones and distance boards, its starts, pit boxes and timing lines, the lines across the roads it does not take. |

`python tools/build_tracks.py` builds both games from them:

- `build/ac/content/tracks/techspeed_fueltech_velopark`, the Assetto Corsa track;
- `build/beamng/levels/techspeed_fueltech_velopark`, one layout for BeamNG.drive (`--layout`, `vp_1500` by default), unpacked: copy `build/beamng` into BeamNG's `mods/unpacked/techspeed_fueltech_velopark` to try it. `--pack` packs its zip in `build/dist` too.

Pass `ac` or `beamng` to build one game only. Only what changed since the last build is made again, so after editing a model a build takes a second or two; `--all` makes everything again. `--install` copies what changed into Assetto Corsa (the folder in `AC_TRACKS`, else the one `release.ps1` uses).

To edit a model in Blender: File > Import > glTF 2.0 the `.glb`, then File > Export > glTF 2.0 over it, as glTF Binary, with Include > Custom Properties on and Images set to None. The build then puts back what Blender changes: names it made unique with `.001`, moved objects, missing tangents, textures it leaves out. Faces with no area are dropped. A new material without `kn5` properties becomes a plain `ksPerPixel` one with its base colour texture; copy the properties from another material to choose its shader.

`tools/extract_sources.py` made `source/` once, from the 0.5.1 track as released, and checked that every model comes back the same, byte for byte. The kn5 reading and writing are in `tools/kn5tree.py`, the glTF ones in `tools/gltf_source.py`.

`tools/export_beamng.py [layout]` converts the Assetto Corsa build in `build/ac` into a BeamNG.drive level, `build/dist/techspeed_fueltech_velopark_beamng.zip`; `release.ps1` runs it every time. To try it: drop the zip in BeamNG's `mods` folder. It carries over the models, textures and surfaces of one layout (`vp_1500` by default), the lights, the tyre stacks' simpler copies for far away as their BeamNG detail levels, a start for every layout to pick from, with its picture, the layouts' racing lines as BeamNG's road network, which its map draws and its AI drives, with the paths across the infield and the pit lanes beside it, drawn as wide as the asphalt round them, and a timed race for each layout (`tools/beamng_races.py`), which BeamNG's Time Trials list too (the level's `quickrace` folder), and an AI race for each, against karts as the player's own (BeamNG's AI race, its racers driving the player's vehicle), from the layout's grid as AC has it, the player at its back, 9 racers at first and up to 35: a marker at its start that offers it, with a countdown, checkpoints and lap times. The start gantry's board shows the best ten laps of the layout raced under it and the race clock, and its start lamps follow the countdown, as `tools/beamng_board.py` lays them out and the level's Lua (`tools/beamng/`) runs them. Road traffic is off. The grass has a ground model of the level's own, BeamNG's grass with a soft layer 1 cm deep instead of 5, so a kart's tyres do not sink out of sight. The cones are loose cones of BeamNG's cones prop, which a level cannot hold as shapes: BeamNG's cone made the Assetto Corsa cone's size, 0.4 m wide and 0.7 m high, and 1.2 kg, as BeamNG's large cone, 4.2 kg, all but flips a kart that hits it. All the cones of a layout are one vehicle, a part of the cones prop of the level's own that `tools/beamng_races.py` writes (each cone still knocked about on its own), as 75 vehicles, one a cone, took a minute to spawn as a race started and stalled the frames as they woke; each race spawns its own layout's, with their distance boards, from its mission's `mainPrefab.prefab.json`, as BeamNG's own time trials do. Free roam has none for now: BeamNG never puts a vehicle to sleep, so all the layouts' cones and boards together cost about as much as another car to simulate, standing or not. Each tree has a trunk to hit, unseen inside its boards. The eucalyptus plantation of `tools/build_trees.py` is BeamNG's forest: each tree as made is a forest item, with its trunk to hit, placed where Assetto Corsa's copies of it stand, turned and sized as they are (`source/models/velopark_trees.json`), swaying a little in the wind; its understory stays shapes. The lights are BeamNG's night lights, as its own floodlit tracks' are, which the time of day switches on at night, with their intensity set as those tracks set theirs, as BeamNG's lights fall off far faster than Custom Shaders Patch's. As BeamNG grows no grass on meshes, `tools/beamng_grass.py` stands clumps of it where Assetto Corsa grows its own, near the racing line.

`tools/build_parking.py` takes the dirt the 0.5.0 track laid over the pits and the car park south of the pit building out of `source/models/velopark_paddock.glb`, so the paddock's asphalt shows there, and paints the car park's bays in `source/models/velopark_lines.glb`.

`tools/build_models.py` makes the models again from the committed ones, step by step (terrain, organize_models, scan_materials, build_roads, shift_pit_entry, restyle_paddock, drape_track, lay_pavers, merge_floor, place_karts, check_normals, check_holes). Only the roads and rumbles shape the terrain, which the merge makes one surface; what stands on the map (pit boxes, grid slots, poles, tyre walls, buildings) is placed by the road editor (top-race-editor) as data, and is never checked for gaps, then the AC track. It keeps in `build/cache` the files each step changed, and runs only the steps whose code or input changed since last time: a change to the buildings runs build_outside and the steps after it, a change to the roads restyle_paddock and on; the pit lanes' lines, pit boxes and their racing lines are laid by paint_pits after the drape, so a change to them runs from there. `--all` runs every step, `--from STEP` that one and those after it, `--until STEP` stops after that one (the models as it leaves them, to work on the next), `--install` copies the track into the game, `--beamng` packs the BeamNG archive too, `--profile` times each step run into `build/cache/<step>.prof`.

`tools/merge_floor.py` makes the whole map's floor one surface with `tools/surface.py`. In order, each over those after it, the regions are:

1. the rumbles described in roads.json;
2. the old kerbs left;
3. the described roads in their order;
4. the other roads and paving;
5. the grass;
6. the far ground;
7. the near ground.

Each region is cut out where those before it lie, and meets its neighbours point for point at one height. Gaps where nothing lies become ground, unless something standing covers them. The ground eases up to every edge, in place of the drape's ramps.

Each region is proved as soon as it is filled (no open edge but on its outline, no point of its outline missing, its area what it should be), then the whole surface (seconds). The first problem stops the build, with where it is.

The merge is kept between builds (`build/cache/merge_state`, `tools/merge_state.py`). The next merge compares each region's triangles with last time and makes again only the area round those that changed (2.5 m round them); the rest is kept face for face, its edge held point for point. An edit to one road merges in seconds; with nothing changed, nothing is merged. Each region's outline and folds are kept too, by its triangles' fingerprint, and so is the world they make. `--all` merges all of it again. The step writes `source/models/merged_floors.json`, the meshes it made: `tools/check_holes.py` and `tools/seal_holes.py` probe only the other floors' edges.

Options:
- `--near X0 Z0 X1 Z1` builds and proves one part of the map without writing it;
- `--models DIR` reads the floors from another folder, such as `build/cache/floors_before_merge`, and never writes.

`tools/test_surface.py` proves the core on made-up cases: a fork, a joint, rumbles, a run-off, a tight turn, a meshed ground, a world with a hole, and the old meshes' faults (a T, no area, a sliver hole, a tiny fold).

`tools/check_holes.py` finds the holes in the floor: every open edge of the ground, the roads, the kerbs and the floors is probed just past it, and must have a floor there at its height, something over it, or another mesh sharing it; else it is a hole or a crack the sky shows through. `tools/seal_holes.py` first welds the seams: where a point of one floor lies on another's open edge (a T, which the probe does not find, but which opens a slit once the floor is laid on the ground point by point), the edge is split there so both share it. Then it closes the holes with a skirt straight down from each such edge, in its mesh's material. `tools/car_parks.py` holds the car parks south of the paddock as the ground photo shows them (their outlines, the grass round their trees, the rows of bays, the road south carried on through the trees to the dirt road): restyle_paddock lays their asphalt and paint_pits their bays' lines. `tools/lay_pavers.py`, after the drape, lays the strips of paving outside a run-off (R41's) again: from the asphalt's edge, smoothly, to just over the grass. The build runs both, the check last: there can be no hole in the models a track is built from.

`tools/check_normals.py` lists the faces of every model drawn the wrong way round: faces turned against their points' normals, and closed pieces (posts, beams, boxes) turned inside out. With `--fix` it turns the first to their normals (not in the parked karts, the kart project's). Run it after any change to the models; it exits 1 while it finds any.

`tools/organize_models.py` puts each thing in the model it belongs to, as the table above has them: the kerbs' tyre walls, lines and run-offs out of `velopark_kerbs.glb`, `velopark_paint.glb` and the layouts' `vp_<layout>_lines.glb` into `velopark_lines.glb` and the layouts' own models, the asphalt round the vp_1000's corners out of the paddock's, `velopark_extra.glb` renamed `velopark_trees.glb`, and the old invisible walls sunk deep under the ground out of `velopark.glb`; the layouts' model lists follow. Run again, it finds them in order.

`tools/clear_walls.py` takes the 0.5.0 track's invisible walls out of `source/models/velopark.glb` where they stand across its asphalt: between the pit building and the car park, round the road by the buildings to the west and along the strip north of them, where the paddock's asphalt was laid past them.

`tools/build_paths.py` lays the two tyre-lined paths across the infield (A, from R20 down into the vp_1500 pit exit at R25, and D, from R19 to R20) as asphalt of their own in `source/models/velopark_paddock.glb`: between their tyre walls, eased along and across, widened into the roads at their ends, but for A's south end, which runs on straight into the pit exit as a road merging with it. Their asphalt is mapped as most of the paddock's is. The paddock's old strips there become ground in `source/models/velopark.glb`, which is lowered under the paths where they ease below it, with the tyre stacks standing there (`source/models/velopark_tyres.glb`). Run once; run again, it finds the paths laid.

`tools/join_paddock.py` then eases the paddock's asphalt into the track's roads where they meet, as at the car park and the pit lane: within 5 m of a road it bends from the road's height and slope at the seam to its own, so a kart crosses without a lip or a change of angle. Its big faces there are cut into a 1 m grid for it, and the paint on it moves with it. Run again, it finds the seams met.

`tools/grade_paths.py` then grades the ground beside the paths into them, as a bank beside a road would be: within 4 m of each side the grass of `source/models/velopark.glb` bends from just under the path's side to the ground as it is past that, meeting each in height and slope, so a path no longer runs through a bank cut beside it or stands on a wall of its skirts. It fills the dents laying the paths left near them too, and the tyre stacks there go up or down with the ground (`source/models/velopark_tyres.glb`). Where a road or anything else lies on the ground, the ground under it stays. Run again, it finds the ground graded.

`tools/plant_trees.py` then lowers the trees whose trunks end in the air: a tree's texture leaves a gap under its trunk, about 0.75 m on the broad leaf trees, so it lowers each tree until the lowest of its trunk that shows is 10 cm into the ground round it. Run again, it finds the trees planted.

`tools/build_trees.py` plants the eucalyptus plantation along the north and the west of the track, in `source/models/velopark_trees.glb`, in place of the 0.5.0 track's broad leaf and ash trees there. The trees are three dimensional, 27 to 37 m high: a trunk and branches of pale bark, bare up to the small clumps of leaves of the crown, on crossed cards. They stand in rows about 3 m apart in the 14 m of the band nearest the track, 6 m apart behind, where the old trees stood and nowhere within 3 m of a road, building, kerb or tyre wall. At their feet, from 4 m before the band to 9 m into it, a ragged mass of shrubs 3.5 to 8 m high, many small clumps of leaves, and young eucalyptus among it. Each trunk has an unseen wall round it up to 3.5 m, which cars hit in both games. The north west, round the L1500's top hairpin, is no plantation: broad leaf trees there, 9 to 16 m high with round crowns down low, wherever the ground's photo shows the crowns of trees, with the meadow round them as the photo has it. Their textures are made there too (`source/textures/euca_*.dds`). Run again, it finds the plantation planted.

`tools/lift_paint.py` then lifts the paint back over the asphalt wherever the asphalt came up over it: the edge line of the straight out of T2 went under the paddock's asphalt that `tools/join_paddock.py` eased into the road beside it. Every point of paint less than 2 mm over the asphalt under it, or whose face dips under it between its points, goes to 2.5 mm over it, as the rest of the paint lies. Run again, it finds the paint over the asphalt.

`tools/build_pavers.py` paves with concrete blocks, as the real track's are, R29, the light concrete beside the L1500's top hairpin, and the run-offs beyond the floor kerbs STRAIGHT4 (R30) and LAST_OUT (R43, R44), with the pieces of the roads beside them 0.5.1 patched them with, which no layout drives on: so each run-off is one paving. Beyond STRAIGHT4 the run-off is first a strip of asphalt along the track, as wide all along, reaching 0.8 m past the kerb, the paving outside it; the faces are cut along its edge. R41's run-off, beyond STRAIGHT5, stays asphalt, and a strip of paving 2.5 m wide is laid outside it, along the straight of its kerb (`1ROAD_PAVERS_STRAIGHT5`), growing out of the asphalt's edge over 5 m at each end. A material of its own, Pavers, and its texture made there (`source/textures/pavers.dds`): blocks 20 by 10 cm in rows a half block apart, mapped flat from above, the texture 6.4 m across so its shades make no pattern seen from far. The paving is driven as kerb: its meshes are named `1KERB_` (as `1KERB_29`), AC's kerb surface, with its rumble, sound and force feedback, which the BeamNG.drive export makes BeamNG's rumble strip. Run again, it finds them paved.

`tools/clear_runoffs.py` makes ground of the pieces of the paddock's asphalt 0.5.1 left round the vp_1000's K1_T3 and K1_T4, some past the tyre walls, which on the real track are ground; the asphalt run-offs it laid there stay. As the ground has holes under them, each becomes ground in its place: the ground's photo mapped as the ground's is, named `1GRASS_` so both games drive it as grass, and grass grown on it. What is left of the paddock's own asphalt there, out of sight under the ground and the run-offs but for its edges, goes. Run again, it finds them cleared.

`tools/build_mouth_lines.py` paints a line across the whole road before T12, where the road the vp_1500 turns off begins to part from it, a few metres before the southernmost of the cones between them: square to the road as it runs before the junction, from edge line to edge line, as the real track has it. Run again, it finds it painted; `tools/lift_paint.py` after it.

`tools/fix_seams.py` takes out the steps where two roads meet a few centimetres apart in height, a seam at a time, as listed in it: the road named is eased into the other within 2 m of where they meet, the grass and anything else sharing its edge going exactly as far, the paint and kerbs on it with it, the road it meets left as it is. So far: R24 into R17, on the vp_1000, up to 4.9 cm; and the ground round the paddock's asphalt into it, which `tools/join_paddock.py` had moved up to 19 cm away from the grass, leaving holes at K1_T8 and K1_T2. Ground eased keeps its edges where it meets a road, kerb or tyre wall, and moves only other ground with it. Run again, it finds them level.

`tools/build_board_objects.py` makes the layouts' distance boards loose objects, as the cones are, that a kart knocks over. They were walls in models of their own (`vp_*_boards.glb`, and a 50 m board in `velopark_board50.glb`); each now goes in its layout's own model, `source/models/vp_<layout>.glb`, beside the cones, as AC takes loose objects as `AC_POBJECT_0`, `AC_POBJECT_1` and on: numbered on from the layout's last cone, each a dummy with the box AC moves it by and the board drawn, standing on the ground under it, none of it in the ground. The layouts load the boards' old models no more, and they are gone. BeamNG.drive knocks the boards about too: each is part of its layout's one vehicle of cones, a box of eight nodes with the board's mesh on them, 2.5 kg. Run again, it finds them loose.

`tools/build_gantry_legs.py` makes the two legs at each end of the start gantries' boards meet each other halfway up the board's end and hold it there, as the real gantry's do, in `source/models/velopark_gantry_*.glb`. Run again, it finds the legs met.

`tools/map_roads.py` maps the asphalt of the roads the 0.5.0 track stretched as the paddock's is. That track maps each road's asphalt along it, the texture's width across it, so on the pit lanes and the other narrow roads it is squeezed across them and stretched along them, about six times as much as on the track. Every road whose asphalt is more than 30% off the paddock's scale, in every model, is mapped flat from above at that scale, 10 by 35 m per texture; the asphalt being the same all over, the seams do not show. Run again, it finds the roads mapped.

`tools/build_round_kerbs.py` lays the round kerbs of the middle of the track in `source/models/velopark_kerbs.glb`: a little higher than the ramp kerbs, 9 cm against their 6, rounded across its 0.6 m, as wide as a ramp kerb, one long hump painted yellow all over, its paint the ramp kerb's made yellow, as round the S2 chicane of the real track. It takes the place of the ramp kerbs `S1_OUT` and `S2_EXIT` and of the middle of `S2_LEFT`, and runs along both sides of `R9` (`R9_NORTH`, `R9_SOUTH`). Its free ends grow out of the road's edge over two blocks, as the ramp kerbs' do; where it meets a ramp kerb, it takes the ramp kerb's shape over its last 1.5 m, so the one carries on into the other. Its texture, `source/textures/kerb_round.dds`, is made there too, from the ramp kerb's. Run again, it finds the kerbs laid.

`tools/place_karts.py` parks Techspeed rental karts in the pit building's workshop and under the white tent north of it, in `source/models/velopark_karts.glb`, added to every layout: the kart project's third level of detail (the farthest has no rims) as built for Assetto Corsa (`d:\projects\kart-techspeed`, or `KART_PROJECT`), drawn within 60 m, with nothing to hit, none within 2.5 m of a pit box or a start. Run it after `tools/build_outside.py`, whose workshop floor they stand on.

`tools/scan_materials.py [--dry-run]` gives surfaces of the track a material scanned by Poly Haven (CC0), as listed in its PLAN: the asphalt, both the track's and the lighter roads', as Aerial Asphalt 01 (a 30 m scan, worn, with tyre marks, hardly repeating; its one crack straight across and its bands taken out, as they crossed the road every 30 m; colour at 4K, normal and shine at 2K), and the concrete pavers, as Brick Pavement 03, the real ones' pattern, each block made the one light grey the real ones are (the scan mixes light and dark ones; its blocks found from its height map). It fetches the scan's colour, normal, roughness and ambient occlusion maps once into `tools/reference/materials/` (kept out of the repository) and makes three textures of them in `source/textures`, named after the material, as `scan_tarmac_dark.dds`, `_nm.dds` and `_maps.dds`: the colour darkened by the occlusion (the asphalt's tinted to the average of the texture it had, so the track keeps its colour and the Custom Shaders Patch's night settings still fit), the normal map as Assetto Corsa reads them (DirectX's, green down) and txMaps from the roughness (red specular, green gloss, blue reflection). The material, its name kept for the Custom Shaders Patch config, takes `ksPerPixelMultiMap`, and each of its meshes gets its uv scaled so the texture repeats at the scan's real size, 30 m for the asphalt, 2.5 m for the pavers. It keeps in `tools/reference/materials/state/<textures>.json` the old texture's colour and the size it covered, and the scan it has: given another scan, it scales the uv from the one to the other, so run again it changes nothing, but scales a mesh some tool made again at that size. Asphalt 02 (3 m) was tried first: its cracks repeated too often; the finer scans (asphalt_track at 2 m) blurred to a flat grey from a kart.

`tools/export_beamng.py` writes every normal map for BeamNG as a PNG with its green turned up: BeamNG reads OpenGL's normal maps (its own levels' are), Assetto Corsa DirectX's (Kunos' are). A txMaps becomes BeamNG's roughness map, its green (the gloss) turned over.

`tools/build_cameras.py [layout ...] [--render]` places each layout's replay cameras for Assetto Corsa, `source/track/<layout>/data/cameras.ini`, from the Assetto Corsa build in `build/ac` and the layout's AI line: one in the race control tower's upper room, one over the start gantry looking back down the grid, one in the judges' tower by the vp_1000, and the rest off the roads round the lap, 3 or 6 m up, where nothing (trees, buildings, tyre walls) stands between them and the karts. Each place is tried against every 2 m of the lap it sees, nearer and from in front being better, and the lap is cut between them with as few cuts as it can, none sooner than 24 m: the gantry takes the grid, the race control room the start straight. Each camera zooms to keep a kart about the same size. `--render` draws each layout's map of its cameras and a picture from each, in `build/cameras`, to check them by eye. Run it again after changing what stands round the track or an AI line.

`tools/build_051.py` copies `base/velopark-0.5.0` to `build/ac/velopark-0.5.1` and rebuilds 0.5.1 there. 0.5.1 was released from it; the track has been made from `source/` since. `tools/check_051.py` renders the corners of that 0.5.1 build.
