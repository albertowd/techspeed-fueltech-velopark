# Track format

The track's own data, the same for every game. Each game's build reads it and writes that game's files:
Assetto Corsa's .kn5 models, layout model lists and Custom Shaders Patch config; BeamNG.drive's level;
rFactor 2's location. Nothing in this data is written for one game. Where a game needs a value of its own,
it sits in that game's block (`targets.ac`, `targets.beamng`, ...), which only that game's build reads.

The editor (top-race-editor) reads and writes this format, and its engine is the track's build: it makes
`track/` and every game's files from it. The track's tools (tools/) made much of the source and read the
same format, but the build no longer runs them. A change to the format is made in this file first.

Every JSON file here has a `version`, so a reader can convert an older one.

## Coordinates

Metres; y up; x west and z north, as the track has always used. Headings are radians from +z
toward +x. A game whose coordinates differ converts them in its build.

## Source and what is made from it

Source is what people write. Everything else is made from it by the build, and is never written back
into source.

| Source | Written by | What it holds |
|---|---|---|
| `tools/reference/track.json` | the editor (its new-track wizard) | The track's settings; each layout's models (`layouts`: `{"vp_1500": {"models": ["velopark", ...]}}`, by their names in `track/models`, in the order a game loads them) |
| `tools/reference/terrain/terrain.json` (and its edits) | the editor | The ground: its heights' files, filters and edits |
| `tools/reference/roads.json` | the editor | The roads, their lines and kerbs |
| `tools/reference/objects.json` | the editor | What stands on the map, as objects of the catalogue; the layouts' grid, pits, hotlap start and timing gates; the paint marks |
| `tools/reference/templates/` | the tools | The catalogue: each model's template (`.npz` parts, or a `.glb` in this format whose meshes keep their roles), and models.json (a tree's `kind` is `tree`: the plantation's variants and its understory; a building's `building`, its collision `model`: its own walls and floors) |
| `source/models/*.glb` | hand-made | One-off models: buildings, the judges' tower, and so on |
| `source/materials.json` | the tools | The material library |
| `source/textures/*.png` | the tools | The textures |
| `source/surfaces.json` | the tools | What each kind of ground is like to drive on |
| `source/lights.json` | by hand, for now | The light series; the lights and screens that aren't part of a catalogue model or placed by the build |
| `source/bumps.json` | the editor (its Bumps tool) | The hand-made bumps, applied to the merged floor in every game |
| `source/map/` | the editor | The map's pictures and heights the editor draws the track over (fetched with its API keys) |

What's made goes in `track/`. `track/models/` is not committed: the editor's build makes it again from the source
every time (the Release action's too). `track/ac/` is committed.
- `track/models/`: every model of the track as the steps leave it (the ground and the floor merged with the
  roads and kerbs, the paint, the objects' copies, the hand-made models as the steps change them), as glTF in
  the format below, with what the steps work out beside them (places.json, lights_outside.json, the trees);
- `track/ac/`: the files of AC's track folder (source/track) the steps change: the layouts' AI lines, until the
  racing lines become data of their own (9).

The steps start from the hand-made: `source/models` and those files of `source/track` are copied to `track/`
first, and only `track/` is written. Each game's build reads `track/` and the source.

The hand-made models in source/models are in this format too: their materials named only, from the library. Some
of their meshes were made for a texture the library has replaced since (the asphalt, the poles, before their scans):
the step that makes the new texture (tools/scan_materials.py) maps them for it, and adds it to the model, as it does
for any model that doesn't hold it yet.

The layouts' places are not models. The grid, the pits, the hotlap start and the timing gates stay in
objects.json. The objects step works out where each stands, a spot or a gate's `y` 0.5 m over the floor under it
(where a game drops its car from), and writes them, with
each layout's physics objects (the knockable cones and boards), to `places.json`:

```json
{ "version": 1, "layouts": { "vp_kids": {
    "spots": [ { "name": "START_0", "x": -94.32, "y": 0.6004, "z": -42.75, "heading": 0.6716 },
               { "name": "TIME_0_L", "x": -88.871, "y": 1.0868, "z": -42.672, "heading": 0.0 } ],
    "pits": 3,
    "objects": [ { "name": "CONE_24", "model": "cone", "x": -77.65, "y": 0.6, "z": -62.89, "heading": 0.0 } ] } } }
```

A spot's name is `START_n`, `PIT_n`, `HOTLAP_START_n` or `TIME_n_L` / `TIME_n_R`. Each game's build makes
its own from there: AC's are empty nodes (`AC_START_0`, ...) and physics objects (`AC_POBJECT_n`), with
the pit boxes' number in its ui_track.json. The file is `track/models/places.json`.

## Models

Standard glTF 2.0. Each mesh node carries `extras.track`:

```json
{ "role": "surface", "surface": "asphalt", "draw": [0, 45], "shadows": true }
```

- `role`:
  - `visual`: seen, not driven on;
  - `surface`: driven on, and seen unless `visible` is false; its kind of ground in `surface`;
  - `wall`: an unseen wall that cars hit.
- `surface`: a key of surfaces.json, for a `surface` mesh. A game drives the mesh as its material's `surface`
  where the material has one (a rumble's paint, a road's tarmac, so a kerb's tarmac bed is asphalt), and as the
  node's where it hasn't. AC's build names the node from the node's `surface` (`1ROAD_`, `1KERB_`).
- `draw`: `[near, far]` in metres, how near and how far away it's drawn, `far` null for no limit. Left
  out, the node is always drawn. It's how a model has a simpler stand-in for far away.
- `shadows`: whether it casts them (true if left out).
- `grass`: true where grass grows on it (AC's Custom Shaders Patch grass, BeamNG's clumps of it).

A node's name is free. No game reads meaning from names. A game's build gives each node the name its
game needs: AC's `1ROAD_` or `1WALL_`, for example.

What decides the surface of what the build makes:
- a road: its `material` in roads.json, whose entry in materials.json has the `surface`;
- a rumble: its kind's material (KerbRamp, KerbConcrete, KerbRound, KerbFloor), the same way (`rumble`);
- the ground: `grass`, until the terrain has zones of its own;
- paint: `visual`, laid just over the road, never a driving surface.

Each material is a glTF PBR material:
- base colour;
- normal (OpenGL: green up);
- metallic and roughness, packed as glTF has them: roughness in green, metallic in blue;
- occlusion, in red;
- emissive, for a glow (`emissiveFactor`, `emissiveTexture`, `KHR_materials_emissive_strength`), so any
  glTF viewer shows it.

Its `extras.track.material` names its entry in materials.json.

UVs are in the texture's own units, as stored, exact: a model read and written again comes back the same.
A material whose meshes are mapped by their size (roads, ground, walls, the light poles' masts) has
`metres` in materials.json: one UV unit is that many metres. A reader that wants metres multiplies by it.
UVs aren't kept in metres themselves. glTF stores them as 32-bit floats, and turning them into metres and
back doesn't land on the same values, which flips the tangents the tools work out from them where a face's
mapping is nearly flat.

## Materials

`source/materials.json`, version and one entry each, by name:

```json
{ "version": 1, "materials": {
  "PoleMast": {
    "colour": "scan_polemast.png", "normal": "scan_polemast_nm.png", "roughness": "scan_polemast_rough.png",
    "metres": 2.0, "alpha": "opaque", "two_sided": false,
    "targets": { "ac": { "shader": "ksPerPixelMultiMap", "specular": 0.2, "specular_exp": 12 } } },
  "PoleLens": { "colour": "pole_lens.png", "glow": [1, 0.933, 0.824], "glow_strength": 0.12, "night_only": true },
  "Tarmac Dark": { "colour": "scan_tarmac_dark.png", "metres": 25.0, "surface": "asphalt" } } }
```

- `colour`, `normal`, `roughness`, `metallic`, `occlusion`: its textures (PNG), each separate. The build
  packs roughness, metallic and occlusion into glTF's channels. Only `colour` is required.
- `metres`: how many metres one UV unit covers, for meshes mapped by their size.
- `alpha`: `opaque`, `mask` (cut out, as leaves and wire mesh) or `blend` (see-through, as glass).
- `glow`, `glow_strength`, `night_only`: how it glows, and whether only at night: a lamp's glass, for
  example.
- `surface`: the kind of ground its driving surfaces are (a key of surfaces.json).
- `covers_grass`: true for what keeps grass off the ground it lies on (the roads, the kerbs, the tyres, the
  buildings): AC's OCCLUDING_MATERIALS, BeamNG's clumps left out under it.
- `targets`: anything a game needs for it that isn't above. Each game's build has its own default for
  everything, from the rest of the entry. The `targets` block holds only what that default gets wrong.

Every material named anywhere is a key here: roads.json's `material`, the kerbs' materials, the
catalogue's and the hand-made models'. The editor can list and check them.

The same file has a `textures` section, one entry per PNG: `{"ac": "DXT1", "normal": false}`. `ac` is the
DDS format AC's build makes it in (DXT1, DXT3, DXT5, or RGBA uncompressed); BeamNG's build takes the same DDS
where it's compressed, and the PNG where it isn't. `normal` marks a normal map,
whose green AC's build turns back down.

Inside the tools, a texture keeps the name AC gives it (`X.dds`). Its file is the PNG of the same name
(`X.png`).

## Textures

PNG, as made: lossless, readable by every tool. Each game's build converts them into its own format:
DDS for AC (colour DXT1, or DXT5 with alpha; normal maps green down, as AC reads them); for BeamNG, the colour as
the same DDS where it's compressed, the normal maps as they are (green up, as BeamNG reads them), and the opacity
of what's see-through as a map of its own; rFactor 2's later.

## Surfaces

`source/surfaces.json`, version and one entry per kind of ground:

```json
{ "version": 1, "surfaces": {
  "asphalt": { "targets": { "ac": "ROAD", "beamng": "ASPHALT" } },
  "rumble":  { "targets": { "ac": "ROAD", "beamng": "RUMBLE_STRIP" } },
  "kerb":    { "targets": { "ac": "KERB", "beamng": "RUMBLE_STRIP" } },
  "grass":   { "targets": { "ac": "GRASS", "beamng": "VELOPARK_GRASS" } } } }
```

A rumble is a kerb laid on the road's edge. AC drives it as ROAD, as the track always has; `kerb` is the
concrete kerb pieces, which AC drives as KERB. How each grips is each game's own: AC's by its surface's key
(`targets.ac`), BeamNG's by its ground model (`targets.beamng`, ASPHALT where it's left out). A ground model the
game lacks is the level's own, made by its build: VELOPARK_GRASS is BeamNG's GRASS with a soft layer 1 cm deep,
not 5, which would hide half a kart's tyre.

## Racing lines

Each layout has its fast line, the way round its lap a car takes at speed, and its pit lane line, the way through
the pits, both set in the editor beside the layout's grid and pits:

```json
"lines": {
  "fast": { "curve": [ { "p": [-77.6, 92.1], "in": [0.0, -1.8], "out": [0.0, 1.8] } ], "closed": true },
  "pit":  { "curve": [ { "p": [-30.2, 60.4], "in": [0.0, 0.0], "out": [-1.1, 2.0] } ], "closed": false,
            "leaves": 1184.5, "joins": 46.2 } }
```

- A line is a curve as the roads' and the objects' are: a Bézier through its points `p` (x, z), each with its
  handles `in` and `out`, relative to it, edited in the editor as they are, point by point and handle by handle.
  Its height is the floor's under it, as a game's build takes it. `fast` is closed, its first point on the start
  line.
- `pit` forks from the fast line and merges back into it: `leaves` and `joins` are where, in metres along the fast
  line from its start. Its first point is the fast line's there where it leaves, its last where it joins, each
  with its handle along the fast line's way, so a car takes the pit lane off the fast line and back onto it with
  no corner; its own points between are the pit lane's. The editor moves a fork or a merge along the fast line,
  and moves it with the fast line when that changes.
- A layout that has none gets one made, which the editor shows and can change:
  - `fast`: the middle line of the roads its lap runs on, each point's handles along the road there (its
    tangents), so the line follows the road's middle with no corners;
  - `pit`: the middle line of the road its pit boxes stand along, the boxes as its reference, forking from the
    fast line where that road leaves the lap and merging into it where it joins it again.
- The lines the track has now (AC's AI files in `track/ac`) are read into this form once, as the editor's first
  lines for each layout.

Each game's build makes its own from them: AC's its AI files (`ai/fast_lane.ai`, `data/ideal_line.ai`,
`ai/pit_lane.ai`: a point about every half metre, with the road's width to either side of it from the roads, its
camber and grade from the floor, and its speed from how tight the line turns), BeamNG's its road network.

AC's AI drives with its kart's middle on the line and knows nothing of how wide the kart is, so the AI files' points
are kept clear of the road's edges first (the editor's `src/engine/aiclear.ts`): moved across the line, smoothly,
till there is at least 1 m of asphalt either side (half a kart and 0.3 m; the middle of a narrower road), and the
corners taken out where the line turns tighter than 6 m. The kerbs don't count as road there, nor in the road's width
the files give either side: a line drawn over a kerb, or round a pit wall's nose, keeps the kart's wheels off it. The
pit lane's line leaves and joins the fast line as that runs after the move.

A layout whose pit lane joins the fast line behind where it leaves it (vp_1000's and the 1500 layouts', whose karts
leave after the start line and come back before it) gets no `ai/pit_lane.ai`. AC takes a pit lane's exit to be ahead
of its entry, and read that one as nearly the whole lap: every AI kart was set for the pits from the grid, and a
driver in the lane was shown going the wrong way. Without it the AI karts don't pit.

## Lights and screens

Colours here are red, green and blue from 0 to 1.

A catalogue model's own lights are in the catalogue (models.json `lights`), in the model's own frame:
`at` and `facing` as [right, up, forward], forward along its heading (sin h, cos h) and right
(cos h, −sin h), up from the floor under it. That's how the light pole's are.

The light series, and the lights and screens that aren't part of a catalogue model, are in
`source/lights.json`, in the world's metres and coordinates:

```json
{ "version": 1,
  "series": { "Paddock wash": { "colour": [1, 0.93, 0.82], "strength": 3.5, "range": 60, "spot": 178,
                                "sharpness": 0.25, "facing": [0, -1, 0], "night_only": true,
                                "targets": { "ac": { "FADE_AT": 600, "DIFFUSE_CONCENTRATION": 0.35 } } } },
  "lights": [ { "series": "Paddock wash", "at": [58.2, 13.993, -27.2] } ],
  "screens": [ { "material": "GantryBoard", "shows": "start_board", "pixels": [2048, 320],
                 "area": [306, 8, 1436, 304], "background": [0, 0, 0] } ] }
```

- A series: how its lights shine. `colour` and `strength`; `range` in metres; `spot`, the beam's whole width
  in degrees (180 or more: every way), and `sharpness`, from 0 (a soft edge) to 1; `facing`, where its lights
  face unless one says otherwise; `night_only`, lit only at night. Its `targets` hold what a game needs beyond
  that (AC's: the Custom Shaders Patch's own keys, as written).
- A light: its `series`, where it is (`at`), and `facing` if it faces its own way.
- A screen: its `material` (the mesh it's drawn on), what it `shows`, its size in `pixels`, the `area` drawn
  on (x, y, width, height; all of it if left out), its `background`, and the `data` it shows, if any. A game
  without a way to draw it shows the material's texture as painted. What each `shows` is:
  - `start_board`: the race's order and lap times, the race clock or the time of day;
  - `session_times`: every car of the session in order of its best lap;
  - `hotlap_times`: a list of lap times (`data`: `times`, rows of rank, brand, car, year, driver, ms and
    edition; the `columns`' x, how many rows are `shown`, each `row`'s height, the list's `top`), the player's
    best among them;
  - `prices`: the fuel prices (`data`: `prices`).

What the build places adds to these, in the same form:
- the catalogue objects' lights, each where its object stands (places.json `lights`, from the objects step);
- the buildings' ceiling lamps and their screens (`lights_outside.json`, from the outside buildings' step).

Both are in `track/models/`.

A material glows where materials.json says so: `glow` (a colour), `glow_strength`, and `night_only`.

AC's build writes all of this as the Custom Shaders Patch's light series, material adjustments and scripted
displays, after AC's own settings (grass, rain, lighting), which `source/track/extension/ext_config.ini` keeps. Its
grass grows on the meshes marked `grass` (GRASS_MESHES) and not under the materials marked `covers_grass`
(OCCLUDING_MATERIALS): the build writes both lists from the models and the library.
A screen's script is `tools/ac/screens/<shows>.lua`, its data put in as Lua. A material's `targets.ac.night`
names the AC values it takes at night and by day (`{"ksDiffuse": [0.45, 0.2]}`). BeamNG's build writes the
lights as its own night lights.

## The builds

The builds are the editor's engine (top-race-editor: its Build dialog, or `--build` from the command line, as
the Release action runs it). The Python builds that came before it (tools/build_models.py, tools/build_tracks.py,
tools/export_beamng.py) are no longer run.

- The track's build makes `track/` from the source.
- Each game's build reads only `track/` and the source:
  - Assetto Corsa: the .kn5 models (names, shaders, flags and DDS textures from this data, the layouts'
    places from objects.json), the layouts' model lists (models_<layout>.ini, from track.json's `layouts`: a model
    a list names that isn't made stops the build) and the Custom Shaders Patch config; packed as
    `build/dist/<track>_ac.7z`.
  - BeamNG.drive: the level, made from this data, not from the AC build: the models of a layout's list, their
    materials and textures from the library, the layouts' places (the knockable cones as BeamNG's own, the marker
    boards as a vehicle), the racing lines as its road network, the lights, and the grass as clumps; packed as
    `build/dist/<track>_beamng.zip`.
  - rFactor 2: the location in `build/rf2/Locations/<track>`, every layout of it: the models as GMTs, their
    textures as DDS, each layout's scene, GDB, AI waypoints and timing beams, the cones and boards as moveable
    objects, the start gantry's lamps as its StartLight, the screens as still pictures, and the layouts' icons
    and loading screens. It runs from rFactor 2's ModDev; it is not packed for a normal install yet.

## Moving there

Each step ends with an AC build the same as before it, model for model:

1. Materials: materials.json, and PNG textures. Every material name in roads.json, the kerbs, the
   catalogue and the models is a key of it. AC's shader settings move from the models into its build.
2. Mesh roles: `extras.track` on every node; AC's name prefixes made by its build, not kept in the models
   (2a). The layouts' places and AC's physics objects made by AC's build from objects.json (2b).
3. Lights and screens as data: lights.json, and the catalogue's lights. AC's build writes the Custom
   Shaders Patch config from them.
4. What's made goes to `track/`, and source/models keeps only the hand-made models.
5. BeamNG made from this data, not from the AC build; the hand-made models in the neutral format; AC's grass
   lists from the models (`grass`) and the library (`covers_grass`).
6. The layouts' model lists as data: track.json's `layouts`, each with its `models`, every one named (no build
   adds one). AC's build writes its models_*.ini from them, BeamNG's build reads them; source/track has none.

7a. The trees as placements: track.json's `trees`, `[{model, x, z, heading, scale}]` (heading as the objects',
   radians from +z toward +x; no height: each stands on the floor), each a catalogue model of kind `tree` (the
   plantation's variants, their hulls their trunks' walls, and its understory, one at each eucalyptus's foot).
   tools/place_trees.py plants them into track/models/velopark_trees.glb, merged by 50 m tile and part, and
   BeamNG's forest of them; source/models has no trees.

Next, proposed:

7. The track's source in one file, `tools/reference/track.json`, with one `version`: its settings as now, and
   `roads` (roads.json's list), `objects` (objects.json's body: `layouts`, `objects`, `marks`), `terrain`
   (terrain.json's description), `trees` (the placements, `[{model, x, z, heading, scale}]`, each a catalogue
   model) and `layouts` (6). Each part's key holds the part itself, or, until it moves in, the path of its file
   (as `"roads": "tools/reference/roads.json"` does now). Files that aren't JSON stay files it names by their
   path, relative to the track's root as track.json's own paths are (`tools/reference/terrain/edit_3.npz`, where
   terrain.json named `edit_3.npz` beside it): the terrain's raster edits (.npz), pictures, heights, map data. The editor reads the separate files and the one file, and writes
   only the one; the tools read the one (each part from it where it has it, from its separate file where not
   yet), then the separate files go.
8. Each model the Velopark's steps make once made data the editor edits: the ground's shape as terrain edits,
   the buildings and the parked karts as models placed in objects, the pit entry as its road's curve. Done:
   the parked karts (the catalogue's kart), the paddock's buildings (each a `.glb` template of kind `building`,
   placed with `"collision": "model"`: each mesh laid as it is, keeping its role, its lamps the template's lights),
   the pit entry, organize_models' moves; the restyled paddock's floors and the gantries' board sides in the
   hand-made models; the outside screens in source/lights.json. The track is built by the editor's engine.
9. The racing lines as data (Racing lines, above): each layout's `lines` in its layout data, read once from
   the AI files; AC's build writes its AI files from them, BeamNG's road network takes them; `track/ac` then
   holds nothing the steps make, and goes.
