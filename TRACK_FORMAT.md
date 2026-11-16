# Track format

The track's own data, the same for every game. Each game's build reads it and writes that game's files:
Assetto Corsa's .kn5 models, layout model lists and Custom Shaders Patch config; BeamNG.drive's level;
rFactor 2's later. Nothing in this data is written for one game. Where a game needs a value of its own,
it sits in that game's block (`targets.ac`, `targets.beamng`, ...), which only that game's build reads.

The track's tools (tools/) and the road editor (top-race-editor) both read this format. The tools are
the reference while the editor's port runs. A change to the format is made in this file first.

Every JSON file here has a `version`, so a reader can convert an older one.

## Coordinates

Metres; y up; x west and z north, as the track has always used. Headings are radians from +z
toward +x. A game whose coordinates differ converts them in its build.

## Source and what is made from it

Source is what people write. Everything else is made from it by the build, and is never written back
into source.

| Source | Written by | What it holds |
|---|---|---|
| `tools/reference/terrain/terrain.json` (and its edits) | the editor | The ground: its heights' files, filters and edits |
| `tools/reference/roads.json` | the editor | The roads, their lines and kerbs |
| `tools/reference/objects.json` | the editor | What stands on the map, as objects of the catalogue; the layouts' grid, pits, hotlap start and timing gates; the paint marks |
| `tools/reference/templates/` | the tools | The catalogue: each model's template, and models.json |
| `source/models/*.glb` | hand-made | One-off models: buildings, the judges' tower, and so on |
| `source/materials.json` | the tools | The material library |
| `source/textures/*.png` | the tools | The textures |
| `source/surfaces.json` | the tools | What each kind of ground is like to drive on |
| `source/lights.json` | the tools | The lights and screens that aren't part of a catalogue model |

What's made goes in `build/track/`: the ground and the floor merged with the roads and kerbs, the paint,
the objects' copies, as glTF in the format below. Each game's build reads `build/track/` and the source.

The layouts' places are not models. The grid, the pits, the hotlap start and the timing gates stay in
objects.json, and each game's build makes its own from there (AC's `AC_START_0`, for example).

## Models

Standard glTF 2.0. Each mesh node carries `extras.track`:

```json
{ "role": "surface", "surface": "asphalt", "draw": [0, 45], "shadows": true }
```

- `role`:
  - `visual`: seen, not driven on;
  - `surface`: driven on, and seen unless `visible` is false; its kind of ground in `surface`;
  - `wall`: an unseen wall that cars hit.
- `surface`: a key of surfaces.json, for a `surface` mesh. If it's left out, the node takes its
  material's `surface`.
- `draw`: `[near, far]` in metres, how near and how far away it's drawn, `far` null for no limit. Left
  out, the node is always drawn. It's how a model has a simpler stand-in for far away.
- `shadows`: whether it casts them (true if left out).

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

UVs are in metres wherever a mesh is mapped by its size (roads, ground, walls, the light poles' masts),
and the material's `metres` scales them. So a texture can change without mapping anything again. A
mesh mapped to a picture of its own (a board, a screen) has UVs in texture units, and `metres` left out.

## Materials

`source/materials.json`, version and one entry each, by name:

```json
{ "version": 1, "materials": {
  "PoleMast": {
    "colour": "scan_polemast.png", "normal": "scan_polemast_nm.png", "roughness": "scan_polemast_rough.png",
    "metres": 2.0, "alpha": "opaque", "two_sided": false,
    "targets": { "ac": { "shader": "ksPerPixelMultiMap", "specular": 0.2, "specular_exp": 12 } } },
  "PoleLens": { "colour": "pole_lens.png", "glow": [255, 238, 210], "glow_strength": 0.12, "night_only": true },
  "Tarmac Dark": { "colour": "scan_tarmac_dark.png", "metres": 25.0, "surface": "asphalt" } } }
```

- `colour`, `normal`, `roughness`, `metallic`, `occlusion`: its textures (PNG), each separate. The build
  packs roughness, metallic and occlusion into glTF's channels. Only `colour` is required.
- `metres`: how many metres one texture covers, for meshes whose UVs are in metres.
- `alpha`: `opaque`, `mask` (cut out, as leaves and wire mesh) or `blend` (see-through, as glass).
- `glow`, `glow_strength`, `night_only`: how it glows, and whether only at night: a lamp's glass, for
  example.
- `surface`: the kind of ground its driving surfaces are (a key of surfaces.json).
- `targets`: anything a game needs for it that isn't above. Each game's build has its own default for
  everything, from the rest of the entry. The `targets` block holds only what that default gets wrong.

Every material named anywhere is a key here: roads.json's `material`, the kerbs' materials, the
catalogue's and the hand-made models'. The editor can list and check them.

The same file has a `textures` section, one entry per PNG: `{"ac": "DXT1", "normal": false}`. `ac` is the
DDS format AC's build makes it in (DXT1, DXT3, DXT5, or RGBA uncompressed). `normal` marks a normal map,
whose green AC's build turns back down.

Inside the tools, a texture keeps the name AC gives it (`X.dds`). Its file is the PNG of the same name
(`X.png`).

## Textures

PNG, as made: lossless, readable by every tool. Each game's build converts them into its own format:
DDS for AC (colour DXT1, or DXT5 with alpha; normal maps green down, as AC reads them), its own for
BeamNG and for rFactor 2.

## Surfaces

`source/surfaces.json`, version and one entry per kind of ground:

```json
{ "version": 1, "surfaces": {
  "asphalt": { "targets": { "ac": "ROAD", "beamng": "ASPHALT" } },
  "rumble":  { "targets": { "ac": "ROAD", "beamng": "RUMBLE_STRIP" } },
  "kerb":    { "targets": { "ac": "KERB" } },
  "grass":   { "targets": { "ac": "GRASS", "beamng": "GRASS" } } } }
```

A rumble is a kerb laid on the road's edge. AC drives it as ROAD, as the track always has; `kerb` is the
concrete kerb pieces, which AC drives as KERB. How each grips, for games that take it from here, is added
with the BeamNG step.

## Lights and screens

A catalogue model's own lights are in the catalogue (models.json `lights`), in the model's own frame:
`at` and `facing` as [right, up, forward], forward along its heading (sin h, cos h) and right
(cos h, −sin h), up from the floor under it. That's how the light pole's are today.

The other lights, and the screens, are in `source/lights.json`, in the world's metres and coordinates:

```json
{ "version": 1,
  "series": { "Paddock wash": { "colour": [255, 237, 209], "strength": 3.5, "range": 60, "spot": 178,
                                "night_only": true } },
  "lights": [ { "series": "Paddock wash", "at": [58.2, 13.993, -27.2], "facing": [0, -1, 0] } ],
  "screens": [ { "material": "OutTiming", "shows": "session_times", "pixels": [1024, 576] } ] }
```

AC's build writes these as the Custom Shaders Patch's light series, material adjustments and scripted
displays. BeamNG's build writes them as its lights.

## The builds

- The track's build makes `build/track/` from the source.
- Each game's build reads only `build/track/` and the source:
  - `build_ac`: the .kn5 models (names, shaders, flags and DDS textures from this data, the layouts'
    places from objects.json), the layouts' model lists and the Custom Shaders Patch config. AI lines
    stay AC's own until the racing lines become data of their own.
  - `build_beamng`: the level, made from this data, not from the AC build.
  - `build_rf2`: later.

## Moving there

Each step ends with an AC build the same as before it, model for model:

1. Materials: materials.json, and PNG textures. Every material name in roads.json, the kerbs, the
   catalogue and the models is a key of it. AC's shader settings move from the models into its build.
2. Mesh roles: `extras.track` on every node, and UVs in metres where meshes are mapped by size. AC's
   name prefixes and places are made by its build, not kept in the models.
3. Lights and screens as data: lights.json, and the catalogue's lights. AC's build writes the Custom
   Shaders Patch config from them.
4. What's made goes to `build/track/`, and source/models keeps only the hand-made models. BeamNG is
   made from this data, not from the AC build.
