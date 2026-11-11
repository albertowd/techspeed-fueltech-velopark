# Surroundings

What lies round the track, 3 km by 3 km (AC's x and z from -1500 to 1500, round the origin), for reference: what could be added round the circuit (trees, roads, fields, water) until the far terrain is made. Made by [tools/surroundings.py](../../surroundings.py); its docstring has the details. Nothing here ships with the track.

| File | What it is |
|---|---|
| `surroundings.jpg`, `.json` | Google's satellite pictures (zoom 17 at scale 2, about half a metre a pixel, laid at a metre a pixel), north up, x falling to the right as the ground photo is, over the track's ground photo, which shows only in the middle where no picture was fetched. The `.json` says where each pixel lies. |
| `google_heights_<date>.json` | Google's heights every 20 m over the 3 km, but where `../elevation/google_*.json` and `google_wide_*.json` already have them. Same columns as those. |
| `relief.png` | Every height there is, from both folders, coloured from low to high with a contour every 2 m, at a metre a pixel to lay over `surroundings.jpg`. |
| `osm_<date>.json` | OpenStreetMap's roads, buildings, land use, woods and water over the 3 km. The area is sparsely mapped: 145 roads and tracks, 20 buildings, 3 natural areas. |
| `tiles/` | The raw pictures, kept out of git (68 MB): `python tools/surroundings.py photo` lays them again, or fetches them if they are missing. |

## 2026-11-11

- The land is low to the west, the river's floodplain at 2 to 8 m above the sea, and rises east of the track into hills up to about 45 m.
- Google's heights are a surface model: they include the tops of trees. The woods stand out as steep "hills" 10 to 15 m high (the plantation north of the track, the woods to the north east), so their heights are not the ground's. The far terrain should take them out as `tools/level_ground.py` does: leave out what stands over the ground round it and fill in from that ground.
- Round the circuit, as the pictures show it: farmland and pasture with lines of trees along the fields, woods to the north and north east, a large fenced site laid out in a grid to the west, a highway along the south west, wetlands to the north west, and a large yard of parked vehicles south east of the car circuit.
