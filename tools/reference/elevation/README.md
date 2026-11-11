# Elevation

The real ground's heights from Google's Elevation API, and how the track's ground compares with them. Made by [tools/elevation.py](../../elevation.py); its docstring has the details.

| File | What it is |
|---|---|
| `location.json` | Where the track lies: AC's x and z to latitude and longitude, fitted from the roads of `velopark.glb` to the kart track's raceways in OpenStreetMap. The roads hold 90.7% of OpenStreetMap's points along them. The track is turned 0.84° from north (x runs west, z north), and AC's origin is at -29.8224384, -51.3243365, 117.5 m south and 8.2 m west of the layouts' geotag. |
| `google_<date>.json` | Google's heights on a 10 m grid over the track (x -260 to 300, z -140 to 380), each point's x, z, latitude, longitude, height and the resolution Google gives. Dated, so a later fetch sits beside it. |
| `google_wide_<date>.json` | Google's heights every 20 m over the whole near ground (x -500 to 500, z -440 to 440), for `tools/level_ground.py`. |
| `osm_property_<date>.json` | OpenStreetMap's buildings, roads, fences and water round the track, for `tools/level_ground.py` and `tools/build_outside.py`. |
| `level_ground.npz`, `.json` | The field `tools/level_ground.py` moved everything up or down by, and its summary, which marks the ground levelled. |
| `compare_<date>_before_levelling.*` | The comparison below, of the track as it was before the levelling. |
| `compare_<date>.json`, `.csv`, `.png` | Each point's track height (top of the ground, roads, kerbs, paths and paddock) against Google's, with the offset between them, the tilt and the differences on the roads, the open ground and the plantation. |
| `google_north_profiles_<date>.json` | Google's heights along five north–south lines from z = 200 to 900, every 20 m, beyond the grid. |
| `google_north_straight_<date>.json` | The road's and Google's heights along the north straight, every 10 m. |
| `methods_<date>.json` | Google's points interpolated six common ways, from points 10 to 120 m apart, and how close each comes to the track's ground. |

```
python tools/elevation.py fetch      # heights again, as google_<today>.json
python tools/elevation.py compare    # the latest fetch against the track
python tools/elevation.py methods
python tools/elevation.py locate     # only if the track is moved or turned
```

`fetch` needs `GOOGLE_API_KEY` in `.env`, with the Elevation API enabled.

## 2026-11-10, before the levelling

Google gives 9.54 m resolution over the whole track. Over the eucalyptus plantation in the north and west, its heights are the tops of the trees, up to 17 m over the track's ground, so they are left out below. Heights are matched by the roads' median, at an offset of 9.64 m: the track's y + 9.64 is the height above sea level.

- **Roads:** within 0.85 m RMS of Google's (5% to 95%: -1.46 to +1.16 m). Fitting a plane takes out a tilt of only 0.13°.
- **Open ground:** 1.24 m RMS (-2.9 to +1.1 m). The track's ground has 5.4 m of relief, Google's open ground about 3.5 m. The track's rises in broad smooth swells, while Google's is flatter with small detail. Google's is 2 to 3 m higher on the west side and up to 4 m lower south of the paddock and along the north edge, where the track's ground climbs toward the trees.
- **North of the circuit:** `google_north_profiles_2026-11-10.json` has five profiles north to z = 900. The ground is flat at about 10–11 m up to the unpaved road (OpenStreetMap's Estrada Porto da Figueira, at z ≈ 319). Right past it, Google's heights step up about 15 m to a 24–29 m plateau. The plateau runs to z ≈ 690, then drops just as sharply back to 7–11 m. Copernicus's 90 m heights (through Open-Meteo) show the same block at 21–23 m. Both are surface models, which include treetops. It is the eucalyptus plantation's canopy, not a hill: from the vp_1000's first corner, looking north, the ground stays level with the track up to the trees, with dense vegetation beyond the road. The track's ground there is not level. It climbs about 3 to 5 m from z = 200 to z = 420, starting before the trees. It sits 1 to 1.7 m over Google's at the north straight (z ≈ 280) and 3 to 5 m over the level ground under the trees. Most likely the old data saw the canopy too, and the interpolation spread its step into a long ramp. Google's heights in front of the road, about 10–11 m (track y ≈ 0.5 to 1.5), are the level to bring that ground to.
- **North straight:** `google_north_straight_2026-11-10.json` has the road and Google's heights every 10 m along it, in driving order (east, x falling), from the vp_2500 AI line. Google's climbs about 1.8 m over 270 m, steadily (0.68%): about 10.0 m at the west end (x ≈ 50), 11.8 m at the east end (x ≈ -160), with a small bump of 0.3 m round x ≈ -40. The track's road climbs about 2 m in the first 150 m to a crest at x ≈ -70, then falls 1.3 m to the east end. Through the middle of the straight it is 1.0 to 1.9 m over Google's, and it meets Google's again at both ends.
- **Interpolation:** with the new data, the method matters little. Nearest, Delaunay linear, Clough-Tocher cubic, inverse distance and thin plate splines all land at 1.1 to 1.2 m RMS from the track's ground, with correlation 0.2 to 0.3. Spacing the points out to 120 m brings it a little nearer (1.03 m with inverse distance), which fits a ground made from sparse points. But no method makes the new data give the track's swells, so the old data's values differed too, not only the way they were interpolated.

## 2026-11-10, after the levelling

`tools/level_ground.py` moved every point of the track up or down by the smooth difference between Google's ground (without the treetops and roofs) and the track's: from -5.35 m (the south west's hill) to +2.3 m, at most 0.23% of grade change a metre along any layout. Compared again (`compare_2026-11-10.*`):

- **Roads:** 0.18 m RMS (5% to 95%: -0.26 to +0.35 m), correlation 0.97.
- **Open ground:** 0.47 m RMS, correlation 0.84.
- **Tilt left:** 0.02°.

