# Techspeed FuelTech Velopark

Version 0.7.0, by Alberto Wollmann Dietrich.

The Techspeed FuelTech Velopark karting circuit in Nova Santa Rita, Brazil, built for go-karts, for Assetto Corsa and BeamNG.drive.

## Layouts

| Layout | Length | Direction | Max karts |
|---|---|---|---|
| Techspeed FuelTech Velopark 1000 | 1004 m | Clockwise | 24 |
| Techspeed FuelTech Velopark 1500 | 1517 m | Clockwise | 16 |
| Techspeed FuelTech Velopark 1500 Amateur | 1477 m | Clockwise | 16 |
| Techspeed FuelTech Velopark 1500 Short | 1217 m | Clockwise | 16 |
| Techspeed FuelTech Velopark 1500 Short Amateur | 1180 m | Clockwise | 16 |
| Techspeed FuelTech Velopark 1500 Hotlap | 1517 m | Clockwise | 36 |
| Techspeed FuelTech Velopark 2500 | 2541 m | Clockwise | 32 |
| Techspeed FuelTech Velopark Kids | 227 m | Anticlockwise | 3 |

Each pit box holds a team of two karts, so the number of karts is twice the number of boxes, except on the Kids layout. Techspeed FuelTech Velopark 1500 is the championship layout, with its pits at the centre building. Techspeed FuelTech Velopark 1500 Amateur is the rental layout with the easier inner corners, Techspeed FuelTech Velopark 1500 Short turns in on the S and skips the northern loop, and Techspeed FuelTech Velopark 1500 Short Amateur is the same with the inner corners. Techspeed FuelTech Velopark 1500 Hotlap has the same route as Techspeed FuelTech Velopark 1500 with its pits at the long pit building.

In hotlap mode you start just after the last sector line, so you reach the start line quickly and your first full lap is timed.

See [CHANGELOG.md](CHANGELOG.md) for what is new.

## Installing

### Assetto Corsa

Copy the `techspeed_fueltech_velopark` folder into `assettocorsa/content/tracks`. If you have an older version in a `techspeed_velopark`, `velopark` or `kart_velopark` folder, delete that folder.

Custom Shaders Patch is recommended. It adds the grass, rain effects, night lights and the live leaderboard on the start gantry. The track still loads without it, but those features are missing.

Servers need this same version of the track. The Kids layout must be added to the server as its own layout.

### BeamNG.drive

Drop `techspeed-fueltech-velopark-<version>-beamng.zip` in BeamNG's `mods` folder as it is, without unzipping it. It is the 1500's track, with every layout's start, a time trial and an AI race for each layout.

## Known limits

- The AI lines are generated from the track geometry, not recorded from a driven lap, so AI pace may need tuning.

## Credits

Barrier tyre: ["Tire"](https://sketchfab.com/3d-models/tire-44168cfdb8044ea8b242c1734d9cb3f9) by [TahirNilin](https://sketchfab.com/TahirNilin), licensed under [CC Attribution 4.0](http://creativecommons.org/licenses/by/4.0/).

Changes made to the model: reduced to fewer sides with hidden faces removed, rescaled, textures reduced in size, and the colour texture cleaned of dirt and stains.

Ground texture: satellite imagery from Google Maps, imagery (c) Google and its providers.
