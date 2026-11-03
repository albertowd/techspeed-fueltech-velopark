# Techspeed Velopark for Assetto Corsa

Version 0.5.1, by Alberto Wollmann Dietrich.

The Techspeed Velopark karting circuit in Nova Santa Rita, Brazil, built for go-karts.

## Layouts

| Layout | Length | Direction | Max karts |
|---|---|---|---|
| Techspeed Velopark 1000 | 1004 m | Clockwise | 24 |
| Techspeed Velopark 1500 | 1517 m | Clockwise | 16 |
| Techspeed Velopark 1500 Amateur | 1477 m | Clockwise | 16 |
| Techspeed Velopark 1500 Short | 1217 m | Clockwise | 16 |
| Techspeed Velopark 1500 Short Amateur | 1180 m | Clockwise | 16 |
| Techspeed Velopark 1500 Hotlap | 1517 m | Clockwise | 36 |
| Techspeed Velopark 2500 | 2541 m | Clockwise | 32 |
| Techspeed Velopark Kids | 227 m | Anticlockwise | 3 |

Each pit box holds a team of two karts, so the number of karts is twice the number of boxes, except on the Kids layout. Techspeed Velopark 1500 is the championship layout, with its pits at the centre building. Techspeed Velopark 1500 Amateur is the rental layout with the easier inner corners, Techspeed Velopark 1500 Short turns in on the S and skips the northern loop, and Techspeed Velopark 1500 Short Amateur is the same with the inner corners. Techspeed Velopark 1500 Hotlap has the same route as Techspeed Velopark 1500 with its pits at the long pit building.

In hotlap mode you start just after the last sector line, so you reach the start line quickly and your first full lap is timed.

See [CHANGELOG.md](CHANGELOG.md) for what is new.

## Installing

Copy the `techspeed_velopark` folder into `assettocorsa/content/tracks`. If you have an older version in a `velopark` or `kart_velopark` folder, delete that folder.

Custom Shaders Patch is recommended. It adds the grass, rain effects, night lights and the live leaderboard on the start gantry. The track still loads without it, but those features are missing.

Servers need this same version of the track. The Kids layout must be added to the server as its own layout.

## Known limits

- The AI lines are generated from the track geometry, not recorded from a driven lap, so AI pace may need tuning.
- Barrier tyres and fences are fixed in place. Cones can be knocked over.

## Credits

Barrier tyre: ["Tire"](https://sketchfab.com/3d-models/tire-44168cfdb8044ea8b242c1734d9cb3f9) by [TahirNilin](https://sketchfab.com/TahirNilin), licensed under [CC Attribution 4.0](http://creativecommons.org/licenses/by/4.0/).

Changes made to the model: reduced to fewer sides with hidden faces removed, rescaled, textures reduced in size, and the colour texture cleaned of dirt and stains.

Ground texture: satellite imagery from Google Maps, imagery (c) Google and its providers.
