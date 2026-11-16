# Roads: one description, everything built from it

Every road on the map (the track, the pit lanes, the side roads, the paths) is described once, as a
centre line with a width each side of it. The road's mesh, its edge lines, its kerbs and its joins to
other roads are all built from that description. A change to a road is a change to its description;
nothing is patched in afterwards. Paint is laid by hand only for what is not a road's edge (pit boxes,
car park bays, start and grid lines).

## A road

A road is a list of points along its middle (x, z), each with:

- `left`, `right`: how far its edges are from the middle there (metres), each side seen going along it;
- optionally `line_left`, `line_right`: the paint along that edge (`white`, `yellow` or none), how far
  in from the edge (default 0.2 m) and how wide (default 0.12 m);
- optionally `kerb_left`, `kerb_right`: a kerb along that edge from here to the next point that says
  otherwise (its kind: `floor`, `ramp`, `concrete`; its width).

Between its points the middle is a smooth curve through them (a spline), and the widths change
smoothly too. So a road's edges are smooth curves, whatever its points' spacing.

The road also says what it is: its name (the mesh's, as `1ROAD_17`), its material (`Tarmac Dark`,
`Tarmac Light`, `Pavers`), and how its texture is mapped (along it, as the track's, or flat from above,
as a paddock road's).

## Its mesh

As the track's own roads are built: a block of asphalt 2 cm over the ground (`ROAD`, laid on it by the
drape), with a small ramp down to the ground along each edge.

- **Across**, in columns of one fixed width (`COL`, 0.5 m) out from
  the middle each side; the last column each side is whatever width is left. Where a road is wider
  there are more columns, never wider ones: where the width grows by a column, a column more starts
  there (its first quad a triangle).
- **Along**, in rows square to the middle, one every `ROW` metres (0.1 m, so the asphalt follows the
  curve smoothly); a line inside a curve keeps fewer of them, so the grid never folds.
- **Its edges**: the top ends at the edge, 2 cm up; the small ramp from there down to the ground is
  the drape's, along every road's edge (`1GRASS_EDGE_RAMP`), as now.

Every road is one such grid; nothing else is added to it.

## Joining roads

Where a road leaves another (a pit entry off the track, a side road off a pit lane), it says so: it
starts on the other's edge, at a point along it, and its edges curve out into the other's over a
radius (`flare`, metres; 0 for a sharp corner). The join is built as part of the road that leaves:

- its first row lies along the other road's edge, and the other road's edge gets a point at each of
  that row's points, so the two share them exactly: no gap, no crack, no overlap;
- its edges curve out of the other's edge over the flare, as part of its grid;
- where the two roads' edge lines meet, they fork at the point between them, on the road, each
  going on along its own edge.

A road that ends on another (a pit exit back onto the track) is the same, the other way round.

In its description an auxiliary road (a pit entry or exit, a side road) says where it leaves and where
it joins: `"from": {"road": "1ROAD_3", "side": "right", "at": 56.0, "flare": 8.0}` (and `"to"` the
same): its first point on that road's edge line at that distance along it, going the same way, its
middle curving away from there over the flare; its last point onto the other's line the same way. So
it leaves and joins smoothly, and the lines fork and merge at the line itself: the main road's edge
line ends at the fork, the auxiliary road's lines begin there.

The exception: asphalt that only lies beside the track (a run-off, as beyond the hairpin's kerb, at
R41, at the last corner) neither leaves nor joins it. It lies along the main road's edge, `"along":
{"road": "1ROAD_3", "side": "left", "from": 120.0, "to": 160.0}`, sharing that edge point for point;
the main road keeps its edge line there, and the run-off has none of its own.

## What is built from it

1. The roads' meshes, a grid each, sharing their points where they join.
2. The edge lines, along each edge at their distance in, forking where roads part.
3. The kerbs, along each edge where the road says, outside the edge line and following it (a guard
   between the road and the grass); a kerb between two roads joins both.
4. The ramps down to the grass and the ground cut under the roads, from the roads' outlines.

The drape, the seal and the checks stay as they are: they work on whatever the roads are.

## Its heights, and one road at a time

A road's description holds its heights too: at each of its points, how high its middle is and how
high each edge (so its camber is kept), as `[x, z, left, right, y, y_left, y_right]`. Its mesh is
then wholly its description's: the drape leaves a described road as it is (the ground, its ramps, the
kerbs and the paint are laid against it, as now), and nothing else moves it.

Each road's mesh is kept (in `build/cache/roads/`) under its description's fingerprint: a build makes
again only the roads whose descriptions changed, and puts the others back as they were. Changing one
road touches that road only.

The heights are read once from a build where the drape laid the roads on the ground, and written into
their descriptions; from then on they are the roads', changed by changing the description.

## Curves and the editor

A road's middle can also be given as a curve of Bézier pieces in place of its points:

```
"curve": [{"p": [x, z], "in": [dx, dz], "out": [dx, dz], "left": w, "right": w}, ...]
```

- **Nodes:** each node is a point the middle goes through.
- **Handles:** `in` and `out` are the node's handles (metres from it).
- **Widths:** `left` and `right` are the widths there, eased from node to node.

The middle is sampled every 0.1 m and the mesh's rows are 0.1 m apart, so curves stay smooth.

The road editor (top-race-editor) edits the curves, the lines (`side`, `from`, `to` in metres along the road, `to` empty for its end, `colour` white or yellow, `pad`, `w`) and the rumbles (`kerbs`: `side`, `from`, `to`, `kind`, `width`, `inset`) over the satellite picture.

A branch says the roads it leaves and joins, as `"joins": [from, to]`:

- the build cuts it where it lies over them;
- where it lies over them, it takes their heights;
- its edge lines fork from theirs at the line itself.

A road with no heights of its own, one new or changed in the editor, gets them recorded by the build on the ground the drape uses. They are kept in `build/cache/road_heights/` by the road's shape.

## What lies over what

The roads are built in their order in `roads.json`, which is the editor's list, with up and down buttons there. A road that joins or lies along another always comes after it.

- **Roads over roads:** each road is cut where it lies under any road above it, except a road joined end to end to it. Its edge points within 5 cm of that road's edge are put on it, and its heights there are that road's. So the two meet point for point.
- **Lines:** a fork leaves no slivers. Gaps under 1 m are filled, pieces under 1.5 m are dropped, and each line runs 0.2 m on past where it gives way to another.
- **Rumbles over roads:** the rumbles (`tools/build_kerbs.py`'s shapes, laid by `tools/merge_floor.py` after the drape) lie over all the roads.
  - **How they're built:** by kind, as the track's kerbs are: ramp, floor, concrete, round, or ramp + concrete.
  - **What they replace:** they take the place of the old kerbs along the roads that have rumbles.
  - **What they cut:** the ground, the roads' edge ramps and any road under them.
- **Roads over the terrain:** the drape cuts the ground under the roads.

## Where the descriptions come from

At first, from the track as it is: each road's middle and widths read from its mesh's outline and the
white lines along its edges, smoothed, written to `tools/reference/roads.json`. From then on that file
is the road network: edited by hand (or from a drawing on the map), and the meshes are built from it.

## Plan

1. The road description and the builder for one road: grid, mapping, edge lines. Proved on the vp1500
   pit entry and its side road (both rebuilt from descriptions, nothing else changed).
2. Joins: a road leaving another and ending on another, with flares and forked lines. Proved on the
   vp1500 entry's mouth, the side road's fork and the hotlap entry.
3. Kerbs from the descriptions.
4. The descriptions of every road read from the track as it is; every road rebuilt from them.
5. The patching in `tools/restyle_paddock.py` and `tools/paint_pits.py` taken out, road by road, as
   each is rebuilt.

Each step is its own build, checked in the game before the next.
