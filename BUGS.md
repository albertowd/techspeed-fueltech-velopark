# Open bugs

What is still to fix or to check in the game. Status: **open** (not fixed yet), **next build** (fixed,
in the next build), **to check** (installed, check it in the game), **waiting** (needs your input).

Where things are drawn and placed (the roads' shapes and lines, the kerbs, the tyre walls, the cones, the
marks, the poles, the pit boxes, the terrain's edits) is the editor's: those are in top-race-editor/BUGS.md,
not listed here. This list is what the build makes of them.

| # | Where | What | Status |
|---|-------|------|--------|
| 1 | Path A merge (north of the pits) | Light blue strips along both road edges, under the tyre walls: the seal's skirts lit like walls facing the sky (no hole under them) | to check (skirts lit as the ground; the hole by the path at x -29.4, z 159.2 filled with ground) |
| 6 | Everywhere | Paint over grass: every line now cut to the road under it, and the build fails if any paint is off the road | to check (the build checks it now) |
| 12 | Car park south of the apron | Bays off the apron and the ramp; the ramp up to the apron's height | to check |
| 13 | Start gantries | Board closed all round (seen from behind) | to check |
| 14 | Path A's cut piece (1ROAD_PATH_A_CUT) | Its mesh is filled any way, not in strips along the road as path A is | open (see 15) |
| 18 | Path A, the grass beside it | The grass ramps' outer edges standing over the ground beside the road, open (the skirt under them showing): their edges now laid just under the ground's mesh | to check |
| 17 | Every gap in the floor | The seal now fills any gap in the floor (no floor at all, round which floors' edges run) with ground, and the hole check finds them | to check |
| 16 | Kerbs (rumbles) | Some lie inside the road, their ends not in line with the road's edge lines (as by LINE_33 at the vp1000's mouth): to be rebuilt along the road grids' edges, outside the edge line and following it (a guard off the grass), but where a kerb lies between two roads (outside T2, R41, R29...). Seen at T3_INSIDE (the line stepping under the kerb), T7 (the kerb not bending with the curve, over the line) and T1_KINK (its middle over the line) | open: next, on R3, R4, R5 |
| 22 | R3, R4, R5 (the track's long pieces) | Built from their descriptions (tools/reference/roads.json): even columns of 0.5 m, fewer cells inside the curves and more outside, their old patches out | to check |
| 23 | R3, R4, R5 edges (before the start line, the grid, outside T2) | Ground over the road's edge and its line: its heights were kept only every 5 m; now every metre, from the ground the drape lays the roads on | to check |
| 24 | R5 / R3 and R4 / R5 seams | A wedge of gap and a step where two rebuilt roads meet, their lines split: their ends now on the original end rows, the same row for both | to check |
| 25 | LAST_OUT, T12, L1500_C, T11, MID_EXIT, R41's two ends | Edge lines doubled or split: one line along each side now, 0.2 m in from the edge all along, broken only at a road's mouth (not at kerbs, patches or run-offs), the old ones cleared 2 m in from the edge | to check |
| 27 | Merges on R3, R4, R5 (as near S1_IN) | A piece of old white line left in the road where two roads merge, just inside the strip cleared along the edge | to check (old lines cleared 2 m in from the edge) |
| 28 | T7, outside | A strip of dark asphalt lying in the grass, off the road | open |
| 26 | T9, outside | A triangle of ground over the road | fixed (asphalt now) |
| 32 | Hotlap pit lane by the boxes | The mesh beside the lane stretched badly, with holes (the pale bands and dark streaks): to find which mesh, and rebuild the lane from its description so its edges and ramps are even | open |
| 34 | R11 (the link by T10) | No white lines along it: built from its description now, a white line along each side | to check |
| 35 | Asphalt | Darker, as the Velopark's is: its colour 0.8 of what it was | to check |
| 36 | R6, R7, R8, R9, R10, R12, R19 | To the new road format: they flare into the track at their ends, which needs the joins (ROADS.md) first | R6, R8, R10: to check (built from the editor's curves, cut into R3/R5, lines forked); R9, R12, R7, R19 open |
| 39 | Rumbles | The old kerbs in the game, not the rumbles placed in the editor: built from roads.json now, by kind, in place of the old kerbs along those roads (K1_T4_JOIN, by R4, went with them) | to check |
| 40 | Beside kerbs | A strip of mixed meshes between the road's edge and its kerb (the old kerbs on the new edges): the ground and edge ramps under the new rumbles cut away | to check |
| 41 | Floor rumbles | Asphalt under them: each thing over what is under it now (rumbles over roads, roads over roads by their order in the editor, roads over the ground) | to check |
| 42 | Merges and junctions | Holes where roads merge or join: a road cut where it lies under one above it, its edge points within 5 cm put on that road's edge, the seam as high as that road there | to check |
| 43 | Lines at forks (R6 / R3) | Thin dotted bits of line in the road and lines a little apart where one road leaves another: no bits under 1.5 m, gaps under 1 m filled, 0.2 m on past the fork | to check |
| 44 | Ramp + concrete rumbles | The concrete laid as a second flat strip after the rumble, in the texture's painted part too: one piece in KerbConcrete now, as the old concrete kerbs, the ramp rising and the concrete sloping back down to the ground | to check |
| 45 | Rumbles carried on to the next road (as S1_IN) | A gap of grass between their parts on the two roads: made one strip now, the rows where they meet made one, the texture on across | to check |
| 46 | Every floor | The roads, rumbles, kerbs, paving, grass and the ground round them merged into one surface, every seam point for point at one height (tools/merge_floor.py); the drape's edge ramps gone | to check |
| 47 | Seal step | Still 964 open edges skirted and 106,000 ground faces laid over gaps after the merge: to find where (the merge's outer edge, the outside's floors, the ground left as it was) | open |
| 48 | Incremental merge (tools/merge_state.py) | A change over most of the map (every ground mesh, the far ground too) left 65,048 open edges over the far ground and "0 points welded" on the rebuilt zone's edge: the kept faces and the remade ones not meeting. A full merge used instead (the last kept merge set aside in build/cache/merge_state_before_terrain, to find why). Again on a small change (the old kids pieces out, 1297 m² made again): an open edge and edges of 3 and 4 faces where the remade zone meets the kept faces, at (-93.3, -40.7); full merge used, the state in build/cache/merge_state_before_kids, the faults in merge_floor_problems_kids.json. A third time, 40 rumbles changed (3413 m² made again): the merge crashed in shapely's difference ("Unable to determine overlay result geometry dimension", surface.build); full merge used, the state in build/cache/merge_state_before_rumbles, the log in merge_floor_crash_rumbles.log | open |
| 49 | Car park by the hotlap's pit apron | Its 10 cm drop under the apron and its ramp still made by the drape, after the roads: to go into the terrain's edits, so every height change is made before the roads | open |
| 50 | Yellow paint (the pit box marks, the roads' yellow lines) | None of it painted: its material was made only by the removed pit lines step, so place_objects left out all 12 yellow marks. The paint is made where it is missing now (roads.paint_mesh) | to check |
| 37 | Pavers R29, R30, R43 + R44 (one road) | To the new road format: run-offs of varying width, beside a kerb: needs run-offs with a width profile | open: after the joins |
| 30 | R41 (the run-off beyond STRAIGHT5) | Its start and end abrupt, its asphalt and pavers bumpy, a seam with R5: built along R5's edge now, from nothing to 2.2 m over 8 m along the road's curve and back, its inner edge on R5's, its heights from R5's edge out to the ground smoothed | to check |
| 29 | Seams between R3, R4, R5 (as at T9, outside) | Sparkles along the seam (its two sides' heights a few millimetres apart at the road's last row) and no white line across it (the lines stopped short of the roads' ends) | to check |
| 21 | Seam between the rebuilt R4 and road 9 (by x 145, z 153) | A gap of about 5 cm between the two roads' edges: to be closed by the roads' joins (ROADS.md, step 2) | open |
| 15 | Every merge and widened road | The asphalt added at merges is patched in with irregular triangles, not in the road's own pattern (strips along it): the root of the seams, slivers and line problems at the merges | open: road rebuild planned |

