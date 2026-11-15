'use strict';
// The road editor's page (see tools/road_editor.py). World: x grows west, z north; the satellite picture
// spans SPAN metres, its pixel (u, v) = ((530 - x) K, (530 - z) K).

const S = {
  roads: [], ctx: null, drawings: null, materials: [],
  view: { cx: 80, cz: 40, ppm: 4 },
  sel: { r: -1, n: -1, line: -1, kerb: -1 },
  mode: 'select', drag: null, hover: null, space: false,
  layers: { sat: true, ctx: true, draw: false, wire: true, lines: true, kerbs: true, names: true, surface: true },
  surf: null,
  snapMark: null,      // (what a dragged point snaps to now: {x, z, label})
  measure: null,       // (a measured line: {a: [x, z], b: [x, z]})          // (the surface checked: its window, its faces in cells, its problems, whether out of date)
  undo: [], redo: [], meshes: {}, dirty: false, newRoad: null, mouse: null, deleted: new Set(),
};
let SPAN = 1060, PX = 8192, TILE = 512, K = PX / SPAN, HALF = SPAN / 2;
// The rumbles (kerbs) the track has, by their material: what each looks like, one of the track's as an example.
const KINDS = [
  ['ramp', 'Ramp', 'Red and white painted kerb sloping up from the road edge (KerbRamp, as at T1 entry)'],
  ['floor', 'Floor', 'Flat painted kerb, level with the road, orange and white (KerbFloor, as outside the hairpin)'],
  ['concrete', 'Concrete', 'Plain raised concrete kerb, shown blue (KerbConcrete, as at the T1 kink)'],
  ['round', 'Round', 'Yellow rounded kerb, a bump along the edge (KerbRound, as at S1 exit)'],
  ['ramp_concrete', 'Ramp + concrete', 'A ramp with a concrete kerb behind it: one piece in KerbConcrete, as the old concrete kerbs (the ramp painted, rising; the concrete behind it sloping back to the ground)'],
];
const KERB_COL = { ramp: ['#d23a3a', '#f2f2f2'], floor: ['#e07a1f', '#f2f2f2'], concrete: ['#3f6fc0', '#5d88d2'], round: ['#f2c200', '#c99f00'], ramp_concrete: ['#d23a3a', '#f2f2f2'] };
const DASH = 0.5, GAP = 0.5;   // metres: a dotted line's dashes and the gaps between them, unless it says
const BACK = 0.5;     // metres: a ramp + concrete rumble's concrete, behind its ramp, unless it says
const kindOf = (k) => (k === 'yellow' ? 'round' : KERB_COL[k] ? k : 'ramp');
const MAT_COL = { 'Tarmac Dark': 'rgba(30,32,38,0.45)', 'Tarmac Light': 'rgba(85,88,96,0.45)', 'Pavers': 'rgba(150,105,70,0.45)', 'Concrete': 'rgba(175,175,170,0.45)' };
const CTX_COL = { road: 'rgba(0,220,255,0.75)', runoff: 'rgba(255,170,60,0.7)', kerb: 'rgba(255,70,70,0.8)', line: 'rgba(255,255,255,0.9)', tyre: 'rgba(0,0,0,0.85)', paddock: 'rgba(200,120,255,0.6)', kids: 'rgba(120,255,140,0.7)' };

const cv = document.getElementById('map');
const g = cv.getContext('2d');
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ view

function W() { return cv.width / devicePixelRatio; }
function H() { return cv.height / devicePixelRatio; }
function toScreen(x, z) { return [W() / 2 + (S.view.cx - x) * S.view.ppm, H() / 2 + (S.view.cz - z) * S.view.ppm]; }
function toWorld(sx, sy) { return [S.view.cx - (sx - W() / 2) / S.view.ppm, S.view.cz - (sy - H() / 2) / S.view.ppm]; }
function viewBox(pad = 0) {
  const [x0, z0] = toWorld(-pad, H() + pad), [x1, z1] = toWorld(W() + pad, -pad);
  return { xmin: Math.min(x0, x1), xmax: Math.max(x0, x1), zmin: Math.min(z0, z1), zmax: Math.max(z0, z1) };
}
let drawQueued = false;
function redraw() { if (!drawQueued) { drawQueued = true; requestAnimationFrame(draw); } }
function resize() {
  const r = cv.getBoundingClientRect();
  cv.width = Math.round(r.width * devicePixelRatio); cv.height = Math.round(r.height * devicePixelRatio);
  redraw();
}

// ------------------------------------------------------------------ geometry

const smooth = (t) => t * t * (3 - 2 * t);
function bezPoint(a, b, t) {
  const P0 = a.p, P3 = b.p;
  const P1 = [P0[0] + a.out[0], P0[1] + a.out[1]], P2 = [P3[0] + b.in[0], P3[1] + b.in[1]];
  const u = 1 - t;
  const x = u * u * u * P0[0] + 3 * u * u * t * P1[0] + 3 * u * t * t * P2[0] + t * t * t * P3[0];
  const z = u * u * u * P0[1] + 3 * u * u * t * P1[1] + 3 * u * t * t * P2[1] + t * t * t * P3[1];
  return [x, z, P0, P1, P2, P3];
}
// A road's middle sampled along it: x, z, left, right, s, the way to its left (nx, nz), which piece and t.
function sampleCurve(nodes, step = 0.2) {
  const X = [], Z = [], L = [], R = [], SEG = [], T = [], DX = [], DZ = [], nodeAt = [0];
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i], b = nodes[i + 1];
    const [, , P0, P1, P2, P3] = bezPoint(a, b, 0);
    const len = Math.hypot(P1[0] - P0[0], P1[1] - P0[1]) + Math.hypot(P2[0] - P1[0], P2[1] - P1[1]) + Math.hypot(P3[0] - P2[0], P3[1] - P2[1]);
    const n = Math.max(2, Math.ceil(len / step));
    for (let k = i ? 1 : 0; k <= n; k++) {
      const t = k / n, [x, z] = bezPoint(a, b, t), e = smooth(t), u = 1 - t;
      // (its way there exactly, from the curve itself: B'(t))
      const dx = 3 * u * u * (P1[0] - P0[0]) + 6 * u * t * (P2[0] - P1[0]) + 3 * t * t * (P3[0] - P2[0]);
      const dz = 3 * u * u * (P1[1] - P0[1]) + 6 * u * t * (P2[1] - P1[1]) + 3 * t * t * (P3[1] - P2[1]);
      X.push(x); Z.push(z); L.push(a.left + (b.left - a.left) * e); R.push(a.right + (b.right - a.right) * e); SEG.push(i); T.push(t); DX.push(dx); DZ.push(dz);
    }
    nodeAt.push(X.length - 1);
  }
  return finishSamples({ X, Z, L, R, SEG, T, DX, DZ, nodeAt });
}
function finishSamples(o) {
  const n = o.X.length, s = new Float64Array(n), nx = new Float64Array(n), nz = new Float64Array(n);
  for (let i = 1; i < n; i++) s[i] = s[i - 1] + Math.hypot(o.X[i] - o.X[i - 1], o.Z[i] - o.Z[i - 1]);
  for (let i = 0; i < n; i++) {
    let a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
    while (a > 0 && Math.hypot(o.X[b] - o.X[a], o.Z[b] - o.Z[a]) < 1e-6) a--;
    while (b < n - 1 && Math.hypot(o.X[b] - o.X[a], o.Z[b] - o.Z[a]) < 1e-6) b++;
    let tx = o.X[b] - o.X[a], tz = o.Z[b] - o.Z[a];
    if (o.DX && Math.hypot(o.DX[i], o.DZ[i]) > 1e-9) { tx = o.DX[i]; tz = o.DZ[i]; }       // (the curve's own way, where it has one)
    const m = Math.hypot(tx, tz) || 1; tx /= m; tz /= m;
    nx[i] = -tz; nz[i] = tx;                       // (its left, as tools/roads.py has it)
  }
  o.s = s; o.nx = nx; o.nz = nz; o.length = s[n - 1] || 0;
  // (where it turns tighter than it is wide: the side inside the curve no farther out than INSIDE of the
  // curve's radius, as tools/roads.py has it, so its edge closes to a fan and never folds back)
  const INSIDE = 0.97, h = new Float64Array(n), k = new Float64Array(n);
  for (let i = 0; i < n; i++) { h[i] = Math.atan2(nx[i], -nz[i]); if (i) { while (h[i] - h[i - 1] > Math.PI) h[i] -= 2 * Math.PI; while (h[i] - h[i - 1] < -Math.PI) h[i] += 2 * Math.PI; } }
  // (its turning measured over BASE metres at least, as tools/roads.py does: a small kink at a node, where
  // the samples lie close, is no tight curve)
  const BASE = 0.2;
  for (let i = 0; i < n; i++) {
    let a = i, b = i;
    while (a > 0 && s[i] - s[a] < BASE) a--;
    while (b < n - 1 && s[b] - s[i] < BASE) b++;
    const ds = s[b] - s[a]; k[i] = ds > 1e-9 ? (h[b] - h[a]) / ds : 0;
  }
  for (let i = 0; i < n; i++) {
    let kl = 0, kr = 0;
    for (let j = Math.max(0, i - 2); j <= Math.min(n - 1, i + 2); j++) { kl = Math.max(kl, k[j]); kr = Math.max(kr, -k[j]); }
    if (kl > 1e-6) o.L[i] = Math.min(o.L[i], INSIDE / kl);
    if (kr > 1e-6) o.R[i] = Math.min(o.R[i], INSIDE / kr);
  }
  let xmin = Infinity, xmax = -Infinity, zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const w = Math.max(o.L[i], o.R[i]) + 1;
    xmin = Math.min(xmin, o.X[i] - w); xmax = Math.max(xmax, o.X[i] + w); zmin = Math.min(zmin, o.Z[i] - w); zmax = Math.max(zmax, o.Z[i] + w);
  }
  o.box = { xmin, xmax, zmin, zmax };
  return o;
}
function byName(name) { return S.roads.find((r) => r.name === name); }
function interpAt(o, s) {
  // a road's samples at how far along (s): index and fraction
  const A = o.s; let lo = 0, hi = A.length - 1;
  if (s <= 0) return [0, 0]; if (s >= o.length) return [hi, 0];
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (A[m] <= s) lo = m; else hi = m; }
  return [lo, (s - A[lo]) / Math.max(A[hi] - A[lo], 1e-9)];
}
function at(o, s) {
  const [i, f] = interpAt(o, s), j = Math.min(i + 1, o.X.length - 1);
  const mix = (A) => A[i] + (A[j] - A[i]) * f;
  let nx = mix(o.nx), nz = mix(o.nz); const m = Math.hypot(nx, nz) || 1;
  return { x: mix(o.X), z: mix(o.Z), l: mix(o.L), r: mix(o.R), nx: nx / m, nz: nz / m };
}
// A run-off (road.along) as samples: its middle on its road's edge, all its width outward.
function sampleAlong(road) {
  const al = road.along, p = byName(al.road);
  if (!p) return null;
  const o = geo(p), sgn = al.side === 'left' ? 1 : -1, ease = Math.max(road.ease || 8, 1e-6);
  const X = [], Z = [], L = [], R = [], SEG = [], T = [];
  const ss = [];
  for (let s = al.from; s < al.to; s += 0.5) ss.push(s);
  ss.push(al.to);
  for (const s of ss) {
    const q = at(o, s), wside = sgn > 0 ? q.l : q.r;
    const e = Math.min(Math.max(Math.min(s - al.from, al.to - s) / ease, 0), 1), w = alongWidth(road, s) * smooth(e);
    X.push(q.x + q.nx * sgn * wside); Z.push(q.z + q.nz * sgn * wside);
    L.push(sgn > 0 ? w : 0); R.push(sgn > 0 ? 0 : w); SEG.push(0); T.push(0);
  }
  return finishSamples({ X, Z, L, R, SEG, T, nodeAt: [] });
}
// A run-off's width along its road (s on that road): its profile's ([[s, width], ...], eased from point to
// point), or its width all along; as tools/roads.py's along_width (before the ease at its ends).
function alongWidth(road, s) {
  const prof = (road.profile || []).slice().sort((a, b) => a[0] - b[0]);
  if (!prof.length) return road.width ?? 2;
  if (s <= prof[0][0] || prof.length === 1) return prof[0][1];
  if (s >= prof[prof.length - 1][0]) return prof[prof.length - 1][1];
  let i = 0; while (prof[i + 1][0] < s) i++;
  const t = (s - prof[i][0]) / Math.max(prof[i + 1][0] - prof[i][0], 1e-9);
  return prof[i][1] + (prof[i + 1][1] - prof[i][1]) * smooth(t);
}
function alongHandles(road) {
  // a run-off's handles: its ends on its road's edge ('afrom', 'ato'), its width points on its outer edge ('aw', n)
  const out = [], al = road && road.along, p = al && byName(al.road), o = p && geo(p);
  if (!o) return out;
  const sgn = al.side === 'left' ? 1 : -1;
  const edge = (s, w) => { const q = at(o, s), e = (sgn > 0 ? q.l : q.r) + w; return { x: q.x + q.nx * sgn * e, z: q.z + q.nz * sgn * e }; };
  out.push(Object.assign({ kind: 'afrom' }, edge(al.from, 0)), Object.assign({ kind: 'ato' }, edge(al.to, 0)));
  const prof = road.profile && road.profile.length ? road.profile : [[(al.from + al.to) / 2, road.width ?? 2]];
  prof.forEach(([s, w], n) => out.push(Object.assign({ kind: 'aw', n }, edge(s, w))));
  return out;
}
const END_BLEND = 4;   // metres from a road's end over which its rows turn into its end row (as tools/roads.py)
function endsTurned(road, o) {
  // its rows' way across turned, within END_BLEND of each end, into its end rows' (start_left / end_left)
  if (!o || (!road.start_left && !road.end_left)) return o;
  for (const [key, fromStart] of [['start_left', true], ['end_left', false]]) {
    const e = road[key]; if (!e) continue;
    for (let i = 0; i < o.X.length; i++) {
      const d = fromStart ? o.s[i] : o.length - o.s[i];
      if (d >= END_BLEND) continue;
      const w = smooth(Math.min(Math.max(1 - d / END_BLEND, 0), 1));
      let nx = o.nx[i] * (1 - w) + e[0] * w, nz = o.nz[i] * (1 - w) + e[1] * w;
      const m = Math.hypot(nx, nz) || 1; o.nx[i] = nx / m; o.nz[i] = nz / m;
    }
  }
  return o;
}
function geo(road) {
  const key = road._v + (road.along ? ':' + ((byName(road.along.road) || {})._v || 0) : '');
  if (!road._geo || road._geoKey !== key) {
    road._geo = road.along ? sampleAlong(road) : (road.curve && road.curve.length > 1 ? endsTurned(road, sampleCurve(road.curve)) : null);
    road._geoKey = key;
  }
  return road._geo;
}
function touched(road, shape = true) {
  road._v = (road._v || 0) + 1;
  if (shape) { road.moved = true; delete S.meshes[road.name]; wantMesh(); if (!syncing) syncJoints(road); }
  setDirty(true);
  if (S.surf) S.surf.stale = true;
  for (const r of S.roads) if (r.along && r.along.road === road.name) { r._v = (r._v || 0) + 1; delete S.meshes[r.name]; }
}

// Joints: a road's first or last node bound to another road's first or last node ('joint' on both:
// {road, node: 'start' | 'end'}). The two ends kept at one point, one road going straight on into the
// other, as wide each side, their end rows the same way across (start_left / end_left): the build makes
// the one's last row the other's first, point for point, with no gap between their meshes.
let syncing = false;
function endOf(road, i) { return !road.curve ? null : i === 0 ? 'start' : i === road.curve.length - 1 ? 'end' : null; }
function endIndex(road, end) { return end === 'start' ? 0 : road.curve.length - 1; }
function travelAt(road, i) {
  // the way a road goes at its end node i (unit)
  const nd = road.curve[i], st = i === 0;
  let h = st ? nd.out : nd.in, m = Math.hypot(h[0], h[1]);
  if (m < 1e-9) { const q = road.curve[st ? 1 : i - 1].p; h = [q[0] - nd.p[0], q[1] - nd.p[1]]; m = Math.hypot(h[0], h[1]) || 1; return st ? [h[0] / m, h[1] / m] : [-h[0] / m, -h[1] / m]; }
  return st ? [h[0] / m, h[1] / m] : [-h[0] / m, -h[1] / m];
}
function endLeft(road, i, t) { road[i === 0 ? 'start_left' : 'end_left'] = [Math.round(-t[1] * 1e6) / 1e6, Math.round(t[0] * 1e6) / 1e6]; }
function follow(src, i, dst, j) {
  // dst's end node j made to go on from src's end node i
  const a = src.curve[i], b = dst.curve[j], ta = travelAt(src, i);
  const sameWay = (i === 0) !== (j === 0);              // (one ends there, the other starts: they run the same way)
  const tb = sameWay ? ta : [-ta[0], -ta[1]];
  b.p = [a.p[0], a.p[1]];
  const key = j === 0 ? 'out' : 'in', L = Math.hypot(b[key][0], b[key][1]) || 3, sg = j === 0 ? 1 : -1;
  b[key] = [tb[0] * L * sg, tb[1] * L * sg];
  if (sameWay) { b.left = a.left; b.right = a.right; } else { b.left = a.right; b.right = a.left; }
  endLeft(src, i, ta); endLeft(dst, j, tb);
}
function syncJoints(road) {
  // after a change to a road, the ends joined to its ends follow them
  if (!road.curve) return;
  syncing = true;
  try {
    for (const i of [0, road.curve.length - 1]) {
      const jt = road.curve[i].joint, other = jt && byName(jt.road);
      if (!other || !other.curve) continue;
      follow(road, i, other, endIndex(other, jt.node));
      touched(other);
    }
  } finally { syncing = false; }
}
function joinNearest(road, i) {
  if (!endOf(road, i)) { alert("Only a road's first or last node joins another road's end."); return; }
  const nd = road.curve[i];
  let best = null;
  for (const r of S.roads) {
    if (r === road || r.along || !r.curve) continue;
    for (const j of [0, r.curve.length - 1]) {
      const d = Math.hypot(r.curve[j].p[0] - nd.p[0], r.curve[j].p[1] - nd.p[1]);
      if (!best || d < best.d) best = { r, j, d };
    }
  }
  if (!best || best.d > 20) { alert("No other road's end within 20 m of this one."); return; }
  remember();
  unjoin(road, i, false);
  if (best.r.curve[best.j].joint) unjoin(best.r, best.j, false);
  follow(best.r, best.j, road, i);                     // (this end onto the other's, the other as it is)
  nd.joint = { road: best.r.name, node: endOf(best.r, best.j) };
  best.r.curve[best.j].joint = { road: road.name, node: endOf(road, i) };
  syncing = true; try { touched(best.r); touched(road); } finally { syncing = false; }
  status('joined ' + road.name + "'s " + endOf(road, i) + ' to ' + best.r.name + "'s " + endOf(best.r, best.j) + (best.d > 0.01 ? ' (moved ' + best.d.toFixed(2) + ' m)' : ''));
  panel(); redraw();
}
function unjoin(road, i, tell = true) {
  const jt = road.curve[i] && road.curve[i].joint;
  if (!jt) return;
  if (tell) remember();
  const other = byName(jt.road);
  if (other && other.curve) { const k = endIndex(other, jt.node); if (other.curve[k].joint && other.curve[k].joint.road === road.name) delete other.curve[k].joint; touched(other, false); }
  delete road.curve[i].joint;
  touched(road, false);
  if (tell) { panel(); redraw(); }
}
function nearestOn(o, x, z) {
  let best = 0, bd = Infinity;
  for (let i = 0; i < o.X.length; i++) { const d = (o.X[i] - x) ** 2 + (o.Z[i] - z) ** 2; if (d < bd) { bd = d; best = i; } }
  return { i: best, d: Math.sqrt(bd), s: o.s[best] };
}

// ------------------------------------------------------------------ data in and out

async function load() {
  const [r, c, d] = await Promise.all([fetch('/api/roads').then((x) => x.json()), fetch('/api/context').then((x) => x.json()), fetch('/api/drawings').then((x) => x.json())]);
  SPAN = r.span; PX = r.px; TILE = r.tile; K = PX / SPAN; HALF = SPAN / 2; S.materials = r.materials;
  setRoads(r.roads);
  S.ctxItems = c.items; buildCtx();
  S.drawings = d;
  if (S.roads.length) { const o = geo(S.roads[0]); if (o) { S.view.cx = o.X[0]; S.view.cz = o.Z[0]; } }
  status('loaded ' + S.roads.length + ' roads');
  panel(); redraw();
}
function setRoads(list) {
  const keepSel = S.sel.r >= 0 && S.roads[S.sel.r] ? S.roads[S.sel.r].name : null;
  S.roads = list.map((r) => Object.assign(r, { _v: 1, was: r.name }));
  for (const r of S.roads) for (const k of r.kerbs || []) k.kind = kindOf(k.kind);   // ('yellow' was the round kerb's old name)
  S.meshes = {};
  S.sel = { r: keepSel ? S.roads.findIndex((r) => r.name === keepSel) : -1, n: -1, line: -1, kerb: -1 };
}
function clean(road) {
  const out = {};
  for (const [k, v] of Object.entries(road)) if (!k.startsWith('_')) out[k] = v;
  return out;
}
function snapshot() { return JSON.stringify(S.roads.map(clean)); }
function remember() {
  S.undo.push(snapshot()); if (S.undo.length > 150) S.undo.shift(); S.redo = [];
}
function restore(snap) {
  const sel = S.sel;
  S.roads = JSON.parse(snap).map((r) => Object.assign(r, { _v: Math.random() }));
  S.meshes = {}; S.sel = { r: Math.min(sel.r, S.roads.length - 1), n: -1, line: -1, kerb: -1 };
  setDirty(true); panel(); wantMesh(); redraw();
}
function undo() { if (!S.undo.length) return; S.redo.push(snapshot()); restore(S.undo.pop()); }
function redo() { if (!S.redo.length) return; S.undo.push(snapshot()); restore(S.redo.pop()); }
function setDirty(v) { S.dirty = v; $('save').classList.toggle('dirty', v); $('state').textContent = v ? 'not saved' : 'saved'; }
async function save() {
  const names = S.roads.map((r) => r.name);
  const dup = names.find((n, i) => names.indexOf(n) !== i);
  if (dup) { alert('Two roads are named ' + dup + '.'); return; }
  status('saving…');
  const res = await fetch('/api/save', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ roads: S.roads.map(clean), deleted: [...S.deleted].filter((n) => !S.roads.some((r) => r.name === n)) }) }).then((x) => x.json());
  if (res.error) { alert('Not saved: ' + res.error); status('not saved'); return; }
  S.deleted.clear(); setRoads(res.roads); buildCtx(); setDirty(false); panel(); wantMesh(); redraw();
  status('saved' + (res.changed.length ? ' — heights recorded again at the next build for: ' + res.changed.join(', ') : ''));
}

// ------------------------------------------------------------------ the wireframe (tools/roads.py's mesh)

let meshTimer = null;
function wired() {
  // the roads whose meshes show: the selected one and those joined to its ends
  const road = S.roads[S.sel.r];
  if (!road) return [];
  const out = [road];
  for (const nd of road.curve || []) { const r2 = nd.joint && byName(nd.joint.road); if (r2 && !out.includes(r2)) out.push(r2); }
  return out;
}
function wantMesh() {
  clearTimeout(meshTimer);
  meshTimer = setTimeout(() => {
    if (!S.layers.wire) return;
    for (const road of wired()) {
      if (S.meshes[road.name]) continue;
      const name = road.name, v = road._v;
      S.meshes[name] = 'loading';
      fetch('/api/mesh', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ road: clean(road), roads: S.roads.map(clean) }) }).then((x) => x.json()).then((res) => {
        if (res.error) throw new Error(res.error);
        if (road._v !== v) { delete S.meshes[name]; wantMesh(); return; }
        S.meshes[name] = { P: new Float32Array(res.P), F: new Uint32Array(res.F) };
        redraw();
      }).catch((e) => { S.meshes[name] = null; status('wireframe: ' + e.message); });
    }
  }, 350);
}

// ------------------------------------------------------------------ drawing

const tiles = new Map();
function tile(L, x, y) {
  const key = L + '/' + x + '/' + y;
  let t = tiles.get(key);
  if (!t) {
    t = new Image(); t.onload = redraw; t.src = '/api/tile?l=' + L + '&x=' + x + '&y=' + y;
    tiles.set(key, t);
    if (tiles.size > 800) tiles.delete(tiles.keys().next().value);
  }
  return t;
}
function drawSatellite() {
  const L = Math.max(0, Math.min(4, Math.floor(Math.log2(K / S.view.ppm))));
  const f = 2 ** L, tw = TILE * f / K;                     // (a tile's width, metres)
  const b = viewBox();
  const u0 = (HALF - b.xmax) * K / (TILE * f), u1 = (HALF - b.xmin) * K / (TILE * f);
  const v0 = (HALF - b.zmax) * K / (TILE * f), v1 = (HALF - b.zmin) * K / (TILE * f);
  const n = Math.ceil(PX / f / TILE);
  for (let ty = Math.max(0, Math.floor(v0)); ty <= Math.min(n - 1, Math.floor(v1)); ty++) {
    for (let tx = Math.max(0, Math.floor(u0)); tx <= Math.min(n - 1, Math.floor(u1)); tx++) {
      const im = tile(L, tx, ty);
      if (!im.complete || !im.naturalWidth) continue;
      const [sx, sy] = toScreen(HALF - tx * tw, HALF - ty * tw);
      g.drawImage(im, sx, sy, im.naturalWidth * f / K * S.view.ppm, im.naturalHeight * f / K * S.view.ppm);
    }
  }
}
function boxVisible(bx, b) { return !(bx.xmax < b.xmin || bx.xmin > b.xmax || bx.zmax < b.zmin || bx.zmin > b.zmax); }
// Drawing in metres: the canvas turned so x grows to the left, z up, ppm pixels a metre.
function worldT() { const p = S.view.ppm, d = devicePixelRatio; g.setTransform(-p * d, 0, 0, -p * d, d * (W() / 2 + S.view.cx * p), d * (H() / 2 + S.view.cz * p)); }
function screenT() { g.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0); }
function grow(box, x, z, r = 0) { box.xmin = Math.min(box.xmin, x - r); box.xmax = Math.max(box.xmax, x + r); box.zmin = Math.min(box.zmin, z - r); box.zmax = Math.max(box.zmax, z + r); }
const newBox = () => ({ xmin: Infinity, xmax: -Infinity, zmin: Infinity, zmax: -Infinity });
// Shapes gathered in cells of size metres (each cell one Path2D a kind): only those in view are drawn.
function cells(size) {
  const m = new Map();
  return {
    m,
    at(kind, x, z) {
      const key = kind + ':' + Math.floor(x / size) + ':' + Math.floor(z / size);
      let c = m.get(key);
      if (!c) { c = { kind, path: new Path2D(), box: newBox() }; m.set(key, c); }
      return c;
    },
  };
}
function buildCtx() {
  const described = new Set(S.roads.map((r) => r.name)), cs = cells(50);
  for (const it of S.ctxItems || []) {
    if (described.has(it.name)) continue;                    // (described: drawn as edited)
    for (const L of it.lines || []) {
      for (let i0 = 0; i0 < L.length - 2; i0 += 126) {        // (in pieces of 64 points)
        const end = Math.min(L.length, i0 + 128), c = cs.at(it.kind, L[i0], L[i0 + 1]);
        c.path.moveTo(L[i0], L[i0 + 1]);
        for (let i = i0; i < end; i += 2) { c.path.lineTo(L[i], L[i + 1]); grow(c.box, L[i], L[i + 1]); }
      }
    }
    const R = it.rings || [];
    for (let i = 0; i < R.length; i += 3) {
      const c = cs.at(it.kind, R[i], R[i + 1]);
      c.path.moveTo(R[i] + R[i + 2], R[i + 1]); c.path.arc(R[i], R[i + 1], R[i + 2], 0, Math.PI * 2); grow(c.box, R[i], R[i + 1], R[i + 2]);
    }
  }
  S.ctx = [...cs.m.values()];
}
function drawContext(b) {
  worldT();
  g.lineWidth = 1 / S.view.ppm;
  for (const c of S.ctx || []) {
    if (!boxVisible(c.box, b)) continue;
    const col = CTX_COL[c.kind] || 'rgba(255,255,255,0.5)';
    if (c.kind === 'tyre') { g.fillStyle = col; g.fill(c.path); } else { g.strokeStyle = col; g.stroke(c.path); }
  }
  screenT();
}
function drawDrawings() {
  for (const [label, d] of Object.entries(S.drawings || {})) {
    for (const [col, css] of [['blue', 'rgba(60,90,255,0.9)'], ['yellow', 'rgba(255,220,0,0.9)']]) {
      for (const st of d[col] || []) {
        g.beginPath();
        st.forEach((p, i) => { const [x, y] = toScreen(p[0], p[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.strokeStyle = css; g.lineWidth = 3; g.stroke();
      }
    }
  }
}
function pathOf(pts, close) { const p = new Path2D(); pts.forEach(([x, z], i) => (i ? p.lineTo(x, z) : p.moveTo(x, z))); if (close) p.closePath(); return p; }
function lineOffsets(o, ln) {
  const sgn = ln.side === 'left' ? 1 : -1, pad = ln.pad ?? 0.2, w = ln.w ?? 0.12;
  return { sgn, a: (q) => sgn * ((sgn > 0 ? q.l : q.r) - pad + w / 2), b: (q) => sgn * ((sgn > 0 ? q.l : q.r) - pad - w / 2) };
}
function kerbOffsets(k, s0 = 0, s1 = Infinity, t0 = k.start === 'triangle', t1 = k.end === 'triangle') {
  // its inner and outer offsets; a triangle end (t0 at s0, t1 at s1) grows from nothing to its width over
  // 'taper' metres
  const sgn = k.side === 'left' ? 1 : -1, inset = k.inset ?? 0, w = k.width ?? 1, tp = Math.max(k.taper ?? 2, 1e-6);
  const f = (s) => Math.min(t0 ? Math.min(Math.max((s - s0) / tp, 0), 1) : 1,
                            t1 ? Math.min(Math.max((s1 - s) / tp, 0), 1) : 1);
  const back = kindOf(k.kind) === 'ramp_concrete' ? (k.back ?? BACK) : 0;
  const edge = (q) => (sgn > 0 ? q.l : q.r) - inset, g = (s) => (s == null ? 1 : f(s));
  return { sgn, a: (q) => sgn * edge(q), m: (q, s) => sgn * (edge(q) + w * g(s)), b: (q, s) => sgn * (edge(q) + (w + back) * g(s)) };
}
// A rumble may go on along another road whose edge carries on from its road's ('before': before its
// start, 'after': after its end, each {road, side, from, to, outer: 'from' | 'to', the rumble's own end
// there}): its pieces, each on its road, its triangle ends at the rumble's own ends only.
function kerbPieces(road, k) {
  const out = [], o = geo(road);
  if (!o) return out;
  const cont = (c, isStart) => {
    const r2 = byName(c.road), o2 = r2 && geo(r2);
    if (!o2) return null;
    const [a, b] = span(o2, c.from, c.to);
    if (b <= a) return null;
    const tri = (isStart ? k.start : k.end) === 'triangle', at0 = c.outer !== 'to';
    return { r: r2, o: o2, s0: a, s1: b, side: c.side, t0: at0 && tri, t1: !at0 && tri, role: isStart ? 'before' : 'after', outer: at0 ? a : b };
  };
  if (k.before) { const pc = cont(k.before, true); if (pc) out.push(pc); }
  const [s0, s1] = span(o, k.from, k.to);
  if (s1 > s0) out.push({ r: road, o, s0, s1, side: k.side, t0: !k.before && k.start === 'triangle', t1: !k.after && k.end === 'triangle', role: 'main' });
  if (k.after) { const pc = cont(k.after, false); if (pc) out.push(pc); }
  return out;
}
const pieceOffsets = (k, pc) => kerbOffsets(Object.assign({}, k, { side: pc.side }), pc.s0, pc.s1, pc.t0, pc.t1);
function continuation(road, k, isStart, wx, wz) {
  // another road under (wx, wz) whose edge goes on from this road's edge where the rumble starts (or ends),
  // the same way: the rumble's piece along it, from there to the mouse
  const o = geo(road), sj = isStart ? Math.max(0, k.from ?? 0) : Math.min(o.length, k.to ?? o.length), q = at(o, sj);
  const sgn = k.side === 'left' ? 1 : -1, w = sgn > 0 ? q.l : q.r, E = [q.x + q.nx * sgn * w, q.z + q.nz * sgn * w], T = [q.nz, -q.nx];
  let best = null;
  for (const r2 of S.roads) {
    if (r2 === road || r2.along) continue;
    const o2 = geo(r2); if (!o2) continue;
    const m = nearestOn(o2, wx, wz), qm = at(o2, m.s);
    if (m.d > Math.max(qm.l, qm.r) + 1.5) continue;                     // (the mouse on it)
    const e = nearestOn(o2, E[0], E[1]), q2 = at(o2, e.s), dot = T[0] * q2.nz - T[1] * q2.nx;
    if (Math.abs(dot) < 0.9) continue;                                   // (going the way this one goes, or straight back)
    for (const side of ['left', 'right']) {
      if ((dot > 0) !== (side === k.side)) continue;                     // (the same edge of the two)
      const sg2 = side === 'left' ? 1 : -1, w2 = sg2 > 0 ? q2.l : q2.r;
      const dd = Math.hypot(q2.x + q2.nx * sg2 * w2 - E[0], q2.z + q2.nz * sg2 * w2 - E[1]);
      if (dd <= 1 && (!best || dd < best.dd)) best = { dd, r2, side, sE: e.s, sM: m.s };
    }
  }
  if (!best) return null;
  const a = Math.min(best.sE, best.sM), b = Math.max(best.sE, best.sM);
  if (b - a < 0.2) return null;
  return { road: best.r2.name, side: best.side, from: Math.round(a * 10) / 10, to: Math.round(b * 10) / 10, outer: best.sM < best.sE ? 'from' : 'to' };
}
function span(o, from, to) { return [Math.max(0, from ?? 0), Math.min(o.length, to ?? o.length)]; }
function band(o, s0, s1, offA, offB, step = 0.25) {
  // a strip along a road from s0 to s1 between offsets offA(q), offB(q): its outline (metres)
  const pts = [], back = [], n = Math.max(2, Math.ceil((s1 - s0) / step));
  for (let k = 0; k <= n; k++) {
    const s = s0 + (s1 - s0) * k / n, q = at(o, s), A = offA(q, s), B = offB(q, s);
    pts.push([q.x + q.nx * A, q.z + q.nz * A]); back.push([q.x + q.nx * B, q.z + q.nz * B]);
  }
  return pts.concat(back.reverse());
}
function shapes(road) {
  // a road's shapes in metres: its outline, its middle, its lines, its rumbles (made again when it changes)
  const o = geo(road);
  if (!o || o.X.length < 2) return null;
  const key = road._geoKey + '|' + (road.kerbs || []).map((k) => [k.before, k.after].map((c) => (c && byName(c.road) ? byName(c.road)._v : '')).join(',')).join(';');
  if (road._shapes && road._shapesKey === key) return road._shapes;
  const left = [], right = [], mid = [];
  for (let i = 0; i < o.X.length; i++) {
    left.push([o.X[i] + o.nx[i] * o.L[i], o.Z[i] + o.nz[i] * o.L[i]]);
    right.push([o.X[i] - o.nx[i] * o.R[i], o.Z[i] - o.nz[i] * o.R[i]]);
    mid.push([o.X[i], o.Z[i]]);
  }
  const sides = new Path2D();
  sides.addPath(pathOf(left)); sides.addPath(pathOf(right));
  const n = left.length, free = (i) => !(road.curve && road.curve[i] && road.curve[i].joint);
  if (free(0)) sides.addPath(pathOf([left[0], right[0]]));                                  // (a free end closed; a joined one open,
  if (road.curve && free(road.curve.length - 1)) sides.addPath(pathOf([left[n - 1], right[n - 1]]));   // the road going on there)
  const sh = { outline: pathOf(left.concat(right.slice().reverse()), true), sides, mid: pathOf(mid), white: new Path2D(), yellow: new Path2D(), kerbs: {} };
  if (!road.along) {
    for (const ln of road.lines || []) {
      const [s0, s1] = span(o, ln.from, ln.to); if (s1 <= s0) continue;
      const off = lineOffsets(o, ln), into = ln.colour === 'yellow' ? sh.yellow : sh.white;
      if (ln.style === 'dotted') {
        const dash = Math.max(ln.dash ?? DASH, 0.05), gap = Math.max(ln.gap ?? GAP, 0.05);
        for (let a = s0; a < s1 - 1e-6; a += dash + gap) into.addPath(pathOf(band(o, a, Math.min(a + dash, s1), off.a, off.b, 0.1), true));
      } else into.addPath(pathOf(band(o, s0, s1, off.a, off.b), true));
    }
    for (const k of road.kerbs || []) {
      const kind = kindOf(k.kind), cols = KERB_COL[kind];
      const put = (col, pts) => (sh.kerbs[col] = sh.kerbs[col] || new Path2D()).addPath(pathOf(pts, true));
      for (const pc of kerbPieces(road, k)) {
        const off = pieceOffsets(k, pc), step = kind === 'concrete' ? (pc.s1 - pc.s0) : 1;
        for (let s = pc.s0, j = 0; s < pc.s1 - 1e-6; s += step, j++) put(cols[j % 2], band(pc.o, s, Math.min(s + step, pc.s1), off.a, off.m, 0.25));
        if (kind === 'ramp_concrete') put(KERB_COL.concrete[0], band(pc.o, pc.s0, pc.s1, off.m, off.b, 0.25));      // (its concrete, behind the ramp)
      }
    }
  }
  road._shapes = sh; road._shapesKey = key;
  return sh;
}
function drawRoad(road, selected) {
  const sh = shapes(road); if (!sh) return;
  const p = S.view.ppm;
  g.fillStyle = MAT_COL[road.material] || MAT_COL['Tarmac Dark']; g.fill(sh.outline);
  g.strokeStyle = selected ? '#4fa3ff' : (road.along ? '#ffaa3c' : road.joins && road.joins.some(Boolean) ? '#7dff9a' : '#ffe14d');
  g.lineWidth = (selected ? 2 : 1.2) / p; g.stroke(sh.sides);
  if (!road.along) {
    g.setLineDash([6 / p, 4 / p]); g.strokeStyle = selected ? 'rgba(79,163,255,0.9)' : 'rgba(255,255,255,0.35)'; g.lineWidth = 1 / p; g.stroke(sh.mid); g.setLineDash([]);
  }
}
function drawPaint(road) {
  const sh = shapes(road); if (!sh) return;
  if (S.layers.kerbs) for (const [col, path] of Object.entries(sh.kerbs)) { g.fillStyle = col; g.fill(path); }
  if (S.layers.lines) { g.fillStyle = '#ffffff'; g.fill(sh.white); g.fillStyle = '#ffd21f'; g.fill(sh.yellow); }
}
function drawSelectedItem(road) {
  // the selected line or rumble outlined
  const o = geo(road); if (!o || road.along) return;
  let pts = null;
  if (S.sel.line >= 0 && road.lines && road.lines[S.sel.line]) { const ln = road.lines[S.sel.line], [s0, s1] = span(o, ln.from, ln.to), off = lineOffsets(o, ln); if (s1 > s0) pts = band(o, s0, s1, (q) => off.a(q) + off.sgn * 0.25, (q) => off.b(q) - off.sgn * 0.25, 0.5); }
  g.strokeStyle = '#4fa3ff'; g.lineWidth = 2 / S.view.ppm;
  if (S.sel.kerb >= 0 && road.kerbs && road.kerbs[S.sel.kerb]) { const k = road.kerbs[S.sel.kerb]; for (const pc of kerbPieces(road, k)) { const off = pieceOffsets(k, pc); g.stroke(pathOf(band(pc.o, pc.s0, pc.s1, off.a, off.b, 0.25), true)); } }
  if (pts) g.stroke(pathOf(pts, true));
}
function wireCells(m) {
  // a mesh's edges in cells of 10 m (Path2D each), made once it comes
  const cs = cells(10), P = m.P, F = m.F;
  for (let f = 0; f < F.length; f += 3) {
    const a = F[f] * 2, c = F[f + 1] * 2, d = F[f + 2] * 2, cell = cs.at('w', P[a], P[a + 1]);
    cell.path.moveTo(P[a], P[a + 1]); cell.path.lineTo(P[c], P[c + 1]); cell.path.lineTo(P[d], P[d + 1]); cell.path.closePath();
    grow(cell.box, P[a], P[a + 1]); grow(cell.box, P[c], P[c + 1]); grow(cell.box, P[d], P[d + 1]);
  }
  return [...cs.m.values()];
}
// ------------------------------------------------------------------ the surface, as the build makes it

const SURF_COL = { rumble: 'rgba(255,90,90,0.8)', road: 'rgba(255,255,255,0.5)', ground: 'rgba(120,230,120,0.35)' };
async function checkSurface() {
  // the surface where one looks (up to 150 m across, round the middle of the view), built and checked
  const b = viewBox(), cx = (b.xmin + b.xmax) / 2, cz = (b.zmin + b.zmax) / 2;
  const hw = Math.min((b.xmax - b.xmin) / 2, 75), hz = Math.min((b.zmax - b.zmin) / 2, 75);
  const window = [cx - hw, cz - hz, cx + hw, cz + hz];
  status('checking the surface here…'); msg('Checking the surface…'); redraw();
  try {
    const res = await fetch('/api/surface', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ roads: S.roads.map(clean), window }) }).then((x) => x.json());
    if (res.error) throw new Error(res.error);
    const cs = cells(10);
    for (const r of res.regions) {
      const P = r.P, F = r.F;
      for (let f = 0; f < F.length; f += 3) {
        const a = F[f] * 2, c = F[f + 1] * 2, d = F[f + 2] * 2, cell = cs.at(r.kind, P[a], P[a + 1]);
        cell.path.moveTo(P[a], P[a + 1]); cell.path.lineTo(P[c], P[c + 1]); cell.path.lineTo(P[d], P[d + 1]); cell.path.closePath();
        grow(cell.box, P[a], P[a + 1]); grow(cell.box, P[c], P[c + 1]); grow(cell.box, P[d], P[d + 1]);
      }
    }
    S.surf = { window, cells: [...cs.m.values()], issues: res.issues, stale: false };
    status(res.issues.length ? res.issues.length + ' seam problems here (ringed red) — ' + res.seconds + ' s' : 'surface here: no seam problems (' + res.regions.length + ' regions, ' + res.seconds + ' s)');
  } catch (e) { status('surface check: ' + e.message); }
  redraw();
}
function drawSurface(b) {
  const sf = S.surf; if (!sf) return;
  const p = S.view.ppm;
  worldT();
  const [x0, z0, x1, z1] = sf.window;
  g.setLineDash([8 / p, 5 / p]); g.strokeStyle = sf.stale ? 'rgba(255,200,0,0.8)' : 'rgba(120,200,255,0.8)'; g.lineWidth = 1.5 / p;
  g.strokeRect(x0, z0, x1 - x0, z1 - z0); g.setLineDash([]);
  if (p >= 3) {
    g.lineWidth = 0.7 / p;
    for (const c of sf.cells) { if (!boxVisible(c.box, b)) continue; g.strokeStyle = sf.stale ? 'rgba(200,200,200,0.25)' : SURF_COL[c.kind] || SURF_COL.road; g.stroke(c.path); }
  }
  screenT();
  for (const [what, x, z] of sf.issues) {
    const [sx, sy] = toScreen(x, z);
    g.beginPath(); g.arc(sx, sy, 9, 0, Math.PI * 2); g.strokeStyle = '#ff3030'; g.lineWidth = 2.5; g.stroke(); g.lineWidth = 1;
  }
  if (sf.stale) msg('The surface shown is out of date: C to check again');
}
function drawWire(road, b) {
  const m = S.meshes[road.name];
  if (!m || m === 'loading') return;
  if (S.view.ppm < 5) { msg('Zoom in to see the wireframe'); return; }
  if (!m.cells) m.cells = wireCells(m);
  g.strokeStyle = 'rgba(255,255,255,0.45)'; g.lineWidth = 0.6 / S.view.ppm;
  for (const c of m.cells) if (boxVisible(c.box, b)) g.stroke(c.path);
}
function nodeHandles(road) {
  // the selected road's handles: [{kind, n, x, z}] (world)
  const out = [];
  if (!road || road.along || !road.curve) return out;
  const o = geo(road);
  road.curve.forEach((nd, i) => {
    const q = o ? at(o, o.s[o.nodeAt[i]]) : null;
    out.push({ kind: 'p', n: i, x: nd.p[0], z: nd.p[1] });
    if (i > 0) out.push({ kind: 'in', n: i, x: nd.p[0] + nd.in[0], z: nd.p[1] + nd.in[1] });
    if (i < road.curve.length - 1) out.push({ kind: 'out', n: i, x: nd.p[0] + nd.out[0], z: nd.p[1] + nd.out[1] });
    if (q) {
      out.push({ kind: 'wl', n: i, x: nd.p[0] + q.nx * nd.left, z: nd.p[1] + q.nz * nd.left });
      out.push({ kind: 'wr', n: i, x: nd.p[0] - q.nx * nd.right, z: nd.p[1] - q.nz * nd.right });
    }
  });
  return out;
}
function endHandles(road) {
  // the selected line's or rumble's ends: [{kind: 'lfrom'|'lto'|'kfrom'|'kto', x, z}]
  const out = [], o = road && geo(road);
  if (!o || road.along) return out;
  const put = (item, kinds, offs) => {
    const [s0, s1] = span(o, item.from, item.to);
    [[kinds[0], s0], [kinds[1], s1]].forEach(([kind, s]) => { const q = at(o, s), w = (offs.a(q) + offs.b(q)) / 2; out.push({ kind, x: q.x + q.nx * w, z: q.z + q.nz * w, nx: q.nx, nz: q.nz }); });
  };
  if (S.sel.line >= 0 && road.lines && road.lines[S.sel.line]) put(road.lines[S.sel.line], ['lfrom', 'lto'], lineOffsets(o, road.lines[S.sel.line]));
  if (S.sel.kerb >= 0 && road.kerbs && road.kerbs[S.sel.kerb]) {
    // a rumble's own ends: its first piece's start, its last piece's end (on another road where it goes on along one)
    const k = road.kerbs[S.sel.kerb], pcs = kerbPieces(road, k);
    if (pcs.length) {
      const first = pcs[0], last = pcs[pcs.length - 1];
      [['kfrom', first, first.role === 'before' ? first.outer : first.s0], ['kto', last, last.role === 'after' ? last.outer : last.s1]].forEach(([kind, pc, s_]) => {
        const off = pieceOffsets(k, pc), q = at(pc.o, s_), w = (off.a(q) + off.b(q)) / 2;
        out.push({ kind, x: q.x + q.nx * w, z: q.z + q.nz * w, nx: q.nx, nz: q.nz });
      });
    }
  }
  return out;
}
function drawHandles(road) {
  const hs = nodeHandles(road);
  g.lineWidth = 1;
  for (const h of hs) if (h.kind === 'in' || h.kind === 'out') {
    const nd = road.curve[h.n], [ax, ay] = toScreen(nd.p[0], nd.p[1]), [bx, by] = toScreen(h.x, h.z);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.strokeStyle = 'rgba(79,163,255,0.8)'; g.stroke();
  }
  for (const h of hs) {
    const [x, y] = toScreen(h.x, h.z), sel = h.n === S.sel.n;
    g.beginPath();
    if (h.kind === 'p') g.rect(x - 5, y - 5, 10, 10);
    else if (h.kind === 'in' || h.kind === 'out') g.arc(x, y, 4.5, 0, Math.PI * 2);
    else { g.moveTo(x, y - 6); g.lineTo(x + 6, y); g.lineTo(x, y + 6); g.lineTo(x - 6, y); g.closePath(); }
    g.fillStyle = h.kind === 'p' ? (sel ? '#4fa3ff' : '#ffffff') : h.kind === 'wl' ? '#7dff9a' : h.kind === 'wr' ? '#ff8ad0' : '#cfe6ff';
    g.fill(); g.strokeStyle = '#000'; g.stroke();
    if (h.kind === 'wl' || h.kind === 'wr') { g.fillStyle = '#000'; g.font = '9px system-ui'; g.fillText(h.kind === 'wl' ? 'L' : 'R', x - 3, y + 3); }
  }
  for (const h of endHandles(road)) {
    const [x, y] = toScreen(h.x, h.z);
    g.beginPath(); g.moveTo(x, y - 7); g.lineTo(x + 6, y + 5); g.lineTo(x - 6, y + 5); g.closePath();
    g.fillStyle = h.kind.endsWith('from') ? '#4fa3ff' : '#ff9f43'; g.fill(); g.strokeStyle = '#000'; g.stroke();
  }
  for (const h of alongHandles(road)) {
    const [x, y] = toScreen(h.x, h.z);
    g.beginPath();
    if (h.kind === 'aw') { g.moveTo(x, y - 6); g.lineTo(x + 6, y); g.lineTo(x, y + 6); g.lineTo(x - 6, y); g.closePath(); g.fillStyle = h.n === S.sel.n ? '#4fa3ff' : '#ffaa3c'; }
    else { g.moveTo(x, y - 7); g.lineTo(x + 6, y + 5); g.lineTo(x - 6, y + 5); g.closePath(); g.fillStyle = h.kind === 'afrom' ? '#4fa3ff' : '#ff9f43'; }
    g.fill(); g.strokeStyle = '#000'; g.lineWidth = 1; g.stroke();
  }
  // joined ends ringed
  (road.curve || []).forEach((nd) => { if (!nd.joint) return; const [x, y] = toScreen(nd.p[0], nd.p[1]); g.beginPath(); g.arc(x, y, 10, 0, Math.PI * 2); g.strokeStyle = '#7dff9a'; g.lineWidth = 2; g.stroke(); g.lineWidth = 1; });
  // first node marked: the road's start
  if (road.curve && road.curve.length) { const [x, y] = toScreen(road.curve[0].p[0], road.curve[0].p[1]); g.fillStyle = '#fff'; g.font = 'bold 11px system-ui'; g.fillText('start', x + 8, y - 8); }
}
function drawNames(b) {
  g.font = '12px system-ui'; g.textAlign = 'center';
  for (const r of S.roads) {
    const o = geo(r); if (!o || !boxVisible(o.box, b)) continue;
    const q = at(o, o.length / 2), [x, y] = toScreen(q.x, q.z);
    g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,0.8)'; g.strokeText(r.name, x, y); g.fillStyle = '#fff'; g.fillText(r.name, x, y);
  }
  g.textAlign = 'left';
}
let msgText = '';
function msg(t) { msgText = t; }
function draw() {
  drawQueued = false; msgText = '';
  screenT();
  g.fillStyle = '#2a2d33'; g.fillRect(0, 0, W(), H());
  const b = viewBox(40);
  if (S.layers.sat) drawSatellite();
  if (S.layers.ctx) drawContext(b);
  if (S.layers.draw) drawDrawings();
  worldT();
  const sel = S.roads[S.sel.r];
  S.roads.forEach((r, i) => { const o = geo(r); if (o && boxVisible(o.box, b) && i !== S.sel.r) drawRoad(r, false); });
  if (sel) drawRoad(sel, true);
  if (sel && S.layers.wire) for (const r of wired()) drawWire(r, b);
  S.roads.forEach((r) => { const o = geo(r); if (o && boxVisible(o.box, b)) drawPaint(r); });
  if (sel) drawSelectedItem(sel);
  screenT();
  if (S.layers.surface) drawSurface(b);
  if (S.layers.names) drawNames(b);
  if (sel) drawHandles(sel);
  if (S.snapMark) {
    const [x, y] = toScreen(S.snapMark.x, S.snapMark.z);
    g.beginPath(); g.arc(x, y, 8, 0, Math.PI * 2); g.strokeStyle = '#3ee8ff'; g.lineWidth = 2; g.stroke(); g.lineWidth = 1;
    g.font = '11px system-ui'; g.fillStyle = '#3ee8ff'; g.fillText(S.snapMark.label, x + 11, y - 9);
  }
  if (S.measure) {
    const m = S.measure, [ax, ay] = toScreen(...m.a), [bx, by] = toScreen(...m.b);
    const dx = m.b[0] - m.a[0], dz = m.b[1] - m.a[1], len = Math.hypot(dx, dz);
    g.beginPath(); g.moveTo(ax, ay); g.lineTo(bx, by); g.strokeStyle = '#ffe14d'; g.lineWidth = 2; g.stroke(); g.lineWidth = 1;
    for (const [x, y] of [[ax, ay], [bx, by]]) { g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fillStyle = '#ffe14d'; g.fill(); }
    const label = len.toFixed(2) + ' m  (dx ' + dx.toFixed(2) + ', dz ' + dz.toFixed(2) + ')';
    g.font = 'bold 12px system-ui'; g.lineWidth = 3; g.strokeStyle = 'rgba(0,0,0,0.8)'; g.strokeText(label, (ax + bx) / 2 + 8, (ay + by) / 2 - 8);
    g.fillStyle = '#ffe14d'; g.fillText(label, (ax + bx) / 2 + 8, (ay + by) / 2 - 8); g.lineWidth = 1;
  }
  if (S.newRoad) {
    const pts = S.newRoad.points.concat(S.mouse ? [S.mouse] : []);
    g.beginPath(); pts.forEach((p, i) => { const [x, y] = toScreen(p[0], p[1]); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.strokeStyle = '#ffe14d'; g.lineWidth = 2; g.setLineDash([5, 4]); g.stroke(); g.setLineDash([]);
    for (const p of S.newRoad.points) { const [x, y] = toScreen(p[0], p[1]); g.fillStyle = '#ffe14d'; g.fillRect(x - 4, y - 4, 8, 8); }
    msg('New road: click its points, Enter to finish, Esc to drop it');
  }
  if (S.mode === 'measure') msg('Measure: drag from one point to another (they snap; Ctrl: not), Esc to stop');
  if (S.mode === 'add' && !S.newRoad) msg(sel ? 'Add node: click on ' + sel.name + ', or past one of its ends' : 'Add node: select a road first');
  const m = $('msg'); m.hidden = !msgText; m.textContent = msgText;
  $('zoom').textContent = S.view.ppm.toFixed(1) + ' px/m';
}

// ------------------------------------------------------------------ editing

function autoHandles(road, only = -1) {
  // handles along the way through each node's neighbours (as a Catmull-Rom curve), a third of each piece long
  const c = road.curve;
  c.forEach((nd, i) => {
    if (only >= 0 && i !== only) return;
    const a = c[Math.max(0, i - 1)].p, b = c[Math.min(c.length - 1, i + 1)].p;
    let tx = b[0] - a[0], tz = b[1] - a[1]; const m = Math.hypot(tx, tz) || 1; tx /= m; tz /= m;
    const dIn = i > 0 ? Math.hypot(nd.p[0] - c[i - 1].p[0], nd.p[1] - c[i - 1].p[1]) / 3 : 0;
    const dOut = i < c.length - 1 ? Math.hypot(c[i + 1].p[0] - nd.p[0], c[i + 1].p[1] - nd.p[1]) / 3 : 0;
    nd.in = [-tx * dIn, -tz * dIn]; nd.out = [tx * dOut, tz * dOut]; delete nd.corner;
  });
}
function splitAt(road, seg, t) {
  // a node in piece seg at t, the curve kept as it is (de Casteljau)
  const a = road.curve[seg], b = road.curve[seg + 1];
  const [, , P0, P1, P2, P3] = bezPoint(a, b, 0);
  const lerp = (p, q) => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
  const A = lerp(P0, P1), B = lerp(P1, P2), C = lerp(P2, P3), D = lerp(A, B), E = lerp(B, C), M = lerp(D, E);
  const e = smooth(t);
  const nd = { p: M, in: [D[0] - M[0], D[1] - M[1]], out: [E[0] - M[0], E[1] - M[1]], left: a.left + (b.left - a.left) * e, right: a.right + (b.right - a.right) * e };
  a.out = [A[0] - P0[0], A[1] - P0[1]]; b.in = [C[0] - P3[0], C[1] - P3[1]];
  road.curve.splice(seg + 1, 0, nd);
  return seg + 1;
}
function addNode(road, x, z) {
  const o = geo(road); if (!o) return;
  const near = nearestOn(o, x, z), last = o.X.length - 1;
  remember();
  if ((near.i === 0 || near.i === last) && near.d > 1) {
    const end = near.i === 0 ? 0 : road.curve.length - 1, nd = road.curve[end];
    if (nd.joint) { S.undo.pop(); alert('That end is joined to ' + nd.joint.road + ': unjoin it first to make the road longer there.'); return; }
    const d = [(x - nd.p[0]) / 3, (z - nd.p[1]) / 3];
    const nn = { p: [x, z], in: [0, 0], out: [0, 0], left: nd.left, right: nd.right };
    if (end === 0) { nn.out = [-d[0], -d[1]]; nd.in = [d[0], d[1]]; road.curve.unshift(nn); S.sel.n = 0; }
    else { nn.in = [-d[0], -d[1]]; nd.out = [d[0], d[1]]; road.curve.push(nn); S.sel.n = road.curve.length - 1; }
  } else {
    S.sel.n = splitAt(road, o.SEG[near.i], o.T[near.i]);
  }
  touched(road); panel(); redraw();
}
function deleteSelected() {
  const road = S.roads[S.sel.r]; if (!road) return;
  remember();
  if (S.sel.line >= 0) { road.lines.splice(S.sel.line, 1); S.sel.line = -1; touched(road, false); }
  else if (S.sel.kerb >= 0) { road.kerbs.splice(S.sel.kerb, 1); S.sel.kerb = -1; touched(road, false); }
  else if (S.sel.n >= 0 && road.curve && road.curve.length > 2) { unjoin(road, S.sel.n, false); road.curve.splice(S.sel.n, 1); S.sel.n = -1; touched(road); }
  else if (S.sel.n >= 0 && road.along && road.profile && road.profile.length) { const w = road.profile[S.sel.n][1]; road.profile.splice(S.sel.n, 1); if (!road.profile.length) { delete road.profile; road.width = w; } S.sel.n = -1; touched(road); }
  else { S.undo.pop(); return; }
  panel(); redraw();
}
function moveRoad(by) {
  // the selected road up or down the list: the order roads are built in, each over those after it
  const i = S.sel.r, j = i + by;
  if (i < 0 || j < 0 || j >= S.roads.length) return;
  remember();
  [S.roads[i], S.roads[j]] = [S.roads[j], S.roads[i]];
  S.sel.r = j;
  for (const r of S.roads) { r.moved = r.moved || false; delete S.meshes[r.name]; }
  setDirty(true); panel(); wantMesh(); redraw();
}
function deleteRoad() {
  const road = S.roads[S.sel.r]; if (!road || !confirm('Delete ' + road.name + ' from the described roads? (Its old mesh stays as it is.)')) return;
  remember(); if (road.curve) { unjoin(road, 0, false); unjoin(road, road.curve.length - 1, false); } S.deleted.add(road.was || road.name); S.roads.splice(S.sel.r, 1); S.sel = { r: -1, n: -1, line: -1, kerb: -1 }; setDirty(true); panel(); redraw();
}
function snapEnds(road) {
  // its first and last nodes onto the centre lines of the roads it joins, along them, as wide as they are
  const j = road.joins || [];
  remember();
  [[0, j[0]], [road.curve.length - 1, j[1]]].forEach(([i, name]) => {
    const p = name && byName(name), o = p && geo(p); if (!o) return;
    const nd = road.curve[i], near = nearestOn(o, nd.p[0], nd.p[1]), q = at(o, near.s);
    const tx = q.nz, tz = -q.nx;                              // (its way along: the left turned back)
    const h = i === 0 ? nd.out : nd.in, len = Math.hypot(h[0], h[1]) || 3;
    const way = i === 0 ? 1 : -1, sgn = Math.sign(tx * h[0] * way + tz * h[1] * way) || 1;
    nd.p = [q.x, q.z];
    const dir = [tx * sgn, tz * sgn];
    if (i === 0) nd.out = [dir[0] * len, dir[1] * len]; else nd.in = [-dir[0] * len, -dir[1] * len];
    const same = sgn > 0;
    nd.left = same ? q.l : q.r; nd.right = same ? q.r : q.l;
  });
  touched(road); panel(); redraw();
}
// Bringing in a road the track has (its mesh, not described yet): read by the server as a curve
let bringMeshes = [];
async function openBring() {
  $('bringbox').hidden = false; $('bringfilter').value = ''; $('bringfilter').focus();
  $('bringlist').replaceChildren(el('li', {}, 'Loading…'));
  const res = await fetch('/api/meshes').then((x) => x.json());
  bringMeshes = (res.meshes || []).filter((m) => !byName(m.name));
  bringList();
}
function bringList() {
  const f = $('bringfilter').value.trim().toLowerCase();
  const items = bringMeshes.filter((m) => !f || (m.name + ' ' + m.model).toLowerCase().includes(f));
  $('bringlist').replaceChildren(...(items.length ? items.map((m) => el('li', { onclick: () => bringIn(m) },
    el('b', {}, m.name), ' ', el('span', { class: 'detail' }, m.model + ', ' + Math.round(m.area) + ' m², ' + m.material))) : [el('li', {}, 'none left')]));
}
async function bringIn(m) {
  status('reading ' + m.name + '…');
  const res = await fetch('/api/bring', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ model: m.model, name: m.name }) }).then((x) => x.json());
  if (res.error || !res.roads) { status('could not bring in ' + m.name + ': ' + (res.error || '?')); return; }
  remember();
  for (const r of res.roads) {
    if (byName(r.name)) continue;
    const road = Object.assign({ lines: [], kerbs: [] }, r, { moved: true, _v: 1 });
    S.roads.push(road); S.deleted.delete(road.name); touched(road);
  }
  S.sel = { r: S.roads.findIndex((r) => r.name === res.roads[0].name), n: -1, line: -1, kerb: -1 };
  S.view.cx = (m.box[0] + m.box[2]) / 2; S.view.cz = (m.box[1] + m.box[3]) / 2;
  $('bringbox').hidden = true;
  status('brought in ' + res.roads.map((r) => r.name).join(' and ') + ' (read ' + (res.how === 'ring' ? 'as a ring: two halves joined end to end' : res.how === 'strip' ? 'from its outline' : 'by its rows') + '); not saved yet');
  setDirty(true); panel(); wantMesh(); redraw();
}
function finishNewRoad() {
  const nr = S.newRoad; S.newRoad = null;
  if (!nr || nr.points.length < 2) { redraw(); return; }
  remember();
  const road = { name: nr.name, material: 'Tarmac Dark', map: 'flat', curve: nr.points.map((p) => ({ p, in: [0, 0], out: [0, 0], left: 4, right: 4 })), lines: [], kerbs: [], moved: true, _v: 1 };
  autoHandles(road);
  S.roads.push(road); S.sel = { r: S.roads.length - 1, n: -1, line: -1, kerb: -1 };
  setMode('select'); touched(road); panel(); redraw();
}

// ------------------------------------------------------------------ mouse and keys

function itemAt(wx, wz) {
  // the line or rumble under (wx, wz), of any road: {kind: 'item', which, road, idx, s}; within a few pixels
  const tol = Math.max(0.05, 4 / S.view.ppm);
  let best = null;
  S.roads.forEach((r, i) => {
    const o = geo(r); if (!o || r.along || !S.layers.lines && !S.layers.kerbs) return;
    const far = wx < o.box.xmin - 2 || wx > o.box.xmax + 2 || wz < o.box.zmin - 2 || wz > o.box.zmax + 2;
    if (far && !(r.kerbs || []).some((k) => k.before || k.after)) return;
    const consider = (which, idx, oo, s0, s1, f) => {
      // (one piece of it, on its road oo)
      const near = nearestOn(oo, wx, wz), q = at(oo, near.s), off = (wx - q.x) * q.nx + (wz - q.z) * q.nz;
      if (near.s < s0 - tol || near.s > s1 + tol) return;
      const a = f.a(q, near.s), b = f.b(q, near.s), lo = Math.min(a, b) - tol, hi = Math.max(a, b) + tol;
      if (off < lo || off > hi) return;
      const chosen = i === S.sel.r && (which === 'line' ? S.sel.line : S.sel.kerb) === idx;
      const score = Math.abs(off - (a + b) / 2) + (which === 'line' ? 0 : 0.01) - (chosen ? 100 : 0);   // (the selected one first; a line over a rumble)
      if (!best || score < best.score) best = { kind: 'item', which, road: i, idx, s: near.s, score };
    };
    if (S.layers.lines) (r.lines || []).forEach((it, idx) => { const [s0, s1] = span(o, it.from, it.to); consider('line', idx, o, s0, s1, lineOffsets(o, it)); });
    if (S.layers.kerbs) (r.kerbs || []).forEach((it, idx) => { for (const pc of kerbPieces(r, it)) consider('kerb', idx, pc.o, pc.s0, pc.s1, pieceOffsets(it, pc)); });
  });
  return best;
}
function hit(sx, sy) {
  const road = S.roads[S.sel.r];
  if (road) {
    for (const h of endHandles(road)) { const [x, y] = toScreen(h.x, h.z); if (Math.hypot(x - sx, y - sy) < 9) return Object.assign({ road: S.sel.r }, h); }
    for (const h of alongHandles(road)) { const [x, y] = toScreen(h.x, h.z); if (Math.hypot(x - sx, y - sy) < 9) return Object.assign({ road: S.sel.r }, h); }
    const hs = nodeHandles(road);
    for (const kind of ['in', 'out', 'wl', 'wr', 'p']) for (const h of hs) {
      if (h.kind !== kind) continue;
      const [x, y] = toScreen(h.x, h.z); if (Math.hypot(x - sx, y - sy) < 8) return Object.assign({ road: S.sel.r }, h);
    }
  }
  const [wx, wz] = toWorld(sx, sy);
  const item = itemAt(wx, wz);
  if (item) return item;
  let best = null;
  S.roads.forEach((r, i) => {
    const o = geo(r); if (!o) return;
    if (wx < o.box.xmin || wx > o.box.xmax || wz < o.box.zmin || wz > o.box.zmax) return;
    const near = nearestOn(o, wx, wz), q = { l: o.L[near.i], r: o.R[near.i] };
    const dx = wx - o.X[near.i], dz = wz - o.Z[near.i], side = dx * o.nx[near.i] + dz * o.nz[near.i];
    const inside = side >= 0 ? side <= q.l + 0.05 : -side <= q.r + 0.05;
    const score = (inside ? near.d : (near.d * S.view.ppm < 6 ? near.d + 100 : Infinity)) - (i === S.sel.r && inside ? 1000 : 0);   // (the selected road first)
    if (score < Infinity && (!best || score < best.score)) best = { kind: 'road', road: i, score, s: near.s };
  });
  return best;
}
function onDown(e) {
  cv.focus();
  const sx = e.offsetX, sy = e.offsetY, [wx, wz] = toWorld(sx, sy);
  if (e.button === 1 || e.button === 2 || S.space) { S.drag = { kind: 'pan', sx, sy, cx: S.view.cx, cz: S.view.cz }; return; }
  if (S.mode === 'new') {
    if (!S.newRoad) {
      const name = prompt('Name of the new road (as its mesh: 1ROAD_…)', '1ROAD_NEW_' + (S.roads.length + 1));
      if (!name) return;
      if (byName(name)) { alert('A road is named ' + name + ' already.'); return; }
      S.newRoad = { name, points: [] };
    }
    S.newRoad.points.push([wx, wz]); redraw(); return;
  }
  if (S.mode === 'measure') {
    const sp = e.ctrlKey ? null : snapPoint(wx, wz, null), a = sp ? [sp.x, sp.z] : [wx, wz];
    S.measure = { a, b: a }; S.drag = { kind: 'measure' }; redraw(); return;
  }
  if (S.mode === 'add') {
    const r = S.roads[S.sel.r];
    if (r && r.curve && !r.along) addNode(r, wx, wz);
    else if (r && r.along) {
      const p = byName(r.along.road), op = p && geo(p); if (!op) return;
      const s_ = Math.round(nearestOn(op, wx, wz).s * 10) / 10;
      if (s_ <= r.along.from || s_ >= r.along.to) return;
      remember();
      if (!r.profile || !r.profile.length) r.profile = [[(r.along.from + r.along.to) / 2, r.width ?? 2]];
      r.profile.push([s_, Math.round(alongWidth(r, s_) * 100) / 100]);
      r.profile.sort((a, b) => a[0] - b[0]);
      S.sel.n = r.profile.findIndex((q) => q[0] === s_);
      touched(r); panel(); redraw();
    }
    return;
  }
  const h = hit(sx, sy);
  if (!h) { S.drag = { kind: 'pan', sx, sy, cx: S.view.cx, cz: S.view.cz, click: true }; return; }
  if (h.kind === 'road') {
    if (h.road !== S.sel.r) { S.sel = { r: h.road, n: -1, line: -1, kerb: -1 }; panel(); wantMesh(); }
    S.drag = { kind: 'pan', sx, sy, cx: S.view.cx, cz: S.view.cz };
    redraw(); return;
  }
  if (h.kind === 'item') {
    // a line or rumble: selected, and dragged along its road as a whole
    const r = S.roads[h.road], it = (h.which === 'line' ? r.lines : r.kerbs)[h.idx], o = geo(r);
    if (h.road !== S.sel.r) wantMesh();
    S.sel = { r: h.road, n: -1, line: h.which === 'line' ? h.idx : -1, kerb: h.which === 'kerb' ? h.idx : -1 };
    panel();
    if (h.which === 'kerb' && (it.before || it.after)) { status('this rumble goes on along another road: drag its ends (the triangles)'); redraw(); return; }
    remember();
    const [s0, s1] = span(o, it.from, it.to);
    S.drag = { kind: 'item', which: h.which, idx: h.idx, road: h.road, s: h.s, from0: s0, to0: s1, toEnd: it.to == null, moved: false };
    cv.style.cursor = 'grabbing'; redraw(); return;
  }
  remember();
  if (['p', 'in', 'out', 'wl', 'wr', 'aw'].includes(h.kind)) { S.sel.n = h.n; panel(); }
  S.drag = Object.assign({ moved: false, alt: e.altKey }, h);
  redraw();
}
// What a point snaps to near (wx, wz): another road's end node, its centre line, its edges; within SNAP_PX
// pixels; the end nodes first. {x, z, label} or null.
const SNAP_PX = 10;
function snapPoint(wx, wz, road) {
  const tol = SNAP_PX / S.view.ppm;
  let best = null;
  const take = (x, z, label, rank) => {
    const d = Math.hypot(x - wx, z - wz);
    if (d <= tol && (!best || rank < best.rank || (rank === best.rank && d < best.d))) best = { x, z, label, rank, d };
  };
  for (const r of S.roads) {
    if (r === road || r.along) continue;
    const o = geo(r); if (!o) continue;
    if (wx < o.box.xmin - tol || wx > o.box.xmax + tol || wz < o.box.zmin - tol || wz > o.box.zmax + tol) continue;
    if (r.curve) for (const i of [0, r.curve.length - 1]) take(r.curve[i].p[0], r.curve[i].p[1], r.name + "'s " + (i ? 'end' : 'start'), 0);
    const n = nearestOn(o, wx, wz), q = at(o, n.s);
    take(q.x, q.z, r.name + "'s centre line, s " + n.s.toFixed(1), 1);
    take(q.x + q.nx * q.l, q.z + q.nz * q.l, r.name + "'s left edge, s " + n.s.toFixed(1), 1);
    take(q.x - q.nx * q.r, q.z - q.nz * q.r, r.name + "'s right edge, s " + n.s.toFixed(1), 1);
  }
  return best;
}
function onMove(e) {
  const sx = e.offsetX, sy = e.offsetY, [wx, wz] = toWorld(sx, sy);
  S.mouse = [wx, wz];
  $('pos').textContent = 'x ' + wx.toFixed(2) + ', z ' + wz.toFixed(2);
  const road = S.roads[S.sel.r], o = road && geo(road);
  if (o) { const n = nearestOn(o, wx, wz); $('along').textContent = road.name + ': s ' + n.s.toFixed(2) + ' m of ' + o.length.toFixed(1) + (n.d > 0.05 ? ', ' + n.d.toFixed(2) + ' m off its middle' : ''); }
  else $('along').textContent = '';
  const d = S.drag;
  if (S.newRoad) redraw();
  if (!d) { if (S.mode === 'select' && !S.space) cv.style.cursor = itemAt(wx, wz) ? 'grab' : 'default'; return; }
  if (d.kind === 'measure') {
    const sp = e.ctrlKey ? null : snapPoint(wx, wz, null);
    S.snapMark = sp; S.measure.b = sp ? [sp.x, sp.z] : [wx, wz]; redraw(); return;
  }
  if (d.kind === 'pan') { S.view.cx = d.cx + (sx - d.sx) / S.view.ppm; S.view.cz = d.cz + (sy - d.sy) / S.view.ppm; if (Math.hypot(sx - d.sx, sy - d.sy) > 3) d.click = false; redraw(); return; }
  const r = S.roads[d.road]; if (!r) return;
  d.moved = true;
  if (d.kind === 'p') {
    const sp = e.ctrlKey ? null : snapPoint(wx, wz, r);
    S.snapMark = sp;
    const nd = r.curve[d.n]; nd.p = sp ? [sp.x, sp.z] : [wx, wz]; touched(r);
  }
  else if (d.kind === 'in' || d.kind === 'out') {
    const nd = r.curve[d.n], v = [wx - nd.p[0], wz - nd.p[1]];
    nd[d.kind] = v;
    if (d.alt || e.altKey) nd.corner = true;
    if (!nd.corner) {
      const other = d.kind === 'in' ? 'out' : 'in', ol = Math.hypot(nd[other][0], nd[other][1]), m = Math.hypot(v[0], v[1]) || 1;
      if (ol > 0) nd[other] = [-v[0] / m * ol, -v[1] / m * ol];
    }
    touched(r);
  } else if (d.kind === 'wl' || d.kind === 'wr') {
    const nd = r.curve[d.n], q = at(geo(r), geo(r).s[geo(r).nodeAt[d.n]]);
    const sgn = d.kind === 'wl' ? 1 : -1, w = Math.max(0.2, ((wx - nd.p[0]) * q.nx + (wz - nd.p[1]) * q.nz) * sgn);
    const round = Math.round(w * 100) / 100;
    if (e.shiftKey) { nd.left = round; nd.right = round; } else nd[d.kind === 'wl' ? 'left' : 'right'] = round;
    touched(r);
  } else if (d.kind === 'afrom' || d.kind === 'ato' || d.kind === 'aw') {
    // a run-off: its ends along its road's edge, its width points out from it (and along it)
    const al = r.along, p = byName(al.road), op = p && geo(p);
    if (op) {
      const n = nearestOn(op, wx, wz), s_ = Math.round(n.s * 10) / 10;
      if (d.kind === 'afrom') al.from = Math.min(s_, al.to - 1);
      else if (d.kind === 'ato') al.to = Math.max(s_, al.from + 1);
      else {
        const sgn = al.side === 'left' ? 1 : -1, q = at(op, n.s), ex = q.x + q.nx * sgn * (sgn > 0 ? q.l : q.r), ez = q.z + q.nz * sgn * (sgn > 0 ? q.l : q.r);
        const w = Math.max(0.1, Math.round(((wx - ex) * q.nx + (wz - ez) * q.nz) * sgn * 100) / 100);
        if (!r.profile || !r.profile.length) r.profile = [[(al.from + al.to) / 2, r.width ?? 2]];
        r.profile[d.n] = [Math.min(Math.max(s_, al.from), al.to), w];
      }
      touched(r);
    }
  } else if (d.kind === 'item') {
    // the whole of it along the road, as long as it was, within the road's ends
    const og = geo(r), L = og.length, len = d.to0 - d.from0;
    const ds = nearestOn(og, wx, wz).s - d.s;
    const from = Math.round(Math.min(Math.max(d.from0 + ds, 0), Math.max(L - len, 0)) * 10) / 10;
    const it = (d.which === 'line' ? r.lines : r.kerbs)[d.idx];
    it.from = from;
    it.to = d.toEnd && from + len >= L - 0.05 ? null : Math.round((from + len) * 10) / 10;
    touched(r, false);
  } else if (d.kind === 'kfrom' || d.kind === 'kto') {
    const k = r.kerbs[S.sel.kerb], isStart = d.kind === 'kfrom', og = geo(r), n = nearestOn(og, wx, wz), qn = at(og, n.s);
    const onIt = n.d <= Math.max(qn.l, qn.r) + 0.3 && (isStart ? n.s > 0.05 : n.s < og.length - 0.05);
    const c = onIt ? null : continuation(r, k, isStart, wx, wz);
    const key = isStart ? 'before' : 'after';
    if (c) k[key] = c;
    else if (onIt || !k[key]) {
      delete k[key];
      const s_ = Math.round(n.s * 10) / 10;
      if (isStart) k.from = Math.min(s_, (k.to ?? og.length) - 0.2); else k.to = s_ >= og.length - 0.05 ? null : Math.max(s_, (k.from ?? 0) + 0.2);
    }
    touched(r, false);
  } else if (['lfrom', 'lto', 'kfrom', 'kto'].includes(d.kind)) {
    const s = Math.round(nearestOn(o, wx, wz).s * 10) / 10;
    const item = d.kind[0] === 'l' ? r.lines[S.sel.line] : r.kerbs[S.sel.kerb];
    item[d.kind.slice(1)] = s >= o.length - 0.05 && d.kind.endsWith('to') ? null : s;
    touched(r, false);
  }
  panelValues(); redraw();
}
function onUp() {
  const d = S.drag; S.drag = null;
  if (S.snapMark) { status('snapped to ' + S.snapMark.label); S.snapMark = null; redraw(); }
  if (d && d.kind === 'item') cv.style.cursor = 'grab';
  if (d && d.kind === 'pan' && d.click && S.mode === 'select') { S.sel = { r: -1, n: -1, line: -1, kerb: -1 }; panel(); redraw(); }
  if (d && d.kind !== 'pan' && d.kind !== 'measure' && !d.moved) S.undo.pop();
  if (d && d.moved) { panel(); wantMesh(); }
}
function onWheel(e) {
  e.preventDefault();
  const [wx, wz] = toWorld(e.offsetX, e.offsetY);
  const f = Math.exp(-e.deltaY * 0.0015);
  S.view.ppm = Math.min(400, Math.max(0.3, S.view.ppm * f));
  const [nx, nz] = toWorld(e.offsetX, e.offsetY);
  S.view.cx += wx - nx; S.view.cz += wz - nz;
  redraw();
}
function setMode(m) {
  S.mode = m;
  if (m !== 'new') S.newRoad = null;
  document.querySelectorAll('#tools button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m));
  cv.style.cursor = m === 'select' ? 'default' : 'crosshair';
  redraw();
}
function onKey(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') { if (e.key === 'Enter') e.target.blur(); return; }
  const k = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && k === 'z') { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
  if ((e.ctrlKey || e.metaKey) && k === 'y') { e.preventDefault(); redo(); return; }
  if ((e.ctrlKey || e.metaKey) && k === 's') { e.preventDefault(); save(); return; }
  if (e.key === ' ') { S.space = true; e.preventDefault(); return; }
  if (e.key === 'Escape' && !$('bringbox').hidden) { $('bringbox').hidden = true; return; }
  if (e.key === 'Escape') { S.newRoad = null; S.measure = null; setMode('select'); S.sel.n = -1; S.sel.line = -1; S.sel.kerb = -1; panel(); return; }
  if (e.key === 'Enter' && S.newRoad) { finishNewRoad(); return; }
  if (e.key === 'Delete' || e.key === 'Backspace') { e.shiftKey ? deleteRoad() : deleteSelected(); return; }
  if (k === 'c') { checkSurface(); return; }
  if (k === 'b') { openBring(); e.preventDefault(); return; }
  if (k === 'j') { const r = S.roads[S.sel.r]; if (r && r.curve && S.sel.n >= 0) joinNearest(r, S.sel.n); }
  if (k === 'v') setMode('select');
  if (k === 'a') setMode('add');
  if (k === 'n') setMode('new');
  if (k === 'm') setMode(S.mode === 'measure' ? 'select' : 'measure');
  if (k === 's') { const r = S.roads[S.sel.r]; if (r && r.curve && !r.along) { remember(); autoHandles(r, S.sel.n); touched(r); panel(); redraw(); } }
  if (k === '?') $('helpbox').hidden = !$('helpbox').hidden;
}

// ------------------------------------------------------------------ the panel

function el(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (k === 'value') e.value = v; else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v); }
  for (const c of kids.flat()) if (c != null) e.append(c.nodeType ? c : String(c));
  return e;
}
let fieldSeq = 0;
const fmt = (v) => (v == null || v === '' || !isFinite(v) ? '' : String(Math.round(+v * 1000) / 1000));
function num(value, set, step = 0.1, attrs = {}) {
  return el('input', Object.assign({ type: 'number', step: 'any', value: fmt(value), 'data-k': 'f' + fieldSeq++,
    onchange: (e) => { const v = e.target.value.trim(); if (v !== '' && !isFinite(parseFloat(v))) return; remember(); set(v === '' ? null : parseFloat(v)); } }, attrs));
}
function pick(value, options, set) {
  return el('select', { 'data-k': 'f' + fieldSeq++, onchange: (e) => { remember(); set(e.target.value); } }, options.map(([v, label]) => el('option', { value: v, selected: v === value }, label)));
}
function fillIn(box, ...kids) {
  // a panel section's contents: lists opened out, nothing for what is not there
  box.replaceChildren(...kids.flat(Infinity).filter((k) => k != null && k !== false));
}
function roadList() {
  const f = $('filter').value.toLowerCase(), ul = $('roadlist');
  ul.replaceChildren(...S.roads.map((r, i) => [r, i]).filter(([r]) => r.name.toLowerCase().includes(f)).map(([r, i]) =>
    el('li', { class: i === S.sel.r ? 'sel' : '', onclick: () => { S.sel = { r: i, n: -1, line: -1, kerb: -1 }; const o = geo(r); if (o) { const q = at(o, o.length / 2); S.view.cx = q.x; S.view.cz = q.z; } panel(); wantMesh(); redraw(); } },
      el('span', {}, (i + 1) + '. ' + r.name + (r.moved ? ' *' : '')),
      el('span', { class: 'tag' }, r.along ? 'run-off' : r.joins && r.joins.some(Boolean) ? 'branch' : r.fitted ? 'fitted' : 'curve'))));
}
function panel() {
  const focused = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.k : null;
  fieldSeq = 0;
  buildPanel();
  if (focused) { const f = document.querySelector('#panel [data-k="' + focused + '"]'); if (f) f.focus(); }
}
function buildPanel() {
  roadList();
  const road = S.roads[S.sel.r], rp = $('roadprops'), np = $('nodeprops'), lp = $('lineprops'), kp = $('kerbprops');
  rp.hidden = np.hidden = lp.hidden = kp.hidden = !road;
  if (!road) return;
  const others = [['', '—']].concat(S.roads.filter((r) => r !== road && !r.along).map((r) => [r.name, r.name]));
  const change = (fn, shape = false) => (v) => { fn(v); touched(road, shape); panel(); redraw(); };
  const o = geo(road);
  fillIn(rp, 
    el('h2', {}, 'Road'),
    el('div', { class: 'row' }, el('span', {}, 'Name'), el('input', { value: road.name, 'data-k': 'name', onchange: (e) => { remember(); const was = road.name; road.name = e.target.value.trim(); for (const r of S.roads) for (const n of r.curve || []) if (n.joint && n.joint.road === was) n.joint.road = road.name; touched(road, false); panel(); } })),
    el('div', { class: 'row' }, el('span', {}, 'Length'), (o ? o.length.toFixed(1) : '–') + ' m' + (road.curve ? ', ' + road.curve.length + ' nodes' : '')),
    el('div', { class: 'row' }, el('span', {}, 'Material'), pick(road.material || 'Tarmac Dark', S.materials.map((m) => [m, m]), change((v) => { road.material = v; }))),
    el('div', { class: 'row' }, el('span', {}, 'Texture'), pick(road.map || 'along', [['along', 'along it'], ['flat', 'flat from above']], change((v) => { road.map = v; }, true))),
    road.along ? [
      el('div', { class: 'row' }, el('span', {}, 'Along'), pick(road.along.road, others.slice(1), change((v) => { road.along.road = v; }, true)), pick(road.along.side, [['left', 'left'], ['right', 'right']], change((v) => { road.along.side = v; }, true))),
      el('div', { class: 'row' }, el('span', {}, 'From–to'), num(road.along.from, change((v) => { road.along.from = v; }, true)), num(road.along.to, change((v) => { road.along.to = v; }, true))),
      road.profile && road.profile.length
        ? el('div', { class: 'row' }, el('span', {}, 'Width'), 'varies (' + road.profile.length + ' points)', el('button', { class: 'small', onclick: () => { remember(); road.width = Math.max(...road.profile.map((q) => q[1])); delete road.profile; touched(road); panel(); redraw(); } }, 'One width'), 'ease', num(road.ease, change((v) => { road.ease = v; }, true)))
        : el('div', { class: 'row' }, el('span', {}, 'Width'), num(road.width, change((v) => { road.width = v; }, true)), 'ease', num(road.ease, change((v) => { road.ease = v; }, true))),
      el('p', { class: 'hint' }, "A run-off: asphalt along its road's edge. Drag its ends (triangles) along that edge and its width points (diamonds) out and along; A then a click on it: a width point more; Delete: the selected one out."),
    ] : [
      el('div', { class: 'row' }, el('span', {}, 'Leaves'), pick((road.joins || [])[0] || '', others, change((v) => { road.joins = [v || null, (road.joins || [])[1] || null]; if (!road.joins.some(Boolean)) delete road.joins; }, true))),
      el('div', { class: 'row' }, el('span', {}, 'Joins'), pick((road.joins || [])[1] || '', others, change((v) => { road.joins = [(road.joins || [])[0] || null, v || null]; if (!road.joins.some(Boolean)) delete road.joins; }, true))),
      el('div', { class: 'row' },
        el('button', { class: 'small', onclick: () => snapEnds(road), disabled: !(road.joins && road.joins.some(Boolean)) }, 'Ends onto the joined roads'),
        el('button', { class: 'small', onclick: () => { remember(); autoHandles(road); touched(road); panel(); redraw(); } }, 'Smooth all'),
        el('button', { class: 'small', onclick: () => { remember(); for (const n of [road.curve[0], road.curve[road.curve.length - 1]]) { const other = n.joint && byName(n.joint.road); if (other && other.curve) { const k = endIndex(other, n.joint.node); if (other.curve[k].joint) other.curve[k].joint.node = other.curve[k].joint.node === 'start' ? 'end' : 'start'; } } if (road.start_left || road.end_left) { [road.start_left, road.end_left] = [road.end_left && road.end_left.map((v) => -v), road.start_left && road.start_left.map((v) => -v)]; if (!road.start_left) delete road.start_left; if (!road.end_left) delete road.end_left; } road.curve.reverse().forEach((n) => { [n.in, n.out] = [n.out, n.in]; [n.left, n.right] = [n.right, n.left]; }); (road.lines || []).forEach((l) => { l.side = l.side === 'left' ? 'right' : 'left'; const L = o.length, f = l.from, t = l.to; l.from = t == null ? 0 : L - t; l.to = f ? L - f : null; }); (road.kerbs || []).forEach((l) => { l.side = l.side === 'left' ? 'right' : 'left'; const L = o.length, f = l.from, t = l.to; l.from = t == null ? 0 : L - t; l.to = f ? L - f : null; }); touched(road); panel(); redraw(); } }, 'Reverse'),
        el('button', { class: 'small', onclick: deleteRoad }, 'Delete road')),
      el('div', { class: 'row' }, el('span', {}, 'Order'), (S.sel.r + 1) + ' of ' + S.roads.length,
        el('button', { class: 'small', title: 'Over more roads: built before them, they are cut where they lie under it', onclick: () => moveRoad(-1) }, '↑ up'),
        el('button', { class: 'small', title: 'Under more roads', onclick: () => moveRoad(1) }, '↓ down')),
      el('p', { class: 'hint' }, 'A road is cut where it lies under any road above it in the list (but those joined end to end to it), and takes their heights there; rumbles lie over them all.'),
      el('p', { class: 'hint' }, road.joins && road.joins.some(Boolean) ? 'A branch: cut where it lies over the roads it leaves and joins; its lines fork from theirs.' : 'Leaves / joins: the roads this one forks off and merges into (its ends on their centre lines).'),
    ]);
  // the selected node
  const nd = road.curve && S.sel.n >= 0 ? road.curve[S.sel.n] : null;
  np.hidden = !nd;
  if (nd) {
    const set = (fn) => change(fn, true);
    fillIn(np, el('h2', {}, 'Node ' + (S.sel.n + 1) + ' of ' + road.curve.length),
      el('div', { class: 'row' }, el('span', {}, 'x, z'), num(nd.p[0], set((v) => { nd.p[0] = v; }), 0.01), num(nd.p[1], set((v) => { nd.p[1] = v; }), 0.01)),
      el('div', { class: 'row' }, el('span', {}, 'Widths'), 'L', num(nd.left, set((v) => { nd.left = v; }), 0.01), 'R', num(nd.right, set((v) => { nd.right = v; }), 0.01)),
      el('div', { class: 'row' }, el('span', {}, 'Handles'), 'in', num(Math.hypot(...nd.in), set((v) => { const m = Math.hypot(...nd.in) || 1; nd.in = [nd.in[0] / m * v, nd.in[1] / m * v]; }), 0.1), 'out', num(Math.hypot(...nd.out), set((v) => { const m = Math.hypot(...nd.out) || 1; nd.out = [nd.out[0] / m * v, nd.out[1] / m * v]; }), 0.1)),
      el('div', { class: 'row' }, el('span', {}, ''), el('label', {}, el('input', { type: 'checkbox', checked: !!nd.corner, onchange: (e) => { remember(); if (e.target.checked) nd.corner = true; else delete nd.corner; touched(road, false); } }), 'corner (handles apart)'),
        el('button', { class: 'small', onclick: () => { remember(); autoHandles(road, S.sel.n); touched(road); panel(); redraw(); } }, 'Smooth')),
      endOf(road, S.sel.n) ? el('div', { class: 'row' }, el('span', {}, 'Joint'),
        nd.joint ? [el('span', {}, 'to ' + nd.joint.road + "'s " + nd.joint.node), el('button', { class: 'small', onclick: () => unjoin(road, S.sel.n) }, 'Unjoin')]
                 : [el('button', { class: 'small', onclick: () => joinNearest(road, S.sel.n), title: "Bind this end to the nearest other road's end (J)" }, "Join to the nearest road's end"),
                    el('span', { class: 'hint' }, 'the same point, straight on, as wide: no gap')]) : null);
  }
  if (road.along) { lp.hidden = kp.hidden = true; return; }
  // its lines and rumbles: a short row each (click it to select it), the selected one's form below
  road.lines = road.lines || []; road.kerbs = road.kerbs || [];
  const set = (fn) => change(fn, false);
  const remove = (list, i) => (e) => { e.stopPropagation(); remember(); list.splice(i, 1); S.sel.line = S.sel.kerb = -1; touched(road, false); panel(); redraw(); };
  const choose = (which, i) => (e) => {
    if (e.target.closest('input, select, button')) return;            // (typing in it: not a click on the row)
    if (which === 'line') { S.sel.line = i; S.sel.kerb = -1; } else { S.sel.kerb = i; S.sel.line = -1; }
    panel(); redraw();
  };
  const sides = [['left', 'L'], ['right', 'R']];
  const add = (which, side) => () => {
    remember();
    const n = road.name.replace(/^1ROAD_/, '');
    if (which === 'line') { road.lines.push({ name: 'LINE_ROAD_' + n + '_' + (side === 'left' ? 'L' : 'R'), side, from: 0, to: null, colour: 'white' }); S.sel.line = road.lines.length - 1; S.sel.kerb = -1; }
    else { road.kerbs.push({ side, from: 0, to: Math.min(10, o ? o.length : 10), kind: 'ramp', width: 1, inset: 0, start: 'square', end: 'square', taper: 2 }); S.sel.kerb = road.kerbs.length - 1; S.sel.line = -1; }
    touched(road, false); panel(); redraw();
  };
  lp.hidden = false;
  const ln = S.sel.line >= 0 ? road.lines[S.sel.line] : null;
  fillIn(lp, el('h2', {}, 'Lines'),
    el('table', { class: 'items' }, el('tr', {}, ['', 'side', 'from (m)', 'to (m)', 'colour', 'style', ''].map((h) => el('th', {}, h))),
      road.lines.map((l, i) => el('tr', { class: S.sel.line === i ? 'sel' : '', onclick: choose('line', i) },
        el('td', {}, String(i + 1)),
        el('td', {}, pick(l.side, sides, set((v) => { l.side = v; }))),
        el('td', {}, num(l.from ?? 0, set((v) => { l.from = v ?? 0; }))),
        el('td', {}, num(l.to, set((v) => { l.to = v; }), 0.1, { placeholder: 'end' })),
        el('td', {}, pick(l.colour || 'white', [['white', 'white'], ['yellow', 'yellow']], set((v) => { l.colour = v; }))),
        el('td', {}, pick(l.style || 'solid', [['solid', 'solid'], ['dotted', 'dotted']], set((v) => { l.style = v; }))),
        el('td', {}, el('button', { class: 'small', title: 'Delete', onclick: remove(road.lines, i) }, '×'))))),
    el('div', { class: 'row' }, el('button', { class: 'small', onclick: add('line', 'left') }, '+ left line'), el('button', { class: 'small', onclick: add('line', 'right') }, '+ right line')),
    ln ? el('div', { class: 'detail' },
      el('div', { class: 'row' }, el('span', {}, 'Mesh'), el('input', { value: ln.name || '', 'data-k': 'lname', onchange: (e) => { remember(); ln.name = e.target.value.trim(); touched(road, false); } })),
      el('div', { class: 'row' }, el('span', {}, 'In (m)'), num(ln.pad ?? 0.2, set((v) => { ln.pad = v ?? 0.2; })), el('span', { class: 'hint' }, "its middle in from the road's edge")),
      el('div', { class: 'row' }, el('span', {}, 'Wide (m)'), num(ln.w ?? 0.12, set((v) => { ln.w = v ?? 0.12; }))),
      ln.style === 'dotted' ? el('div', { class: 'row' }, el('span', {}, 'Dashes (m)'), num(ln.dash ?? DASH, set((v) => { ln.dash = v ?? DASH; })), 'gaps', num(ln.gap ?? GAP, set((v) => { ln.gap = v ?? GAP; })), el('span', { class: 'hint' }, 'counted from its start')) : null) : null,
    el('p', { class: 'hint' }, "From / to: metres along the road from its start (to empty: its end). Click a row to select a line: its ends show as triangles on the map, to drag."));
  kp.hidden = false;
  const kb = S.sel.kerb >= 0 ? road.kerbs[S.sel.kerb] : null;
  const ends = [['square', 'square'], ['triangle', 'triangle (tapers in)']];
  fillIn(kp, el('h2', {}, 'Rumbles'),
    el('table', { class: 'items' }, el('tr', {}, ['', 'side', 'from (m)', 'to (m)', 'kind', ''].map((h) => el('th', {}, h))),
      road.kerbs.map((k, i) => el('tr', { class: S.sel.kerb === i ? 'sel' : '', onclick: choose('kerb', i) },
        el('td', {}, el('span', { class: 'swatch', style: 'background:' + (kindOf(k.kind) === 'ramp_concrete' ? 'linear-gradient(90deg,' + KERB_COL.ramp[0] + ' 60%,' + KERB_COL.concrete[0] + ' 60%)' : KERB_COL[kindOf(k.kind)][0]) })),
        el('td', {}, pick(k.side, sides, set((v) => { k.side = v; }))),
        el('td', {}, num(k.from ?? 0, set((v) => { k.from = v ?? 0; }))),
        el('td', {}, num(k.to, set((v) => { k.to = v; }), 0.1, { placeholder: 'end' })),
        el('td', {}, pick(kindOf(k.kind), KINDS.map(([v, label]) => [v, label]), set((v) => { k.kind = v; }))),
        el('td', {}, el('button', { class: 'small', title: 'Delete', onclick: remove(road.kerbs, i) }, '×'))))),
    el('div', { class: 'row' }, el('button', { class: 'small', onclick: add('kerb', 'left') }, '+ left rumble'), el('button', { class: 'small', onclick: add('kerb', 'right') }, '+ right rumble')),
    kb ? el('div', { class: 'detail' },
      el('div', { class: 'row' }, el('span', {}, 'Wide (m)'), num(kb.width ?? 1, set((v) => { kb.width = v ?? 1; })), el('span', { class: 'hint' }, kindOf(kb.kind) === 'ramp_concrete' ? 'its ramp, out from its inner side' : 'out from its inner side')),
      kindOf(kb.kind) === 'ramp_concrete' ? el('div', { class: 'row' }, el('span', {}, 'Concrete'), num(kb.back ?? BACK, set((v) => { kb.back = v ?? BACK; })), el('span', { class: 'hint' }, 'metres of concrete kerb behind the ramp')) : null,
      el('div', { class: 'row' }, el('span', {}, 'In (m)'), num(kb.inset ?? 0, set((v) => { kb.inset = v ?? 0; })), el('span', { class: 'hint' }, "its inner side in from the road's edge (0: on the edge)")),
      el('div', { class: 'row' }, el('span', {}, 'Start'), pick(kb.start || 'square', ends, set((v) => { kb.start = v; }))),
      el('div', { class: 'row' }, el('span', {}, 'End'), pick(kb.end || 'square', ends, set((v) => { kb.end = v; }))),
      el('div', { class: 'row' }, el('span', {}, 'Taper (m)'), num(kb.taper ?? 2, set((v) => { kb.taper = v ?? 2; })), el('span', { class: 'hint' }, 'how long a triangle end is'))) : null,
    el('div', { class: 'legend' }, KINDS.map(([v, label, what]) => el('div', { class: 'row' }, el('span', { class: 'swatch', style: 'background:linear-gradient(90deg,' + KERB_COL[v][0] + ' 33%,' + KERB_COL[v][1] + ' 33% 66%,' + (v === 'ramp_concrete' ? KERB_COL.concrete[0] : KERB_COL[v][0]) + ' 66%)' }), el('b', {}, label), ' ', what))),
    el('p', { class: 'hint' }, "A rumble lies outside the road's edge, going outward. Click a row to select it: its ends show as triangles on the map, to drag. Kept in roads.json; the build makes the kerbs from it once the kerbs are rebuilt."));
}
function panelValues() {
  // the node's numbers as it is dragged (without rebuilding the panel)
  const road = S.roads[S.sel.r], nd = road && road.curve && S.sel.n >= 0 ? road.curve[S.sel.n] : null;
  if (!nd) return;
  const inputs = $('nodeprops').querySelectorAll('input[type=number]');
  const vals = [nd.p[0], nd.p[1], nd.left, nd.right, Math.hypot(...nd.in), Math.hypot(...nd.out)];
  inputs.forEach((inp, i) => { if (i < vals.length && document.activeElement !== inp) inp.value = (+vals[i]).toFixed(i < 4 ? 2 : 1); });
}
function status(t) { $('state').textContent = t; }

// ------------------------------------------------------------------ start

window.addEventListener('resize', resize);
cv.tabIndex = 0;
cv.addEventListener('mousedown', onDown);
window.addEventListener('mousemove', (e) => { if (e.target === cv || S.drag) { const r = cv.getBoundingClientRect(); onMove({ offsetX: e.clientX - r.left, offsetY: e.clientY - r.top, shiftKey: e.shiftKey, altKey: e.altKey }); } });
window.addEventListener('mouseup', onUp);
cv.addEventListener('wheel', onWheel, { passive: false });
cv.addEventListener('contextmenu', (e) => e.preventDefault());
cv.addEventListener('dblclick', () => { if (S.newRoad) finishNewRoad(); });
window.addEventListener('keydown', onKey);
window.addEventListener('keyup', (e) => { if (e.key === ' ') S.space = false; });
window.addEventListener('beforeunload', (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ''; } });
document.querySelectorAll('#tools button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
document.querySelectorAll('#layers input').forEach((c) => c.addEventListener('change', () => { S.layers[c.dataset.layer] = c.checked; if (c.dataset.layer === 'wire') wantMesh(); redraw(); }));
$('undo').onclick = undo; $('redo').onclick = redo; $('save').onclick = save;
$('checksurf').onclick = checkSurface;
$('bring').onclick = openBring;
$('delroad').onclick = () => { if (S.sel.r < 0) { status('select a road to delete first'); return; } deleteRoad(); };
$('bringclose').onclick = () => { $('bringbox').hidden = true; };
$('bringfilter').oninput = bringList;
$('help').onclick = () => { $('helpbox').hidden = !$('helpbox').hidden; };
$('helpclose').onclick = () => { $('helpbox').hidden = true; };
$('filter').oninput = roadList;
resize();
setDirty(false);
load().catch((e) => { status('could not load: ' + e.message); console.error(e); });
