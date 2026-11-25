// The README's renders (docs/renders): the track as the Top Race Editor renders it (its src/engine-window/render.ts, the AC build's models,
// sun, shadows, sky and grass), from the Assetto Corsa build in build/ac: the whole circuit from above, the 1500's and the 1000's start gantries, and the kids
// track. Run by render.mjs, in Electron.
import { resolve } from "node:path";
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { readAiFile } from "../../../top-race-editor/src/engine/acai";
import { layoutViews, overview, type Shot, trackScene, type V3, type View } from "../../../top-race-editor/src/engine/trackrender";
import { render } from "../../../top-race-editor/src/engine-window/render";

const ROOT = resolve(__dirname, "../..");
const AC = `${ROOT}/build/ac/content/tracks/techspeed_fueltech_velopark`;
const OUT = `${ROOT}/docs/renders`;
const SIZE: [number, number] = [1920, 1080];
const files = {
	read: async (p: string) => new Uint8Array(readFileSync(p)),
	list: async (d: string) => readdirSync(d),
	size: async (p: string) => statSync(p).size,
	exists: async (p: string) => existsSync(p),
	write: async (p: string, b: Uint8Array) => writeFileSync(p, b),
};
const lineOf = (l: string): V3[] => {
	const ai = readAiFile(new Uint8Array(readFileSync(`${AC}/${l}/ai/fast_lane.ai`)));
	const out: V3[] = [];
	for (let i = 0; i + 2 < ai.points.length; i += 3) out.push([ai.points[i] as number, ai.points[i + 1] as number, ai.points[i + 2] as number]);
	return out;
};

async function main() {
	const layouts = ["vp_2500", "vp_1500", "vp_1000", "vp_kids"];
	const { scene, drawn } = await trackScene(files as never, AC, layouts);
	const lines = new Map(layouts.map((l) => [l, lineOf(l)]));
	const gantry = layoutViews(scene, drawn, lines, SIZE);



	const shots: [string, View, string][] = [
		["1-overview", overview(lines.get("vp_2500") as V3[], { size: SIZE, pitch: 38, margin: 0.04 }), "vp_2500"],
		["2-gantry-1500", gantry.get("vp_1500") as View, "vp_1500"],
		["3-gantry-1000", gantry.get("vp_1000") as View, "vp_1000"],
		["4-kids", overview(lines.get("vp_kids") as V3[], { size: SIZE, pitch: 40, margin: 0.12 }), "vp_kids"],
	];
	const made = await render(
		scene,
		shots.map(([, v, l]): Shot => ({ ...v, models: drawn[l] as string[], size: SIZE, format: "jpeg" })),
	);
	made.forEach((b, i) => writeFileSync(`${OUT}/${shots[i]?.[0]}.jpg`, b));
	console.log(`rendered ${made.length}`);
}

main().then(
	() => require("electron").ipcRenderer.send("done", 0),
	(e) => {
		console.log(`failed: ${e?.stack ?? e}`);
		require("electron").ipcRenderer.send("done", 1);
	},
);
