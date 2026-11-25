// Renders the README's pictures (docs/renders) with the Top Race Editor beside this repository (../top-race-editor, its packages installed):
// node tools/readme_renders/render.mjs, after a build (build/ac). Bundles entry.ts with the editor's rolldown, then runs it in the editor's Electron.
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const editor = resolve(here, "../../../top-race-editor");
const { build } = await import(pathToFileURL(join(editor, "node_modules/rolldown/dist/index.mjs")).href);
await build({ cwd: here, input: join(here, "entry.ts"), platform: "node", external: ["electron", /^node:/], output: { file: join(here, "bundle.cjs"), format: "cjs" } });
const electron = createRequire(join(editor, "package.json"))("electron");
// (ELECTRON_RUN_AS_NODE, as VS Code's terminals set it, would run main.cjs as plain Node)
const { ELECTRON_RUN_AS_NODE: _, ...env } = process.env;
const run = spawnSync(electron, [join(here, "main.cjs")], { stdio: "inherit", env });
process.exit(run.status ?? 1);
