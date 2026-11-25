// The renders' window: hidden, the bundle run in it with Node's require (its WebGL the GPU's), the app ended with its exit code.
const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("node:path");
app.whenReady().then(() => {
	const w = new BrowserWindow({ show: false, width: 640, height: 480, webPreferences: { nodeIntegration: true, contextIsolation: false, sandbox: false, backgroundThrottling: false } });
	w.webContents.on("console-message", (...a) => console.log(a[0]?.message ?? a[2]));
	ipcMain.on("done", (_e, code) => app.exit(code));
	w.loadFile(path.join(__dirname, "index.html"));
});
