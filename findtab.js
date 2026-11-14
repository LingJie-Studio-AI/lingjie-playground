const browserPluginRoot = process.env.ZCODE_PLUGIN_ROOT ?? process.env.CLAUDE_PLUGIN_ROOT;
const { join } = await import("node:path");
const { pathToFileURL } = await import("node:url");
const { setupBrowserRuntime } = await import(pathToFileURL(join(browserPluginRoot, "scripts", "browser-client.mjs")).href);
await setupBrowserRuntime({ globals: globalThis });
const browser = await agent.browsers.getForUrl("http://127.0.0.1:8901/index.html");
const tabs = await browser.tabs.list();
let target = null;
for (let i = tabs.length - 1; i >= 0; i--) {
  const t = await browser.tabs.get(tabs[i].id);
  const has = await t.playwright.evaluate(`!!window.__ljdbg`).catch(() => false);
  if (has) { target = {id: tabs[i].id, url: tabs[i].url}; break; }
}
require("fs").writeFileSync("tabfound.log", JSON.stringify(target));
