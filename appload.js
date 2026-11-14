/* 在 node 里用最小 DOM 垫片加载 lj-app.js，抓加载期运行时错误。 */
function stubEl() {
  return {
    classList: { toggle() {}, add() {}, remove() {}, contains: () => false },
    style: {}, dataset: {},
    addEventListener() {}, removeEventListener() {},
    appendChild() {}, setAttribute() {}, getAttribute: () => null,
    innerHTML: "", textContent: "", value: "", disabled: false,
    children: [], querySelector: () => null, querySelectorAll: () => [],
    focus() {}, click() {},
  };
}
global.window = globalThis;
global.navigator = { gpu: null };
global.document = {
  querySelector: () => stubEl(),
  getElementById: () => stubEl(),
  addEventListener() {},
  documentElement: stubEl(),
  body: stubEl(),
  createElement: () => stubEl(),
};
global.localStorage = { getItem: () => null, setItem() {} };
global.location = { hash: "" };
global.fetch = () => new Promise(() => {});
global.performance = require("perf_hooks").performance;
global.requestAnimationFrame = (f) => setTimeout(f, 0);
global.matchMedia = () => ({ matches: false, addEventListener() {} });
global.URL = { createObjectURL: () => "", revokeObjectURL() {} };

window.matchMedia = global.matchMedia;
window.addEventListener = () => {};
window.document = global.document;

require("./lj-tokenizer.js");
require("./lj-cpu.js");
require("./lj-gpu.js");
try {
  require("./lj-app.js");
  console.log("lj-app loaded OK");
} catch (e) {
  console.log("LOAD ERROR:", e.message);
  console.log(e.stack.split("\n").slice(0, 6).join("\n"));
}
