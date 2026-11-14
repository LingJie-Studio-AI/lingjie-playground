const fs = require("fs");
const s = fs.readFileSync("lj-app.js", "utf8");
const i = s.indexOf("async function initBackend");
fs.writeFileSync("initb.log", s.slice(i - 200, i + 500));
const j = s.indexOf("new LJVGPU");
fs.writeFileSync("initb2.log", "new LJVGPU @ " + j + "\n" + s.slice(j - 120, j + 80));
