const s = require("fs").readFileSync("lj-gpu.js", "utf8").split("\n");
const out = [];
for (let i = 655; i < Math.min(684, s.length); i++) out.push((i + 1) + ": " + s[i]);
require("fs").writeFileSync("tail.log", out.join("\n"));
