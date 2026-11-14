const s = require("fs").readFileSync("lj-gpu.js", "utf8").split("\n");
const out = [];
s.forEach(function (l, i) {
  if (l.indexOf("xb") >= 0) out.push((i + 1) + ": " + l.trim().slice(0, 110));
});
require("fs").writeFileSync("xbr.log", out.join("\n"));
