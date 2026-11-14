const s = require("fs").readFileSync("lj-gpu.js", "utf8");
const i = s.indexOf("scratch(key, bytes)");
require("fs").writeFileSync("sm.log", s.slice(i - 60, i + 750));
