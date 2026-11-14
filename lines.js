const s = require("fs").readFileSync("lj-app.js", "utf8").split("\n");
require("fs").writeFileSync("lines.log",
  s.slice(85, 100).map((l, i) => (i + 86) + ": " + l).join("\n"));
