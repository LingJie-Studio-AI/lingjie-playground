const fs = require("fs");
const s = fs.readFileSync("lj-gpu.js", "utf8");
let depth = 0, line = 1, inStr = null;
for (let i = 0; i < s.length; i++) {
  const c = s[i], p = s[i - 1];
  if (c === "\n") line++;
  if (inStr) { if (c === inStr && p !== "\\") inStr = null; continue; }
  if (c === '"' || c === "'" || c === "`") { inStr = c; continue; }
  if (c === "/" && s[i + 1] === "*") { inStr = "*/"; continue; }
  if (inStr === "*/" && c === "*" && s[i + 1] === "/") { inStr = null; continue; }
  if (c === "/" && s[i + 1] === "/") { inStr = "//"; continue; }
  if (inStr === "//" && c === "\n") { inStr = null; continue; }
  if (c === "{") depth++;
  if (c === "}") { depth--; if (depth < 0) { console.log("extra } at line", line); process.exit(0); } }
}
console.log("final depth", depth, "last line", line);
