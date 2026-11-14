const s = require("fs").readFileSync("lj-gpu.js", "utf8");
const i = s.indexOf("gemm(aBuf");
require("fs").writeFileSync("gemm.log",
  "f16 in shader: " + (s.includes("array<f16>") ? "YES" : "NO") + "\n" +
  "enable f16: " + s.split("enable f16").length - 1 + " 处\n" +
  s.slice(i - 60, i + 900));
