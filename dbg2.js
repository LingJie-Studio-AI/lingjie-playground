globalThis.window = globalThis;
require("./lj-tokenizer.js");
const fs = require("fs");
const tok = new LJVTokenizer(fs.readFileSync("tokenizer.json", "utf-8"));
console.log("ranks has 'ä ¸':", tok.ranks.get("ä ¸"));
console.log("ranks has 'ä ½':", tok.ranks.get("ä ½"));
// 找出包含 ½ 的前 5 个 merge
let n = 0;
for (const m of tok.merges) {
  if (m[0] === "ä" || m[1] === "½") { console.log("merge:", JSON.stringify(m)); if (++n > 5) break; }
}
// python 参考的 3214 对应什么 token
console.log("tok 3214:", JSON.stringify(tok.idToTok[3214]));
console.log("tok 266:", JSON.stringify(tok.idToTok[266]));
console.log("tok 166:", JSON.stringify(tok.idToTok[166]));
