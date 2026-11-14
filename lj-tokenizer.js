/* LingJie 字节级 BPE 分词器（从 tokenizer.json 加载，纯 JS，无依赖）。 */
(function (root) {
  const BYTES_TO_UNICODE = (() => {
    const bs = [];
    for (let i = 0; i < 256; i++) bs.push(i);
    let cs = [];
    for (let b = 33; b <= 126; b++) cs.push(b);
    for (let b = 161; b <= 172; b++) cs.push(b);
    for (let b = 174; b <= 255; b++) cs.push(b);
    let n = 0;
    const out = [];
    for (const b of bs) {
      if (cs.includes(b)) { out[b] = b; }
      else { out[b] = 256 + n; n++; }
    }
    const m = new Map();
    for (let b = 0; b < 256; b++) m.set(b, String.fromCharCode(out[b]));
    return m;
  })();
  const UNICODE_TO_BYTES = (() => {
    const m = new Map();
    for (const [b, c] of BYTES_TO_UNICODE) m.set(c, b);
    return m;
  })();
  const PRETOK = /(?:'s|'t|'re|'ve|'m|'ll|'d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const SPECIALS = ["<|pad|>", "<|end|>", "<|user|>", "<|assistant|>", "<|image|>"];

  class LJVTokenizer {
    constructor(tokenizerJson) {
      const j = typeof tokenizerJson === "string"
        ? JSON.parse(tokenizerJson) : tokenizerJson;
      this.vocab = j.model.vocab;                 // str(byte-unicode) -> id
      this.merges = j.model.merges;               // ["a b", ...]
      this.ranks = new Map();
      for (let i = 0; i < this.merges.length; i++) {
        const m = this.merges[i];
        const pair = Array.isArray(m) ? m[0] + " " + m[1] : m;
        this.ranks.set(pair, i);
      }
      this.idToTok = new Array(j.model.vocab.length || 0);
      for (const [t, id] of Object.entries(this.vocab)) this.idToTok[id] = t;
      if (j.added_tokens) {
        for (const at of j.added_tokens) {
          this.vocab[at.content] = at.id;
          this.idToTok[at.id] = at.content;
        }
      }
      this.specialIds = new Set();
      for (const s of SPECIALS) {
        if (this.vocab[s] !== undefined) this.specialIds.add(this.vocab[s]);
      }
      this.byteDec = new TextDecoder("utf-8");
    }

    id(tok) { return this.vocab[tok]; }
    token(id) { return this.idToTok[id]; }

    _byteEncode(s) {
      const out = [];
      const bytes = new TextEncoder().encode(s);
      for (const b of bytes) out.push(BYTES_TO_UNICODE.get(b));
      return out.join("");
    }

    _bpe(piece) {
      if (piece.length <= 1) return [piece];
      let parts = piece.split("");
      while (true) {
        let bestRank = Infinity, bestIdx = -1;
        for (let i = 0; i < parts.length - 1; i++) {
          const key = parts[i] + " " + parts[i + 1];
          const r = this.ranks.get(key);
          if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
        }
        if (bestIdx < 0) break;
        parts.splice(bestIdx, 2, parts[bestIdx] + parts[bestIdx + 1]);
      }
      return parts;
    }

    encodePiece(s) {
      const enc = this._byteEncode(s);
      const out = [];
      for (const t of this._bpe(enc)) {
        const id = this.vocab[t];
        if (id !== undefined) out.push(id);
      }
      return out;
    }

    encode(text) {
      // 先按特殊 token 切分
      const out = [];
      let rest = text;
      const re = /<\|pad\|>|<\|end\|>|<\|user\|>|<\|assistant\|>|<\|image\|>/g;
      let last = 0, m;
      while ((m = re.exec(rest)) !== null) {
        if (m.index > last) this._encodePlain(rest.slice(last, m.index), out);
        out.push(this.vocab[m[0]]);
        last = m.index + m[0].length;
      }
      if (last < rest.length) this._encodePlain(rest.slice(last), out);
      return out;
    }

    _encodePlain(s, out) {
      for (const piece of s.match(PRETOK) || []) {
        for (const id of this.encodePiece(piece)) out.push(id);
      }
    }

    decode(ids) {
      let s = "";
      for (const id of ids) {
        const t = this.idToTok[id];
        if (t === undefined) continue;
        s += t;
      }
      const bytes = new Uint8Array(s.length);
      for (let i = 0; i < s.length; i++) bytes[i] = UNICODE_TO_BYTES.get(s[i]) || 0;
      return this.byteDec.decode(bytes);
    }
  }

  root.LJVTokenizer = LJVTokenizer;
})(typeof window !== "undefined" ? window : globalThis);
