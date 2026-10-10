// Browser copy of python/features.py plus the model math. Kept line for line in sync
// so a message gets the same score here as it did during training (see tests/check_parity.mjs).

const LABELS = {
  __url__: "a link",
  __phone__: "a phone number",
  __money__: "a dollar amount",
  __num__: "a number",
  __email__: "an email address",
};

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export function createScorer(rules, model) {
  const weak = new Set(["me", "in", "at", "be", "co", "us", "gd", "gl", "cc", "tv", "ca", "au", "de", "fr", "eu", "mobi", "ly"]);
  const strongTlds = rules.tlds.filter((t) => !weak.has(t));
  const byLen = (a, b) => b.length - a.length;
  const shorteners = [...rules.shorteners].sort(byLen).map(esc).join("|");
  const tlds = [...strongTlds].sort(byLen).join("|");

  const EMAIL = /[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}/g;
  const URL = new RegExp(
    String.raw`https?://[^\s]+|www\.[^\s]+|\b(?:` + shorteners + String.raw`)(?:/[^\s]*)?|\b(?:[a-z0-9-]+\.)+(?:` + tlds + String.raw`)\b(?:/[^\s]*)?`,
    "g"
  );
  const MONEY = /(?:[£$€₹]|\brs\.?|\binr|\busd)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:dollars|pounds|usd|gbp)\b/g;
  const PHONE = /\+?\d(?:[\s().-]?\d){6,}/g;
  const NUM = /\d+/g;
  const TOKEN = /__[a-z]+__|[a-z]+/g;
  const IP = /^\d{1,3}(?:\.\d{1,3}){3}$/;
  const FLAG_RES = Object.entries(rules.flags).map(([name, p]) => [name, new RegExp(p)]);
  const filler = rules.brandFillers.join("|");
  const brandPatterns = Object.entries(rules.brands).map(([brand, official]) => [
    official,
    new RegExp("^(?:(?:" + filler + ")?" + brand + "(?:" + filler + ")?)$"),
  ]);

  const vocab = new Map(model.terms.map((t, i) => [t, i]));
  const nText = model.terms.length;
  const severe = new Set(rules.severe);

  const domainOf = (url) =>
    url.replace(/^https?:\/\//, "").split(/[/?#:]/)[0].replace(/^www\./, "");

  const brandLookalike = (domain) => {
    const labels = domain.split(/[.-]/);
    return brandPatterns.some(([official, rx]) => {
      if (official.some((o) => domain === o || domain.endsWith("." + o))) return false;
      return labels.some((l) => rx.test(l));
    });
  };

  function analyze(raw) {
    const text = raw.normalize("NFKC");
    const letters = text.match(/[A-Za-z]/g) || [];
    const upper = letters.filter((c) => c >= "A" && c <= "Z").length;
    const lower = text.toLowerCase();

    const flags = Object.fromEntries(model.flagNames.map((n) => [n, 0]));
    flags.shouting = letters.length >= 12 && upper / letters.length >= 0.4 ? 1 : 0;

    let s = lower.replace(EMAIL, " __email__ ");
    const links = [];
    for (const m of s.matchAll(URL)) {
      const url = m[0].replace(/[.,!?)'"]+$/, "");
      const d = domainOf(url);
      const link = { url, domain: d, problems: [] };
      flags.url_present = 1;
      if (rules.shorteners.includes(d)) { flags.url_shortener = 1; link.problems.push("url_shortener"); }
      if (rules.suspiciousTlds.includes(d.split(".").pop())) { flags.suspicious_tld = 1; link.problems.push("suspicious_tld"); }
      if (IP.test(d)) { flags.ip_url = 1; link.problems.push("ip_url"); }
      if (brandLookalike(d)) { flags.brand_lookalike = 1; link.problems.push("brand_lookalike"); }
      links.push(link);
    }
    s = s.replace(URL, " __url__ ");

    for (const [name, rx] of FLAG_RES) flags[name] = rx.test(s) ? 1 : 0;

    if (s.search(MONEY) >= 0) flags.has_money = 1;
    s = s.replace(MONEY, " __money__ ");
    if (s.search(PHONE) >= 0) flags.has_phone = 1;
    s = s.replace(PHONE, " __phone__ ");
    s = s.replace(NUM, " __num__ ");

    return { tokens: s.match(TOKEN) || [], flags, links };
  }

  function score(raw) {
    const { tokens, flags, links } = analyze(raw);
    const grams = [...tokens];
    for (let i = 0; i + 1 < tokens.length; i++) grams.push(tokens[i] + " " + tokens[i + 1]);

    const counts = new Map();
    for (const g of grams) if (vocab.has(g)) counts.set(g, (counts.get(g) || 0) + 1);
    const x = new Map();
    let norm = 0;
    for (const [g, c] of counts) {
      const j = vocab.get(g);
      const w = (1 + Math.log(c)) * model.idf[j];
      x.set(j, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm) || 1;
    for (const [j, w] of x) x.set(j, w / norm);
    model.flagNames.forEach((n, k) => { if (flags[n]) x.set(nText + k, flags[n] * model.flagScale); });

    const logits = model.weights.map((row, c) => {
      let z = model.intercept[c];
      for (const [j, v] of x) z += row[j] * v;
      return z;
    });
    const top = Math.max(...logits);
    const exps = logits.map((z) => Math.exp(z - top));
    const sum = exps.reduce((a, b) => a + b, 0);
    const probs = exps.map((e) => e / sum);

    // how much each word pushed the message toward "scam" and away from "legit"
    const words = [];
    for (const [j, v] of x) {
      if (j >= nText) continue;
      const push = v * (model.weights[2][j] - model.weights[0][j]);
      if (push > 0.15) words.push({ term: model.terms[j], push });
    }
    words.sort((a, b) => b.push - a.push);

    const pScam = probs[2], pSpam = probs[1];
    const t = model.thresholds;
    const hasSevere = [...severe].some((f) => flags[f]);
    let verdict = "okay";
    if (pScam >= t.scam) verdict = "scam";
    else if (hasSevere || pScam >= t.careful) verdict = "careful";
    else if (pSpam >= t.spam) verdict = "spam";

    return { probs, verdict, flags, links, words, tokens };
  }

  const describe = (term) =>
    term.split(" ").map((p) => LABELS[p] || p).join(" ");

  return { analyze, score, describe, explanations: rules.explanations, severe };
}
