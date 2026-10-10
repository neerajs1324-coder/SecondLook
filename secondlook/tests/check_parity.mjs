// Proves the browser scorer matches the Python model on every test message.
// Run from the project root: node tests/check_parity.mjs

import { copyFileSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

// copy to .mjs so Node treats it as a module without needing a package.json in web/
const tmp = join(mkdtempSync(join(tmpdir(), "parity-")), "scorer.mjs");
copyFileSync(new URL("../web/scorer.js", import.meta.url), tmp);
const { createScorer } = await import(pathToFileURL(tmp));

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const scorer = createScorer(read("../web/rules.json"), read("../web/model.json"));
const fixtures = read("./fixtures.json");

let worst = 0, flagMismatches = 0;
for (const f of fixtures) {
  const r = scorer.score(f.text);
  r.probs.forEach((p, i) => { worst = Math.max(worst, Math.abs(p - f.probs[i])); });
  const flags = Object.keys(r.flags).filter((k) => r.flags[k]).sort().join(",");
  if (flags !== [...f.flags].sort().join(",")) {
    flagMismatches++;
    if (flagMismatches <= 5) console.log("flag mismatch:", f.text.slice(0, 80), "\n  js:", flags, "\n  py:", f.flags.join(","));
  }
}

console.log(`${fixtures.length} messages checked`);
console.log(`largest probability difference: ${worst.toExponential(2)}`);
console.log(`red flag mismatches: ${flagMismatches}`);
const ok = worst < 1e-4 && flagMismatches === 0;
console.log(ok ? "PASS  browser and Python agree" : "FAIL");
process.exit(ok ? 0 : 1);
