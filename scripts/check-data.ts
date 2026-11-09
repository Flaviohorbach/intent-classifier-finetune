// Cleans data/generated.jsonl into data/train.jsonl and data/val.jsonl:
// drops duplicates and anything too close to a test message, then splits by label.
import { writeFileSync } from "node:fs";
import { LABELS, readJsonl, type Example } from "./labels";
import { LEAK_THRESHOLD, normalize, similarity, trigrams } from "./text";

const generated = readJsonl<Example & { batch: string }>("data/generated.jsonl");
const test = readJsonl("data/test.jsonl").map((t) => ({ ...t, grams: trigrams(t.text) }));

const seen = new Set<string>();
const kept: Example[] = [];
const leaks: { train: string; test: string; score: number }[] = [];
let duplicates = 0;

for (const row of generated) {
  const key = normalize(row.text);
  if (seen.has(key)) {
    duplicates++;
    continue;
  }
  seen.add(key);
  const grams = trigrams(row.text);
  let closest = { score: 0, text: "" };
  for (const t of test) {
    const score = similarity(grams, t.grams);
    if (score > closest.score) closest = { score, text: t.text };
  }
  if (closest.score >= LEAK_THRESHOLD) {
    leaks.push({ train: row.text, test: closest.text, score: Number(closest.score.toFixed(2)) });
    continue;
  }
  kept.push({ text: row.text, label: row.label });
}

// Stable split: every 10th message of each label goes to validation.
const train: Example[] = [];
const val: Example[] = [];
for (const label of LABELS) kept.filter((r) => r.label === label).forEach((r, i) => (i % 10 === 9 ? val : train).push(r));

const toJsonl = (rows: Example[]) => rows.map((r) => JSON.stringify(r)).join("\n") + "\n";
writeFileSync("data/train.jsonl", toJsonl(train));
writeFileSync("data/val.jsonl", toJsonl(val));

const count = (rows: Example[]) => Object.fromEntries(LABELS.map((l) => [l, rows.filter((r) => r.label === l).length]));
const report = {
  generated: generated.length,
  duplicates,
  tooCloseToTest: leaks.length,
  train: { total: train.length, ...count(train) },
  val: { total: val.length, ...count(val) },
  averageWords: Number((kept.reduce((n, r) => n + r.text.split(/\s+/).length, 0) / kept.length).toFixed(1)),
  closestLeaks: leaks.sort((a, b) => b.score - a.score).slice(0, 20),
};
writeFileSync("results/data-report.json", JSON.stringify(report, null, 1) + "\n");
console.log(JSON.stringify({ ...report, closestLeaks: report.closestLeaks.slice(0, 5) }, null, 1));
