// Builds data/test.jsonl: the 30 cases from project 9 plus the hand-written ones.
// Written before any training data existed, so nothing in it was shaped by the generator.
import { readFileSync, writeFileSync } from "node:fs";
import { LABELS, type Example } from "./labels";

const fromProject9 = JSON.parse(readFileSync("data/project9-cases.json", "utf8")) as { id: string; text: string; category: string }[];
const handwritten = JSON.parse(readFileSync("data/test-handwritten.json", "utf8")) as { text: string; label: string }[];

const rows: Example[] = [
  ...fromProject9.map((c) => ({ text: c.text, label: c.category, source: "project9" })),
  ...handwritten.map((c) => ({ text: c.text, label: c.label, source: "handwritten" })),
] as Example[];

for (const row of rows) if (!LABELS.includes(row.label)) throw new Error(`Unknown label: ${row.label}`);
const texts = new Set(rows.map((r) => r.text));
if (texts.size !== rows.length) throw new Error("Duplicate test messages");

writeFileSync("data/test.jsonl", rows.map((r) => JSON.stringify(r)).join("\n") + "\n");
console.log(`${rows.length} test messages`);
