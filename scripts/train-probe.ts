// Trains the browser model: frozen multilingual-e5-small embeddings plus a
// linear layer. Embeds with Transformers.js and the same quantized model the
// page downloads, so the numbers here are the numbers visitors get.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { pipeline } from "@huggingface/transformers";
import { predict, trainHead, type Head } from "../src/lib/head";
import { LABELS, readJsonl, type Example } from "./labels";

const MODEL = "Xenova/multilingual-e5-small";
const DTYPE = "q8";
const PREFIX = "query: ";
const CACHE = ".cache/embeddings.json";

const train = readJsonl("data/train.jsonl");
const val = readJsonl("data/val.jsonl");
const test = readJsonl("data/test.jsonl");

const embed = await pipeline("feature-extraction", MODEL, { dtype: DTYPE });
const cache: Record<string, number[]> = existsSync(CACHE) ? JSON.parse(readFileSync(CACHE, "utf8")) : {};

async function vectors(rows: Example[]): Promise<number[][]> {
  const missing = [...new Set(rows.map((r) => r.text).filter((t) => !cache[t]))];
  for (let i = 0; i < missing.length; i += 32) {
    const chunk = missing.slice(i, i + 32);
    const output = await embed(chunk.map((t) => PREFIX + t), { pooling: "mean", normalize: true });
    const list = output.tolist() as number[][];
    chunk.forEach((t, j) => (cache[t] = list[j]!));
    process.stdout.write(`\rembedded ${Math.min(i + 32, missing.length)}/${missing.length}`);
  }
  if (missing.length) process.stdout.write("\n");
  mkdirSync(".cache", { recursive: true });
  writeFileSync(CACHE, JSON.stringify(cache));
  return rows.map((r) => cache[r.text]!);
}

const [xTrain, xVal, xTest] = [await vectors(train), await vectors(val), await vectors(test)];
const yTrain = train.map((r) => LABELS.indexOf(r.label));
const accuracy = (head: Head, xs: number[][], rows: Example[]) => xs.filter((x, i) => predict(head, x).label === rows[i]!.label).length / rows.length;

// Pick the regularization on the validation set only. The test set is not looked at until the end.
let best: { head: Head; l2: number; val: number } | null = null;
for (const l2 of [0, 1e-4, 1e-3, 1e-2]) {
  const { weights, bias } = trainHead(xTrain, yTrain, LABELS.length, { l2, epochs: 300, rate: 2 });
  const head: Head = { embeddingModel: MODEL, dtype: DTYPE, prefix: PREFIX, labels: [...LABELS], weights, bias };
  const score = accuracy(head, xVal, val);
  console.log(`l2=${l2}: validation accuracy ${(score * 100).toFixed(1)}%`);
  if (!best || score > best.val) best = { head, l2, val: score };
}

const round = (v: number) => Math.round(v * 1e5) / 1e5;
const head = { ...best!.head, weights: best!.head.weights.map((r) => r.map(round)), bias: best!.head.bias.map(round) };
writeFileSync("public/model/head.json", JSON.stringify(head));

const predictions = test.map((row, i) => ({ text: row.text, label: row.label, predicted: predict(head, xTest[i]!).label }));
writeFileSync("results/probe-predictions.json", JSON.stringify({ model: `${MODEL} (${DTYPE}) + linear layer, l2=${best!.l2}`, predictions }, null, 1) + "\n");
console.log(`chosen l2=${best!.l2}, validation ${(best!.val * 100).toFixed(1)}%. Test predictions saved; run npm run eval for the scores.`);
