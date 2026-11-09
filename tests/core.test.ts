import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { classifyWithKeywords } from "../scripts/keywords";
import { LABELS } from "../scripts/labels";
import { score } from "../scripts/metrics";
import { LEAK_THRESHOLD, normalize, similarity, trigrams } from "../scripts/text";
import { predict, softmax, trainHead } from "../src/lib/head";

describe("linear head", () => {
  it("turns scores into probabilities that sum to 1, even for large values", () => {
    const p = softmax([1000, 1001, 999]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(p[1]).toBeGreaterThan(p[0]!);
  });

  it("learns to separate simple clusters", () => {
    const xs = [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]];
    const ys = [0, 0, 1, 1];
    const { weights, bias } = trainHead(xs, ys, 2, { l2: 0, epochs: 200, rate: 1 });
    const head = { embeddingModel: "", dtype: "", prefix: "", labels: ["a", "b"], weights, bias };
    expect(predict(head, [0.95, 0.05]).label).toBe("a");
    expect(predict(head, [0.05, 0.95]).label).toBe("b");
  });
});

describe("leak check", () => {
  it("treats accents, case and punctuation as the same text", () => {
    expect(normalize("Cadê o PEDIDO?!")).toBe("cade o pedido");
  });

  it("flags a reworded copy and passes a different message", () => {
    const test = trigrams("Meu pedido não chegou e o prazo venceu anteontem");
    expect(similarity(trigrams("meu pedido nao chegou e o prazo venceu ontem"), test)).toBeGreaterThanOrEqual(LEAK_THRESHOLD);
    expect(similarity(trigrams("Vocês aceitam Pix?"), test)).toBeLessThan(LEAK_THRESHOLD);
  });
});

describe("metrics", () => {
  it("scores accuracy, recall and invalid answers", () => {
    const s = score([
      { text: "a", label: "troca", predicted: "troca" },
      { text: "b", label: "troca", predicted: "atraso" },
      { text: "c", label: "atraso", predicted: "atraso" },
      { text: "d", label: "outro", predicted: "não sei" },
    ]);
    expect(s.accuracy).toBe(0.5);
    expect(s.invalid).toBe(1);
    expect(s.perLabel.troca.recall).toBe(0.5);
    expect(s.perLabel.atraso.precision).toBe(0.5);
    expect(s.confusion.outro!.invalido).toBe(1);
  });
});

describe("data", () => {
  const test = readFileSync("data/test.jsonl", "utf8").trim().split("\n").map((l) => JSON.parse(l) as { text: string; label: string });

  it("has only known labels in the test set", () => {
    for (const row of test) expect(LABELS).toContain(row.label);
  });

  it("keeps the keyword baseline from project 9", () => {
    expect(classifyWithKeywords("Meu pedido ainda não chegou")).toBe("atraso");
    expect(classifyWithKeywords("oi")).toBe("outro");
  });
});
