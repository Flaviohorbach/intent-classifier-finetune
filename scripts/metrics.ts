import { LABELS, type Label } from "./labels";

export interface Prediction {
  text: string;
  label: string;
  predicted: string;
}

export interface Scores {
  n: number;
  accuracy: number;
  macroF1: number;
  invalid: number;
  perLabel: Record<Label, { precision: number; recall: number; f1: number; support: number }>;
  /** rows: true label, columns: predicted label (plus "invalido" when a model answered something else). */
  confusion: Record<string, Record<string, number>>;
  errors: Prediction[];
}

const round = (v: number) => Math.round(v * 1000) / 1000;

export function score(predictions: Prediction[]): Scores {
  const columns = [...LABELS, "invalido"];
  const confusion = Object.fromEntries(LABELS.map((l) => [l, Object.fromEntries(columns.map((c) => [c, 0]))])) as Scores["confusion"];
  for (const p of predictions) {
    const column = (LABELS as readonly string[]).includes(p.predicted) ? p.predicted : "invalido";
    confusion[p.label]![column]!++;
  }
  const perLabel = Object.fromEntries(
    LABELS.map((l) => {
      const tp = confusion[l]![l]!;
      const support = Object.values(confusion[l]!).reduce((a, b) => a + b, 0);
      const predictedAs = LABELS.reduce((n, row) => n + confusion[row]![l]!, 0);
      const precision = predictedAs ? tp / predictedAs : 0;
      const recall = support ? tp / support : 0;
      const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
      return [l, { precision: round(precision), recall: round(recall), f1: round(f1), support }];
    }),
  ) as Scores["perLabel"];
  const right = predictions.filter((p) => p.predicted === p.label).length;
  return {
    n: predictions.length,
    accuracy: round(right / predictions.length),
    macroF1: round(LABELS.reduce((s, l) => s + perLabel[l].f1, 0) / LABELS.length),
    invalid: predictions.filter((p) => !(LABELS as readonly string[]).includes(p.predicted)).length,
    perLabel,
    confusion,
    errors: predictions.filter((p) => p.predicted !== p.label),
  };
}
