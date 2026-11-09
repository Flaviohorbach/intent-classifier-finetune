/**
 * The trained part of the browser model: a linear layer on top of frozen
 * sentence embeddings (a "linear probe"). Used by the training script and by
 * the page, so both compute exactly the same thing.
 */
export interface Head {
  embeddingModel: string;
  dtype: string;
  /** multilingual-e5 expects "query: " before every input. */
  prefix: string;
  labels: string[];
  /** One row of weights per label. */
  weights: number[][];
  bias: number[];
}

export function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

export function logits(head: Pick<Head, "weights" | "bias">, embedding: ArrayLike<number>): number[] {
  return head.weights.map((row, k) => {
    let sum = head.bias[k]!;
    for (let i = 0; i < row.length; i++) sum += row[i]! * embedding[i]!;
    return sum;
  });
}

export function predict(head: Head, embedding: ArrayLike<number>): { label: string; probabilities: { label: string; p: number }[] } {
  const p = softmax(logits(head, embedding));
  const probabilities = head.labels.map((label, k) => ({ label, p: p[k]! })).sort((a, b) => b.p - a.p);
  return { label: probabilities[0]!.label, probabilities };
}

/**
 * Multinomial logistic regression with L2, by full-batch gradient descent.
 * A few thousand 384-dimension vectors train in seconds, with no library.
 */
export function trainHead(xs: ArrayLike<number>[], ys: number[], classes: number, options: { l2: number; epochs: number; rate: number }) {
  const dim = xs[0]!.length;
  const weights = Array.from({ length: classes }, () => new Array<number>(dim).fill(0));
  const bias = new Array<number>(classes).fill(0);
  const n = xs.length;
  for (let epoch = 0; epoch < options.epochs; epoch++) {
    const gw = Array.from({ length: classes }, () => new Array<number>(dim).fill(0));
    const gb = new Array<number>(classes).fill(0);
    for (let j = 0; j < n; j++) {
      const x = xs[j]!;
      const p = softmax(logits({ weights, bias }, x));
      for (let k = 0; k < classes; k++) {
        const err = p[k]! - (ys[j] === k ? 1 : 0);
        gb[k]! += err;
        const row = gw[k]!;
        for (let i = 0; i < dim; i++) row[i]! += err * x[i]!;
      }
    }
    for (let k = 0; k < classes; k++) {
      bias[k]! -= (options.rate * gb[k]!) / n;
      const row = weights[k]!;
      const g = gw[k]!;
      for (let i = 0; i < dim; i++) row[i]! -= options.rate * (g[i]! / n + options.l2 * row[i]!);
    }
  }
  return { weights, bias };
}
