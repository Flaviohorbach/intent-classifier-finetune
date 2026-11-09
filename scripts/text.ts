/** Lowercase, no accents, no punctuation: "Cadê o pedido?!" and "cade o pedido" are the same message. */
export const normalize = (text: string) =>
  text.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9 ]+/g, " ").replace(/\s+/g, " ").trim();

export function trigrams(text: string): Set<string> {
  const t = ` ${normalize(text)} `;
  const set = new Set<string>();
  for (let i = 0; i < t.length - 2; i++) set.add(t.slice(i, i + 3));
  return set;
}

/** Jaccard similarity of character trigrams: 1 for the same text, near 0 for unrelated ones. */
export function similarity(a: Set<string>, b: Set<string>): number {
  let shared = 0;
  for (const g of a) if (b.has(g)) shared++;
  return shared / (a.size + b.size - shared || 1);
}

/** Above this, a training message counts as a copy of a test message. */
export const LEAK_THRESHOLD = 0.6;
