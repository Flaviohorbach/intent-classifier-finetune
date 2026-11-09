import { readFileSync } from "node:fs";

export const LABELS = ["troca", "atraso", "duvida", "reclamacao_grave", "outro"] as const;
export type Label = (typeof LABELS)[number];

export interface Example {
  text: string;
  label: Label;
  source?: string;
}

export const readJsonl = <T = Example>(path: string): T[] =>
  readFileSync(path, "utf8")
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line) as T);

/** The labeling guide, shared by the data generator and every prompt-based classifier. */
export const LABEL_GUIDE = readFileSync(new URL("../data/LABELS.md", import.meta.url), "utf8");
