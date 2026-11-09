/// <reference lib="webworker" />
// Runs the model off the main thread, so the page keeps responding while it
// downloads 118 MB and computes embeddings.
import { env, pipeline, type FeatureExtractionPipeline } from "@huggingface/transformers";
import { predict, type Head } from "./lib/head";

export type WorkerRequest = { type: "load"; headUrl: string } | { type: "classify"; id: number; text: string };
export type WorkerResponse =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; loadMs: number }
  | { type: "result"; id: number; label: string; probabilities: { label: string; p: number }[]; ms: number }
  | { type: "error"; message: string };

env.allowLocalModels = false;

let extractor: FeatureExtractionPipeline | null = null;
let head: Head | null = null;
const post = (message: WorkerResponse) => self.postMessage(message);

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      const start = performance.now();
      head = (await (await fetch(message.headUrl)).json()) as Head;
      const files = new Map<string, { loaded: number; total: number }>();
      extractor = await pipeline("feature-extraction", head.embeddingModel, {
        dtype: head.dtype as "q8",
        progress_callback: (p: { status: string; file?: string; loaded?: number; total?: number }) => {
          if (p.status !== "progress" || !p.file) return;
          files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
          const all = [...files.values()];
          post({ type: "progress", loaded: all.reduce((n, f) => n + f.loaded, 0), total: all.reduce((n, f) => n + f.total, 0) });
        },
      });
      post({ type: "ready", loadMs: Math.round(performance.now() - start) });
    } else if (message.type === "classify") {
      if (!extractor || !head) throw new Error("O modelo ainda não carregou.");
      const start = performance.now();
      const output = await extractor(head.prefix + message.text, { pooling: "mean", normalize: true });
      const result = predict(head, output.data as Float32Array);
      post({ type: "result", id: message.id, ...result, ms: Math.round(performance.now() - start) });
    }
  } catch (error) {
    post({ type: "error", message: (error as Error).message });
  }
};
