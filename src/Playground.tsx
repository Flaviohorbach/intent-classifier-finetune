import { useEffect, useRef, useState } from "react";
import { LABEL_NAMES } from "./lib/label-names";
import type { WorkerRequest, WorkerResponse } from "./worker";

const EXAMPLES = [
  "vcs entregam no sábado?",
  "o fone só funciona de um lado",
  "Previsão era dia 3, hoje é dia 10. Alguma novidade?",
  "Fui cobrado duas vezes e vou no Procon",
  "Vocês estão contratando?",
];

type State =
  | { status: "idle" }
  | { status: "loading"; loaded: number; total: number }
  | { status: "ready"; loadMs: number }
  | { status: "error"; message: string };

interface Result {
  text: string;
  label: string;
  probabilities: { label: string; p: number }[];
  ms: number;
}

const mb = (bytes: number) => `${(bytes / 1e6).toLocaleString("pt-BR", { maximumFractionDigits: 0 })} MB`;

export function Playground() {
  const worker = useRef<Worker | null>(null);
  const nextId = useRef(0);
  const [state, setState] = useState<State>({ status: "idle" });
  const [text, setText] = useState(EXAMPLES[0]!);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const pending = useRef(new Map<number, string>());

  useEffect(() => () => worker.current?.terminate(), []);

  function load() {
    const w = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    worker.current = w;
    w.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const m = event.data;
      if (m.type === "progress") setState({ status: "loading", loaded: m.loaded, total: m.total });
      else if (m.type === "ready") setState({ status: "ready", loadMs: m.loadMs });
      else if (m.type === "error") {
        setBusy(false);
        setState({ status: "error", message: m.message });
      } else if (m.type === "result") {
        const sent = pending.current.get(m.id) ?? "";
        pending.current.delete(m.id);
        setBusy(false);
        setResult({ text: sent, label: m.label, probabilities: m.probabilities, ms: m.ms });
      }
    };
    setState({ status: "loading", loaded: 0, total: 0 });
    w.postMessage({ type: "load", headUrl: `${import.meta.env.BASE_URL}model/head.json` } satisfies WorkerRequest);
  }

  function classify(value: string) {
    if (!worker.current || state.status !== "ready" || value.trim().length < 2) return;
    const id = nextId.current++;
    pending.current.set(id, value);
    setBusy(true);
    worker.current.postMessage({ type: "classify", id, text: value.trim() } satisfies WorkerRequest);
  }

  return (
    <section aria-labelledby="try-title" className="mt-8 rounded-xl border border-line bg-surface p-4 sm:p-5">
      <h2 id="try-title" className="font-display text-xl font-semibold">
        Teste no seu navegador
      </h2>

      {state.status === "idle" && (
        <div className="mt-3">
          <p className="text-sm text-muted">O modelo tem 118 MB e é baixado uma vez, quando você pedir. Depois fica no cache do navegador.</p>
          <button type="button" onClick={load} className="mt-3 rounded-lg bg-accent px-4 py-2 font-medium text-accent-ink">
            Baixar o modelo (118 MB)
          </button>
        </div>
      )}

      {state.status === "loading" && (
        <div className="mt-3" role="status">
          <p className="text-sm text-muted">{state.total ? `Baixando: ${mb(state.loaded)} de ${mb(state.total)}` : "Preparando…"}</p>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-surface-2">
            <div className="h-full bg-accent transition-[width]" style={{ width: `${state.total ? (state.loaded / state.total) * 100 : 2}%` }} />
          </div>
        </div>
      )}

      {state.status === "error" && (
        <p role="alert" className="mt-3 text-sm text-danger">
          Não foi possível usar o modelo neste navegador: {state.message}
        </p>
      )}

      {state.status === "ready" && (
        <div className="mt-3">
          <p className="text-xs text-muted">Pronto em {(state.loadMs / 1000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} s.</p>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {EXAMPLES.map((e) => (
              <button
                key={e}
                type="button"
                onClick={() => {
                  setText(e);
                  classify(e);
                }}
                className="rounded-full border border-line px-3 py-1 text-xs hover:border-accent"
              >
                {e}
              </button>
            ))}
          </div>
          <form
            className="mt-3 flex flex-col gap-2 sm:flex-row"
            onSubmit={(event) => {
              event.preventDefault();
              classify(text);
            }}
          >
            <label htmlFor="message" className="sr-only">
              Mensagem de cliente
            </label>
            <input id="message" value={text} onChange={(e) => setText(e.target.value)} maxLength={500} className="min-w-0 flex-1 rounded-lg border border-line bg-surface p-2" />
            <button type="submit" disabled={busy} className="rounded-lg bg-accent px-4 py-2 font-medium text-accent-ink disabled:opacity-60">
              Classificar
            </button>
          </form>

          {result && (
            <div className="mt-4" aria-live="polite">
              <p>
                <span className="font-display text-2xl font-semibold">{LABEL_NAMES[result.label]}</span>
                <span className="ml-2 text-sm text-muted">em {result.ms} ms, no seu aparelho</span>
              </p>
              <ul className="mt-2 space-y-1">
                {result.probabilities.map((p) => (
                  <li key={p.label} className="grid grid-cols-[8rem_1fr_3rem] items-center gap-2 text-sm">
                    <span className="text-muted">{LABEL_NAMES[p.label]}</span>
                    <span className="h-2 overflow-hidden rounded-full bg-surface-2">
                      <span className="block h-full bg-accent" style={{ width: `${p.p * 100}%` }} />
                    </span>
                    <span className="text-right tabular-nums text-muted">{Math.round(p.p * 100)}%</span>
                  </li>
                ))}
              </ul>
              {result.probabilities[0]!.p - result.probabilities[1]!.p < 0.08 && (
                <p className="mt-2 text-sm text-warn">
                  Quase empate com {LABEL_NAMES[result.probabilities[1]!.label]}: o modelo está inseguro. Num atendimento real, uma mensagem assim iria para uma pessoa.
                </p>
              )}
              <p className="mt-2 text-xs text-muted">As porcentagens são a confiança do modelo, não a chance real de acerto: a avaliação mostra quanto ele erra.</p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
