import { useState } from "react";
import summary from "../results/summary.json";
import { LABEL_NAMES } from "./lib/label-names";
import { Playground } from "./Playground";


type Model = (typeof summary.models)[number];
const percent = (v: number) => `${(v * 100).toLocaleString("pt-BR", { maximumFractionDigits: 1 })}%`;

function latency(ms: number | null) {
  if (ms === null) return "–";
  if (ms === 0) return "< 1 ms";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toLocaleString("pt-BR", { maximumFractionDigits: 1 })} s`;
}

function Confusion({ model }: { model: Model }) {
  const rows = summary.labels;
  const columns = model.scores.invalid > 0 ? [...rows, "invalido"] : rows;
  const confusion = model.scores.confusion as Record<string, Record<string, number>>;
  return (
    <div className="overflow-x-auto">
      <table className="min-w-[30rem] border-collapse text-sm">
        <caption className="mb-2 text-left text-xs text-muted">Linhas: rótulo certo. Colunas: o que o modelo respondeu.</caption>
        <thead>
          <tr>
            <th className="p-1.5" />
            {columns.map((c) => (
              <th key={c} scope="col" className="p-1.5 text-xs font-medium text-muted">
                {LABEL_NAMES[c]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r}>
              <th scope="row" className="p-1.5 text-left text-xs font-medium text-muted">
                {LABEL_NAMES[r]}
              </th>
              {columns.map((c) => {
                const n = confusion[r]?.[c] ?? 0;
                const right = r === c;
                return (
                  <td key={c} className={`p-1.5 text-center tabular-nums ${n === 0 ? "text-muted/50" : right ? "bg-accent-soft font-semibold" : "bg-danger/10 font-semibold text-danger"}`}>
                    {n}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function App() {
  const models = summary.models;
  const [selected, setSelected] = useState(models[models.length - 1]!.id);
  const model = models.find((m) => m.id === selected) ?? models[0]!;

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
      <p className="text-sm font-medium text-accent">Classificador de intenções</p>
      <h1 className="mt-1 font-display text-3xl font-bold text-balance sm:text-4xl">Um modelo que roda no seu navegador contra um LLM ajustado e o Gemini</h1>
      <p className="mt-3 max-w-3xl text-muted">
        Cinco formas de dizer de que trata a mensagem de um cliente, medidas nas mesmas {summary.testSize} mensagens escritas à mão antes de qualquer treino. O modelo abaixo roda aqui, no seu
        aparelho: nada do que você digitar sai do navegador.
      </p>

      <Playground />

      <section aria-labelledby="results-title" className="mt-10">
        <h2 id="results-title" className="font-display text-xl font-semibold">
          Resultados no teste
        </h2>
        <div className="mt-3 overflow-x-auto rounded-xl border border-line bg-surface">
          <table className="w-full min-w-[40rem] border-collapse text-sm">
            <thead>
              <tr className="border-b border-line text-left text-xs text-muted">
                <th scope="col" className="p-3 font-medium">Modelo</th>
                <th scope="col" className="p-3 font-medium">Acertos</th>
                <th scope="col" className="p-3 font-medium">F1 médio</th>
                <th scope="col" className="p-3 font-medium">Graves detectadas</th>
                <th scope="col" className="p-3 font-medium">Tempo por mensagem</th>
                <th scope="col" className="p-3 font-medium">Onde roda</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => {
                const grave = m.scores.perLabel.reclamacao_grave;
                return (
                  <tr key={m.id} className={`border-b border-line last:border-0 ${m.id === selected ? "bg-accent-soft" : ""}`}>
                    <th scope="row" className="p-3 text-left font-medium">
                      <button type="button" onClick={() => setSelected(m.id)} className="text-left underline-offset-2 hover:underline" aria-pressed={m.id === selected}>
                        {m.name}
                      </button>
                      <span className="block text-xs font-normal text-muted">{m.how}</span>
                    </th>
                    <td className="p-3 tabular-nums">{percent(m.scores.accuracy)}</td>
                    <td className="p-3 tabular-nums">{m.scores.macroF1.toLocaleString("pt-BR", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
                    <td className="p-3 tabular-nums">
                      {Math.round(grave.recall * grave.support)} de {grave.support}
                    </td>
                    <td className="p-3 tabular-nums">{latency(m.latencyMs)}</td>
                    <td className="p-3">{m.runsOn}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted">Escolha um modelo para ver onde ele erra. F1 médio: a média, entre as cinco categorias, do equilíbrio entre não deixar passar e não acusar à toa.</p>

        <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <div className="min-w-0 rounded-xl border border-line bg-surface p-4">
            <h3 className="font-medium">{model.name}: confusões</h3>
            <div className="mt-3">
              <Confusion model={model} />
            </div>
          </div>
          <div className="min-w-0 rounded-xl border border-line bg-surface p-4">
            <h3 className="font-medium">
              {model.scores.errors.length} {model.scores.errors.length === 1 ? "erro" : "erros"}
            </h3>
            <ul className="mt-3 max-h-80 space-y-2 overflow-y-auto pr-1 text-sm">
              {model.scores.errors.map((e) => (
                <li key={e.text} className="rounded-lg bg-surface-2 p-2">
                  <span className="block break-words">“{e.text}”</span>
                  <span className="mt-1 block text-xs text-muted">
                    Certo: {LABEL_NAMES[e.label]} · respondeu: <span className="text-danger">{LABEL_NAMES[e.predicted] ?? e.predicted}</span>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <footer className="mt-12 border-t border-line pt-4 text-sm text-muted">
        Projeto de portfólio. Os dados de treino foram gerados por IA e o teste foi escrito à mão.{" "}
        <a className="text-accent underline" href="https://github.com/Flaviohorbach/intent-classifier-finetune">
          Código e relatório no GitHub
        </a>
        .
      </footer>
    </main>
  );
}
