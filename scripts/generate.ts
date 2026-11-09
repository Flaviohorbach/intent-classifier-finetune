// Generates labeled training messages with Gemini. Resumable: each batch is
// appended to data/generated.jsonl, and a rerun skips the batches already there.
//   npm run data:generate            all batches
//   npm run data:generate -- --dry   print the batch plan only
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { generateText, Output } from "ai";
import { z } from "zod";
import { LABEL_GUIDE, LABELS, type Label } from "./labels";
import { createShopModel } from "./model";

const PER_BATCH = 40;
const OUT = "data/generated.jsonl";

const STYLES = [
  "WhatsApp, curta, sem pontuação e tudo minúsculo",
  "formal, educada, frases completas",
  "com erros de digitação e abreviações (vc, q, pq, tb, msm)",
  "irritada, com letras maiúsculas e pontos de exclamação",
  "gírias de jovem e emoji",
  "pessoa idosa, educada, explicando o contexto",
  "longa, contando a história antes de chegar ao ponto",
  "muito curta, 2 a 5 palavras",
  "irônica ou sarcástica",
  "mistura dois assuntos, mas um deles é claramente o principal",
];

const PRODUCTS = ["eletrônicos e celulares", "moda e calçados", "casa e cozinha", "beleza e perfumaria", "brinquedos e bebês", "esporte e suplementos", "móveis e decoração", "livros e papelaria"];

const LABEL_HINTS: Record<Label, string> = {
  troca: "troca, devolução, arrependimento, produto errado, com defeito, quebrado ou incompleto",
  atraso: "pedido atrasado, parado, extraviado, marcado como entregue sem ter chegado, devolvido ao remetente",
  duvida: "perguntas antes ou depois da compra sem nenhum problema: pagamento, frete, prazo, produto, garantia, loja, cupom",
  reclamacao_grave: "cobrança indevida ou duplicada, fraude, vazamento de dados, dano à saúde ou segurança, ameaça de Procon, advogado ou processo, cliente furioso com problema sério",
  outro: "saudações, agradecimentos, elogios, spam, vagas de emprego, parcerias, sugestões, pedidos sobre cadastro ou dados, texto sem sentido",
};

interface Batch {
  key: string;
  label: Label;
  style: string;
  product: string;
}

// Every label meets every style; products rotate so each batch gets a different one.
const batches: Batch[] = LABELS.flatMap((label, l) =>
  STYLES.map((style, s) => ({ key: `${label}|${s}`, label, style, product: PRODUCTS[(s + l * 3) % PRODUCTS.length]! })),
);

const done = new Set(existsSync(OUT) ? readFileSync(OUT, "utf8").split("\n").filter(Boolean).map((l) => (JSON.parse(l) as { batch: string }).batch) : []);
const todo = batches.filter((b) => !done.has(b.key));
console.log(`${batches.length} batches, ${done.size} done, ${todo.length} to go (${PER_BATCH} messages each)`);
if (process.argv.includes("--dry")) process.exit(0);

const model = createShopModel();
for (const batch of todo) {
  try {
    const { output } = await generateText({
      model,
      temperature: 1,
      maxRetries: 0,
      system: `Você escreve mensagens realistas que clientes brasileiros mandam para o atendimento de uma loja online. Siga este guia de rótulos:\n\n${LABEL_GUIDE}`,
      prompt: `Escreva ${PER_BATCH} mensagens diferentes entre si, todas do rótulo "${batch.label}" (${LABEL_HINTS[batch.label]}).
Estilo: ${batch.style}.
Produtos, quando houver: ${batch.product}.
Varie o assunto, o tamanho, o começo das frases e o vocabulário. Não numere, não repita estruturas, não use nomes reais de empresas.
Cada mensagem deve pertencer sem dúvida ao rótulo "${batch.label}" pelo guia.`,
      output: Output.object({ schema: z.object({ messages: z.array(z.string().min(2).max(500)).min(10) }) }),
    });
    for (const text of output.messages) appendFileSync(OUT, JSON.stringify({ text: text.trim(), label: batch.label, batch: batch.key }) + "\n");
    process.stdout.write(`${batch.key}: ${output.messages.length}\n`);
  } catch (error) {
    console.error(`\n${batch.key}: ${(error as Error).message.slice(0, 160)}\nStopped. Run again later to resume.`);
    process.exitCode = 1;
    break;
  }
}
