# Conceitos do módulo 10, explicados pelo projeto

Guia de estudo em português: como saber se um modelo é bom e quando vale a pena treinar um.

## 1. Avaliação primeiro

Antes de treinar qualquer coisa, decida **como vai medir**. Aqui isso virou três regras:

- **O teste veio antes do treino.** As 120 mensagens de `data/test.jsonl` foram escritas e commitadas antes de existir qualquer dado de treino. O histórico do Git prova a ordem.
- **O teste nunca é usado para ajustar nada.** A regularização da camada (`l2`) foi escolhida pela validação (`data/val.jsonl`). O teste roda uma vez por modelo, no fim.
- **O teste é diferente do treino.** O treino é sintético, gerado pelo Gemini; o teste foi escrito à mão. Se os dois viessem do mesmo gerador, o modelo seria bom em imitar o gerador, não em entender clientes.

**Vazamento** é quando o modelo vê no treino algo do teste. `scripts/check-data.ts` compara cada mensagem gerada com todas as do teste (trigramas de caracteres) e remove as parecidas demais. Saíram 4, como "bom dia" e "oi".

## 2. As métricas

Veja `scripts/metrics.ts`:

- **Acurácia:** a fração de acertos. Simples, mas esconde em que o modelo erra.
- **Precisão** de um rótulo: das vezes que o modelo disse "grave", quantas eram graves.
- **Revocação** (*recall*): das mensagens graves, quantas o modelo pegou. Para reclamação grave, é a métrica que importa, porque deixar passar custa caro.
- **F1:** o equilíbrio entre precisão e revocação. O **F1 médio** (*macro-F1*) faz a média entre os rótulos, para um rótulo raro pesar tanto quanto um comum.
- **Matriz de confusão:** a tabela de "era isso, respondeu aquilo". Ela mostra, por exemplo, que o modelo do navegador confunde dano à saúde com troca.

## 3. Prompt, ajuste fino ou camada treinada?

| | Como aprende | Custo para usar | Quando escolher |
| --- | --- | --- | --- |
| **Prompt num modelo grande** | Não aprende: segue o guia no prompt | Paga por chamada, depende de rede | Poucos dados, regras que mudam, precisa ser bom já |
| **Ajuste fino (LoRA)** | Ajusta uma parte pequena dos pesos com exemplos | Precisa de GPU para treinar e para rodar | Muitos exemplos, tarefa estável, custo ou privacidade importam |
| **Camada sobre embeddings** | Só a última camada aprende; o resto fica congelado | Quase zero: roda no navegador | Classificação simples, precisa ser barato, rápido e privado |

**O que o projeto mostrou:** o Qwen de 1,5B só com prompt acertou 42,5% e nem seguia o formato. Depois de 7 minutos de LoRA, foi a 95,0%, perto do Gemini (98,3%). O modelo do navegador ficou em 82,5%. Ajustar um modelo pequeno não costuma ganhar do grande em qualidade, mas chega perto, e ganha em custo, velocidade e privacidade.

## 4. LoRA em uma frase

Em vez de mudar os 1,5 bilhão de pesos do Qwen, o LoRA congela todos e acrescenta, ao lado de algumas camadas, duas matrizes pequenas cujo produto vira a "correção". Só elas são treinadas, cerca de 1% dos parâmetros. Por isso cabe numa GPU gratuita, e o resultado é um arquivo de dezenas de MB, não de 3 GB.

No notebook, a perda conta só os tokens do rótulo (`labels = -100` no prompt). Sem isso, o modelo gastaria o treino aprendendo a repetir mensagens de clientes.

## 5. Linear probe

O multilingual-e5-small transforma uma frase em 384 números (um *embedding*), de forma que frases parecidas ficam perto. Por cima, uma **regressão logística** aprende cinco pesos por dimensão, um conjunto para cada rótulo. São 5 × 384 números e 5 vieses: o arquivo `public/model/head.json` tem 16 KB.

A regressão logística foi escrita à mão em `src/lib/head.ts`: *softmax* e descida do gradiente, sem biblioteca. Ela treina 1.798 exemplos em segundos.

## 6. Dados sintéticos: o que ganham e o que custam

Gerar dados é rápido e barato, mas o gerador tem vieses e o modelo herda todos. Aqui, o Gemini escreveu "reclamação grave" quase sempre sobre dinheiro (72 casos) e raramente sobre saúde (17), e rotulou 3 casos de saúde como troca. Resultado: o modelo do navegador deixou passar uma alergia e um corte em criança.

A lição é **ler amostras dos dados antes de treinar**, e contar os casos que importam, não só o total por rótulo.

## 7. O mesmo modelo, números diferentes

O modelo quantizado em 8 bits calcula a escala dos números por lote. Um texto processado junto com outros 31 sai um pouco diferente do mesmo texto processado sozinho. No treino, os textos iam em lotes; no navegador, um por vez. A avaliação, então, media um modelo um pouco diferente do que o visitante usa.

A regra geral é **avaliar o modelo exatamente como ele roda em produção**: mesmo arquivo, mesma quantização, mesmo pré-processamento, mesmo tamanho de lote.

## 8. Limites deste projeto, para dizer em entrevista

- **120 mensagens de teste** escritas por uma pessoa só: dá para ver diferenças grandes, não para ordenar modelos separados por poucos pontos.
- **Uma rodada por modelo**, sem intervalo de confiança.
- **O Gemini classificou em lotes de 10** para caber na cota gratuita, e isso pode mudar respostas.

## Exercícios

1. **Corrija o ponto cego.** Gere 40 mensagens de reclamação grave só sobre saúde e segurança, retreine (`npm run train:probe`) e meça de novo. A revocação de "grave" sobe? A acurácia geral muda?
2. **Limiar de confiança.** Se o modelo do navegador só responder quando a maior probabilidade passar de 50%, e mandar o resto para uma pessoa, quantas mensagens ele resolve e com que acurácia?
3. **Quanto dado basta?** Treine a camada com 10%, 25%, 50% e 100% do treino e faça um gráfico da acurácia no teste.
4. **Matriz de confusão.** Qual par de rótulos o modelo do navegador mais confunde? Isso é culpa do modelo ou de um rótulo mal definido no guia?
