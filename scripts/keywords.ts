import type { Label } from "./labels";

// The keyword baseline from project 9, unchanged: free, instant, and the floor every model has to beat.
const KEYWORDS: [Label, RegExp][] = [
  ["reclamacao_grave", /procon|advogad|processar|processo|justi[cç]a|reclame ?aqui|golpe|fraude|cobra(do|ram|n[cç]a)[^.]{0,30}(duas|2) vezes|cobran[cç]a indevida|estorno|absurdo|vergonha/i],
  ["troca", /troc(a|ar)|devolv|devolu[cç]|defeito|quebrad|errad|n[aã]o serv|tamanho/i],
  ["atraso", /atras|n[aã]o (chegou|recebi)|ainda n[aã]o|cad[eê] (o|meu)|parad[oa]|demor/i],
  ["duvida", /\?|como |qual |quais |quando |posso |aceita|parcel|pix|boleto|rastre|prazo/i],
];

export const classifyWithKeywords = (text: string): Label => KEYWORDS.find(([, pattern]) => pattern.test(text))?.[0] ?? "outro";
