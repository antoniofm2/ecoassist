"""Agente de decisão baseado em utilidade e regras.

Combina a saída da Visão Computacional (top-k com probabilidades) com a do PLN
(intenção, material citado, trechos recuperados) e produz a resposta final.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

BASE = json.loads((Path(__file__).resolve().parents[2] / "dados" / "conhecimento.json")
                  .read_text(encoding="utf-8"))["classes"]

LIMIAR_CONFIANCA = 0.60
LIMIAR_INTENCAO = 0.40

CAMPO_POR_INTENCAO = {
    "como_descartar": "descarte", "cuidados": "cuidados",
    "curiosidade": "curiosidade",
}


@dataclass
class Decisao:
    classe: str | None
    resposta: str
    regra: str
    avisos: list = field(default_factory=list)


def decidir(top_visual, intencao, p_intencao, material_texto=None, trechos=None) -> Decisao:
    """top_visual: [(classe, prob)]; material_texto: classe citada no texto ou None."""
    trechos = trechos or []
    classe, p = top_visual[0]
    avisos = []

    # R1: baixa confiança visual e nada no texto -> pedir nova foto
    if p < LIMIAR_CONFIANCA and not material_texto:
        alt = ", ".join(f"{BASE[c]['nome']} ({q:.0%})" for c, q in top_visual[:2])
        return Decisao(None, f"Não tenho certeza do material (candidatos: {alt}). "
                             "Envie outra foto com melhor iluminação ou informe o material na pergunta.",
                       "R1_baixa_confianca")

    regra = "R4_visual"
    # R2: baixa confiança visual, mas o texto cita material -> usar o texto
    if p < LIMIAR_CONFIANCA and material_texto:
        classe, regra = material_texto, "R2_texto_resolve_incerteza"
        avisos.append("Confiança visual baixa; usei o material citado na pergunta.")
    # R3: conflito entre texto e imagem confiante -> prevalece a imagem, com aviso
    elif material_texto and material_texto != classe:
        regra = "R3_conflito_imagem_prevalece"
        avisos.append(f"Você citou {BASE[material_texto]['nome']}, mas a imagem indica "
                      f"{BASE[classe]['nome']} ({p:.0%}). Considerei a imagem.")

    info = BASE[classe]
    nome = info["nome"]
    if p_intencao < LIMIAR_INTENCAO:
        # R5: intenção incerta -> resposta geral em vez de assumir uma intenção
        intencao = "resumo"
        regra = regra + "+R5_intencao_incerta"
        avisos.append("Não tenho certeza do que você quer saber; dei um resumo.")

    if intencao == "resumo":
        rec = "reciclável" if info["reciclavel"] else "não reciclável"
        corpo = (f"Material: {nome} ({rec}). Lixeira {info['lixeira']}. "
                 f"{info['descarte']}")
    elif intencao == "identificar_material":
        corpo = f"Parece ser {nome}."
    elif intencao == "reciclavel":
        corpo = (f"Sim, {nome} é reciclável." if info["reciclavel"]
                 else f"{nome.capitalize()} não é reciclável; vai para o lixo comum.")
    elif intencao == "cor_lixeira":
        corpo = f"Para {nome}, use a lixeira {info['lixeira']}."
    else:
        corpo = info[CAMPO_POR_INTENCAO.get(intencao, "descarte")]

    resposta = f"{corpo} (material: {nome}, lixeira {info['lixeira']})"
    if trechos:
        resposta += "\n\nReferência: " + trechos[0][0]
    return Decisao(classe, resposta, regra, avisos)
