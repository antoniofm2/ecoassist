"""Avaliação ponta a ponta: foto (conjunto de TESTE do TrashNet) + pergunta livre.

Monta 8 imagens de teste por classe (48 pares), cada uma com uma pergunta de
dados/perguntas_livres.csv (intenções alternadas). Um par é correto quando:
(a) a classe final da decisão é a classe real da imagem e
(b) a intenção prevista é a intenção esperada da pergunta.
Também registra a regra de decisão acionada e a acurácia por etapa.
Uso: python -m src.avaliacao_ponta_a_ponta
"""
import json
import random
from collections import Counter

import pandas as pd
from PIL import Image
from torchvision import datasets

from src.machine_learning.treinar_imagens import dividir
from src.pipeline import EcoAssist
from src.pln.busca_semantica import Buscador
from src.visao_computacional.classificador import ClassificadorVisual

RAIZ = __import__("pathlib").Path(__file__).resolve().parents[1]
POR_CLASSE = 8


def main():
    rng = random.Random(42)
    base = datasets.ImageFolder(RAIZ / "dados" / "trashnet")
    _, _, te = dividir(base)
    por_classe = {}
    for i in te:
        caminho, y = base.samples[i]
        por_classe.setdefault(base.classes[y], []).append(caminho)
    perguntas = pd.read_csv(RAIZ / "dados" / "perguntas_livres.csv")
    por_int = {k: g.texto.tolist() for k, g in perguntas.groupby("intencao")}
    intents = sorted(por_int)

    sistema = EcoAssist(ClassificadorVisual(), Buscador())
    linhas, k = [], 0
    for classe, caminhos in sorted(por_classe.items()):
        for caminho in rng.sample(caminhos, min(POR_CLASSE, len(caminhos))):
            it = intents[k % len(intents)]
            q = rng.choice(por_int[it]); k += 1
            r = sistema.responder(Image.open(caminho), q)
            d = r["decisao"]
            linhas.append({
                "imagem": str(caminho.replace("\\", "/").split("trashnet/")[-1]),
                "classe_real": classe, "visual": r["top_visual"][0][0],
                "conf": round(r["top_visual"][0][1], 3), "pergunta": q,
                "intencao_esperada": it, "intencao_prevista": r["intencao"][0],
                "classe_final": d.classe, "regra": d.regra,
                "ok_visual": r["top_visual"][0][0] == classe,
                "ok_intencao": r["intencao"][0] == it,
                "ok_final": d.classe == classe and r["intencao"][0] == it,
                "ok_classe_final": d.classe == classe,
            })
    df = pd.DataFrame(linhas)
    out = RAIZ / "resultados"
    df.to_csv(out / "avaliacao_ponta_a_ponta.csv", index=False, encoding="utf-8")
    resumo = {
        "n_pares": len(df), "busca_semantica_modo": sistema.buscador.modo,
        "acuracia_visual": df.ok_visual.mean(), "acuracia_intencao": df.ok_intencao.mean(),
        "acuracia_final": df.ok_final.mean(), "acuracia_classe_final": df.ok_classe_final.mean(),
        "regras": dict(Counter(df.regra)),
    }
    (out / "metricas_ponta_a_ponta.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
