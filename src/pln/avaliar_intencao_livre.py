"""Avalia o classificador de intenção em perguntas escritas livremente
(dados/perguntas_livres.csv), fora dos modelos de frase do treino.

Rascunho inicial escrito pelo assistente de IA; a equipe deve revisar/ampliar
as frases e rótulos antes de citar os números no artigo.
Uso: python -m src.pln.avaliar_intencao_livre
"""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.pln import intencao

CSV = intencao.RAIZ / "dados" / "perguntas_livres.csv"


def main():
    modelo = intencao.carregar()
    df = pd.read_csv(CSV)
    pred = modelo.predict(df.texto)
    rel = classification_report(df.intencao, pred, output_dict=True, zero_division=0)
    print(classification_report(df.intencao, pred, zero_division=0))
    erros = df[pred != df.intencao].assign(previsto=pred[pred != df.intencao])
    print("Erros:\n", erros.to_string(index=False))
    out = {"n": len(df), "acuracia": accuracy_score(df.intencao, pred),
           "f1_macro": f1_score(df.intencao, pred, average="macro"), "relatorio": rel}
    (intencao.RAIZ / "resultados" / "metricas_intencao_livre.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
