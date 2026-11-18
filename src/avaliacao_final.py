"""Avaliação final com pares (foto de celular, pergunta) escritos pelo grupo.

Passa cada par pelo mesmo fluxo do Streamlit (EcoAssist.responder) e grava um CSV
de resultados com a coluna `resposta_correta` em branco, para o grupo marcar
(1 = correta, 0 = incorreta). Acertos de classe e de intenção são automáticos;
o acerto da resposta vem exclusivamente da marcação do grupo.

Uso:
    python -m src.avaliacao_final            # gera dados/avaliacao_final/resultados_final.csv
    python -m src.avaliacao_final --resumo   # recalcula métricas depois da marcação
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "dados" / "avaliacao_final"
CLASSES = {"cardboard", "glass", "metal", "paper", "plastic", "trash"}
INTENCOES = {"como_descartar", "identificar_material", "reciclavel",
             "cor_lixeira", "cuidados", "curiosidade"}
OBRIGATORIAS = ["foto", "pergunta", "classe_esperada", "intencao_esperada", "resposta_esperada"]
COLUNAS_SAIDA = OBRIGATORIAS + [
    "classe_prevista", "confianca_visual", "intencao_prevista", "confianca_intencao",
    "regra", "resposta_app", "fonte", "acerto_classe", "acerto_intencao", "resposta_correta"]


def validar(pares: pd.DataFrame):
    faltam = [c for c in OBRIGATORIAS if c not in pares.columns]
    if faltam:
        raise ValueError(f"Colunas ausentes em pares.csv: {faltam}")
    for i, r in pares.iterrows():
        if r.classe_esperada not in CLASSES:
            raise ValueError(f"Linha {i + 2}: classe_esperada inválida: {r.classe_esperada!r}")
        if r.intencao_esperada not in INTENCOES:
            raise ValueError(f"Linha {i + 2}: intencao_esperada inválida: {r.intencao_esperada!r}")


def avaliar(pares: pd.DataFrame, fotos: Path, sistema) -> pd.DataFrame:
    """Executa o fluxo em cada par. `sistema` é um EcoAssist (ou equivalente)."""
    from PIL import Image
    validar(pares)
    linhas = []
    for _, r in pares.iterrows():
        caminho = fotos / str(r.foto)
        if not caminho.exists():
            raise FileNotFoundError(f"Foto não encontrada: {caminho}")
        out = sistema.responder(Image.open(caminho), str(r.pergunta))
        d = out["decisao"]
        # a classe prevista é a que o sistema usou na resposta; se pediu nova foto (R1), fica vazia
        classe = d.classe or ""
        linhas.append({
            **{c: r[c] for c in OBRIGATORIAS},
            "classe_prevista": classe,
            "confianca_visual": round(out["top_visual"][0][1], 3),
            "intencao_prevista": out["intencao"][0],
            "confianca_intencao": round(out["intencao"][1], 3),
            "regra": d.regra,
            "resposta_app": d.resposta.replace("\n", " "),
            "fonte": out["trechos"][0][0] if out["trechos"] else "",
            "acerto_classe": int(classe == r.classe_esperada),
            "acerto_intencao": int(out["intencao"][0] == r.intencao_esperada),
            "resposta_correta": "",
        })
    return pd.DataFrame(linhas, columns=COLUNAS_SAIDA)


def resumir(df: pd.DataFrame) -> dict:
    n = len(df)
    marc = pd.to_numeric(df["resposta_correta"], errors="coerce")
    marcadas = int(marc.notna().sum())
    res = {
        "n_pares": n,
        "acerto_classe": float(df.acerto_classe.mean()) if n else None,
        "acerto_intencao": float(df.acerto_intencao.mean()) if n else None,
        "respostas_marcadas": marcadas,
        "acerto_resposta": float(marc.mean()) if marcadas else None,
    }
    res["avaliacao_completa"] = marcadas == n and n > 0
    return res


def imprimir(res: dict):
    f = lambda v: "pendente" if v is None else f"{v:.1%}"
    print(f"Pares: {res['n_pares']}")
    print(f"Acerto de classe:    {f(res['acerto_classe'])}")
    print(f"Acerto de intenção:  {f(res['acerto_intencao'])}")
    print(f"Acerto da resposta:  {f(res['acerto_resposta'])} "
          f"({res['respostas_marcadas']}/{res['n_pares']} marcadas pelo grupo)")
    if not res["avaliacao_completa"]:
        print("Marque 1 (correta) ou 0 (incorreta) em `resposta_correta` e rode com --resumo.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pares", type=Path, default=PASTA / "pares.csv")
    ap.add_argument("--fotos", type=Path, default=PASTA / "fotos")
    ap.add_argument("--saida", type=Path, default=PASTA / "resultados_final.csv")
    ap.add_argument("--resumo", action="store_true", help="só recalcula métricas a partir do CSV de resultados")
    ap.add_argument("--forcar", action="store_true", help="sobrescreve resultados que já têm marcações")
    a = ap.parse_args(argv)

    if a.resumo:
        if not a.saida.exists():
            sys.exit(f"{a.saida} não existe. Rode sem --resumo primeiro.")
        imprimir(resumir(pd.read_csv(a.saida, encoding="utf-8")))
        return

    pares = pd.read_csv(a.pares, encoding="utf-8")
    if pares.empty:
        sys.exit(f"{a.pares} está vazio. Preencha os pares antes de rodar (veja LEIAME.md).")
    if a.saida.exists() and not a.forcar:
        antigo = pd.read_csv(a.saida, encoding="utf-8")
        if pd.to_numeric(antigo["resposta_correta"], errors="coerce").notna().any():
            sys.exit(f"{a.saida} já tem respostas marcadas. Use --forcar para sobrescrever "
                     "(as marcações serão perdidas) ou --saida para outro arquivo.")

    from src.pipeline import EcoAssist
    from src.pln.busca_semantica import Buscador
    from src.visao_computacional.classificador import ClassificadorVisual
    sistema = EcoAssist(ClassificadorVisual(), Buscador())
    df = avaliar(pares, a.fotos, sistema)
    a.saida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.saida, index=False, encoding="utf-8")
    print(f"Resultados gravados em {a.saida}")
    imprimir(resumir(df))


if __name__ == "__main__":
    main()
