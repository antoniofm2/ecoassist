"""Gera figuras do artigo em resultados/graficos/.

- arquitetura.pdf/png: diagrama do fluxo (não depende de dados)
- matriz_confusao_*.pdf, curva_treino.pdf, comparacao_modelos.pdf: exigem
  resultados/metricas_imagens.json (gerado por treinar_imagens.py)

Uso: python -m src.machine_learning.gerar_graficos
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES = Path(__file__).resolve().parents[2] / "resultados"
OUT = RES / "graficos"


def caixa(ax, x, y, w, h, texto, cor):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=cor, ec="#333", lw=1))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=7)


def seta(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1, color="#333"))


def arquitetura():
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, 10); ax.axis("off")
    caixa(ax, 0.2, 7.8, 3.6, 1.6, "Foto do\nresíduo", "#e8f0fe")
    caixa(ax, 6.2, 7.8, 3.6, 1.6, "Pergunta\n(português)", "#e8f0fe")
    caixa(ax, 0.2, 4.9, 3.6, 1.9, "Visão Comp.\nMobileNetV2\n(top-3)", "#d9f2d9")
    caixa(ax, 6.2, 4.9, 3.6, 1.9, "PLN\nintenção, material,\nbusca semântica", "#fff2cc")
    caixa(ax, 2.2, 2.0, 5.6, 1.9, "Agente de decisão\n(regras R1–R4 + confiança)", "#f8d7da")
    caixa(ax, 2.2, 0.0, 5.6, 1.2, "Resposta contextualizada", "#e8f0fe")
    seta(ax, (2, 7.8), (2, 6.85)); seta(ax, (8, 7.8), (8, 6.85))
    seta(ax, (2, 4.9), (3.8, 3.95)); seta(ax, (8, 4.9), (6.2, 3.95))
    seta(ax, (5, 2.0), (5, 1.25))
    fig.savefig(OUT / "arquitetura.pdf", bbox_inches="tight")
    fig.savefig(OUT / "arquitetura.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def matriz(m, classes, titulo, nome):
    m = np.array(m)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, m[i, j], ha="center", va="center", fontsize=6,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xlabel("Previsto", fontsize=7); ax.set_ylabel("Real", fontsize=7)
    ax.set_title(titulo, fontsize=8)
    fig.savefig(OUT / f"{nome}.pdf", bbox_inches="tight")
    plt.close(fig)


def metricas():
    arq = RES / "metricas_imagens.json"
    if not arq.exists():
        print("metricas_imagens.json não encontrado; gerados só o diagrama.")
        return
    d = json.loads(arq.read_text(encoding="utf-8"))
    matriz(d["mobilenetv2"]["matriz"], d["classes"], "MobileNetV2", "matriz_confusao_mobilenetv2")
    matriz(d["hog_svm"]["matriz"], d["classes"], "HOG + SVM", "matriz_confusao_hog_svm")

    h = d["historico"]
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.plot([e["epoca"] for e in h], [e["perda"] for e in h], label="perda (treino)")
    ax.plot([e["epoca"] for e in h], [e["acc_val"] for e in h], label="acurácia (val.)")
    ax.set_xlabel("Época", fontsize=8); ax.legend(fontsize=7)
    fig.savefig(OUT / "curva_treino.pdf", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    nomes = ["HOG + SVM", "MobileNetV2"]
    chaves = ["hog_svm", "mobilenetv2"]
    x = np.arange(2)
    ax.bar(x - 0.18, [d[k]["acuracia"] for k in chaves], 0.36, label="Acurácia")
    ax.bar(x + 0.18, [d[k]["f1_macro"] for k in chaves], 0.36, label="F1 macro")
    ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=8); ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    fig.savefig(OUT / "comparacao_modelos.pdf", bbox_inches="tight"); plt.close(fig)
    print("Gráficos de métricas gerados em", OUT)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    arquitetura()
    metricas()
