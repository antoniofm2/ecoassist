"""Gera o dataset de intenções (dados/intencoes.csv) a partir de modelos de frases.

O dataset é sintético (templates x objetos x variações) e deve ser descrito
como tal no artigo. Execução: python -m src.pln.gerar_intencoes
"""
import csv
import itertools
import random
from pathlib import Path

SEED = 42
OBJETOS = ["isso", "esse item", "este material", "essa embalagem", "esse resíduo",
           "essa garrafa", "essa caixa", "essa lata", "esse objeto", "esse lixo"]

TEMPLATES = {
    "como_descartar": [
        "Como descarto {o}?", "Como devo descartar {o}?", "Onde jogo {o}?",
        "Qual o jeito certo de descartar {o}?", "O que faço com {o}?",
        "Como me livrar de {o} corretamente?", "Onde coloco {o}?"],
    "identificar_material": [
        "Do que é feito {o}?", "Qual é o material de {o}?", "O que é {o}?",
        "Que tipo de material é {o}?", "Você sabe identificar {o}?",
        "Qual a classificação de {o}?"],
    "reciclavel": [
        "{o} é reciclável?", "Dá para reciclar {o}?", "Posso reciclar {o}?",
        "{o} pode ser reciclado?", "A cooperativa aceita {o}?",
        "Isso pode ir para a reciclagem, {o}?"],
    "cor_lixeira": [
        "Qual a cor da lixeira para {o}?", "Em qual lixeira coloco {o}?",
        "Qual lixeira devo usar para {o}?", "Vai na lixeira de qual cor, {o}?",
        "Qual é a cor certa da lixeira de {o}?"],
    "cuidados": [
        "Preciso lavar {o} antes?", "Tem algum cuidado com {o}?",
        "Existe risco ao descartar {o}?", "Como preparo {o} para descarte?",
        "Preciso tomar cuidado com {o}?", "{o} precisa estar limpo para descartar?"],
    "curiosidade": [
        "Me conte uma curiosidade sobre {o}", "Por que reciclar {o} é importante?",
        "Qual o impacto ambiental de {o}?", "O que acontece quando se recicla {o}?",
        "Qual a vantagem de reciclar {o}?"],
}


def gerar(caminho: Path, por_intencao: int = 60) -> int:
    rng = random.Random(SEED)
    linhas = []
    for intencao, modelos in TEMPLATES.items():
        combos = list(itertools.product(modelos, OBJETOS))
        rng.shuffle(combos)
        for modelo, obj in combos[:por_intencao]:
            frase = modelo.format(o=obj)
            if rng.random() < 0.2:
                frase = frase.lower().replace("?", "")
            linhas.append((frase, intencao))
    rng.shuffle(linhas)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["texto", "intencao"])
        w.writerows(linhas)
    return len(linhas)


if __name__ == "__main__":
    destino = Path(__file__).resolve().parents[2] / "dados" / "intencoes.csv"
    print(f"{gerar(destino)} frases gravadas em {destino}")
