"""Busca semântica na base de conhecimento.

Usa embeddings multilíngues (sentence-transformers) quando disponíveis; caso
contrário recorre a TF-IDF, para a demonstração rodar em qualquer máquina.
"""
import json
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
BASE = RAIZ / "dados" / "conhecimento.json"
MODELO_EMB = "paraphrase-multilingual-MiniLM-L12-v2"


def carregar_base():
    return json.loads(BASE.read_text(encoding="utf-8"))


class Buscador:
    def __init__(self, usar_embeddings: bool = True):
        self.trechos = carregar_base()["trechos"]
        self.textos = [t["texto"] for t in self.trechos]
        self.modo = "tfidf"
        if usar_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self.enc = SentenceTransformer(MODELO_EMB)
                self.mat = self.enc.encode(self.textos, normalize_embeddings=True)
                self.modo = "embeddings"
            except Exception:
                pass
        if self.modo == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vet = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
            self.mat = self.vet.fit_transform(self.textos)

    def buscar(self, consulta: str, classe: str | None = None, k: int = 2):
        if self.modo == "embeddings":
            sims = self.mat @ self.enc.encode([consulta], normalize_embeddings=True)[0]
        else:
            q = self.vet.transform([consulta])
            sims = (self.mat @ q.T).toarray().ravel()
        sims = np.asarray(sims, dtype=float)
        # só trechos gerais ou da classe visual prevista são candidatos
        if classe:
            for i, t in enumerate(self.trechos):
                if t["classe"] not in ("geral", classe):
                    sims[i] = -1.0
                elif t["classe"] == classe:
                    sims[i] += 0.1
        idx = np.argsort(-sims)[:k]
        return [(self.trechos[i]["texto"], float(sims[i])) for i in idx]
