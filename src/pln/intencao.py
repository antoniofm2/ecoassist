"""Reconhecimento de intenção: TF-IDF (n-gramas) + Regressão Logística."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

RAIZ = Path(__file__).resolve().parents[2]
CSV = RAIZ / "dados" / "intencoes.csv"
MODELO = RAIZ / "resultados" / "modelo_intencao.joblib"


def treinar(avaliar: bool = True):
    df = pd.read_csv(CSV)
    X_tr, X_te, y_tr, y_te = train_test_split(
        df.texto, df.intencao, test_size=0.2, stratify=df.intencao, random_state=42)
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True),
        LogisticRegression(max_iter=1000))
    pipe.fit(X_tr, y_tr)
    relatorio = classification_report(y_te, pipe.predict(X_te), output_dict=True)
    if avaliar:
        print(classification_report(y_te, pipe.predict(X_te)))
    MODELO.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELO)
    return pipe, relatorio


def carregar():
    if not MODELO.exists():
        return treinar(avaliar=False)[0]
    return joblib.load(MODELO)


def prever(texto: str, modelo=None):
    """Retorna (intenção, probabilidade)."""
    modelo = modelo or carregar()
    proba = modelo.predict_proba([texto])[0]
    i = int(proba.argmax())
    return modelo.classes_[i], float(proba[i])


if __name__ == "__main__":
    treinar()
