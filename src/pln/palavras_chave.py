"""Extração de palavras-chave e menção a materiais no texto da pergunta."""
import re
import unicodedata

# sinônimos em português -> classe do TrashNet
MATERIAIS = {
    "cardboard": ["papelao", "caixa de papelao", "caixa", "embalagem de papelao"],
    "glass": ["vidro", "garrafa de vidro", "pote de vidro", "caco"],
    "metal": ["metal", "lata", "aluminio", "aco", "ferro"],
    "paper": ["papel", "jornal", "revista", "folha", "caderno"],
    "plastic": ["plastico", "pet", "garrafa pet", "sacola", "garrafa", "copo plastico"],
    "trash": ["lixo", "rejeito", "fralda", "esponja"],
}
STOP = set("a o as os um uma de do da dos das em no na para por com e que é como posso pode isso esse essa este esta qual onde".split())


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def palavras_chave(texto: str):
    try:
        import spacy
        nlp = spacy.load("pt_core_news_sm")
        return [t.lemma_ for t in nlp(texto) if t.pos_ in {"NOUN", "PROPN", "ADJ"}]
    except Exception:
        toks = re.findall(r"\w+", texto.lower())
        return [t for t in toks if t not in STOP and len(t) > 2]


def material_mencionado(texto: str):
    """Retorna a classe citada no texto (maior sinônimo casado) ou None."""
    t = _norm(texto)
    melhor, tam = None, 0
    for classe, sins in MATERIAIS.items():
        for s in sins:
            if re.search(rf"\b{re.escape(s)}\b", t) and len(s) > tam:
                melhor, tam = classe, len(s)
    return melhor
