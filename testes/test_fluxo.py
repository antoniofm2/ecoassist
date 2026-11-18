import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pln.gerar_intencoes import gerar
from src.pln import intencao
from src.pln.palavras_chave import material_mencionado
from src.pipeline import EcoAssist


def setup_module(_):
    gerar(intencao.CSV)
    intencao.treinar(avaliar=False)


def test_intencao():
    assert intencao.prever("Dá para reciclar essa lata?")[0] == "reciclavel"
    assert intencao.prever("Qual a cor da lixeira para isso?")[0] == "cor_lixeira"


def test_material_texto():
    assert material_mencionado("Essa garrafa de vidro vai onde?") == "glass"
    assert material_mencionado("Como descarto isso?") is None


def test_baixa_confianca_pede_nova_foto():
    r = EcoAssist().responder(None, "Como descarto isso?", top_visual=[("glass", .4), ("plastic", .35)])
    assert r["decisao"].regra == "R1_baixa_confianca"


def test_texto_resolve_incerteza():
    r = EcoAssist().responder(None, "Essa lata é reciclável?", top_visual=[("glass", .4), ("metal", .3)])
    assert r["decisao"].classe == "metal" and r["decisao"].regra == "R2_texto_resolve_incerteza"


def test_conflito_imagem_prevalece():
    r = EcoAssist().responder(None, "Como descarto esse papel?", top_visual=[("plastic", .95)])
    assert r["decisao"].classe == "plastic" and r["decisao"].avisos


def test_cor_lixeira():
    r = EcoAssist().responder(None, "Qual a cor da lixeira?", top_visual=[("paper", .9)])
    assert "azul" in r["decisao"].resposta
