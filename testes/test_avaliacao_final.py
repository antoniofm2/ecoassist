import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import avaliacao_final as af
from src.pipeline import EcoAssist


class VisualFalso:
    """Devolve sempre a mesma previsão, sem carregar o modelo."""
    def __init__(self, top):
        self.top = top

    def prever(self, imagem, k=3):
        return self.top


def _pares(tmp_path, linhas):
    Image.new("RGB", (32, 32), "white").save(tmp_path / "a.jpg")
    return pd.DataFrame(linhas, columns=af.OBRIGATORIAS)


def test_acertos_automaticos_e_resposta_pendente(tmp_path):
    pares = _pares(tmp_path, [
        ["a.jpg", "Qual a cor da lixeira?", "paper", "cor_lixeira", "azul"],
        ["a.jpg", "Dá para reciclar isso?", "glass", "reciclavel", "sim"]])
    sis = EcoAssist(VisualFalso([("paper", .9)]))
    df = af.avaliar(pares, tmp_path, sis)
    assert list(df.columns) == af.COLUNAS_SAIDA
    assert df.acerto_classe.tolist() == [1, 0]
    assert df.acerto_intencao.tolist() == [1, 1]
    assert (df.resposta_correta == "").all()
    res = af.resumir(df)
    assert res["acerto_resposta"] is None and not res["avaliacao_completa"]


def test_resposta_vem_da_marcacao_do_grupo(tmp_path):
    pares = _pares(tmp_path, [["a.jpg", "Qual a cor da lixeira?", "paper", "cor_lixeira", "azul"]] * 4)
    df = af.avaliar(pares, tmp_path, EcoAssist(VisualFalso([("paper", .9)])))
    df["resposta_correta"] = [1, 1, 0, 1]
    res = af.resumir(df)
    assert res["acerto_resposta"] == pytest.approx(0.75) and res["avaliacao_completa"]


def test_validacao_e_foto_ausente(tmp_path):
    ruim = _pares(tmp_path, [["a.jpg", "oi", "papel", "cor_lixeira", "x"]])
    with pytest.raises(ValueError):
        af.avaliar(ruim, tmp_path, EcoAssist(VisualFalso([("paper", .9)])))
    sem_foto = _pares(tmp_path, [["nao_existe.jpg", "oi", "paper", "cor_lixeira", "x"]])
    with pytest.raises(FileNotFoundError):
        af.avaliar(sem_foto, tmp_path, EcoAssist(VisualFalso([("paper", .9)])))
