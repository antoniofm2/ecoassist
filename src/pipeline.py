"""Fluxo integrado: imagem + pergunta -> resposta."""
from src.decisao.agente import decidir
from src.pln import intencao as pln_intencao
from src.pln.palavras_chave import material_mencionado, palavras_chave


class EcoAssist:
    def __init__(self, classificador=None, buscador=None):
        self.visual = classificador
        self.intencao = pln_intencao.carregar()
        self.buscador = buscador

    def responder(self, imagem, pergunta: str, top_visual=None):
        top = top_visual or self.visual.prever(imagem)
        intent, p_int = pln_intencao.prever(pergunta, self.intencao)
        material = material_mencionado(pergunta)
        trechos = self.buscador.buscar(pergunta, classe=top[0][0]) if self.buscador else []
        d = decidir(top, intent, p_int, material, trechos)
        return {"decisao": d, "top_visual": top, "intencao": (intent, p_int),
                "material_texto": material, "palavras_chave": palavras_chave(pergunta),
                "trechos": trechos}
